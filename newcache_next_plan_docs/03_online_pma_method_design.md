# 03. Online PMA-Cache 方法设计

## 0. 这一章解决什么问题

你现在最关键的断点是：PMA 在 E4 里表现出了很强的 oracle 上限，但 oracle 不能直接用于真实推理。因为 DINO / LPIPS drift 需要先得到相邻 call 的 `xhat`，而 `xhat` 是完整 denoiser forward 之后才有的。如果为了判断是否 refresh 先跑完整 denoiser，就没有加速意义。

因此，下一阶段的主方法必须满足三个条件：

1. **在线**：当前 call 到来时，就能决定 refresh / skip。
2. **便宜**：不能在每个 call 运行完整 denoiser、DINO、LPIPS。
3. **感知相关**：score 要尽量预测 clean-image perceptual drift，而不是只预测 noisy feature drift。

推荐主方法暂名：`PMA-Cache`，全称 `Perceptual-Manifold-Aware Cache for x-prediction Pixel Diffusion`。中文可写作“感知流形感知的 x-pred cache”。

---

## 1. 方法总直觉

x-prediction 模型每次 denoiser 会预测一张 clean image estimate：

```text
xhat_t = f_theta(x_t, t, c)
v_t    = (xhat_t - x_t) / (1 - t)
```

这和普通 velocity-pred / noise-pred 模型不同。x-pred 模型的输出本身更接近最终图像，所以我们可以问：

> 相邻两个 call 的 clean-image prediction 在人眼/语义上有没有明显变化？

如果变化很小，就可以 skip；如果变化很大，就 refresh。

但是真实推理时我们不能提前知道 `xhat_t`，所以需要一个便宜代理：

```text
cheap feature  --->  predict perceptual drift  --->  cache refresh decision
```

SeaCache 已经说明，经过频域滤波后的输入特征比 raw feature 更适合表示“内容相关变化”；PixelGen 则说明 LPIPS / DINO 这样的感知空间可以把模型训练引向 perceptual manifold。你的贡献是把这两者用于 **推理阶段的 cache gate**，并且强调这是 **x-prediction 模型特有的 clean-image trajectory**。

---

## 2. 感知流形的工程化定义

“感知流形”听起来抽象，但论文里必须定义得足够具体。建议这样写：

> 我们不直接在 RGB 像素或 noisy latent/input 上度量变化，而是在由局部感知特征、全局语义特征和频谱信号共同诱导的空间中度量 denoising trajectory 的变化。

工程上，你可以把 oracle perceptual drift 定义为：

```text
D_pma(c-1, c) = w_sea   * Norm(D_sea)
              + w_dino  * Norm(D_dino)
              + w_lpips * Norm(D_lpips)
```

其中：

- `D_sea`：SEA-filtered input proxy 的相对距离，负责频谱/内容信号；
- `D_dino`：DINO clean-image feature drift，负责语义结构；
- `D_lpips`：LPIPS clean-image drift，负责局部纹理和人眼感知差异。

注意：在线版本不直接计算 `D_dino` 和 `D_lpips`，它们只作为 calibration label 或分析指标。

---

## 3. Online PMA 的核心公式

在线方法可以写成两层。

### 3.1 感知 drift 预测器

给每个 call transition 提取便宜特征：

```text
z_c = [
    log1p(SEA_c),
    Raw_c,
    t_c,
    dt_c,
    call_index / num_calls,
    is_predictor,
    is_corrector,
    transition_kind,
    proxy_norm,
    proxy_delta_p95,
    cache_age,
    last_refresh_interval,
]
```

预测：

```text
s_c = g_phi(z_c)
```

其中 `g_phi` 可以是：

- heuristic：无学习，只用 stage-kind normalized SEA；
- ridge regression：线性加权；
- uncertainty-aware ridge：预测均值 + 风险补偿。

主论文推荐用 ridge 或 quantile ridge。这样足够简单，不会被 reviewer 质疑“又训练了一个黑盒模型”。

### 3.2 Accumulated risk gate

和 TeaCache / SeaCache 类似，用累积风险触发 refresh：

```text
A_c = A_{c-1} + s_c

if forced_refresh(c) or A_c >= delta or cache_age >= max_skip or uncertainty_c >= tau:
    refresh
    A_c = 0
else:
    skip / reuse cached output
```

这里 `delta` 用 calibration set 调到目标 RR。比如 target RR=0.30，就找一个 delta 让 calibration 上实际 refresh ratio 约为 0.30。

---

## 4. Stage-kind normalization：非常重要

你的 E2 已经显示 predictor→corrector 和 corrector→predictor 的统计分布不同，而且第一个 transition 有巨大 spike。因此直接把所有 score 放在一起比大小，会导致 early spike 或某一种 transition 支配整个 schedule。

建议对 score 做 stage-kind normalization：

```text
s_norm = (s_raw - median[bin]) / (mad[bin] + eps)
s_norm = softplus(s_norm)
```

其中 `bin` 可以由下面几项组成：

```text
bin = (stage_bin, transition_kind)
stage_bin ∈ {early, middle, late}
transition_kind ∈ {predictor_to_corrector, corrector_to_predictor}
```

中文解释：

- `median` 是中位数，比均值更不怕异常值；
- `MAD` 是 median absolute deviation，即偏离中位数的典型幅度；
- `softplus(x)=log(1+exp(x))`，可以把 score 变成非负，同时比 ReLU 平滑。

最小实现也可以只做：

```text
s_norm = s_raw / (median[bin] + eps)
```

---

## 5. 不确定性 gate：保护最差样本

cache 方法最怕平均指标很好，但少数样本崩掉。AAAI reviewer 很可能会问：你的方法会不会因为误 skip 造成严重 artifacts？

建议加一个很轻的 uncertainty gate：

```text
s_safe = mu(z) + k * sigma(z)
```

几种实现：

### 5.1 Bootstrap Ridge

从 calibration set 里重复采样，训练 5 个 ridge model：

```text
mu    = mean(model_i(z))
sigma = std(model_i(z))
score = mu + k * sigma
```

如果 5 个模型意见不一致，说明这个 transition 不确定，就提高 refresh 概率。

### 5.2 Quantile Regression

直接预测 0.75 或 0.90 分位数：

```text
score = q90(z)
```

中文解释：模型不是预测“平均风险”，而是预测“较坏情况下可能的风险”。这对 cache 更安全。

### 5.3 Stage residual table

最简单：在 calibration set 上统计每个 `stage_kind` 的残差标准差：

```text
score = ridge_mu(z) + k * residual_std[stage_kind]
```

这不需要多个模型，很稳。

---

## 6. Budget controller：让 RR 精准可控

很多 cache paper 都会报 matched refresh ratio 或 matched latency。你需要让方法在 target RR 下稳定运行。

推荐做一个两步控制：

### 6.1 离线 delta calibration

在 calibration set 上二分搜索 delta：

```text
for target_rr in [0.2, 0.3, 0.4, 0.5, 0.6]:
    find delta so that actual_rr(calibration, delta) ~= target_rr
```

### 6.2 在线轻微自适应

如果某个 batch 的实际 RR 偏离太多，可以轻微缩放 score：

```text
if running_rr < target_rr - margin:
    score *= 1.05
if running_rr > target_rr + margin:
    score *= 0.95
```

但是主实验建议先不用在线自适应，避免引入额外复杂性。把它放到 ablation 或 appendix。

---

## 7. skip 时到底复用什么

你的 HeunSamplerJiT 最终使用的是 velocity，但模型原始输出是 `xhat`。因此有两个选择。

### 7.1 复用 velocity

```text
v_cached = last_v
```

优点：sampler 直接用，最简单。

缺点：velocity 里有 `(1-t)` 的缩放，late stage 可能更敏感。

### 7.2 复用 xhat 后重新转 velocity

```text
xhat_cached = last_xhat
v_current   = (xhat_cached - x_t_current) / (1 - t_current)
```

优点：符合 x-pred 的语义，当前 `x_t` 和当前 `t` 仍然参与转换。

缺点：如果 `xhat_cached` 太旧，可能造成奇怪的 velocity。

建议 E6 先保持当前代码的复用逻辑；E7 再加 `reuse_xhat_then_convert` 作为 ablation。如果 `reuse_xhat_then_convert` 明显更好，这就是 x-pred-specific 的好贡献。

---

## 8. 伪代码

```python
class PMAOnlineCacheController:
    def __init__(self, proxy_model, normalizer, target_rr, warmup_calls=5,
                 max_skip_calls=4, force_final=True, use_uncertainty=True):
        self.proxy_model = proxy_model
        self.normalizer = normalizer
        self.delta = load_calibrated_delta(target_rr)
        self.acc = 0.0
        self.cache_age = 0

    def observe_transition(self, prev_proxy, cur_proxy, meta):
        # 1. cheap features
        z = build_features(prev_proxy, cur_proxy, meta, self.cache_age)

        # 2. predict perceptual risk
        if self.use_uncertainty:
            mu, sigma = self.proxy_model.predict_mu_sigma(z)
            score = mu + self.k * sigma
        else:
            score = self.proxy_model.predict(z)

        # 3. stage-kind normalization
        score = self.normalizer(score, meta.stage_bin, meta.transition_kind)

        # 4. accumulated gate
        self.acc += score
        refresh = False
        if meta.call_index < self.warmup_calls:
            refresh = True
        if meta.is_final_call:
            refresh = True
        if self.cache_age >= self.max_skip_calls:
            refresh = True
        if self.acc >= self.delta:
            refresh = True
        if sigma is not None and sigma >= self.uncertainty_threshold:
            refresh = True

        if refresh:
            self.acc = 0.0
            self.cache_age = 0
        else:
            self.cache_age += 1

        return refresh, score
```

---

## 9. 论文中可以怎样表述贡献

推荐贡献点写法：

1. 我们指出 x-prediction pixel diffusion 的 clean-image prediction trajectory 为 cache 提供了一个新的 perceptual refresh criterion。
2. 我们通过 oracle 分析证明，clean-image perceptual drift 比 raw noisy feature drift 更能反映 cache 误差；SEA-filtered input 是连接 online proxy 与 perceptual manifold 的关键桥梁。
3. 我们提出一个训练自由/轻量校准的 PMA-Cache，用 cheap online features 预测 perceptual drift，并用 solver-aware accumulated gate 进行刷新。
4. 在 matched refresh ratio / latency 下，PMA-Cache 比 Uniform、RawInput、SEAInput 和若干现有 cache-style baseline 有更好的 fidelity 和 tail robustness。

---

## 10. 最容易被 reviewer 攻击的问题与回答

### Q1：这是不是只是 SeaCache + DINO/LPIPS？

回答思路：不是。SeaCache 的刷新判据定义在频谱滤波后的输入特征上；我们研究的是 x-prediction 模型暴露出的 clean-image prediction trajectory，并用 DINO/LPIPS 定义 oracle perceptual drift，再学习/校准一个 cheap online proxy。SEA 在我们这里是 proxy 的一部分，而不是全部方法。

### Q2：在线推理是否要跑 DINO/LPIPS？

回答：不需要。DINO/LPIPS 只在 calibration 或分析阶段作为标签和评估，主方法在线只用 first-block proxy、timestep、call kind 和轻量回归器。

### Q3：为什么只在 PixelGen / x-pred 上有效？

回答：方法的核心依赖 clean-image prediction。普通 v-pred / noise-pred 可以转换出 `xhat`，但 x-pred 模型的训练目标直接对齐 clean image，因此 clean-image perceptual drift 更自然、更稳定。可以在 appendix 做一个 v-pred latent model 的初步对比，证明 x-pred 上更明显。

### Q4：calibration 是否违反 training-free？

回答：可以称为 calibration-only，不训练扩散模型，不改动权重，样本量很小。很多推理加速方法也需要少量阈值校准。为了稳妥，可以提供无学习 heuristic 版本作为 fully training-free baseline。

