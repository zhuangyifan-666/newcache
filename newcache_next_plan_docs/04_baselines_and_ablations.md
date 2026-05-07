# 04. Baselines 与 Ablations 清单

## 0. 为什么 baseline 很关键

顶会论文最容易被拒的原因之一是 baseline 不充分。你的想法很有潜力，但 reviewer 会自然地问：

- 你是否只是在 SeaCache 上改了一个小权重？
- 你是否比 TeaCache、MagCache、TaylorSeer 这类方法更强？
- 你的感知流形设计到底是哪一部分有效？
- 如果不用 DINO / LPIPS，只用 SEA，是否已经足够？

所以实验设计要主动回答这些问题。

---

## 1. 必须有的 baseline

### 1.1 Full / No Cache

完整 99 calls 的 Heun exact reference。

用途：

- paired fidelity 的 reference；
- FID / IS 的质量上界；
- latency 的基准。

表中写法：

```text
Full, RR=1.00, Speedup=1.00x, LPIPS=0, PSNR=inf
```

### 1.2 Uniform Cache

固定间隔 refresh。

用途：证明“不是只要少跑几步就行，refresh 位置很重要”。

### 1.3 RawInput-online

你已经有的 first-block proxy raw relative L1。

用途：传统 noisy feature drift baseline。

### 1.4 SEAInput-online

你现在最强的真实在线 baseline。

用途：证明 PMA-Cache 不只是 SeaCache 的复现，而是在 SEAInput 基础上进一步利用 perceptual proxy。

### 1.5 SEAInput-oracle

仅作为 upper bound / diagnostic，不作为在线 baseline。

用途：展示 online method 与 oracle gap。

### 1.6 PMA-oracle

E4 的 PMA-no-gate、soft-gate candidates。

用途：说明 perceptual manifold 的潜在上限，但必须明确 oracle 不可部署。

### 1.7 PMA-Proxy online

你的主方法。

用途：证明 oracle perceptual signal 能被 cheap proxy 转化为真实在线 cache。

---

## 2. 应该尽量补的外部方法风格 baseline

不一定要完整复现所有论文；如果算力和代码精力有限，可以实现 “style baseline”，但论文里要诚实命名。例如 `TeaCache-style` 而不是直接声称完整 TeaCache。

### 2.1 TeaCache-style

思想：用 timestep embedding modulated input 的相邻差异估计 output drift，再用累积阈值刷新。

你已经有 RawInput-online 和 OnlineInputCacheController，TeaCache-style 可以加入 polynomial rescale：

```text
score = poly(raw_rel_l1, t)
```

或者：

```text
score = a0 + a1 * raw + a2 * raw^2 + a3 * t + a4 * raw * t
```

用 calibration set 拟合，让它预测 output / perceptual drift。

### 2.2 SeaCache-style

你已有 SEAInput-online，但需要确保实现细节和 SeaCache 尽量一致：

- SEA filter 有 gain normalization；
- beta 做 ablation；
- cache metric 是 filtered feature relative L1；
- 使用 accumulated-distance refresh rule。

这可以作为“强 baseline”。

### 2.3 MagCache-style

思想：利用 denoising residual magnitude 的变化规律进行 cache。

x-pred 上可以定义：

```text
m_t = ||xhat_t - x_t||_1 / numel
r_t = m_t / (m_prev + eps)
```

在线时只能在 refresh 后更新 `m_t`，skip 时可用外推值：

```text
m_pred = m_last * expected_ratio(stage)
score_mag = |m_pred - m_last| / (m_last + eps)
```

这可能不是很强，但能回应 reviewer：你比较过 magnitude-aware cache。

### 2.4 TaylorSeer-style forecast

TaylorSeer 的强点不是只判断是否 skip，而是在 skip 时预测未来特征。你可以实现一个轻量版本：

```text
out_forecast = out_last + alpha * (out_last - out_prev_refresh)
```

不建议一开始作为主 baseline，因为实现和调参复杂。可以作为 E7 extension。

### 2.5 DiCache-style shallow probe

DiCache 强调用模型自己的浅层/中间层差异预测 cache 风险。你可以做一个轻量 ablation：

- first block proxy；
- middle block proxy；
- first + middle proxy。

如果 middle block 更准但成本高，就可以说明你选择 first-block proxy 是 speed-quality trade-off。

### 2.6 ToCa / FreqCa 只做分析即可

Token-wise / frequency-wise cache 比较复杂，会把主线带偏。建议只做两张 appendix 图：

- token delta heatmap；
- frequency band delta curve。

如果效果明显，写 future work。

---

## 3. 主表建议

### 3.1 Matched RR 主表

至少做 RR=0.30 和 RR=0.50 两张表，或者一张表里每个方法两行。

| Method | Online? | RR ↓ | Speedup ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | DINO ↓ | p95 LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full | yes | 1.00 | 1.00x | ∞ | 1.000 | 0.000 | 0.000 | 0.000 |
| Uniform | yes | 0.30 | ... | ... | ... | ... | ... | ... |
| RawInput | yes | 0.30 | ... | ... | ... | ... | ... | ... |
| SEAInput | yes | 0.30 | ... | ... | ... | ... | ... | ... |
| TeaCache-style | yes | 0.30 | ... | ... | ... | ... | ... | ... |
| MagCache-style | yes | 0.30 | ... | ... | ... | ... | ... | ... |
| PMA-oracle | no | 0.30 | n/a | ... | ... | ... | ... | ... |
| PMA-Proxy | yes | 0.30 | ... | ... | ... | ... | ... | ... |

注意 PMA-oracle 的 Speedup 不建议和 online 方法同表比较，或者标 `oracle schedule`。

### 3.2 Pareto curve

横轴：latency 或 actual RR。纵轴：LPIPS / PSNR。

比较：

- Uniform
- RawInput
- SEAInput
- TeaCache-style
- PMA-Proxy
- PMA-oracle upper bound

这张图非常重要，因为 cache 方法常常在某个 RR 点偶然好，但整条曲线才说明稳健性。

---

## 4. 必做 ablation

### 4.1 去掉感知标签

| Variant | 说明 |
|---|---|
| PMA-Proxy full | 用 SEA + predicted DINO/LPIPS risk |
| w/o DINO label | 不预测语义 drift |
| w/o LPIPS label | 不预测局部感知 drift |
| SEA only | 只用 SEA |
| Raw only | 只用 Raw |

目的：证明 DINO / LPIPS 的 perceptual label 对 online proxy 有帮助。

### 4.2 去掉 stage-kind normalization

| Variant | 预期 |
|---|---|
| full | 最稳 |
| w/o stage bins | early/middle/late 混在一起，可能 early spike 支配 |
| w/o transition kind | predictor/corrector 差异被忽略 |
| global z-score only | 比 raw 好但不如 full |

### 4.3 去掉 uncertainty gate

| Variant | 关注指标 |
|---|---|
| full | mean + p95/p99 |
| w/o uncertainty | 平均可能相近，但 p95/p99 变差 |
| uncertainty only | 过保守，RR 不稳定 |

特别要报 `p95 LPIPS` 和 worst-case grid。这个 ablation 能体现方法对 tail failure 的处理。

### 4.4 proxy model 类型

| Proxy | 优点 | 风险 |
|---|---|---|
| heuristic | 完全可解释 | 上限低 |
| ridge | 稳、便宜、可解释 | 非线性不足 |
| random forest / XGBoost | 捕捉非线性 | reviewer 可能觉得工程化 |
| small MLP | 灵活 | 容易过拟合、需要训练解释 |
| quantile ridge | tail 安全 | 实现稍复杂 |

推荐主文只放 heuristic / ridge / uncertainty ridge 三个。

### 4.5 SEA filter beta

| beta | 含义 |
|---|---|
| 1 | 较弱自然图像先验 |
| 2 | 图像常用，默认 |
| 3 | 更强低频偏置 |
| learned/calibrated | 从数据拟合 |

如果 beta=2 足够好，就保留默认。不要过度调参。

### 4.6 warmup / max_skip

| 参数 | 建议范围 |
|---|---|
| warmup_calls | 0 / 3 / 5 / 8 |
| max_skip_calls | 2 / 4 / 6 / 8 |
| force_final | yes / no |

预期：

- warmup 太小，early stage 崩；
- max_skip 太大，tail samples 崩；
- force final 可能保护最后细节。

### 4.7 reuse target

| Variant | 说明 |
|---|---|
| reuse velocity | 当前默认，sampler 直接使用 |
| reuse xhat then convert | 更 x-pred-specific |
| forecast velocity | Taylor-style |
| forecast xhat then convert | x-pred + forecast |

如果 `reuse xhat then convert` 好，你的 x-pred 专用性更强。

---

## 5. 负结果也能写，但要放对位置

你可能会遇到：

- PMA-Proxy 在 RR=0.30 不如 SEAInput；
- DINO branch 容易误导 early stage；
- LPIPS branch 对 class-to-image 纹理有效，但对语义帮助有限；
- MagCache-style 在 PixelGen 上不稳定。

这些不是坏事。只要主方法在重要设置下赢，负结果可以变成分析：

> perceptual drift must be stage-aware and budget-aware; naively injecting perceptual scores can over-refresh early calls or reduce diversity.

这会让论文更可信。

---

## 6. 推荐最终实验优先级

如果时间很紧，按下面顺序做：

1. SEAInput-online vs PMA-Proxy online：RR 0.30/0.40/0.50，192 samples。
2. 同样实验扩到 1024 samples。
3. 加 TeaCache-style 和 MagCache-style。
4. 做 ablation：w/o stage-kind、w/o uncertainty、SEA only。
5. 跑 5k FID。
6. 做 forecast extension。
7. 扩 10k/50k FID 或 text-to-image generalization。

