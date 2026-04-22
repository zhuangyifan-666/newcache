# Perceptual-Manifold-Aware Cache for Pixel Diffusion
## 第一阶段实验计划（2–3 × RTX 3090 版本）

> 目标不是一开始就做“可投论文的完整方法”，而是先用最低成本回答一个关键科学问题：  
> **在 PixelGen / JiT 这类 x-pred pixel diffusion 中，cache 的 refresh 判据如果定义在 clean-image perceptual manifold 上，而不是 raw hidden/input space 上，是否会在相同 refresh ratio 下更接近 full trajectory？**

---

## 0. 结论先行：第一阶段应该做什么

**第一阶段只做“oracle + 小规模在线复现”，不训练新主干，不上大规模 T2I，不追 FID SOTA。**

你的算力只有 2–3 张 3090，这意味着最正确的策略不是一开始训练 PixelGen 或重做大模型，而是：

1. **固定一个已经能跑通的 PixelGen 推理代码路径**（最好就是你现在已经接入 SeaCache 的版本）。
2. **固定缓存位置**（不要同时改 block、改 layer、改 sampler）。
3. **先做 oracle 验证**：看“感知流形”是不是比 raw input / SEA(input) 更适合作为 PixelDiffusion 的 cache 判据。
4. **只有 oracle 赢了，第二阶段才去做低成本 proxy**（比如 shallow preview 或轻量 probe）。

一句话说：  
**第一阶段不是做最终方法，而是做“值得继续做”的证据。**

---

## 1. 核心想法，再细化一层

### 1.1 你的题目真正有意思的地方
SeaCache 的 insight 是：

- 相邻 step 的 raw feature distance 混合了 **content** 和 **noise**；
- diffusion 过程存在 **spectral evolution**；
- 所以 cache 判据应该先做 **spectral-aware filtering**，再决定 skip / refresh。

PixelGen 的 insight 是：

- pixel diffusion 不应该死磕 full image manifold；
- x-pred 直接预测 clean image，更适合建模 image manifold；
- LPIPS 强化 **local texture**，P-DINO 强化 **global semantics**；
- 而且 **高噪声早期不适合直接上 perceptual constraint**。

把两者合起来，最自然的问题就是：

> **Pixel diffusion 的 cache 判据，是否应该是“随 SNR 变化的 spectral + perceptual gate”？**

也就是：

- **高噪声早期**：主要信 spectral / content 稳定性；
- **中期**：开始信 semantic drift（DINO）；
- **低噪声后期**：更信 local texture / edge / LPIPS。

这比“把 SeaCache 生搬到 PixelGen”更有研究味道，因为它直接利用了 PixelGen 的 **x-pred + perceptual manifold** 结构。

---

## 2. 我建议的完整研究路线（但第一阶段只做 A）

### A. Oracle 层：验证“判据是否存在”
你先假装自己“能看到 full trajectory”，比较不同判据在 matched refresh ratio 下能否更接近 full run。

这一步回答的问题是：

- PixelDiffusion 是否真的有额外的 perceptual caching signal？
- 这个 signal 是 global semantic 为主，还是 local detail 为主？
- 它是不是必须和 SEA 结合，而不是单独使用？

### B. Proxy 层：把 oracle 蒸馏成低成本在线信号
如果 A 成立，再做一个低成本在线 proxy，例如：

- shallow block preview；
- low-res image preview；
- 或者 shallow feature + tiny head。

### C. Online 层：真正部署成训练免费 / 轻训练 cache
最后才做可部署的 online schedule。

---

## 3. 第一阶段的**明确研究问题**

第一阶段你只回答下面 4 个问题：

### Q1
在 **相同 refresh ratio** 下，  
`SEA(input)` 是否比 `raw input` 更适合 PixelGen？

### Q2
在 **full predicted clean image** 上定义的 perceptual drift：

- `DINO(x_pred)`
- `LPIPS(x_pred)`
- `Edge(x_pred)`

是否能比 `SEA(input)` 更接近 full trajectory？

### Q3
这些 perceptual drift 是否像 PixelGen 训练现象那样，**只在后半段低噪声阶段有用**？

### Q4
`spectral + perceptual` 的 stage-aware fusion 是否优于任一单独分支？

---

## 4. 你第一阶段的方法定义（建议叫 **PMA-Cache Oracle**）

这里先不做最终 online 方法，只定义 oracle 版本。

### 4.1 记号
- `I_t`：你当前 SeaCache 使用的 input feature（建议保持完全一致，别改 hook 点）
- `xhat_t`：第 `t` 步模型预测的 clean image（PixelGen / JiT 的 x-pred 输出）
- `P(G_t, I_t)`：SeaCache 的 SEA filter 作用在 `I_t` 上后的结果
- `RR`：refresh ratio，full denoiser 真正运行的 step 占比

### 4.2 距离分支
定义下面几个相邻步距离：

#### (1) Raw input distance
```math
\Delta_t^{raw} = L1_{rel}(I_t, I_{t+1})
```

#### (2) SEA input distance
```math
\Delta_t^{sea} = L1_{rel}(P(G_t^{norm}, I_t), P(G_{t+1}^{norm}, I_{t+1}))
```

#### (3) DINO perceptual drift
```math
\Delta_t^{dino} = 1 - \cos(\phi_{dino}(xhat_t), \phi_{dino}(xhat_{t+1}))
```

#### (4) LPIPS perceptual drift
```math
\Delta_t^{lpips} = LPIPS(xhat_t, xhat_{t+1})
```

#### (5) Edge / high-frequency drift
```math
\Delta_t^{edge} = || \nabla(xhat_t) - \nabla(xhat_{t+1}) ||_1
```

其中 `xhat_t` 建议先 resize 到 128 或 224 再算，以降低开销。

### 4.3 归一化
不同分支的量纲不同，所以先做 calibration normalization：

```math
\bar{\Delta}_t^m = \frac{\Delta_t^m}{\mu_m + \epsilon}
```

- `m ∈ {sea, dino, lpips, edge}`
- `μ_m` 用一个小 calibration set 上的平均 step distance 估计
- 第一阶段不要学 normalization，直接用 calibration statistics 即可

### 4.4 Stage-aware fusion（手工版）
定义一个基于 **SNR / time** 的分段融合：

#### 初始推荐权重
设 `s_t` 表示当前步的 clean-signal 强度（可直接用采样时间 `t`，或用 `snr(t)=t^2/((1-t)^2+eps)`）。

- **低 SNR 段（前 30% denoising）**
  ```math
  \Delta_t^{pma} = \bar{\Delta}_t^{sea}
  ```

- **中 SNR 段（30%–70%）**
  ```math
  \Delta_t^{pma} = 0.6 \bar{\Delta}_t^{sea} + 0.4 \bar{\Delta}_t^{dino}
  ```

- **高 SNR 段（后 30%）**
  ```math
  \Delta_t^{pma} = 0.2 \bar{\Delta}_t^{sea} + 0.3 \bar{\Delta}_t^{dino}
                    + 0.3 \bar{\Delta}_t^{lpips} + 0.2 \bar{\Delta}_t^{edge}
  ```

> 第一阶段不追求最优权重，只追求“有没有趋势”。  
> 如果手工权重都能赢，就说明这个方向真的值得做。

### 4.5 Refresh 规则
仍然沿用 TeaCache / SeaCache 的 accumulated-distance 规则：

```math
A_t = A_{t-1} + \Delta_t, \quad A_t > \delta \Rightarrow refresh,\; A_t \leftarrow 0
```

注意第一阶段的重点不是阈值本身，而是：

> **在 matched refresh ratio 下，哪种距离定义更好。**

---

## 5. 为什么第一阶段必须先做 oracle，而不是直接做 online proxy

因为你现在算力有限，而 proxy 训练会引入很多无关变量：

- preview head 到底多深？
- 64x64 还是 128x128 preview？
- DINO-S 还是 DINO-B？
- shallow feature 能否稳定预测 semantic drift？

如果 oracle 本身就赢不了，后面这些工程全都白做。

所以第一阶段的顺序必须是：

1. **先证明 perceptual manifold 确实包含更好的 cache 信号；**
2. 再考虑怎么低成本地近似它。

---

## 6. 第一阶段实验范围：一定要收缩

### 6.1 只做一个主战场
**只做 class-to-image，256×256。**

理由：

- PixelGen 论文中这部分最干净；
- 不需要 text encoder；
- 没有 CFG 时更便宜；
- 同 seed 的 full-reference 比较最直接。

### 6.2 只做一个主模型
**优先用你现在已经跑通的 PixelGen checkpoint。**

建议优先级：

1. **PixelGen-L/16**（如果有 release，优先这个）
2. PixelGen-XL/16（如果只有这个且能在 3090 上 batch=1 跑）
3. 如果 PixelGen release 不稳，再退到你当前能运行的 JiT / PixelGen 兼容版本

### 6.3 只固定一个 cache hook 点
**保持和你现在 SeaCache-on-PixelGen 的 hook 点完全一致。**

不要第一阶段同时研究：

- cache 在第几层；
- cache 哪几个 block；
- 是否 block-wise nonuniform；
- sampler 改成别的 solver。

这些都会把变量搞炸。

---

## 7. 具体实验设计（第一阶段）

## 7.1 实验总表

| ID | 实验 | 必做/可选 | 目的 |
|---|---|---|---|
| E0 | Full inference 跑通 + 输出对齐检查 | 必做 | 确保 uncached reference 可复现 |
| E1 | RawInput vs SEAInput 基线复现 | 必做 | 先确认 SeaCache 移植到 PixelGen 是成立的 |
| E2 | Oracle distance bank 提取 | 必做 | 为后续所有 schedule 提供同一批 full trajectory 数据 |
| E3 | Offline matched-RR oracle 比较 | 必做 | 验证 perceptual metric 是否优于 SEA |
| E4 | 在线 cache rerun（cheating schedule） | 建议做 | 看 oracle 优势是否传导到真实 cache 输出 |
| E5 | JiT 对照 | 可选 | 证明“perceptual supervision 提升 cacheability” |
| E6 | 小规模 FID/IS | 可选 | 早期不必做主结论，避免浪费算力 |

---

## 8. 每个实验该怎么做

## 8.1 E0：Full inference sanity check
### 做法
- 固定 32 个 class label + seed 对
- 跑 full inference 两次
- 检查最终图像是否 bitwise / 近似一致
- 保存：
  - final image
  - 每步 `xhat_t`（只对 8 个样本完整保存即可）
  - 当前 SeaCache 使用的 `I_t`

### 通过标准
- 同 seed 输出稳定
- 你抓到的 `I_t` 与目前 SeaCache 使用的 tensor 一致
- `xhat_t` 的范围、归一化方式清楚（例如 `[-1,1]` 还是 `[0,1]`）

---

## 8.2 E1：RawInput vs SEAInput 基线复现
### 做法
在同一批 128 个样本上比较：

- Raw input metric
- SEA input metric

目标 refresh ratio 设三个点：

- `RR ≈ 0.30`
- `RR ≈ 0.40`
- `RR ≈ 0.50`

### 输出
- wall-clock
- avg RR
- final-image PSNR / LPIPS / SSIM vs uncached output
- 8 组可视化

### 目的
先确认你当前 SeaCache-on-PixelGen 的 gain 不是偶然，也为后面 oracle 比较建立基线。

---

## 8.3 E2：Oracle distance bank 提取（最关键）
### 核心原则
**只保存“相邻步距离序列”，不要傻存全量中间特征。**

对每个样本，full inference 一次，在相邻步之间直接在线计算并保存：

- `Δ_raw`
- `Δ_sea`
- `Δ_dino`
- `Δ_lpips`
- `Δ_edge`

以及一些辅助量：

- 当前 step index
- 当前 `t`
- 当前 `snr(t)`
- `xhat_t` 的 low-res 缩略图（只给少量可视化样本保存）

### 为什么这么做
因为第一阶段的核心是比较 **schedule metric**，不是重放所有特征。  
你真正需要的是每个样本长度为 `T-1` 的标量序列，而不是 TB 级中间 tensor。

### 推荐样本规模
分三档：

- **sanity**：64
- **pilot**：256
- **main**：512

如果 512 太慢，先在 256 上完成主结论。

---

## 8.4 E3：Offline oracle 比较（第一阶段主结果）
### 目标
对每种 metric，在 matched RR 下生成 refresh schedule，再比较它们与 full trajectory 的接近程度。

### 候选 metric
1. `RawInput`
2. `SEAInput`
3. `DINO(x_pred)`
4. `LPIPS(x_pred)`
5. `Edge(x_pred)`
6. `PMA-no-gate`（直接固定权重融合）
7. `PMA-stage-aware`（推荐主方法）

### matched RR 的做法
先用 32 个 calibration 样本找 threshold `δ`，使每种 metric 分别落在：

- `RR ≈ 0.30`
- `RR ≈ 0.40`
- `RR ≈ 0.50`

然后在主测试集上比较。

### 主指标
#### A. trajectory fidelity
- per-step PSNR vs full trajectory
- per-step LPIPS vs full trajectory
- final-image PSNR / LPIPS / SSIM vs uncached output

#### B. schedule behavior
- avg RR
- refresh heatmap over timestep
- early / mid / late 三段 refresh density

### 你最想看到的现象
- `SEAInput > RawInput`
- `DINO/LPIPS/Edge` 各自只在部分阶段有优势
- `PMA-stage-aware` 在大多数 RR 下最好
- `PMA-no-gate` 不如 `PMA-stage-aware`

这会非常漂亮，因为它直接把 PixelGen 的“高噪声早期别上 perceptual”结论转化成 inference-time cache principle。

---

## 8.5 E4：在线 cache rerun（cheating schedule）
> 这个实验不是必须，但我很建议做。

### 做法
对 E3 中表现最好的 3 个 metric：

- `SEAInput`
- 最好的 perceptual 单分支（通常可能是 DINO 或 LPIPS）
- `PMA-stage-aware`

根据 oracle 得到的 refresh schedule，重新跑一遍真实 cache inference：

- refresh step：正常 full denoiser
- skip step：复用缓存输出

### 目的
验证“oracle 上更好的 metric”是否真的转化为更好的最终输出。

### 说明
这一步仍然是“cheating schedule”，不是最终可部署方法。  
但它能回答一个重要问题：

> 改善 cache 判据，是否真的能改善生成结果，而不仅仅是离线距离更漂亮？

---

## 8.6 E5：JiT 对照（可选，但很有价值）
### 你想证明的额外结论
如果能拿到 JiT checkpoint 或者 PixelGen 的无 perceptual 版本，那么做同样 E2/E3：

- PixelGen
- JiT / Baseline

比较同一 cache metric 在两个模型上的收益。

### 最强结论
如果 `PMA-stage-aware` 在 PixelGen 上的提升明显大于 JiT，  
你就能讲出一句非常强的话：

> **perceptual supervision 不仅提升了 pixel diffusion 的生成质量，还提升了它的 inference-time cacheability。**

这个点非常有 AAAI 味道。

---

## 9. 推荐的数据规模与算力安排（按 2–3 张 3090 设计）

## 9.1 强烈建议的资源分配
### 如果只能用 2 张卡
- **GPU0/1**：轮流跑 generation / cache rerun
- DINO / LPIPS 离线算，不和生成混在一个进程里

### 如果能用 3 张卡
- **GPU0/1**：full trajectory 提取
- **GPU2**：离线计算 DINO / LPIPS / Edge + 作图

---

## 9.2 实际建议 batch
对 256×256、PixelGen-L/XL、50 steps：

- 先从 `batch=1 / GPU` 起步
- 能跑稳再试 `batch=2`
- 全程 `torch.no_grad()` + fp16/bf16
- 第一阶段不要开任何训练图

---

## 9.3 第一阶段建议时间预算
### 第 1–2 天
- 跑通 full inference
- 抓到 `I_t`
- 做 E0 / E1 的 32–64 样本 sanity

### 第 3–4 天
- 跑 E2 的 256 样本 distance bank
- 写 threshold search 和 matched-RR 脚本

### 第 5–6 天
- 做 E3 主结果图
- 看 `PMA-stage-aware` 是否显著优于 `SEAInput`

### 第 7 天
- 如果结果好，做 E4 小规模在线 rerun
- 如果结果差，立刻停，进入分析与改方向

> 换句话说，第一阶段不是“一个月大工程”，而应该是 **一周左右能看到 go / no-go 信号**。

---

## 10. 具体实现细节（很重要）

## 10.1 `xhat_t` 的处理
- LPIPS：建议输入范围统一成 `[-1,1]`
- DINO：统一 resize 到 224，再做标准 mean/std normalize
- Edge：建议用灰度 Sobel magnitude，便宜且稳定

## 10.2 DINO 特征怎么取
优先级：

1. **最后一层 patch token 平均后做 cosine**
2. 如果实现麻烦，再退到 CLS token cosine

第一阶段只要稳定，不必追求和 PixelGen 训练损失 100% 完全一致。

## 10.3 SeaCache 的 FFT 怎么做
- 保持与你当前实现一致
- patch token reshape 回 `H/p × W/p × C`
- 只在空间维做 FFT
- class token / extra token 一律固定策略（要么删掉，要么一直保留）

## 10.4 不要让 calibration 泄漏
- calibration set：32 样本
- main result set：256 或 512 样本
- 阈值 `δ` 只能在 calibration 上定，不能在主测试集上来回调

---

## 11. 第一阶段的**最小可交付结果**

如果你只做出下面这些，其实已经足够决定要不要继续：

### 图 1
`PSNR vs Refresh Ratio` 曲线：

- RawInput
- SEAInput
- DINO
- LPIPS
- Edge
- PMA-stage-aware

### 图 2
不同 metric 的 per-step distance 曲线 + refresh heatmap

### 表 1
在 `RR ≈ 0.30/0.40/0.50` 下的：

- wall-clock
- PSNR
- LPIPS
- SSIM

### 图 3
8 组 qualitative 对比：
- uncached
- SEAInput
- PMA-stage-aware

---

## 12. 第一阶段的成功标准（Go / No-Go）

## Go
满足下面任意两条，就值得继续到第二阶段：

1. `PMA-stage-aware` 在 **至少两个 RR 档位** 上优于 `SEAInput`
2. 提升不是只体现在一个指标上，而是 **PSNR + LPIPS** 同时更好
3. 刷新分布明显呈现“早期 + 后期更密，中期更 sparse”的结构
4. `PMA-no-gate < PMA-stage-aware`，说明 stage-aware 不是装饰

## No-Go
出现下面情况，建议及时止损或转方向：

1. `SEAInput` 已经稳定最好，所有 perceptual metric 都不赢
2. perceptual 分支只在极高 RR（几乎没加速）时才有帮助
3. DINO / LPIPS 波动很大，结论不稳定
4. PMA 的 gain 小于 `0.2~0.3 dB PSNR` 且没有一致的 LPIPS 改善

---

## 13. 如果第一阶段成功，第二阶段该怎么接

第二阶段不要一下子训练大模型，而是：

### 路线 1：轻量 preview proxy（推荐）
- 只跑前 `K` 个 shallow blocks
- 接一个 tiny preview head
- 输出 `64×64` 或 `128×128` 预览图
- 用这个预览图近似 `DINO / LPIPS / Edge`

### 路线 2：shallow probe 直接回归 oracle distance
- 输入 shallow features
- 输出 `\Delta^{dino}, \Delta^{lpips}, \Delta^{edge}`
- 类似 sample-specific cache，但 target 是 perceptual drift

### 路线 3：完全 training-free surrogate
- 用 input feature 的低频分量估计 semantic drift
- 用 token variance / edge energy 估计细节 drift

但这些都必须建立在第一阶段 oracle 成立的前提上。

---

## 14. 你第一阶段**不要做**的事

1. **不要训练 PixelGen 主干**
2. **不要一上来做 text-to-image**
3. **不要一开始就追 FID**
4. **不要同时比较多个 cache hook 点**
5. **不要同时改 sampler / block / threshold 逻辑**
6. **不要把 proxy head 提前到 oracle 之前**

---

## 15. 一个最实用的执行顺序（我最推荐）

### Step 1
用你当前 `SeaCache + PixelGen` 代码，先做：

- 64 个样本
- `RawInput vs SEAInput`
- `RR≈0.3/0.5`

### Step 2
在相同 64 个样本上加 oracle distance 提取：

- DINO(x_pred)
- LPIPS(x_pred)
- Edge(x_pred)

### Step 3
做 `PMA-stage-aware`，先别调权重，直接用我上面给的手工权重

### Step 4
如果 64 样本趋势清楚，再扩到 256 / 512

### Step 5
只有当 `PMA-stage-aware` 稳定赢 `SEAInput`，你才进入第二阶段

---

## 16. 我对你这个课题的判断

如果第一阶段跑出来下面这种结果：

- `SEAInput` 比 raw 好；
- `DINO` / `LPIPS` 各自在后半段有明显优势；
- `PMA-stage-aware` 在 matched RR 下稳定最好；

那这个故事就非常像一篇像样的 paper：

> PixelGen 的 perceptual manifold 不只是训练时的 supervision target，  
> 它也是 inference-time caching 的正确几何。

这句话就是你的题眼。

如果第一阶段跑不出来，也不是坏事——至少你能很快知道：

- PixelGen 的 perceptual gain 可能主要作用在训练目标，而不提供更好的缓存判据；
- 那后面就应该转向 **blockwise nonuniform cache** 或 **detail-branch cache**，而不是继续堆 perceptual metric。

---

## 17. 建议的代码文件组织

```text
project/
├── scripts/
│   ├── run_full_reference.py
│   ├── run_cache_baseline.py
│   ├── extract_oracle_distances.py
│   ├── calibrate_thresholds.py
│   ├── rerun_with_oracle_schedule.py
│   └── summarize_results.py
├── metrics/
│   ├── sea_metric.py
│   ├── dino_metric.py
│   ├── lpips_metric.py
│   ├── edge_metric.py
│   └── pma_fusion.py
├── outputs/
│   ├── sanity/
│   ├── pilot/
│   └── main/
└── notebooks/
    └── analysis_phase1.ipynb
```

---

## 18. 相关文献定位（第一阶段够用的版本）

### Pixel / x-pred 方向
- **JiT / Back to Basics: Let Denoising Generative Models Denoise**  
  核心：pixel x-pred，让模型直接预测 clean image。
- **PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss**  
  核心：LPIPS + P-DINO，把 pixel diffusion 拉到 perceptual manifold；高噪声早期不要硬上 perceptual constraint。

### Cache / 加速方向
- **DeepCache**  
  早期代表性的 training-free cache baseline。
- **TeaCache**  
  用 timestep-aware input distance 做 accumulated-threshold refresh。
- **SeaCache**  
  用 spectral-evolution-aware filter 先去除 raw distance 里的 noise 干扰。
- **TaylorSeer**  
  从“复用过去特征”走向“预测未来特征”。
- **DiCache**  
  用 shallow online probe 做 sample-specific cache 调度。

> 但第一阶段并不需要把所有 baselines 都复现。  
> **你的真正 baseline 只需要：RawInput / SEAInput / PMA-stage-aware。**

---

## 19. 最后的建议

以你现在的算力条件，**最优策略不是“做大”，而是“把问题问准”**。

这个课题最危险的地方，不是实现难，而是很容易做成：

- 频域阈值小修小补；
- 跑了很多图，但没有清楚回答“为什么 PixelDiffusion 需要专门的 cache”。

所以第一阶段最重要的不是结果多漂亮，而是：

> **用最便宜的实验，证明 PixelGen 的 perceptual manifold 确实能提供比 SeaCache 更好的 cache 判据。**

只要你把这个点证实，后面的 proxy / online 设计、甚至论文标题，都会自然很多。
