# Perceptual-Manifold-Aware Cache for Pixel Diffusion
## 第一阶段实验计划 v2：更紧凑、更可验证的 2–3 × RTX 3090 版本

> 这版计划的核心修改是：**第一阶段不再把“离线 distance bank 比较”当成最终证据，而是把“oracle schedule 的真实 cache rerun”提升为必做主实验。**  
> 因为只看相邻步距离序列，只能说明某个 metric 看起来合理；真正能说服人的证据是：在相同 refresh ratio 下，用这个 metric 产生的 refresh schedule 重新跑 cache 后，最终图像是否更接近 uncached full trajectory。

---

## 0. 这一版相比原计划改了什么

### 保留的核心
- 仍然只做 **PixelGen / JiT 这类 x-pred pixel diffusion**。
- 仍然先做 **oracle**，不训练 PixelGen 主干。
- 仍然以 **SeaCache / TeaCache 的 accumulated-distance refresh rule** 为基础。
- 仍然围绕问题：

> 在 x-pred pixel diffusion 中，refresh 判据定义在 clean-image perceptual space 上，是否比定义在 raw / SEA-filtered input feature space 上更适合？

### 主要删减
- **删掉第一阶段的小规模 FID / IS。**  
  它不适合做第一阶段主证据。cache 论文第一阶段更应该证明 cached result 是否接近 uncached result，而不是追生成分布指标。FID 需要大量样本，2–3 张 3090 不划算。

- **把 JiT 对照移出第一阶段。**  
  JiT 对照很有论文价值，但它会引入 checkpoint、实现、采样器和训练差异。第一阶段先证明 PixelGen 上 PMA 判据是否成立；JiT 放到 Phase 1.5 或第二阶段。

- **Edge 从主线降级为可选诊断。**  
  Edge / Sobel 很便宜，但它容易把高频细节和高频噪声混在一起。第一阶段主线只保留 `SEA(input) + DINO(x_pred) + LPIPS(x_pred)`，Edge 只作为 failure analysis。

### 主要新增
- **新增 Uniform / Random matched-RR 控制组。**  
  这是为了确认 PMA 不是靠“刚好多刷新了某些 step”偶然赢，而是真的比简单 schedule 更好。

- **新增 SEA-oracle schedule 与 SEA-online schedule 的区分。**  
  SeaCache 真正部署时是 online schedule；但为了公平比较 metric 本身，也要用 full trajectory 上的 `SEA(input)` 生成一个 cheating schedule。这样可以区分：
  - PMA 赢，是因为 metric 更好；
  - 还是只是因为 online schedule 轨迹漂移导致 SeaCache 吃亏。

- **把 oracle cache rerun 从“建议做”改成“必做”。**  
  这是第一阶段最关键的结论来源。

- **新增 paired bootstrap / paired test。**  
  样本量小的时候不要只看平均值。每个 seed 都有 SEA 与 PMA 的成对结果，应该报告 paired difference 和置信区间。

- **新增轻量 proxy feasibility check。**  
  不训练大模型，只用 distance bank 做一个线性 / logistic 回归，看 `SEA(input), t, accumulator` 是否能预测 PMA oracle 的 refresh decision。这个实验很便宜，但能提前判断第二阶段 proxy 是否有希望。

---

## 1. 第一阶段最终目标

第一阶段不要求做出可部署的完整 PMA-Cache。  
第一阶段只回答一个更严格的问题：

> **如果我们允许一个 oracle 使用 full trajectory 中的 clean-image prediction 来决定 refresh schedule，那么这种 perceptual schedule 是否能在相同 denoiser refresh budget 下，比 RawInput / SEAInput 更接近 uncached PixelGen 的完整生成轨迹？**

这里的“相同 budget”主要指：

```text
同样 50 个 sampling steps，大家都只允许 refresh 约 30% / 40% / 50% 的 step。
```

第一阶段比较的是 **判据是否有上限价值**，不是比较真实在线推理延迟。PMA-oracle 的 perceptual metric 计算成本不计入最终速度，因为它目前是 cheating oracle；真正可部署版本要等第二阶段 proxy。

---

## 2. 推荐的核心假设

### 2.1 专业表述

在 x-pred pixel diffusion 中，模型每一步直接预测 clean image estimate：

```math
\hat{x}_t = f_\theta(x_t, t, c)
```

因此，cache refresh 判据可以不只定义在 noisy input / hidden feature space 上，而可以定义在 predicted clean image 的 perceptual representation 上。我们希望验证：

```math
\Delta_t^{PMA}
= g_{snr}(t) \cdot
\Big[\Delta_t^{SEA(input)},\; \Delta_t^{DINO(\hat{x})},\; \Delta_t^{LPIPS(\hat{x})}\Big]
```

是否比

```math
\Delta_t^{raw} = d(I_t, I_{t+1})
```

或

```math
\Delta_t^{sea} = d(P(G_t, I_t), P(G_{t+1}, I_{t+1}))
```

更适合作为 refresh trigger。

### 2.2 直白解释

主流 cache 方法大多看：

> 模型内部输入 / 特征数字变了多少？

你的方法想看：

> 模型当前认为的“干净图像”在语义、纹理、感知质量上有没有真的变？

PixelGen 的 x-pred 结构让这个想法变得自然，因为它每一步本来就输出 clean image estimate。PixelGen 又用 LPIPS 强化局部纹理、用 P-DINO 强化全局语义，因此 inference-time cache 判据也可以沿着这个 perceptual geometry 设计。

---

## 3. 第一阶段只回答 3 个问题

原计划有 4 个问题，这里压缩成 3 个更有判别力的问题。

### Q1. SeaCache 在 PixelGen 上是否真的优于 raw input cache？
必须先确认：

```text
SEA(input) > RawInput
```

如果 SeaCache 移植到 PixelGen 都不稳定，那 PMA 的 baseline 没站住。

### Q2. PMA-oracle schedule 是否优于 SEA-oracle schedule？
这是最关键的问题。

不是只比较 PMA 与 SeaCache online，而是要比较：

```text
SEA(input) oracle schedule vs PMA oracle schedule
```

因为二者都使用 full trajectory 生成 schedule，区别只在 metric 本身。  
如果 PMA 能赢 SEA-oracle，说明 perceptual metric 本身确实有额外信息。

### Q3. PMA 的优势是否依赖 SNR / stage-aware gate？
需要比较：

```text
PMA-no-gate vs PMA-stage-aware
```

如果 stage-aware 明显更好，就能支撑你的核心论点：

> Pixel diffusion 的 cache 判据不能静态地看 perceptual drift，而应该根据 denoising stage 调整 spectral / semantic / texture 的权重。

---

## 4. 方法定义：PMA-Cache Oracle v2

### 4.1 记号

- `T`：sampling step 数，默认 50。
- `I_t`：SeaCache / TeaCache 使用的 timestep-modulated input feature。
- `P(G_t, I_t)`：SEA filter 后的 input feature。
- `xhat_t`：PixelGen / JiT 在第 `t` 步预测的 clean image。
- `RR`：refresh ratio，即真正 full denoiser evaluation 的比例。
- `A_t`：accumulated distance accumulator。

---

## 4.2 必做 metric

### Metric 1：RawInput

```math
\Delta_t^{raw} = L1_{rel}(I_t, I_{t+1})
```

作用：最原始的 input-space 动态 cache baseline。

---

### Metric 2：SEAInput

```math
\Delta_t^{sea} = L1_{rel}(P(G_t^{norm}, I_t), P(G_{t+1}^{norm}, I_{t+1}))
```

作用：SeaCache 风格 baseline。它仍然在 input / feature space 上比较变化，但先通过 SEA filter 抑制噪声成分、强调内容相关成分。

---

### Metric 3：DINO clean-image drift

```math
\Delta_t^{dino}
= 1 - \cos(\phi_{DINO}(\hat{x}_t), \phi_{DINO}(\hat{x}_{t+1}))
```

建议实现：

- `xhat_t` resize 到 224。
- 用 DINOv2-B 或 DINOv2-S。
- 第一阶段优先取最后一层 patch tokens 的平均特征。
- 如果显存 / 实现不方便，再用 CLS token。

作用：近似 global semantic drift。

---

### Metric 4：LPIPS clean-image drift

```math
\Delta_t^{lpips} = LPIPS(\hat{x}_t, \hat{x}_{t+1})
```

建议实现：

- `xhat_t` resize 到 128 或 224。
- 输入范围统一成 `[-1, 1]`。
- 第一阶段用 AlexNet LPIPS 更便宜；如果结果敏感，再换 VGG LPIPS。

作用：近似 local perceptual / texture drift。

---

## 4.3 可选 metric：Edge drift

Edge 只放在诊断里，不放第一阶段主方法。

```math
\Delta_t^{edge} = \|Sobel(\hat{x}_t) - Sobel(\hat{x}_{t+1})\|_1
```

使用场景：

- LPIPS 不稳定；
- PMA 在文字、边缘、细纹理样本上失败；
- 需要分析 late-stage 高频漂移。

第一阶段主表不要强行加入 Edge，否则方法看起来像堆 metric。

---

## 4.4 归一化：用 robust calibration，不要用可学习权重

不同 metric 的数值尺度差很多，必须归一化。推荐：

```math
\tilde{\Delta}_t^m = \frac{\Delta_t^m}{\operatorname{median}_{(i,t)\in \mathcal{C}}(\Delta_{i,t}^m) + \epsilon}
```

其中：

- `m ∈ {sea, dino, lpips}`。
- `C` 是 calibration set，例如 32 个样本。
- 使用 median 而不是 mean，避免极端样本污染。
- 不做 per-timestep normalization，因为那会抹掉不同 timestep 的自然重要性。

如果某个 metric 的 early-stage 数值极端不稳定，可以额外做 percentile clipping：

```text
clip normalized distance to [0, p99]
```

---

## 4.5 PMA-no-gate

作为反例 / 消融，定义一个不区分阶段的固定融合：

```math
\Delta_t^{pma\_nogate}
= 0.4\tilde{\Delta}_t^{sea}
+ 0.3\tilde{\Delta}_t^{dino}
+ 0.3\tilde{\Delta}_t^{lpips}
```

作用：证明“感知分支有用”还不够，还要证明 **stage-aware gate** 有用。

---

## 4.6 PMA-stage-aware：第一阶段主方法

把 50 个 sampling steps 按 denoising 进度分三段：

```text
early:  0%–30% denoising
middle: 30%–70% denoising
late:   70%–100% denoising
```

推荐第一版手工权重：

### Early stage：只信 SEA

```math
\Delta_t^{pma} = \tilde{\Delta}_t^{sea}
```

理由：高噪声阶段 `xhat_t` 的感知特征不可靠，PixelGen 训练中也观察到 early high-noise 阶段不宜直接施加 perceptual loss。

### Middle stage：SEA + DINO

```math
\Delta_t^{pma}
= 0.5\tilde{\Delta}_t^{sea}
+ 0.5\tilde{\Delta}_t^{dino}
```

理由：中期开始形成主体结构，global semantics 开始有意义。

### Late stage：DINO + LPIPS 为主，SEA 为辅

```math
\Delta_t^{pma}
= 0.25\tilde{\Delta}_t^{sea}
+ 0.35\tilde{\Delta}_t^{dino}
+ 0.40\tilde{\Delta}_t^{lpips}
```

理由：低噪声后期主要在修语义一致性、局部纹理和细节，LPIPS 的作用应该更明显。

---

## 4.7 Refresh rule

所有 metric 共用同一个 accumulated-distance 规则：

```math
A_t = A_{t-1} + \Delta_t
```

当：

```math
A_t > \delta
```

则：

```text
refresh at step t; A_t = 0
```

每种 metric 的 threshold `δ` 只在 calibration set 上搜索，使其达到目标 RR：

```text
RR ≈ 0.30 / 0.40 / 0.50
```

pilot 阶段只做：

```text
RR ≈ 0.30 / 0.50
```

趋势明确后再补 `RR ≈ 0.40`。

---

# 5. 修改后的实验总表

## 5.1 总体实验结构

| ID | 实验 | 状态 | 目的 | 是否进入主结论 |
|---|---|---|---|---|
| E0 | Deterministic full inference sanity check | 必做 | 保证 reference 稳定 | 否 |
| E1 | Online RawInput / SEAInput baseline | 必做 | 确认 SeaCache-on-PixelGen baseline | 是 |
| E2 | Oracle distance bank extraction | 必做 | 提取 full trajectory 上的 metric 序列 | 否，作为中间数据 |
| E3 | Schedule-level analysis | 必做 | 看 metric 形状、threshold、refresh heatmap | 辅助结论 |
| E4 | Oracle-schedule real cache rerun | 必做 | 第一阶段主结果 | 是 |
| E5 | Lightweight proxy feasibility check | 可选但推荐 | 预判第二阶段是否可做 online proxy | 否，作为后续依据 |
| E6 | Edge / failure diagnostic | 可选 | 分析失败样本 | 否 |

### 从原计划删除 / 延后

| 原实验 | 处理 | 原因 |
|---|---|---|
| 小规模 FID / IS | 删除 | 样本量不足时误导性强，且不回答 cache fidelity 问题 |
| JiT 对照 | 延后到 Phase 1.5 | 有价值但变量太多，不适合第一阶段主线 |
| 大规模 T2I | 延后 | text encoder 与 prompt 复杂度会污染第一阶段结论 |
| 多 hook 点 / 多 block 策略 | 延后 | 会把变量从“判据”变成“系统工程” |

---

# 6. 具体实验设计

## E0. Deterministic full inference sanity check

### 做法

固定：

```text
32 个 ImageNet class label × seed pair
50-step sampler
相同 checkpoint
相同 precision
相同 sampler config
```

跑两次 full inference，检查：

- final image 是否近似一致；
- `xhat_t` 范围是否正确；
- `I_t` hook 是否稳定；
- sampler step 顺序和 `t` 定义是否清楚。

### 保存

只保存：

```text
final image
每个 step 的 xhat_t low-res preview（仅 8 个样本）
每个 step 的必要 scalar metadata
```

不要保存全量 hidden tensor。

### 通过标准

- 同 seed final PSNR 接近无穷或非常高。
- `xhat_t` 没有明显归一化错误。
- `I_t` hook 与 SeaCache 代码中的输入特征一致。

---

## E1. Online RawInput / SEAInput baseline

### 目的

确认在 PixelGen 上：

```text
SEAInput online cache > RawInput online cache
```

如果这个都不成立，说明 SeaCache 移植、hook、FFT reshape 或 sampler 适配可能有问题。

### 设置

Pilot：

```text
64 samples
RR ≈ 0.30 / 0.50
```

Main：

```text
128 或 256 samples
RR ≈ 0.30 / 0.40 / 0.50
```

### Baselines

- Full inference reference
- Uniform matched-RR cache
- RawInput online cache
- SEAInput online cache

### 输出指标

与 uncached final image 比较：

```text
PSNR ↑
LPIPS ↓
SSIM ↑
actual RR
number of full denoiser calls
wall-clock latency（只对 online baseline 有意义）
```

### 判断

如果 `SEAInput` 不稳定优于 `RawInput` 和 `Uniform`，先不要做 PMA，优先 debug SeaCache 移植。

---

## E2. Oracle distance bank extraction

### 目的

在 full trajectory 上提取每个样本、每个 step 的 metric 序列。

每个样本最后应该得到一个长度为 `T-1` 的 scalar 序列：

```text
Δ_raw[0:T-1]
Δ_sea[0:T-1]
Δ_dino[0:T-1]
Δ_lpips[0:T-1]
可选：Δ_edge[0:T-1]
```

### 推荐实现

不要保存所有 `xhat_t`。推荐在线计算相邻步距离：

```python
prev_xhat = None
for t in timesteps:
    xhat_t, I_t = full_denoiser(...)
    if prev_xhat is not None:
        compute DINO(prev_xhat, xhat_t)
        compute LPIPS(prev_xhat, xhat_t)
        compute raw/SEA distance(prev_I, I_t)
    prev_xhat = xhat_t.detach()
    prev_I = I_t.detach()
```

如果 DINO / LPIPS 太慢，可以先把 `xhat_t` 的 low-res 版本暂存成 fp16 CPU tensor，再批量计算。

### 样本规模

```text
Sanity: 32–64
Pilot: 128
Main: 256
Optional extended: 512
```

我建议第一阶段主结果先用 256，不要强行上 512。

---

## E3. Schedule-level analysis

### 目的

E3 不是最终证据，只是看：

- metric 的 timestep pattern 是否合理；
- threshold 是否能稳定控制 RR；
- PMA-stage-aware 的 refresh 分布是否符合预期；
- DINO / LPIPS 是否主要在 middle / late stage 起作用。

### 候选 schedule

必做：

1. Uniform
2. RawInput-oracle
3. SEAInput-oracle
4. DINO-oracle
5. LPIPS-oracle
6. PMA-no-gate-oracle
7. PMA-stage-aware-oracle

可选：

8. Edge-oracle
9. Random matched-RR
10. TopK-PMA upper bound

### 注意

这里的 `RawInput-oracle` 和 `SEAInput-oracle` 指的是：

```text
在 full trajectory 上提前算好 metric，再生成 cheating schedule。
```

它们不是 online RawInput / SeaCache。

这样做的目的是公平比较 metric 本身。

### 输出图

1. 每种 metric 的 average step distance curve。
2. 每种 schedule 的 refresh heatmap。
3. early / middle / late 三段 refresh density。
4. threshold vs achieved RR 曲线。

---

## E4. Oracle-schedule real cache rerun（第一阶段主实验）

### 为什么 E4 必做

这是原计划里最需要加强的地方。  
只看 E3 的 schedule-level 结果是不够的，因为 cache 会改变后续 sampler trajectory。一个 schedule 离线看起来合理，不代表实际复用后 final image 一定更接近 full reference。

因此必须用 E3 得到的 schedule 重新跑真实 cached inference。

### 做法

对每个 test sample：

1. 已经有 full inference reference。
2. 根据 full trajectory metric 生成 oracle refresh schedule。
3. 重新跑一遍 cache inference：
   - schedule 指定 refresh：正常 full denoiser；
   - schedule 指定 skip：复用 cached output / cached feature；
   - cache 位置、复用逻辑完全沿用你的 SeaCache-on-PixelGen 实现。
4. 比较 final image 与 full reference。

### 主比较对象

最小主表只放这些：

| 方法 | schedule 来源 | 是否可部署 | 作用 |
|---|---|---|---|
| Uniform | 固定间隔 | 可部署 | 简单控制组 |
| RawInput-online | 当前 cached trajectory | 可部署 | 原始动态 baseline |
| SEAInput-online | 当前 cached trajectory | 可部署 | SeaCache-style baseline |
| SEAInput-oracle | full trajectory | 不可部署 | 公平 metric baseline |
| PMA-no-gate-oracle | full trajectory | 不可部署 | 证明 gate 是否必要 |
| PMA-stage-aware-oracle | full trajectory | 不可部署 | 你的第一阶段主方法 |

可选加：

```text
DINO-oracle
LPIPS-oracle
Random matched-RR
```

### 目标 RR

Pilot：

```text
RR ≈ 0.30 / 0.50
```

Main：

```text
RR ≈ 0.30 / 0.40 / 0.50
```

实际 RR 允许误差：

```text
±0.02
```

超过这个范围，要重新校准 threshold。

### 主指标

与 uncached final image 比较：

```text
PSNR ↑
LPIPS ↓
SSIM ↑
```

报告：

```text
actual RR
full denoiser call count
paired ΔPSNR = PSNR(PMA) - PSNR(SEA)
paired ΔLPIPS = LPIPS(PMA) - LPIPS(SEA)
bootstrap 95% CI
```

注意：PMA-oracle 的 DINO / LPIPS schedule 计算成本不要算进 runtime。这里只比较同样 denoiser refresh budget 下的 fidelity upper bound。

### 可选 per-step trajectory fidelity

只对 16 个样本保存 cached intermediate states，画：

```text
per-step PSNR vs full trajectory
per-step LPIPS vs full trajectory
```

这个图很有解释力，但不必对所有样本保存。

---

## E5. Lightweight proxy feasibility check（可选但推荐）

这个实验很便宜，不需要训练主干，也不需要新增生成。

### 目的

如果 PMA-oracle 赢了，你马上会遇到第二阶段问题：

> 既然 PMA 需要 full trajectory 的 xhat，它真实推理时怎么在线获得？

在进入 preview head 训练前，可以先做一个超轻量可行性检查。

### 做法

从 E2 的 distance bank 构造数据：

输入特征：

```text
Δ_raw(t)
Δ_sea(t)
t / snr(t)
current accumulator A_t
previous few step distances
```

标签：

```text
PMA-stage-aware oracle 是否在 step t refresh
```

训练一个很小的模型：

```text
logistic regression
linear SVM
small MLP on CPU/GPU
```

### 指标

```text
AUC
F1
predicted schedule 的 achieved RR
predicted schedule rerun 的 final PSNR / LPIPS（可选）
```

### 判断

如果只用 `SEA + time + accumulator` 就能较好预测 PMA refresh decision，说明第二阶段可能不需要复杂 preview image head；可以先做 training-free / tiny-probe 版本。

如果预测很差，第二阶段再考虑 shallow preview head。

---

## E6. Failure diagnostic：Edge 与样本类别分析

只在 E4 结果不清楚或 PMA 失败时做。

### 可以分析

- 哪些 class 上 PMA 赢？
- 哪些 class 上 PMA 输？
- 失败样本是否集中在：
  - 细纹理；
  - 小物体；
  - 高频背景；
  - early-stage 结构漂移；
  - late-stage 颜色 / 边缘漂移。

### Edge 的使用方式

不要把 Edge 一开始塞进主方法。先看失败样本：

```text
如果 PMA 输在 late-stage fine edge，那么再加入 Edge 作为 PMA-Edge variant。
```

---

# 7. 推荐算力安排

## 7.1 2 张 3090

```text
GPU0: full inference / cache rerun
GPU1: full inference / cache rerun 或 DINO/LPIPS batch metric
CPU: threshold search / bootstrap / plotting
```

不要让生成和 DINO / LPIPS 抢同一张卡，容易造成显存碎片和速度不稳定。

## 7.2 3 张 3090

```text
GPU0: full reference extraction
GPU1: cache rerun
GPU2: DINO / LPIPS / metric bank computation
CPU: threshold search / schedule generation / stats
```

## 7.3 batch 建议

```text
PixelGen-L/16, 256×256: 从 batch=1/GPU 开始
如果稳定，再试 batch=2/GPU
全程 torch.no_grad()
fp16 或 bf16
不要开训练图
```

---

# 8. 建议执行顺序

## Day 1：E0 + E1 sanity

- 32–64 samples。
- 跑通 full reference。
- 跑 Uniform / RawInput-online / SEAInput-online。
- 只做 RR≈0.30 和 RR≈0.50。

如果 `SEAInput-online` 没有明显优于 `RawInput-online`，暂停 PMA，先 debug SeaCache 移植。

---

## Day 2–3：E2 distance bank

- 提取 128 samples 的 distance bank。
- 算 `Δ_raw / Δ_sea / Δ_dino / Δ_lpips`。
- 画 average distance curves。

如果 DINO / LPIPS 在 early stage 极端噪声化，确认 stage-aware gating 是否把 early perceptual branch 关掉。

---

## Day 4：E3 schedule-level analysis

- 用 32 calibration samples 搜 threshold。
- 在 128 test samples 上生成 schedules。
- 画 refresh heatmap。
- 先检查：

```text
PMA-stage-aware 的 refresh pattern 是否不同于 SEA？
PMA-no-gate 是否过度刷新 early stage？
DINO / LPIPS 是否主要影响 middle / late stage？
```

---

## Day 5–6：E4 oracle-schedule cache rerun

- 先跑 64 samples pilot。
- 如果趋势清楚，再扩到 128 或 256。
- 主比较：

```text
SEAInput-online
SEAInput-oracle
PMA-no-gate-oracle
PMA-stage-aware-oracle
```

优先做 RR≈0.30 / RR≈0.50。  
如果 PMA 在两档都赢，再补 RR≈0.40。

---

## Day 7：统计与决策

- paired bootstrap。
- qualitative cases。
- failure analysis。
- 决定 Go / Weak-Go / No-Go。

---

# 9. 第一阶段最小主结果

## 表 1：Online baseline sanity

| Method | RR | PSNR↑ | LPIPS↓ | SSIM↑ | Calls | Latency |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 0.30 | | | | | |
| RawInput-online | 0.30 | | | | | |
| SEAInput-online | 0.30 | | | | | |
| Uniform | 0.50 | | | | | |
| RawInput-online | 0.50 | | | | | |
| SEAInput-online | 0.50 | | | | | |

---

## 表 2：Oracle-schedule real cache rerun（主表）

| Method | Schedule source | Deployable now? | RR | PSNR↑ | LPIPS↓ | SSIM↑ | ΔPSNR vs SEA-oracle | ΔLPIPS vs SEA-oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SEAInput-online | cached trajectory | Yes | 0.30 | | | | | |
| SEAInput-oracle | full trajectory | No | 0.30 | | | | 0 | 0 |
| PMA-no-gate-oracle | full trajectory | No | 0.30 | | | | | |
| PMA-stage-aware-oracle | full trajectory | No | 0.30 | | | | | |
| SEAInput-online | cached trajectory | Yes | 0.50 | | | | | |
| SEAInput-oracle | full trajectory | No | 0.50 | | | | 0 | 0 |
| PMA-no-gate-oracle | full trajectory | No | 0.50 | | | | | |
| PMA-stage-aware-oracle | full trajectory | No | 0.50 | | | | | |

---

## 图 1：Quality vs refresh ratio

横轴：RR。  
纵轴分别画：

```text
PSNR ↑
LPIPS ↓
SSIM ↑
```

至少包含：

```text
Uniform
RawInput-online
SEAInput-online
SEAInput-oracle
PMA-stage-aware-oracle
```

---

## 图 2：Refresh heatmap

展示不同 schedule 在 50 steps 上的 refresh density：

```text
SEAInput-oracle
PMA-no-gate-oracle
PMA-stage-aware-oracle
```

这个图应该能说明 PMA-stage-aware 是否真的改变了 refresh 位置，而不是只改变 threshold。

---

## 图 3：Qualitative cases

至少 8 个 seed：

```text
Full reference
SEAInput-online
SEAInput-oracle
PMA-stage-aware-oracle
```

每个 case 标注：

```text
RR
PSNR
LPIPS
class label
```

---

# 10. Go / Weak-Go / No-Go 标准

## Strong Go

满足下面条件，可以进入第二阶段并考虑论文方法设计：

1. `PMA-stage-aware-oracle` 在至少两个 RR 档位上优于 `SEAInput-oracle`。
2. 提升同时体现在 PSNR 和 LPIPS，而不是单一指标。
3. paired bootstrap 95% CI 显示提升不是偶然。
4. `PMA-stage-aware-oracle > PMA-no-gate-oracle`，说明 stage-aware gate 有必要。
5. qualitative cases 中 PMA 不只是数值好，确实更接近 full reference。

推荐最低强度：

```text
ΔPSNR ≥ 0.3 dB 或 LPIPS 相对下降 ≥ 3%–5%
```

不一定每档都达到，但至少在两个 RR budget 上趋势一致。

---

## Weak Go

出现下面情况，说明方向有希望，但论文故事要谨慎：

1. `PMA-stage-aware-oracle` 明显优于 `SEAInput-online`，但不明显优于 `SEAInput-oracle`。
2. 这说明 PMA 可能赢在“cheating full trajectory schedule”，而不是 metric 本身明显强于 SEA。
3. 后续要重点做 online proxy / schedule stability，而不是继续堆 perceptual metric。

---

## No-Go

出现下面任意情况，建议及时调整方向：

1. `SEAInput-oracle` 稳定优于 PMA。
2. PMA 在 E3 schedule-level 看起来好，但 E4 actual rerun 后不赢。
3. PMA 只在 RR≈0.50 这种高 refresh budget 下有微弱提升，RR≈0.30 完全失效。
4. `PMA-no-gate` 与 `PMA-stage-aware` 没区别，说明 stage-aware 不是关键。
5. DINO / LPIPS 的 early-stage 波动导致 schedule 极不稳定。

---

# 11. 如果 No-Go，怎么转向

## 情况 A：SEAInput-oracle 已经最好

说明 clean-image perceptual drift 没提供明显额外信号。可以转向：

```text
PixelGen-specific SEA filter
blockwise nonuniform cache
x-pred output residual cache
frequency-decoupled cache
```

也就是不再强调 perceptual manifold，而是强调 pixel-space spectral / residual dynamics。

## 情况 B：PMA schedule-level 赢，但 rerun 输

说明 metric 不是问题，问题是 cache 后 trajectory drift 会累积，oracle schedule 不稳定。可以转向：

```text
short-horizon refresh correction
multi-step cache alignment
DiCache-style trajectory alignment
PMA + cache correction
```

## 情况 C：DINO 赢、LPIPS 不赢

说明你的方法应该主打 semantic refresh，而不是 local texture。第二阶段 proxy 可以更轻，不一定需要 preview image。

## 情况 D：LPIPS 赢、DINO 不赢

说明 late-stage texture / detail 是关键。可以考虑只在后 30% 引入 LPIPS / edge-aware refresh。

---

# 12. 如果 Strong Go，第二阶段怎么接

## 路线 1：先做 training-free surrogate

如果 E5 表明 `SEA + time + accumulator` 能预测 PMA decision，就先做：

```text
PMA-Surrogate Cache
```

优点：

- 不训练 preview head；
- 更接近 SeaCache / TeaCache 的 training-free 风格；
- 对 AAAI 审稿更容易解释为 plug-and-play。

## 路线 2：shallow probe 回归 PMA distance

如果 training-free surrogate 不够好，做一个小 probe：

```text
input: shallow features + timestep
output: Δ_pma 或 refresh probability
loss: regression / BCE against PMA-oracle decision
```

这比 preview image head 更简单，建议优先。

## 路线 3：low-res preview head

只有在前两条都不行时，再做 preview image：

```text
shallow blocks -> tiny decoder -> 64×64 preview xhat
```

它最贴近“clean-image perceptual manifold”的故事，但实现成本最高。

---

# 13. 代码组织建议 v2

```text
project/
├── scripts/
│   ├── 00_sanity_full_reference.py
│   ├── 01_run_online_baselines.py
│   ├── 02_extract_distance_bank.py
│   ├── 03_calibrate_thresholds.py
│   ├── 04_make_oracle_schedules.py
│   ├── 05_rerun_with_fixed_schedule.py
│   ├── 06_bootstrap_stats.py
│   └── 07_plot_phase1.py
├── metrics/
│   ├── raw_metric.py
│   ├── sea_metric.py
│   ├── dino_metric.py
│   ├── lpips_metric.py
│   ├── edge_metric.py
│   └── pma_metric.py
├── schedules/
│   ├── uniform.py
│   ├── accumulated_threshold.py
│   ├── random_matched_rr.py
│   └── topk_oracle.py
├── outputs/
│   ├── references/
│   ├── distance_banks/
│   ├── schedules/
│   ├── reruns/
│   └── figures/
└── notebooks/
    └── phase1_analysis.ipynb
```

---

# 14. 最重要的实现注意事项

## 14.1 不要混淆 online schedule 和 oracle schedule

这是第一阶段最容易写错的地方。

- `SEAInput-online`：在 cached inference 过程中实时计算 metric。
- `SEAInput-oracle`：先跑 full trajectory，再用 full trajectory 的 metric 生成 fixed schedule。
- `PMA-oracle`：先跑 full trajectory，利用 full `xhat_t` 生成 fixed schedule。

主结论应该比较：

```text
PMA-oracle vs SEAInput-oracle
```

辅助说明再比较：

```text
PMA-oracle vs SEAInput-online
```

---

## 14.2 不要报告 PMA-oracle 的真实 latency

PMA-oracle schedule 是 cheating 的，它需要先知道 full trajectory。  
所以第一阶段只能说：

```text
At the same refresh ratio / same number of denoiser calls, PMA-oracle preserves the full trajectory better.
```

不能说：

```text
PMA-oracle is faster in real deployment.
```

第二阶段有 proxy 之后，才能报告真实 latency。

---

## 14.3 calibration set 和 test set 必须分开

推荐：

```text
calibration: 32 samples
test-pilot: 64 or 128 samples
test-main: 256 samples
```

Threshold、normalization median、stage weights不要在 test set 上反复调。

---

## 14.4 只调很少的东西

第一阶段不要网格搜索很多权重。  
最多比较：

```text
PMA-no-gate
PMA-stage-aware
```

不要再额外调 20 组权重，否则很容易变成过拟合实验。

---

# 15. 我现在对这个计划的判断

原计划的方向是对的，但实验层级需要更严谨：

- **distance bank 只能证明 metric 有形状，不足以证明 cache 结果更好；**
- **oracle-schedule rerun 才是第一阶段最关键证据；**
- **PMA 必须和 SEA-oracle 比，而不只是和 SEA-online 比；**
- **Edge、JiT、FID/IS 都不该挤进第一阶段主线。**

如果这一版计划跑出 Strong Go，那么你的论文主张会非常清楚：

> PixelGen 的 perceptual manifold 不只是 training-time supervision target，也可以作为 inference-time cache scheduling geometry。现有 cache 方法主要在 raw / filtered input feature space 中判断 redundancy，而 PMA-Cache 证明了 x-pred pixel diffusion 可以利用 clean-image perceptual drift 产生更优的 refresh schedule。

如果跑不出来，也能在一周左右明确止损，并知道该转向 spectral residual / blockwise cache，而不是继续堆 DINO、LPIPS、Edge。

---

# 16. 参考依据

- PixelGen: Pixel Diffusion Beats Latent Diffusion with Perceptual Loss. 核心依据：x-pred clean image、LPIPS local texture、P-DINO global semantics、high-noise early stage 不适合 perceptual constraint。
- SeaCache: Spectral-Evolution-Aware Cache for Accelerating Diffusion Models. 核心依据：raw feature distance 会混合 content/noise，SEA(input) 通过频谱过滤改善 refresh metric。
- TeaCache: Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model. 核心依据：用 timestep-modulated input 估计 output difference，并用 accumulated threshold 做 cache。
- DiCache: Let Diffusion Model Determine Its Own Cache. 核心依据：shallow-layer online probe 可作为 sample-specific cache indicator，启发第二阶段 proxy 设计。
