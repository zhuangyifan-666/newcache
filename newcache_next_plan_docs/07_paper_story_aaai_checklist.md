# 07. AAAI 论文故事线与检查清单

## 0. 先说现实判断

你现在的方向是有论文潜力的，但不要把目标定成“我一定要证明一个复杂大方法全面超过所有 cache paper”。更稳的路线是：

> 把 x-prediction 模型的 clean-image prediction trajectory 作为新的 cache 研究对象，提出感知流形刷新判据，并给出可部署的 online proxy。

这条线的好处是：它不是泛化到所有 diffusion 的大而空主张，而是非常明确地针对 PixelGen / JiT 这类 x-pred pixel diffusion。范围更窄，创新更清楚，也更适合本科生在有限算力下完成。

---

## 1. 推荐论文标题

可以考虑：

1. **PMA-Cache: Perceptual-Manifold-Aware Caching for x-Prediction Pixel Diffusion**
2. **Caching on the Perceptual Trajectory of x-Prediction Diffusion Models**
3. **Perceptual Refresh: Cache Acceleration for Pixel Diffusion via Clean-Image Prediction Drift**
4. **xP-Cache: Exploiting Clean-Image Prediction for Efficient Pixel Diffusion Sampling**

我最推荐第 1 或第 3 个。第 1 个像方法名，第 3 个更容易让 reviewer 立刻理解。

---

## 2. Abstract 应该讲什么

结构：

1. 现有 cache 方法多在 noisy input / hidden feature 上判断相邻 timestep 是否冗余。
2. 对 x-prediction pixel diffusion，这种判据忽略了模型直接预测 clean image 的特殊轨迹。
3. 我们发现 clean-image perceptual drift 更能反映 cache 风险；SEA-filtered input 与 perceptual drift 有强相关性。
4. 提出 PMA-Cache，用轻量 online proxy 预测 perceptual drift，并用 solver-aware accumulated gate 刷新。
5. 在 PixelGen / ImageNet 上，在相同 refresh ratio / latency 下优于 Uniform、RawInput、SEAInput 和若干 cache baselines。

避免写：

- “我们解决了所有扩散模型推理加速问题”；
- “完全不需要任何校准”如果你用了 calibration；
- “感知流形”但不给具体定义。

---

## 3. Introduction 逻辑

### 3.1 第一段：背景

多模态生成模型质量越来越高，但 diffusion / flow sampling 仍然需要多步 denoising，推理慢。Cache acceleration 是训练自由的方向，通过复用相邻 timestep 的中间结果减少 denoiser forward。

### 3.2 第二段：现有方法问题

现有 cache 主要用 raw input feature、timestep embedding modulated feature、magnitude、frequency 或 token sensitivity 来判断是否 refresh。这些方法大多没有利用 x-prediction 模型的一个特殊事实：模型每一步直接输出 clean image estimate。

### 3.3 第三段：x-pred 的机会

PixelGen / JiT 类模型中，denoiser 输出 `xhat_t`，再转换成 velocity 用于采样。因此，cache 决策可以从“noisy feature 是否变化”转为“clean-image prediction 在感知上是否变化”。

### 3.4 第四段：挑战

直接用 DINO / LPIPS 计算 `xhat_t` 的感知 drift 是 oracle，不能在线加速。因此需要 cheap proxy。

### 3.5 第五段：你的方法

提出 PMA-Cache：先用 full trajectory 分析感知 drift；再用 SEA-filtered first-block proxy、timestep、call-kind 等轻量特征预测 perceptual risk；最后用 solver-aware accumulated gate 做在线 cache。

### 3.6 Contributions

建议写三条：

1. **Observation / Analysis**：首次系统分析 x-prediction pixel diffusion 的 clean-image perceptual drift 与 cache 风险之间的关系。
2. **Method**：提出 PMA-Cache，一个轻量校准、在线可部署的感知流形 cache 策略。
3. **Experiments**：在 PixelGen ImageNet 上，PMA-Cache 在 matched RR / latency 下优于多个 cache baseline，并改善 tail failure。

---

## 4. Method 章节结构

### 4.1 x-prediction trajectory

说明：

```text
xhat_t = f_theta(x_t, t, c)
v_t = (xhat_t - x_t)/(1-t)
```

强调这是你方法针对 x-pred 的基础。

### 4.2 Perceptual-manifold cache criterion

定义 oracle drift：

```text
D_pma = w1 * SEA + w2 * DINO(xhat) + w3 * LPIPS(xhat)
```

解释每一项：

- SEA：频谱内容信号；
- DINO：全局语义；
- LPIPS：局部感知。

### 4.3 Oracle diagnosis

用 E2/E4 结果说明：

- Raw 与 perceptual drift 弱相关；
- SEA 与 perceptual drift 更相关；
- PMA oracle 在高 RR 有明显上限；
- refresh 位置比数量重要。

### 4.4 Online perceptual proxy

介绍：

```text
s_c = g_phi(z_c)
```

其中 `z_c` 是 cheap features。说明 calibration-only，不训练 diffusion 模型。

### 4.5 Solver-aware refresh gate

解释 99 calls 和 predictor/corrector 的问题，给出 accumulated gate。

### 4.6 Complexity

说明在线开销：

- first-block proxy 已有；
- SEA filter 是轻量 FFT；
- ridge proxy 是几乎免费的矩阵乘；
- 不运行 DINO / LPIPS。

---

## 5. Experiments 章节结构

### 5.1 Setup

- PixelGen XL / ImageNet 256；
- Heun exact 50 steps = 99 calls；
- no CFG；
- same seeds / class labels；
- RTX 3090 或你的硬件；
- metrics。

### 5.2 Main results

表：matched RR 0.30 / 0.50。

图：speed-quality curve。

### 5.3 Analysis of perceptual trajectory

放 E2 correlation、per-call curves、refresh heatmap。

### 5.4 Ablations

- w/o stage-kind normalization；
- w/o uncertainty；
- SEA only；
- heuristic vs ridge；
- beta；
- reuse xhat vs reuse velocity。

### 5.5 Generation quality

FID / IS / Precision / Recall。

### 5.6 Qualitative

正常样本 + worst cases。

---

## 6. 你应当避免的论文表述

### 6.1 不要把 PMA-oracle 当方法

PMA-oracle 是 upper bound，不是可部署方法。主方法必须是 PMA-Proxy online。

### 6.2 不要夸大“训练自由”

如果用了 ridge calibration，写：

> training-free with respect to the diffusion model; only a lightweight calibration on a small set is used.

或者提供 fully training-free heuristic 版本。

### 6.3 不要声称“感知流形”是数学严格流形

你可以写：

> We use the term perceptual manifold operationally, referring to the feature space induced by LPIPS, DINO and spectrum-aware signal components.

中文：这里的“流形”是工程化定义，不是证明了一个严格的微分几何流形。

### 6.4 不要横向夸大 FID

你的主要贡献是 cache 加速，不是训练一个新生成模型。

---

## 7. Rebuttal 预案

### Reviewer：为什么不直接用 SeaCache？

回答：SeaCache 关注 spectral evolution，并在 filtered input space 做 distance。我们的分析显示，在 x-pred 模型中，clean-image perceptual drift 是更接近 cache error 的目标；SEA 是有效 proxy，但不是完整 perceptual risk。PMA-Cache 利用 x-pred-specific clean image trajectory 和 solver-aware proxy，在 high RR / tail metrics 上进一步改进。

### Reviewer：DINO / LPIPS 在线太慢。

回答：主方法在线不运行它们。DINO / LPIPS 只用于 oracle analysis、calibration labels 和 evaluation。

### Reviewer：只在 PixelGen 上测，泛化不足。

回答：本文目标是 x-prediction pixel diffusion。可以补一个 JiT 或不同 checkpoint / steps 的小实验。若暂时没有，承认未来工作，但强调这是首个针对该类模型的 cache 分析。

### Reviewer：calibration set 会不会过拟合？

回答：使用小 calibration set，测试集严格分离；报告多 seed、多 split；heuristic fully training-free 版本也有效。

### Reviewer：paired metric 不是最终质量。

回答：补 FID / IS / Precision / Recall，并展示 cached distribution 没有明显退化。

---

## 8. 最终投稿前 checklist

### 方法完整性

- [ ] 有真正 online 的 PMA-Cache。
- [ ] 主方法不在线运行 DINO / LPIPS。
- [ ] 有明确伪代码。
- [ ] 有复杂度分析。
- [ ] 有 target RR calibration 方法。

### 实验完整性

- [ ] Full / Uniform / Raw / SEA / PMA-Proxy。
- [ ] 至少一个 TeaCache-style 或 MagCache-style baseline。
- [ ] 1024 paired samples。
- [ ] 5k 或 10k FID。
- [ ] RR 0.30 / 0.50 主表。
- [ ] speed-quality curve。
- [ ] tail failure 分析。
- [ ] ablation。

### 论文可信度

- [ ] 不把 oracle 当主方法。
- [ ] 不夸大泛化。
- [ ] 代码和配置可复现。
- [ ] 所有表格写清楚 actual RR 和 latency。
- [ ] Qualitative 图不只 cherry-pick。

---

## 9. 论文成败的关键判断

如果 E6 出现下面结果，论文很有希望：

```text
PMA-Proxy online 在 RR=0.40/0.50 明显优于 SEAInput-online，
在 RR=0.30 至少不差，并且 p95 LPIPS 更稳。
```

如果 RR=0.30 也赢，那就是强结果。

如果 PMA-Proxy 平均不赢，但 worst-case 明显改善，也可以转成“robust perceptual cache”的故事。

如果 PMA-Proxy 完全不赢 SEAInput，那么退路是写成分析型论文：证明 x-pred 的 perceptual oracle 上限存在，并提出更强的 solver-aware SEAInput / xhat reuse 策略。

