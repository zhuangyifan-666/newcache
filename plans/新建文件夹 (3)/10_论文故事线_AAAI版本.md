# 10｜论文故事线：AAAI 版本怎么讲 xWPCache-v2？

这份文档讲如何把实验组织成论文故事。重点是避免写成“我调了一个 cache schedule”。

---

## 1. 论文核心问题

现有 cache 方法大多问：

```text
相邻 timestep 的 feature 变了多少？
```

你的方法要问：

```text
对 x-prediction pixel diffusion，复用过期 clean prediction 会对最终感知结果造成多大 solver error？
```

这两个问题不同。

---

## 2. 三个 observation

### Observation 1：x-pred 模型暴露 clean prediction trajectory

PixelGen / JiT 类模型直接预测 clean image：

```text
xhat = net(x_t, t, c)
```

这使得 cache 误差可以从 clean prediction 角度分析，而不只是从内部 feature 分析。

---

### Observation 2：感知损伤应该看最终图，而不是只看中间距离

E5 的单点干预测：

```text
skip 一个 call 后，最终图坏多少？
```

这比“相邻 feature drift”更接近真实目标。

---

### Observation 3：单点安全不代表连续复用安全

E5.5 证明：

```text
连续 skip 的损伤可以累积。
```

这自然引出 accumulated solver risk。

---

## 3. 方法贡献

可以写成三条：

### Contribution 1：Causal perceptual cache diagnosis

提出 E5/E5.5 这种干预实验：

```text
用最终图 LPIPS/DINO 衡量 skip 某个 call/window 的真实因果损伤。
```

这不是最终 cache 方法，但它是可靠分析工具。

---

### Contribution 2：x-prediction cache risk derivation

从 PixelGen 的 x-prediction 和 velocity conversion 推导：

```text
cache error ∝ clean prediction error / clip(1-t)
```

再乘 step size 得到 solver error。

---

### Contribution 3：xWPCache-v2

提出：

```text
Wiener clean proxy
+ perceptual frequency weighting
+ anchor-relative vector accumulated solver error
+ E5.5-calibrated threshold
```

作为 training-free x-pred-specific cache refresh rule。

---

## 4. 和 SeaCache 的关系怎么写？

SeaCache 的思想是：

```text
raw feature distance 混合 content 和 noise，
所以用 timestep-dependent spectral filter 得到更合理的 cache metric。
```

你的方法不是简单换滤波器，而是进一步利用 x-pred 模型结构：

```text
SeaCache:
    feature-side spectral redundancy

xWPCache:
    output-side clean prediction solver error
```

对比表：

| 方法 | 判断空间 | 数学来源 | 是否 x-pred 专属 |
|---|---|---|---|
| TeaCache | timestep-modulated input feature | input-output change proxy | 否 |
| SeaCache | SEA-filtered input feature | linear denoising spectral evolution | 否 |
| xWPCache | clean prediction perceptual solver residual | x-pred velocity conversion + Wiener proxy | 是 |

---

## 5. 论文实验结构

### Section 4.1：E5/E5.5 causal analysis

展示：

```text
单点 skip PIS heatmap
连续 skip window PIS heatmap
单点安全 vs 连续危险案例
```

结论：

```text
需要 window-level accumulated risk。
```

---

### Section 4.2：E6-D0 risk diagnostic

主表：

```text
不同 risk score 对 E5.5 PIS 的预测能力。
```

这是方法可信度的核心证据。

---

### Section 4.3：Online cache results

在相同 RR/latency 下比较：

```text
Uniform
Raw/Tea-style
SEA-style
xWPCache-v2
```

指标：

```text
LPIPS/DINO/PSNR/SSIM/latency/RR
```

---

### Section 4.4：Ablation

验证：

```text
Wiener proxy
perceptual weight
ODE factor
vector accumulation
uncertainty
trust-region
```

---

### Section 4.5：Generalization

如果有算力：

```text
不同 seeds
不同 class subset
不同 step count
不同 x-pred checkpoint
```

如果没有算力，至少做：

```text
main64 + leave-one-sample-out calibration
```

---

## 6. 论文标题备选

```text
Causal Perceptual Caching for x-Prediction Pixel Diffusion

xWPCache: Wiener Perceptual Cache for x-Prediction Diffusion Inference

From Feature Drift to Clean-Prediction Risk: Cache Acceleration for Pixel Diffusion
```

---

## 7. reviewer 可能问什么？

### Q1：为什么不用 SeaCache 就够了？

答：SeaCache 是 feature-side redundancy metric，而 xWPCache 利用 x-prediction 的 clean prediction output 和 velocity conversion，直接建模 cache-induced solver error。E6-D0 和 online 实验证明它在 x-pred PixelGen 上更能预测/避免 causal perceptual damage。

---

### Q2：Wiener proxy 太简单，真的能代表 xhat 吗？

答：我们不声称它精确重建 xhat，而是作为 training-free approximation。我们用 Full-xhat oracle 作为上限，并在 E6-D0 中比较 Wiener proxy 与 oracle 的差距。如果差距存在，也可作为未来 work 引入 prefix probe。

---

### Q3：是不是只学了 timestep？

答：我们加入 time-only baseline 和 controlled correlation。若 xWPCache 在控制 start_call/window_len 后仍能预测 PIS，并且 schedule 在不同样本间有差异，说明它不仅是 timestep rule。

---

### Q4：阈值是不是调出来的？

答：阈值只用 E5.5 window labels 做 validation，且采用 leave-one-sample-out。online 结果在 heldout samples / larger sample set 上报告。

---

## 8. 一句话总结

> **论文不要讲“我们发现某些 timestep 可以 skip”，而要讲“我们从 x-prediction 的 clean prediction error 推导 cache-induced solver risk，并用 E5.5 的因果感知损伤验证这个 risk 能指导刷新”。**
