# 06. 指标与评测协议

## 0. 核心原则

cache 加速论文的评测不能只看一两个平均指标。你需要同时证明：

1. **快**：真实 latency / denoiser calls / FLOPs 降低；
2. **像 full model**：cached output 接近 uncached trajectory；
3. **质量没塌**：FID / IS / Precision / Recall 不显著下降；
4. **最差样本稳**：不是平均好看但少数样本崩坏；
5. **matched budget 公平**：在相同 RR 或相同 latency 下比较。

---

## 1. Paired full-reference fidelity

### 1.1 定义

Paired fidelity 是把同一个 seed、同一个 class/prompt 下：

```text
full uncached sample  vs  cached sample
```

做成一一对应比较。

它回答的问题是：

> cache 后的采样轨迹是否接近原模型的完整采样轨迹？

这对 cache 论文非常重要，因为 cache 的目标通常不是创造一个新生成器，而是尽量复现原生成器同时减少计算。

### 1.2 必报指标

| 指标 | 越大/小 | 解释 |
|---|---|---|
| PSNR | ↑ | 像素级接近程度；对小误差敏感 |
| SSIM | ↑ | 结构相似度；比 PSNR 更接近视觉结构 |
| LPIPS | ↓ | 感知距离；更接近人眼感受 |
| DINO distance | ↓ | 语义特征差异 |
| p95 LPIPS | ↓ | 最差 5% 样本是否稳定 |
| p99 LPIPS | ↓ | 极端失败风险 |

PSNR 很高不一定代表感知质量好；LPIPS 和 DINO 更重要。PMA 方法的卖点是感知流形，所以主文里建议重点报 LPIPS / DINO / p95 LPIPS。

### 1.3 样本规模

| 阶段 | 样本数 | 用途 |
|---|---:|---|
| debug | 32 / 64 | 检查代码 |
| pilot | 192 | 和 E4 对齐 |
| main | 1024 | 主表推荐 |
| stress | 2048+ | 算力允许时做 tail analysis |

---

## 2. Refresh ratio 与速度

### 2.1 RR 定义必须统一

你的 Heun exact 有：

```text
50 predictor calls + 49 corrector calls = 99 denoiser opportunities
```

所以：

```text
RR = number_of_refresh_calls / 99
```

不要按 50 solver steps 算 RR，否则会和真实 denoiser forward 数不一致。

### 2.2 速度指标

| 指标 | 说明 |
|---|---|
| Actual RR | 实际刷新比例 |
| Denoiser calls | 实际完整 forward 次数 |
| Wall-clock latency | 端到端生成耗时 |
| Denoiser-only time | 只统计 denoiser forward 时间 |
| Cache overhead | proxy / FFT / controller 的额外时间 |
| Peak memory | 可选，但有助于说明部署性 |

主文建议写：

```text
Speedup = latency_full / latency_cache
```

并单独报 overhead，证明 PMA proxy 没有抵消加速收益。

### 2.3 matched RR vs matched latency

两种比较都要有：

- **matched RR**：大家刷新同样比例，比较质量；
- **matched latency**：大家耗时接近，比较质量。

如果只做 matched RR，reviewer 可能说你的 proxy overhead 更高；如果只做 matched latency，reviewer 可能说 refresh budget 不公平。

---

## 3. Unpaired generation quality

### 3.1 为什么需要 FID

paired fidelity 只能说明 cached sample 接近 full sample，但不能说明整体生成分布仍然好。比如某方法可能很接近 full output，但 full output 本身在小样本上偶然有偏差；或者 cache 稍微改变分布却 paired metric 看不出来。

所以必须补：

- FID；
- IS；
- Precision / Recall。

### 3.2 样本规模建议

| 规模 | 建议用途 |
|---|---|
| 1k | debug，不建议作为主结论 |
| 5k | quick FID，可放 appendix 或 pilot |
| 10k | 算力紧张时主文可用，但需说明 |
| 50k | 标准 ImageNet 评测，最好最终有 |

如果你没有足够算力跑 50k，可以在论文里把重点放在 matched-fidelity cache，不要和大模型论文的 FID 做横向强结论。

### 3.3 FID 表建议

| Method | RR | FID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---:|---:|---:|---:|---:|
| Full | 1.00 | ... | ... | ... | ... |
| Uniform | 0.30 | ... | ... | ... | ... |
| SEAInput | 0.30 | ... | ... | ... | ... |
| PMA-Proxy | 0.30 | ... | ... | ... | ... |
| SEAInput | 0.50 | ... | ... | ... | ... |
| PMA-Proxy | 0.50 | ... | ... | ... | ... |

重点是同 checkpoint、同 sampler、同 seeds / class distribution 下比较。

---

## 4. Tail failure 分析

cache 方法很容易平均指标不错，但某些 class 失败。建议单独做：

### 4.1 per-sample error 分布

画 CDF：

```text
x-axis: per-sample LPIPS
 y-axis: fraction of samples <= x
```

如果 PMA-Cache 的曲线整体在 SEAInput 左边，说明更稳。

### 4.2 worst-k grid

每个方法选 LPIPS 最高的 16 个样本，展示：

```text
Full | SEAInput | PMA-Cache | error map
```

如果 PMA-Cache 对最差样本更稳，这张图很有说服力。

### 4.3 class-wise failure

ImageNet class-to-image 可以统计每个 class 的平均 LPIPS / failure count。看看：

- 细纹理类：鸟、鱼、昆虫；
- 结构类：车辆、建筑；
- 多物体类：食物、场景。

PMA 的 DINO branch 可能对结构类更有效，LPIPS branch 对细纹理类更有效。

---

## 5. Proxy quality 评测

E5 proxy 的指标不要混到 E6 真实采样表里，单独一节分析。

| 指标 | 为什么有用 |
|---|---|
| Spearman | cache schedule 更关心排序 |
| Pearson | 看数值预测是否线性相关 |
| Top-20% risk recall | 能否抓住高风险 transition |
| Schedule overlap | 与 PMA-oracle 的 refresh 决策重合度 |
| RR calibration error | target RR 与 actual RR 差距 |
| Stage-kind ECE | 不同阶段是否预测偏差 |

Top-k risk recall 的定义：

```text
oracle_high_risk = top 20% transitions by oracle PMA score
proxy_selected   = transitions refreshed by proxy schedule
recall = |oracle_high_risk ∩ proxy_selected| / |oracle_high_risk|
```

---

## 6. 统计显著性

建议用 bootstrap 置信区间。

### 6.1 怎么做

对 1024 个 samples 重采样 1000 次，每次算平均 LPIPS / PSNR，得到 95% CI。

```python
for b in range(1000):
    idx = np.random.choice(N, size=N, replace=True)
    metric_b = metric[idx].mean()
```

表中可写：

```text
LPIPS 0.046 ± 0.003
```

或者：

```text
95% CI: [0.043, 0.049]
```

### 6.2 为什么重要

你的方法如果只比 SEAInput 好一点点，没有 CI 就很难说服 reviewer。如果 CI 不重叠，说服力强很多。

---

## 7. Stress tests

### 7.1 不同 RR

至少：

```text
RR = 0.20, 0.30, 0.40, 0.50, 0.60
```

低 RR 看极限加速，高 RR 看高质量场景。

### 7.2 不同随机种子

至少 3 个 seed 做 192-sample pilot：

```text
seed = 0, 1, 2
```

主表可以用 seed=0 的 1024 samples，但 appendix 放多 seed 稳定性。

### 7.3 不同 class subset

不要只用前 192 个 class 或固定顺序。建议：

- balanced across 1000 classes；
- 或固定 random class list 并公开。

### 7.4 不同 sampler / steps：可选

如果时间允许，比较：

- Heun 50；
- Euler 50；
- Heun 25。

这能证明方法不是只对一个采样器过拟合。

---

## 8. Qualitative figures

### 8.1 主文图

建议展示 4–8 组：

```text
Full | Uniform | SEAInput | PMA-Cache
```

选择原则：

- 不要只挑 PMA 最好看的；
- 包含普通样本和困难样本；
- 展示局部纹理、语义结构、文字/物体边界等。

### 8.2 Error map

可以用：

```text
abs(cache - full).mean(channel)
```

或者 LPIPS patch map。如果实现复杂，用 RGB absolute error map 即可。

---

## 9. 常见评测坑

### 9.1 不要把 oracle 当 online speedup

PMA-oracle 用了 full trajectory 里的 DINO / LPIPS，只能当上限。主方法必须是 PMA-Proxy online。

### 9.2 不要只报 RR，不报 latency

FFT / proxy 虽然便宜，但必须实测。

### 9.3 不要用不同 seeds 比 paired fidelity

paired fidelity 必须同 seed、同 class、同 sampler。

### 9.4 不要只跑 192 samples 就下强结论

192 可以探索，主结论至少 1024 paired。

### 9.5 不要横向比较不同论文的 FID

你的 FID 主要用于同模型同设置下比较 cache 方法，不要声称全面超过某个大模型，除非评测协议完全一致。

---

## 10. 推荐最终评测组合

最小组合：

```text
192 samples: all ablations
1024 samples: main paired table
5000 samples: quick FID
RR: 0.30, 0.50
methods: Full, Uniform, RawInput, SEAInput, TeaCache-style, PMA-Proxy
```

强组合：

```text
1024/2048 samples: main paired + tail
10000/50000 samples: FID/IS/Prec/Rec
RR: 0.20–0.60 curve
methods: add MagCache-style, forecast extension, oracle upper bound
```

