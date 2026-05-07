# 09. 算力有限时的最小可发表路线

## 0. 你应该优先做什么

如果你的算力、时间和基础都有限，千万不要一开始就跑大规模 FID 或实现一堆复杂 baseline。最小可发表路线是：

> 先证明 PMA-oracle 能被 cheap online proxy 近似，并且 PMA-Proxy online 在真实 cache rerun 中超过 SEAInput-online。

这一步不成立，后面所有大实验都没有意义。

---

## 1. 四周计划

### Week 1：E5 proxy dataset

目标：把 E2 distance bank 变成 proxy fitting 数据。

任务：

- 保存 cheap features：Raw、SEA、t、call kind、stage、proxy norm、delta p95；
- 保存 labels：DINO、LPIPS、PMA；
- 完成 64/64/128 split；
- 计算 Spearman、Top-k recall。

通过标准：

```text
PMA proxy 的 Spearman > RawInput
Top-20% risk recall > RawInput
```

如果不通过：先做 stage-kind normalization 和 log1p(SEA)。

### Week 2：E6 online PMA pilot

目标：真实在线跑 192 samples。

方法：

- Uniform；
- RawInput-online；
- SEAInput-online；
- PMA-Proxy heuristic；
- PMA-Proxy ridge。

RR：

```text
0.30, 0.40, 0.50
```

通过标准：

```text
RR=0.40/0.50: PMA-Proxy LPIPS < SEAInput-online
RR=0.30: PMA-Proxy 不明显差，或 p95 LPIPS 更好
```

### Week 3：ablation 和 tail

任务：

- w/o stage-kind normalization；
- w/o uncertainty；
- SEA only；
- heuristic vs ridge；
- worst-case grid；
- refresh heatmap。

通过标准：

```text
能解释为什么 PMA-Proxy 有效，而不是偶然调参。
```

### Week 4：扩展到 1024 paired

任务：

- 主方法和强 baseline 跑 1024 samples；
- bootstrap CI；
- speed-quality curve；
- 初版论文图。

通过标准：

```text
主结论在 1024 samples 上仍然成立。
```

---

## 2. 八周计划

在四周计划基础上增加：

### Week 5：外部 baseline 风格

- TeaCache-style polynomial；
- MagCache-style residual magnitude；
- 如果时间允许，加 Taylor-style forecast。

### Week 6：FID quick eval

- 5k FID；
- Full / SEAInput / PMA-Proxy；
- RR=0.30 和 0.50。

### Week 7：solver-aware extension

- two-threshold accumulator；
- predictor-corrector paired refresh；
- reuse xhat then convert。

### Week 8：论文初稿

- 写 Introduction / Method / Experiments；
- 整理图表；
- 找老师或同学内部 review。

---

## 3. 十二周强路线

如果结果好，并且有更多算力：

1. 10k 或 50k FID。
2. 多 seed paired evaluation。
3. 不同 sampler steps。
4. 另一个 x-pred checkpoint 或 text-to-image 小实验。
5. 完整开源代码和 reproducibility 文档。

---

## 4. Go / No-Go 决策点

### 决策点 1：E5 proxy 是否有效

如果 proxy 预测 oracle PMA 完全不如 SEAInput：

- 不要继续 PMA-Proxy；
- 改成 stage-kind SEAInput + solver-aware cache；
- PMA oracle 作为分析上限。

### 决策点 2：E6 online 是否超过 SEAInput

如果 PMA-Proxy online 平均指标不赢，但 p95/p99 赢：

- 论文故事改成 robust perceptual cache；
- 主打 tail failure protection。

如果平均和 tail 都不赢：

- 说明 DINO/LPIPS oracle 不能被 cheap proxy 稳定预测；
- 转向 xhat reuse / forecast。

### 决策点 3：FID 是否下降明显

如果 paired fidelity 好，但 FID 变差：

- 检查是否过度保守或过度复用导致分布偏移；
- 提高 RR 或限制 max_skip；
- 把主贡献定位为 faithful acceleration under paired reference，不夸大 generation quality。

---

## 5. 三条备选论文路线

### 路线 A：最理想

**PMA-Proxy online 全面优于 SEAInput / TeaCache-style / MagCache-style。**

论文主张：

> Perceptual-manifold-aware cache is a stronger refresh criterion for x-prediction pixel diffusion.

### 路线 B：稳健性路线

**平均指标接近 SEAInput，但 p95/p99 明显更好。**

论文主张：

> Perceptual proxy improves robustness and reduces tail artifacts in aggressive cache acceleration.

### 路线 C：分析型路线

**online proxy 不够强，但 oracle analysis 很清楚，SEAInput solver-aware 很强。**

论文主张：

> We provide the first systematic analysis of cache criteria for x-prediction pixel diffusion and propose a solver-aware spectral cache baseline.

这条路线也能投稿 workshop 或中文会议，再继续打磨顶会。

---

## 6. 最小主表模板

如果只做最小可发表版本，主表可以是：

| Method | Online | RR | Speedup | LPIPS ↓ | DINO ↓ | p95 LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| Uniform | yes | 0.30 | ... | ... | ... | ... |
| RawInput | yes | 0.30 | ... | ... | ... | ... |
| SEAInput | yes | 0.30 | ... | ... | ... | ... |
| PMA-oracle | no | 0.30 | - | ... | ... | ... |
| PMA-Proxy | yes | 0.30 | ... | ... | ... | ... |
| Uniform | yes | 0.50 | ... | ... | ... | ... |
| RawInput | yes | 0.50 | ... | ... | ... | ... |
| SEAInput | yes | 0.50 | ... | ... | ... | ... |
| PMA-oracle | no | 0.50 | - | ... | ... | ... |
| PMA-Proxy | yes | 0.50 | ... | ... | ... | ... |

加一张 curve 图和一张 refresh heatmap，论文就有基本骨架。

---

## 7. 每天工作建议

一个非常实际的节奏：

- 上午：跑实验或检查日志；
- 下午：分析 csv / 画图；
- 晚上：写实验记录，不要只改代码。

每天至少记录：

```text
今天跑了什么 config？
结果是否符合预期？
失败原因可能是什么？
下一步只改一个变量是什么？
```

不要同时改多个变量，否则结果好坏都无法解释。

---

## 8. 对你最重要的建议

你现在不要焦虑“我是不是一定能发 AAAI”。你应该把问题拆成一个个可验证的小命题：

1. SEA 是否是 perceptual drift 的好 proxy？你已经有证据。
2. PMA oracle 是否有上限？你已经有证据。
3. Cheap proxy 能否预测 PMA oracle？E5 回答。
4. Online PMA 是否真实加速且质量更好？E6 回答。
5. 结果是否足够稳定和完整？E8 回答。

只要第 3 和第 4 步做出来，你就有一篇论文的核心。

