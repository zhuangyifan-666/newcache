# 06｜阈值校准与非经验化 refresh 规则

这份文档讲：当 E6-D0 证明某个 xWPCache risk score 有效后，如何把它变成 online refresh rule。

重点：

> **阈值不能用“看起来某个 call 段安全”来定。阈值要用 E5.5 的 window-level PIS 做校准。**

---

## 1. Online rule 的基本形式

每个 call 之前，计算如果继续 skip 会得到的风险：

\[
Risk_{next}
\]

然后：

```text
if no cache:
    refresh
elif Risk_next > delta:
    refresh
elif trust_region_broken:
    refresh
else:
    skip
```

其中：

```text
delta 是风险阈值。
trust_region 是防止 proxy 失效的安全约束。
```

---

## 2. delta 怎么从 E5.5 校准？

对每个 E5.5 window W，你有：

```text
Y(W) = 真实 PIS
R(W) = xWPCache 预测 risk
```

我们希望：

```text
R(W) 高 → 应该 refresh
R(W) 低 → 可以 skip
```

定义损失：

\[
\mathcal{L}(\delta)
=
\sum_W Y(W)\mathbf{1}[R(W)\leq\delta]
+
\lambda\sum_W \mathbf{1}[R(W)>\delta]
\]

这句话翻译成中文：

```text
如果真实损伤 Y(W) 很大，但 risk 小于阈值，被认为可以 skip：重罚。
如果 risk 大于阈值，被 refresh：损失一点速度，也罚，但罚得轻。
```

其中 \(\lambda\) 控制质量和速度的权衡。

---

## 3. 用 matched budget 选 delta

论文里更常用的是 matched refresh ratio。

例如目标：

```text
RR = 0.50
RR = 0.40
RR = 0.30
```

在 E5.5 windows 上选择 delta，使得：

```text
被判定 refresh 的 window 比例约等于目标 RR
```

然后比较不同 score 在相同 RR 下的：

```text
SkippedPIS
Dangerous false negative
CapturedPIS
```

这能公平比较 xWPCache、SEAInput、raw、time-only。

---

## 4. train/validation 划分

现在 main8 样本很少，但仍然要避免过拟合。

推荐：leave-one-sample-out。

```text
每次拿 7 个 sample 的 windows 调 delta。
在剩下 1 个 sample 上测试。
循环 8 次，取平均。
```

输出：

```text
score_metrics_loso.csv
```

字段：

```text
heldout_sample
score_name
target_rr
delta_selected
actual_rr
skipped_pis
captured_pis
false_negative_rate
```

---

## 5. trust-region 不是经验分段

E5.5 证明连续 skip 可能非线性累积，所以 online 里可以加入 trust-region。

但不能写成：

```text
call 50-77 max_age=8
call 88-97 max_age=2
```

这又回到了经验方法。

推荐两种非经验 trust-region。

---

### 5.1 全局 max_age

```text
if cache_age >= M:
    refresh
```

其中 M 是全局值，不依赖 call 区间。

M 通过 E5.5 validation 选择：

```text
M ∈ {2, 4, 6, 8, 12, 16, inf}
```

如果没有 max_age 时 false negative 很高，而 M=8 明显降低 false negative，说明它作为 trust-region cap 是合理的。

---

### 5.2 risk-based age penalty

比硬 max_age 更数学一点：

\[
Risk_{next}=Risk_{xwp}+\mu\left(\frac{age}{M}\right)^p
\]

含义：

```text
cache 复用越久，对 proxy 的信任越低。
```

推荐初始：

```text
p = 2
M = 8
mu 通过 validation 选
```

这不是按具体 call 段手工调，而是一个全局 trust-region 惩罚。

---

## 6. uncertainty gate 怎么定？

如果使用 SNR / uncertainty：

```text
Risk = Risk_xwp + eta * U(t)
```

eta 也应该通过 E5.5 validation 选。

候选：

```text
eta ∈ {0, 0.01, 0.03, 0.1, 0.3}
```

判断标准：

```text
early / transition false negative 是否下降？
整体 refresh ratio 是否过高？
```

如果 eta 太大，算法会退化成早期强制 refresh。不是不能用，但要证明它不是纯经验：

```text
加 eta 后，E5.5 false negative 明显下降；
controlled correlation 仍然高于 time-only；
```

---

## 7. threshold_sweep.csv

建议输出：

```text
score_name
target_rr
delta
actual_rr
skipped_pis_lpips
skipped_pis_dino
skipped_pis_total
captured_pis_total
false_negative_rate
wasted_refresh_rate
```

其中：

```text
wasted_refresh_rate = 被 refresh 的 windows 中，真实 PIS 很低的比例
```

这个指标帮助你判断算法是否太保守。

---

## 8. 进入 online 前的选择标准

选主公式时，不要只看 Spearman。

推荐综合顺序：

```text
1. false_negative_rate 低
2. skipped_pis 低
3. PR-AUC 高
4. Spearman 高
5. 公式复杂度适中
6. 运行开销低
```

为什么 false negative 第一？

因为 cache 方法最怕：

```text
把真实危险的 window 当成安全，导致图坏。
```

多 refresh 一点只是速度损失；漏掉危险 window 是质量损失。

---

## 9. online 初始 delta 设置

假设 E6-D0 在 main8 上得到：

```text
RR=0.50 对应 delta=0.12
RR=0.40 对应 delta=0.18
RR=0.30 对应 delta=0.26
```

那 online E6 的第一批实验就跑：

```text
delta in {0.12, 0.18, 0.26}
```

然后看真实 online RR 是否匹配。

如果 online RR 偏差很大，说明：

```text
full trajectory 上的 risk 分布和 online skip trajectory 上的 risk 分布不同。
```

这很正常，需要做 online delta calibration：

```text
小样本 online pilot 调整 delta，使 RR 对齐。
```

但注意：调 delta 只能对齐 RR，不能根据最终图质量手工调 call 段。

---

## 10. 一句话总结

> **delta、max_age、uncertainty 都应该通过 E5.5 window PIS 做 validation，而不是根据肉眼观察的 call 区间手工设置。这样 xWPCache 才不是经验 schedule。**
