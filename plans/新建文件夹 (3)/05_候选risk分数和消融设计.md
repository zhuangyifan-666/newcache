# 05｜候选 risk 分数和消融设计：证明 xWPCache 不是换皮 SeaCache，也不是 time-only

这份文档列出 E6-D0 必须比较的 risk 分数。目的不是把所有东西都塞进最终方法，而是通过消融证明：

```text
每个数学组件是否真的有用？
```

---

## 1. 总原则

你需要比较三类东西：

```text
1. 很弱但必要的 baseline：time-only、raw x_t。
2. 强 baseline：SEA-like、SEAInput、Full-xhat oracle。
3. xWPCache 逐步增强版本：Wiener → perceptual → ODE → vector accumulation → uncertainty。
```

如果 xWPCache 只比 raw 强，但不比 SEAInput 强，贡献不够。

如果 xWPCache 不比 time-only 强，说明方法其实只是学到了 timestep pattern。

如果 Full-xhat oracle 很强但 xWiener 很弱，说明 clean prediction 目标是对的，但 proxy 需要改。

---

## 2. Baseline 0：time-only

### 2.1 为什么必须有？

E5/E5.5 的结果具有明显阶段性，所以一个只看 call index 的方法可能也能取得不错分数。

如果不加 time-only baseline，reviewer 可能会说：

```text
你的 xWPCache 只是变相利用 timestep prior，并没有真正看内容。
```

### 2.2 版本

```text
R_time_len = window_len
R_time_ode = sum_i abs(h_i) / max(1-t_i, eps)
R_time_poly = ridge([start_call, end_call, window_len, mean_t])
```

`R_time_poly` 必须使用 leave-one-sample-out：

```text
用 7 个 sample 拟合，用剩下 1 个 sample 测试。
```

不能在训练集上报告。

---

## 3. Baseline 1：raw x_t

### 3.1 adjacent

\[
R_{raw-adj}(W)=\sum_{i=s}^{e}\text{RelL1}(x_i,x_{i-1})
\]

### 3.2 anchor

\[
R_{raw-anchor}(W)=\sum_{i=s}^{e}\text{RelL1}(x_i,x_{s-1})
\]

这个 baseline 回答：

```text
直接看当前 sampler state 是否足够？
```

---

## 4. Baseline 2：SEA-like

SeaCache 的核心思想是：raw feature distance 混合了 signal 和 noise，所以先用 timestep-dependent spectral filter 得到 spectrum-aware representation，再用 accumulated distance 做 refresh。

你需要实现一个近似 SEA-like baseline：

```text
R_sea_xt_adj
R_sea_xt_anchor
R_seainput_adj
R_seainput_anchor
```

如果你的仓库已有 SEAInput-online 逻辑，优先复用。

### 4.1 为什么它是强 baseline？

因为 SeaCache 已经证明，SEA-filtered input 比 raw input 更能贴近 SEA-filtered output distance。

你的 xWPCache 必须解释自己和 SeaCache 的区别：

```text
SeaCache:     对 input feature 做 spectral filtering，估计 redundancy。
xWPCache:     对 x-pred clean prediction error 做 solver-level perceptual risk modeling。
```

所以 E6-D0 里必须比较：

```text
SEA-style risk vs xWPCache risk
```

---

## 5. Candidate 1：xW-basic

只做 Wiener proxy，不做感知权重，不做 ODE factor。

\[
\bar{x}_i=\text{Wiener}(x_i,t_i)
\]

\[
R_{xw-basic}=\sum_i \text{RelL1}(\bar{x}_i,\hat{x}_{anchor})
\]

它回答：

```text
仅仅把 x_t 映射到 clean proxy 是否有用？
```

---

## 6. Candidate 2：xW-perceptual

加感知频率权重：

\[
z_i=\Phi_{t_i}(\bar{x}_i)
\]

\[
z_a=\Phi_{t_i}(\hat{x}_{anchor})
\]

\[
R_{xwp}=\sum_i \text{RelL1}(z_i,z_a)
\]

它回答：

```text
感知频域权重是否比普通 RGB / lowres L1 更能预测 LPIPS/DINO PIS？
```

---

## 7. Candidate 3：xW-perceptual-ODE

加入 solver factor：

\[
c_i=\frac{|h_i|}{\max(1-t_i,\epsilon)}
\]

\[
R_{xwp-ode}=\sum_i c_i\text{RelL1}(z_i,z_a)
\]

它回答：

```text
x-pred velocity conversion 推导出的 ODE factor 是否有用？
```

如果这个版本比不加 ODE 的差，可能说明：

```text
1. h_i 或 t_i 方向保存错了；
2. epsilon_clip 需要调；
3. corrector 的有效 solver coefficient 不是简单 abs(h)；
4. Wiener proxy 在 tail 不稳定，ODE factor 过度放大错误。
```

---

## 8. Candidate 4：xW-perceptual-ODE-vector

这是新版最重要的候选。

单步 residual：

\[
\tilde{e}_i=c_i(z_a-z_i)
\]

vector accumulated risk：

\[
R_{vec}=\left\|\sum_i\tilde{e}_i\right\|
\]

为什么重要？

因为 sampler state 的误差是带方向累积的，不一定等于每一步误差绝对值相加。

E5.5 正好可以检验：

```text
连续 skip 的真实最终损伤更像 scalar accumulation，还是 vector accumulation？
```

---

## 9. Candidate 5：xW-perceptual-ODE-vector + uncertainty

加入：

\[
U(t)=\frac{1}{\sqrt{SNR^P_t+\xi}}
\]

最终：

\[
R=R_{vec}+\eta\sum_i U(t_i)
\]

它回答：

```text
早期高噪声阶段 proxy 不可靠，是否需要 uncertainty 补偿？
```

如果 early false negative 很多，这个项通常有帮助。

---

## 10. Upper bound：Full-xhat oracle

用真实 full denoiser 输出：

```text
xhat_i，而不是 xbar_i
```

计算：

```text
R_oracle_scalar
R_oracle_vector
```

解释结果：

| 现象 | 含义 |
|---|---|
| oracle 好，xW 差 | clean prediction drift 是好目标，但 proxy 不够好 |
| oracle 差，xW 也差 | final damage 可能不只由 clean prediction drift 决定 |
| oracle 和 xW 都好 | 说明路线很强 |
| xW 比 oracle 好 | 检查代码，通常不应该稳定发生 |

---

## 11. 必做消融表

建议输出：

| score | Spearman ↑ | PR-AUC ↑ | FNR@30 ↓ | SkippedPIS@30 ↓ | controlled rho ↑ |
|---|---:|---:|---:|---:|---:|
| time-only |  |  |  |  |  |
| raw-anchor |  |  |  |  |  |
| SEAInput-anchor |  |  |  |  |  |
| xW-basic |  |  |  |  |  |
| xW-perceptual |  |  |  |  |  |
| xW-perceptual-ODE scalar |  |  |  |  |  |
| xW-perceptual-ODE vector |  |  |  |  |  |
| xW-perceptual-ODE vector + U |  |  |  |  |  |
| full-xhat oracle |  |  |  |  |  |

`FNR@30` 的意思是：在只 refresh top 30% risk windows 的情况下，真实危险 window 被误判安全的比例。

---

## 12. 通过条件

推荐判断：

```text
xW-v2 主分数至少要：
1. 明显超过 raw-anchor。
2. 至少接近或超过 SEAInput-anchor。
3. 超过 time-only controlled baseline。
4. false negative 低于 SEAInput。
5. 与 full-xhat oracle 的差距可以解释。
```

如果 xW-v2 没有超过 SEAInput，但 full-xhat oracle 超过 SEAInput，那么下一步应该改进 proxy，而不是放弃 clean-prediction risk。

---

## 13. 一句话总结

> **消融不是装饰，而是判断 xWPCache 是否真正成立的核心证据。你必须证明：Wiener proxy、感知权重、ODE factor、vector accumulation 每一层都不是凭空加的。**
