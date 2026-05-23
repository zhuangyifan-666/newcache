# 02｜xWPCache-v2 数学定义：从 clean prediction 误差到累计 solver error

这份文档给出新版 xWPCache 的核心数学公式。目标是把方法从经验规则变成一个有来源的 cache 刷新准则。

---

## 1. x-prediction 模型里，cache 误差是什么？

PixelGen / JiT 类模型直接预测 clean image：

$$
\hat{x}_i = f_\theta(x_i, t_i, c)
$$

其中：

| 符号 | 含义 |
|---|---|
| $x_i$ | 当前 sampler state，也就是当前 noisy image |
| $t_i$ | 当前 timestep |
| $c$ | 条件信息，比如类别或文本 |
| $\hat{x}_i$ | 模型预测的 clean image，也就是 xhat |

PixelGen 的 velocity conversion 是：

$$
v_i = \frac{\hat{x}_i - x_i}{1 - t_i}
$$

实际实现里要对 $1-t_i$ 做 clip：

$$
v_i = \frac{\hat{x}_i - x_i}{\max(1-t_i, \epsilon)}
$$

如果当前 call 使用 cache，不算新的 $\hat{x}_i$，而是复用上一次 refresh 的：

$$
\hat{x}_r
$$

那么 velocity 误差大约是：

$$
\tilde{v}_i - v_i
=
\frac{\hat{x}_r - \hat{x}_i}{\max(1-t_i,\epsilon)}
$$

采样器更新还要乘 step size $h_i$：

$$
\Delta x_{i+1}
\approx
h_i(\tilde{v}_i - v_i)
=
\frac{h_i}{\max(1-t_i,\epsilon)}(\hat{x}_r - \hat{x}_i)
$$

这说明：

> **对 x-prediction 模型，cache 风险的核心不是 raw feature 变化，而是 cached clean prediction 和当前 clean prediction 的差异，并且这个差异会被 solver 系数放大。**

---

## 2. 问题：当前 $\hat{x}_i$ 不能直接算

如果你完整 forward 得到 $\hat{x}_i$，那这一步的计算成本已经花掉了，cache 就没有意义。

所以 xWPCache 的关键是：

```text
不用完整 denoiser forward，便宜地估计当前 clean prediction 的位置。
```

我们把这个便宜估计叫：

$$
\bar{x}_i
$$

读作：x bar。它不是模型输出，而是一个 proxy。

---

## 3. Wiener clean proxy

PixelGen 的 noising 过程是线性插值：

$$
x_t = t x_0 + (1-t)\epsilon
$$

其中：

| 符号 | 含义 |
|---|---|
| $x_0$ | 干净图像 |
| $\epsilon$ | 高斯噪声 |
| $x_t$ | 当前 noisy image |
| $t=0$ | 接近纯噪声 |
| $t=1$ | 接近干净图 |

在自然图像频谱先验下：

$$
S_x(f) \propto (f^2 + f_0^2)^{-\beta/2}
$$

可以得到一个 Wiener-like clean estimator：

$$
G_t(f)
=
\frac{t S_x(f)}{t^2 S_x(f) + (1-t)^2}
$$

于是：

$$
\bar{x}_t
=
\mathcal{F}^{-1}\left[ G_t(f) \mathcal{F}(x_t) \right]
$$

其中：

| 符号 | 含义 |
|---|---|
| $\mathcal{F}$ | FFT，傅里叶变换 |
| $\mathcal{F}^{-1}$ | iFFT，逆傅里叶变换 |
| $G_t(f)$ | timestep-dependent Wiener filter |
| $\bar{x}_t$ | 从当前 noisy image 估计出来的 clean proxy |

直观理解：

```text
早期噪声大，只相信较稳定的低频结构；
后期图更干净，允许更多频率通过。
```

---

## 4. 感知频域表示

我们不直接比较 RGB，而是比较感知加权后的表示：

$$
z_i = \Phi_{t_i}(\bar{x}_i)
$$

$$
z_r = \Phi_{t_i}(\hat{x}_r)
$$

其中 $\Phi_t$ 是：

```text
低分辨率下采样
→ FFT
→ 乘感知频率权重 W_t(f)
→ iFFT
→ 可选归一化
```

也就是：

$$
\Phi_t(u)
=
\mathcal{F}^{-1}\left[W_t(f)\mathcal{F}(u)\right]
$$

感知权重可以写成：

$$
W_t(f)=\lambda_D W_D(f)+\lambda_L q(t)W_L(f)
$$

其中：

| 项 | 含义 |
|---|---|
| $W_D$ | DINO-like，全局结构，偏低频 |
| $W_L$ | LPIPS-like，局部纹理，偏中高频 |
| $q(t)$ | noise gate，高噪声早期减少纹理权重 |

初始设置：

$$
q(t)=\text{clip}\left(\frac{t-0.3}{0.7},0,1\right)
$$

不要把这当成最终真理。它是一个有 PixelGen 动机的初始设计，后面必须通过 E6-D0 验证。

---

## 5. 单步 anchor residual

假设上一次 refresh 是 call $r$，当前候选 skip 是 call $i$。

定义感知残差：

$$
e_i
=
z_r - z_i
=
\Phi_{t_i}(\hat{x}_r)-\Phi_{t_i}(\bar{x}_i)
$$

这是：

```text
cached clean prediction 和当前 clean proxy 在感知空间里的差。
```

再乘 ODE factor：

$$
c_i
=
\frac{|h_i|}{\max(1-t_i,\epsilon)}
$$

得到 solver residual：

$$
\tilde{e}_i = c_i e_i
$$

这一步对应前面的推导：

```text
clean prediction 误差
→ velocity 误差
→ sampler state 误差
```

---

## 6. 标量累计 vs 向量累计

这是 v2 方案最重要的修改。

旧版只写：

$$
A = \sum_i \|\tilde{e}_i\|
$$

这叫 **scalar accumulated risk**。

但从 ODE 误差角度看，连续 skip 的 state error 更像：

$$
E = \sum_i \tilde{e}_i
$$

然后再取 norm：

$$
R_{vec}=\left\|\sum_i \tilde{e}_i\right\|
$$

这叫 **vector accumulated solver error**。

两者区别：

| 方式 | 公式 | 含义 |
|---|---|---|
| scalar | $\sum_i \|\tilde{e}_i\|$ | 每一步误差绝对值相加，不允许抵消 |
| vector | $\|\sum_i \tilde{e}_i\|$ | 先按方向累计，再看总误差，允许部分抵消 |

哪一个更好不能靠猜。E5.5 正好可以检验。

**新版 E6-D0 必须同时比较 scalar 和 vector 两种 accumulated risk。**

---

## 7. 推荐主风险公式

对一个连续 skip window $W=[s,e]$，假设 anchor 是 $r=s-1$。

### 7.1 scalar risk

$$
R_{scalar}(W)
=
\sum_{i=s}^{e}
\frac{|h_i|}{\max(1-t_i,\epsilon)}
\cdot
D_P(\bar{x}_i,\hat{x}_r;t_i)
$$

其中 $D_P$ 是感知距离。

---

### 7.2 vector risk

$$
R_{vec}(W)
=
\frac{
2\left\|\sum_{i=s}^{e} c_i\left[\Phi_{t_i}(\hat{x}_r)-\Phi_{t_i}(\bar{x}_i)\right]\right\|_1
}{
\sum_{i=s}^{e} c_i\left(\|\Phi_{t_i}(\hat{x}_r)\|_1+\|\Phi_{t_i}(\bar{x}_i)\|_1\right)+\xi
}
$$

这个公式看起来复杂，但意思很简单：

```text
每个 skipped call 都会产生一个带方向的 solver residual。
把这些 residual 累加起来。
最后看累计 residual 有多大。
```

---

### 7.3 uncertainty 项

早期高噪声阶段，$\bar{x}_i$ 本身不可靠，所以还需要 uncertainty：

$$
R(W)=R_{vec}(W)+\eta\sum_{i=s}^{e}U(t_i)
$$

其中 $U(t_i)$ 可以由 perceptual SNR 推出：

$$
U(t)=\frac{1}{\sqrt{SNR^P_t+\xi}}
$$

如果实现太复杂，第一版可以先用一个简单形式：

```python
U(t) = 1.0 / (t + 1e-4)
```

但正式版最好使用频域 SNR。

---

## 8. Online refresh rule

Online 时，从上一次 refresh 后维护：

```text
cached_xhat
accum_vector
accum_scalar
cache_age
```

每个 call：

```text
1. 用当前 x_t 计算 Wiener clean proxy xbar_t。
2. 计算 z_cur = Phi_t(xbar_t)。
3. 计算 z_cache = Phi_t(cached_xhat)。
4. 计算 residual = ode_factor * (z_cache - z_cur)。
5. 预测如果继续 skip，accum_vector_next 会多大。
6. 如果 risk 超过 threshold，就 refresh。
7. 否则 skip，并把 residual 累入 accum_vector。
```

公式：

$$
E_{next}=E_{current}+c_i(z_r-z_i)
$$

$$
Risk_{next}=\frac{2\|E_{next}\|_1}{Norm_{next}+\xi}+\eta U(t_i)
$$

如果：

$$
Risk_{next}>\delta
$$

则 refresh。

否则 skip。

---

## 9. 为什么这不是经验方法？

因为核心公式来自三层结构：

```text
PixelGen x-prediction:
    模型输出 clean image xhat。

Velocity conversion:
    xhat 误差会通过 1/(1-t) 进入 velocity。

ODE update:
    velocity 误差还要乘 step size h，并在连续 skip 中累积。
```

E5.5 的作用是检验：

```text
这个数学累计误差 R(W) 是否预测真实最终感知损伤 Y(W)。
```

而不是直接告诉算法：

```text
哪些 call 段该 refresh。
```

---

## 10. 最小实现版本

第一版不要一次全上。按这个顺序：

```text
v2-D0-a: Full-xhat oracle vector risk
v2-D0-b: Wiener proxy vector risk
v2-D0-c: Wiener + perceptual weight
v2-D0-d: Wiener + perceptual + ODE factor
v2-D0-e: Wiener + perceptual + ODE + uncertainty
```

先看哪个版本对 E5.5 window PIS 预测最好，再决定 online 主公式。

---

## 11. 一句话总结

> **xWPCache-v2 不再是“单步距离累计”，而是“相对于 cache anchor 的感知 solver residual 累计”。E5.5 用来判断这个累计 residual 是否真的对应连续 cache 复用后的最终感知损伤。**
