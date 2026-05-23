# HeunSamplerJiT 数学原理与实现说明

本文档解释 PixelGen 当前实验中使用的 `HeunSamplerJiT`：

```text
src/diffusion/flow_matching/sampling.py
class HeunSamplerJiT
```

它是 PixelGen / JiT 这套 **x-pred pixel diffusion / flow matching** 推理链路里的核心采样器。E0-E5 里反复提到的：

```text
predictor call
corrector call
99 denoiser opportunities
call-level cache
single-skip PIS
```

都直接来自这个采样器的实现。

---

## 1. 一句话版本

`HeunSamplerJiT` 做的事情是：

```text
从初始噪声 x_0 出发，沿着 t: 0 -> 1 的时间轴积分一个 ODE。

每个采样 step 先用 denoiser 在当前点估计一次方向，得到 predictor slope；
再沿 predictor slope 走出一个临时点；
然后在这个临时点重新跑一次 denoiser，得到 corrector slope；
最后用两个 slope 的平均值更新状态。
```

用最经典的 Heun 公式写就是：

```text
k1 = f(x_i, t_i)
x_tilde = x_i + dt * k1
k2 = f(x_tilde, t_{i+1})
x_{i+1} = x_i + dt * (k1 + k2) / 2
```

在当前 PixelGen 配置里：

```text
num_steps = 50
exact_henu = true
```

所以一张图的 denoiser 调用次数不是 50，而是：

```text
50 predictor calls + 49 corrector calls = 99 calls
```

这就是 E1-E5 都按 **call-level** 而不是 step-level 统计 refresh ratio 的原因。

---

## 2. 源码位置

核心实现分布在几个文件里：

```text
src/diffusion/flow_matching/sampling.py
  shift_respace_fn
  ode_step_fn
  HeunSamplerJiT

src/diffusion/flow_matching/scheduling.py
  LinearScheduler

src/diffusion/base/scheduling.py
  BaseScheduler

src/diffusion/base/guidance.py
  simple_guidance_fn

configs_c2i/PixelGen_XL_without_CFG.yaml
  diffusion_sampler 配置
```

当前 E0-E5 主要使用的配置是：

```yaml
diffusion_sampler:
  class_path: src.diffusion.flow_matching.sampling.HeunSamplerJiT
  init_args:
    exact_henu: true
    num_steps: 50
    guidance: 1.0
    timeshift: 2.0
    guidance_interval_min: 0.1
    guidance_interval_max: 0.9
    scheduler: src.diffusion.flow_matching.scheduling.LinearScheduler
    w_scheduler: src.diffusion.flow_matching.scheduling.LinearScheduler
    guidance_fn: src.diffusion.base.guidance.simple_guidance_fn
    step_fn: src.diffusion.flow_matching.sampling.ode_step_fn
```

注意字段名是 `exact_henu`，不是 `exact_heun`。这是代码里的历史拼写，但含义就是 exact Heun。

---

## 3. 时间轴：为什么是从 0 到 1？

PixelGen 这里的 flow matching 采样使用时间变量：

```text
t = 0 表示噪声端
t = 1 表示图像端
```

所以推理过程是：

```text
initial noise -> generated image
```

代码里先构造一个线性时间网格：

```python
timesteps = torch.linspace(0.0, 1 - last_step, num_steps)
timesteps = torch.cat([timesteps, torch.tensor([1.0])], dim=0)
```

默认：

```text
last_step = 1 / num_steps
num_steps = 50
```

因此原始网格包含 51 个时间点，对应 50 个区间：

```text
0.00, 0.02, 0.04, ..., 0.98, 1.00
```

但 PixelGen-XL 配置里还有：

```text
timeshift = 2.0
```

所以真实使用的时间点会经过：

```python
def shift_respace_fn(t, shift=3.0):
    return t / (t + (1 - t) * shift)
```

数学上：

```math
\tau(t) = \frac{t}{t + (1 - t) \cdot s}
```

其中 `s = timeshift`。

当 `s > 1` 时，中间大部分时间点会被压向更小的 t，也就是在更靠近噪声端的位置分配更多步长密度。例如 `s=2` 时：

```text
t=0.50 -> tau=0.333...
t=0.80 -> tau=0.666...
t=0.98 -> tau≈0.9608
```

这解释了 E5 结果中 `t_values` 不是简单的 0.00, 0.02, 0.04，而是经过 timeshift 后的值。

---

## 4. Scheduler：alpha / sigma 的含义

采样器依赖一个 scheduler，定义：

```text
alpha(t)
sigma(t)
d alpha / dt
d sigma / dt
```

当前配置使用 `LinearScheduler`：

```python
class LinearScheduler(BaseScheduler):
    def alpha(self, t):
        return t

    def sigma(self, t):
        return 1 - t

    def dalpha(self, t):
        return 1

    def dsigma(self, t):
        return -1
```

它对应一个非常直观的线性路径：

```math
x_t = \alpha(t) x_\text{clean} + \sigma(t) \epsilon
    = t x_\text{clean} + (1 - t) \epsilon
```

其中：

```text
x_clean : 最终干净图像
epsilon : 初始噪声
```

当：

```text
t = 0 -> x_t = epsilon
t = 1 -> x_t = x_clean
```

这正好符合从噪声到图像的推理方向。

`BaseScheduler` 还提供两个组合量：

```python
def dalpha_over_alpha(self, t):
    return self.dalpha(t) / self.alpha(t)

def dsigma_mul_sigma(self, t):
    return self.dsigma(t) * self.sigma(t)
```

它们被 `HeunSamplerJiT` 用来从 velocity 推出 score-like 项 `s`。

---

## 5. JiT 为什么叫 x-pred？

`HeunSamplerJiT` 和普通 flow-matching sampler 的关键区别是：

```text
JiT denoiser 直接输出 clean image prediction，也就是 xhat。
```

代码中：

```python
out = net(cfg_x, cfg_t_cur, cfg_condition)
```

这里的 `out` 不是 velocity，而是模型预测的干净图像：

```math
\hat{x}_\text{clean} = f_\theta(x_t, t, c)
```

但采样 ODE 需要的是速度：

```math
v_t \approx \frac{dx_t}{dt}
```

所以代码紧接着做转换：

```python
out = (out - cfg_x) / (1.0 - cfg_t_cur.view(-1, 1, 1, 1)).clamp_min(self.t_eps)
```

也就是：

```math
v_\theta(x_t, t, c)
= \frac{\hat{x}_\text{clean} - x_t}{1 - t}
```

直观解释：

```text
如果当前状态是 x_t，而模型认为最终干净图像应该是 xhat，
那么从当前点走到 xhat 还剩下 1 - t 的时间。

所以速度大约是：

方向差 / 剩余时间
```

代码里还有：

```python
self.t_eps = 5e-2
```

也就是分母会 clamp 到至少 `0.05`。这是为了避免非常接近 `t=1` 时：

```text
1 - t 太小
velocity 爆炸
```

在当前 50-step timeshift=2 的配置里，最后一次 predictor 的 `t` 大约是 `0.9608`，此时 `1-t≈0.0392`，确实会触发 `0.05` 的 clamp。

---

## 6. Classifier-Free Guidance 在这里怎么做？

采样器每次 denoiser 调用都把 unconditional 和 conditional 拼在 batch 维度上：

```python
cfg_condition = torch.cat([uncondition, condition], dim=0)
cfg_x = torch.cat([x, x], dim=0)
cfg_t = t.repeat(2)
out = net(cfg_x, cfg_t, cfg_condition)
```

因此模型输出 `out` 的 batch 维度前一半是 unconditional，后一半是 conditional。

`simple_guidance_fn` 是：

```python
uncondition, condition = out.chunk(2, dim=0)
out = uncondition + cfg * (condition - uncondition)
```

数学上：

```math
v_\text{guided}
= v_\text{uncond}
+ w_\text{cfg} \cdot (v_\text{cond} - v_\text{uncond})
```

当前 `PixelGen_XL_without_CFG.yaml` 里：

```text
guidance = 1.0
```

所以：

```math
v_\text{guided} = v_\text{cond}
```

也就是说，虽然代码仍然为了兼容 CFG 跑了 unconditional/conditional 双 batch，但 guidance scale 为 1 时，最终等价于使用 conditional 输出。

采样器还有 guidance interval：

```text
guidance_interval_min = 0.1
guidance_interval_max = 0.9
```

代码逻辑是：

```python
if t_cur[0] > min and t_cur[0] <= max:
    use self.guidance
else:
    use 1.0
```

当前 guidance 本身就是 1.0，因此 interval 对最终数值没有影响。但如果未来用 `guidance=2.25` 之类的 CFG 配置，这个 interval 会决定哪些时间段启用 CFG。

一个实现细节：corrector 分支里判断 guidance interval 仍然用的是 `t_cur`，不是 `t_hat`。这意味着 corrector 的 guidance 开关跟当前 step 的起点时间一致，而不是 corrector 实际 denoiser 调用所在的 `t_next`。

---

## 7. ODE step function

当前配置使用：

```python
def ode_step_fn(x, v, dt, s, w):
    return x + v * dt
```

也就是标准 ODE 显式更新：

```math
x_{t+\Delta t} = x_t + \Delta t \cdot v_t
```

虽然函数签名里有：

```text
s, w
```

但 `ode_step_fn` 不使用它们。

这些参数是为其他 SDE / score-based step function 保留的。例如同一个文件里还有：

```python
sde_step_fn
sde_mean_step_fn
sde_preserve_step_fn
```

这些才会使用 `s` 和 `w`。

当前 PixelGen E0-E5 的 `HeunSamplerJiT` 实际就是：

```text
Heun + ODE step
```

---

## 8. Heun 方法的数学原理

考虑 ODE：

```math
\frac{dx}{dt} = f(x, t)
```

Euler 方法只在区间起点估计一次斜率：

```math
k_1 = f(x_i, t_i)
```

然后：

```math
x_{i+1}^{Euler}
= x_i + \Delta t \cdot k_1
```

这是一阶方法，局部误差较大。

Heun 方法可以理解成二阶 Runge-Kutta 方法。它先用 Euler 走一个临时点：

```math
\tilde{x}_{i+1}
= x_i + \Delta t \cdot k_1
```

再在临时点上估计第二个斜率：

```math
k_2 = f(\tilde{x}_{i+1}, t_{i+1})
```

最后用两个斜率的平均值：

```math
x_{i+1}
= x_i + \Delta t \cdot \frac{k_1 + k_2}{2}
```

直觉：

```text
Euler 只看“出发时的方向”。
Heun 同时看“出发时的方向”和“到达临时终点后的方向”。
如果两者不一致，就用平均方向修正。
```

在 diffusion / flow matching 采样中，这通常比纯 Euler 更稳定，因为 denoiser 预测的速度场会随 `x` 和 `t` 改变。

---

## 9. HeunSamplerJiT 中 predictor / corrector 的严格对应

源码中的一个 step 对应下面流程。

### 9.1 当前 step 的时间

```python
for i, (t_cur, t_next) in enumerate(zip(steps[:-1], steps[1:])):
    dt = t_next - t_cur
```

数学上：

```text
i        : step index
t_i      : t_cur
t_{i+1}  : t_next
dt       : t_{i+1} - t_i
```

### 9.2 Predictor call

如果：

```python
if i == 0 or self.exact_henu:
```

当前实验 `exact_henu=true`，所以每个 step 都会跑 predictor：

```python
cfg_x = torch.cat([x, x], dim=0)
cfg_t_cur = t_cur.repeat(2)
out = net(cfg_x, cfg_t_cur, cfg_condition)
out = (out - cfg_x) / (1.0 - cfg_t_cur).clamp_min(t_eps)
out = guidance_fn(out, guidance)
v = out
```

这对应：

```math
k_1 = v_\theta(x_i, t_i, c)
```

然后计算：

```python
s = ((alpha_over_dalpha) * v - x) / (...)
```

如果当前 step function 是 ODE，这个 `s` 不影响更新；但后面为了兼容 SDE step，也会跟 `v` 一起做 Heun 平均。

### 9.3 临时状态 x_hat

代码：

```python
x_hat = self.step_fn(x, v, dt, s=s, w=w)
```

当前 `step_fn=ode_step_fn`，所以：

```math
\tilde{x}_{i+1} = x_i + \Delta t \cdot v_i
```

重要：这里源码变量名叫 `x_hat`，但它不是 PixelGen 的 clean-image prediction `xhat`。

为了避免混淆：

```text
sampler 里的 x_hat:
  Heun predictor 走出来的临时状态。

PixelGen 里的 xhat / clean prediction:
  denoiser 直接预测的最终干净图像。
```

这两个概念非常容易混。

### 9.4 Corrector call

除了最后一个 step，每个 step 都有 corrector：

```python
if i < self.num_steps - 1:
    cfg_x_hat = torch.cat([x_hat, x_hat], dim=0)
    cfg_t_hat = t_hat.repeat(2)
    out = net(cfg_x_hat, cfg_t_hat, cfg_condition)
    out = (out - cfg_x_hat) / (1.0 - cfg_t_hat).clamp_min(t_eps)
    out = guidance_fn(out, guidance)
    v_hat = out
```

这对应：

```math
k_2 = v_\theta(\tilde{x}_{i+1}, t_{i+1}, c)
```

然后：

```python
v = (v + v_hat) / 2
s = (s + s_hat) / 2
x = self.step_fn(x, v, dt, s=s, w=w)
```

在 ODE 情况下就是：

```math
x_{i+1}
= x_i
+ \Delta t \cdot \frac{k_1 + k_2}{2}
```

### 9.5 最后一个 step 为什么没有 corrector？

代码：

```python
if i < self.num_steps - 1:
    corrector...
else:
    x = self.last_step_fn(x, v, dt, s=s, w=w)
```

当 `num_steps=50` 时，step index 是：

```text
0, 1, 2, ..., 49
```

只有：

```text
i = 0..48
```

有 corrector。

最后：

```text
i = 49
```

只有 predictor，然后直接走到 `t=1`。

因此 call 数量是：

```text
predictor: 50
corrector: 49
total: 99
```

---

## 10. call index 如何映射到 step / predictor / corrector

在 E1-E5 的 cache 实验里，我们把每次 denoiser 调用称为一个 call。

对于 exact Heun：

```text
call 0  = step 0 predictor
call 1  = step 0 corrector
call 2  = step 1 predictor
call 3  = step 1 corrector
call 4  = step 2 predictor
call 5  = step 2 corrector
...
call 96 = step 48 predictor
call 97 = step 48 corrector
call 98 = step 49 predictor
```

所以：

```text
偶数 call 基本是 predictor
奇数 call 是 corrector
最后 call 98 是 predictor
```

更形式化地说：

```text
predictor call for step i = 2 * i
corrector call for step i = 2 * i + 1, 仅当 i < num_steps - 1
```

这就是为什么 E5 的 PIS bank shape 是：

```text
[num_samples, 99]
```

而不是：

```text
[num_samples, 50]
```

---

## 11. exact_henu=true 和 exact_henu=false 的差别

源码里 predictor 分支是：

```python
if i == 0 or self.exact_henu:
    run predictor denoiser
else:
    v = v_hat
    s = s_hat
```

当：

```text
exact_henu = true
```

每个 step 都重新在当前真实状态 `x` 上跑 predictor。

当：

```text
exact_henu = false
```

只有第一个 step 会显式跑 predictor。后续 step 会复用上一轮 corrector 的 `v_hat, s_hat` 作为当前 step 的 predictor。

这是一种省 denoiser call 的近似做法。它把 Heun 结构从：

```text
50 predictor + 49 corrector = 99 calls
```

变成接近：

```text
1 initial predictor + 49 corrector = 50 calls
```

但当前 E0-E5 统一使用：

```text
exact_henu = true
```

原因是它更接近标准 Heun，并且 full reference 更稳定。cache 实验也都是围绕这个 99-call full trajectory 展开的。

---

## 12. 为什么还要计算 s？

`HeunSamplerJiT` 每次得到 velocity `v` 后，会计算：

```python
s = ((alpha_over_dalpha) * v - x) / (
    sigma**2 - (alpha_over_dalpha) * dsigma_mul_sigma
)
```

其中：

```python
alpha_over_dalpha = 1 / scheduler.dalpha_over_alpha(t)
dsigma_mul_sigma = scheduler.dsigma_mul_sigma(t)
```

这个公式不是 Heun ODE 更新所必需的。当前 `ode_step_fn` 完全忽略 `s`。

但如果使用 SDE 类 step function，`s` 会参与 drift / noise 修正。

### 12.1 这个公式从哪里来？

假设 forward path 是：

```math
x_t = \alpha(t) x_0 + \sigma(t) \epsilon
```

那么速度是：

```math
v_t = \frac{dx_t}{dt}
    = \alpha'(t) x_0 + \sigma'(t) \epsilon
```

令：

```math
A(t) = \frac{\alpha'(t)}{\alpha(t)}
```

有：

```math
\frac{v_t}{A(t)}
= \alpha(t)x_0 + \frac{\sigma'(t)}{A(t)} \epsilon
```

再减去：

```math
x_t = \alpha(t)x_0 + \sigma(t)\epsilon
```

得到：

```math
\frac{v_t}{A(t)} - x_t
= \left(\frac{\sigma'(t)}{A(t)} - \sigma(t)\right)\epsilon
```

代码里的分母：

```math
\sigma(t)^2 - \frac{1}{A(t)}\sigma'(t)\sigma(t)
= \sigma(t)\left(\sigma(t) - \frac{\sigma'(t)}{A(t)}\right)
```

所以：

```math
s_t
= \frac{\frac{v_t}{A(t)} - x_t}
       {\sigma(t)^2 - \frac{1}{A(t)}\sigma'(t)\sigma(t)}
= -\frac{\epsilon}{\sigma(t)}
```

这就是一个 score-like 项。

### 12.2 在线性 scheduler 下的简化

当前：

```math
\alpha(t) = t
\sigma(t) = 1 - t
\alpha'(t) = 1
\sigma'(t) = -1
```

所以：

```math
A(t) = 1/t
\frac{1}{A(t)} = t
```

公式可以化简成：

```math
s_t = \frac{t v_t - x_t}{1 - t}
```

再强调一次：当前配置用的是 `ode_step_fn`，所以这个 `s_t` 算出来后不会改变 ODE 更新结果；它只是为 SDE step 函数保留。

---

## 13. HeunSamplerJiT 的伪代码

下面是贴近源码、但更容易读的伪代码：

```python
def sample(net, noise, condition, uncondition):
    steps = shifted_timesteps
    cfg_condition = concat(uncondition, condition)
    x = noise

    for i in range(num_steps):
        t_cur = steps[i]
        t_next = steps[i + 1]
        dt = t_next - t_cur

        # ---------- predictor ----------
        if i == 0 or exact_henu:
            cfg_x = concat(x, x)
            cfg_t = repeat(t_cur, 2)

            pred_clean = net(cfg_x, cfg_t, cfg_condition)
            v1_all = (pred_clean - cfg_x) / clamp(1 - cfg_t, min=t_eps)
            v1 = cfg(v1_all)
            s1 = score_like(v1, x, t_cur)
        else:
            v1 = previous_corrector_v
            s1 = previous_corrector_s

        # Euler provisional point
        x_tilde = x + dt * v1

        # ---------- corrector ----------
        if i < num_steps - 1:
            cfg_x_tilde = concat(x_tilde, x_tilde)
            cfg_t_next = repeat(t_next, 2)

            pred_clean_2 = net(cfg_x_tilde, cfg_t_next, cfg_condition)
            v2_all = (pred_clean_2 - cfg_x_tilde) / clamp(1 - cfg_t_next, min=t_eps)
            v2 = cfg(v2_all)
            s2 = score_like(v2, x_tilde, t_next)

            v = (v1 + v2) / 2
            s = (s1 + s2) / 2

            x = step_fn(x, v, dt, s=s, w=w)
        else:
            x = last_step_fn(x, v1, dt, s=s1, w=w)

    return x
```

在当前 ODE 配置下，`step_fn` 就是：

```python
x = x + v * dt
```

---

## 14. 和 cache 实验的关系

### 14.1 为什么 refresh ratio 要按 call 算？

因为真实 denoiser 机会是 99 次，而不是 50 次。

如果一个 cache 方法刷新 30 次，它的 refresh ratio 是：

```math
RR = 30 / 99 \approx 0.303
```

而不是：

```math
30 / 50 = 0.6
```

所以 E1-E5 的所有 RR 都是 call-level：

```text
queries = 99 * num_samples
refreshes = 实际跑 denoiser 的次数
RR = refreshes / queries
```

### 14.2 predictor 和 corrector 都能 cache 吗？

当前 cache controller 把 predictor 和 corrector 都视作普通 denoiser call。

也就是说：

```text
predictor 可以 refresh 或 skip
corrector 也可以 refresh 或 skip
```

这和 E5 的 single-skip PIS 完全一致：

```text
只 skip 某一个 call
其他 call 仍然 full refresh
看最终图像损伤
```

### 14.3 为什么 E5 里 corrector 可能更危险？

Heun corrector 不是“附赠的一次可有可无计算”。

它的作用是：

```text
在 predictor 走出的临时位置上重新估计方向，
然后修正整个 step 的积分方向。
```

如果 skip corrector，相当于这个 step 的二阶修正被破坏。尤其 early step 中速度场变化很剧烈，corrector 的修正更可能影响后续轨迹。

这正好解释了你当前 E5 main8 结果：

```text
predictor mean LPIPS : 0.000375
corrector mean LPIPS : 0.001931
```

corrector skip 的 final perceptual damage 明显更大。

---

## 15. E5 single-skip 如何对应 HeunSamplerJiT

E5 的 PIS 定义是：

```text
只在某一个 call c 上 skip 一次，
其他 call 全部正常 full compute，
最后比较 final image 与 full reference 的差异。
```

在 Heun call 结构下：

```text
如果 c 是 predictor:
  当前 step 的起点方向被替换成 cached output。

如果 c 是 corrector:
  当前 step 的临时终点修正方向被替换成 cached output。
```

E5 脚本为了加速，不是每个 call 都从 t=0 重新跑，而是：

```text
1. 先跑 full reference，并保存每个 step 开始前的 x。
2. 对 call c，定位它属于哪个 step。
3. 从该 step 的 full x 开始重放 suffix。
4. 只在 call c 复用前一个 cached output。
5. c 之后所有 call 都正常 full compute。
```

这样得到的仍然是 single-skip intervention，只是避免重复计算 intervention 之前完全相同的 prefix。

---

## 16. 容易混淆的变量名

### 16.1 `x`

当前采样状态：

```text
x = x_t
```

它从噪声逐渐变成图像。

### 16.2 `out`

在 `HeunSamplerJiT` 里，`out` 先是 denoiser 原始输出：

```text
out = predicted clean image
```

随后立刻被覆盖成 velocity：

```python
out = (out - cfg_x) / (1 - cfg_t)
```

所以读代码时要注意：

```text
转换前 out = xhat clean prediction
转换后 out = velocity
```

### 16.3 `x_hat`

源码里的：

```python
x_hat = self.step_fn(x, v, dt, s=s, w=w)
```

这个 `x_hat` 是 Heun predictor 得到的临时状态，不是 PixelGen clean prediction。

建议阅读时把它理解成：

```text
x_tilde
```

### 16.4 `v_hat`

源码里的 `v_hat` 是 corrector slope：

```text
v_hat = velocity at temporary state x_hat and time t_next
```

它不是 clean image prediction。

### 16.5 `exact_henu`

拼写是 `henu`，含义是 `heun`。

```text
exact_henu = true:
  每个 step 都重新算 predictor。

exact_henu = false:
  后续 step 复用上一轮 corrector 作为 predictor。
```

---

## 17. 当前实现的一些细节和后续注意点

### 17.1 `steps = self.timesteps.to(noise.device)`

`HeunSamplerJiT` 中这里没有显式传 dtype：

```python
steps = self.timesteps.to(noise.device)
```

多数情况下 `self.timesteps` 是 float32，noise 也是 float32，所以没问题。EulerSamplerJiT 里写得更明确：

```python
steps = self.timesteps.to(noise.device, noise.dtype)
```

如果以后做 mixed precision 或改 dtype，要留意这里。

### 17.2 `w_scheduler` 对当前 ODE 没有效果

配置里有：

```yaml
w_scheduler: src.diffusion.flow_matching.scheduling.LinearScheduler
step_fn: ode_step_fn
```

源码会打印 warning：

```text
current sampler is ODE sampler, but w_scheduler is enabled
```

原因是 `ode_step_fn` 不使用 `w`。

所以当前配置下：

```text
w_scheduler 存在，但不影响最终 ODE 更新。
```

### 17.3 corrector 的 guidance interval 使用 `t_cur`

corrector denoiser 实际是在 `t_hat = t_next` 上跑，但 guidance interval 判断用的是：

```python
if t_cur[0] > min and t_cur[0] <= max:
```

这和 predictor 一样使用 step 起点时间。当前 guidance=1.0 没影响；如果未来 guidance>1，需要知道这个细节。

### 17.4 `v_trajs` 不是 call-level trajectory

`BaseSampler.forward(..., return_v_trajs=True)` 返回的 `v_trajs` 是每个 step 的最终平均 velocity：

```python
v_trajs.append(v)
```

其中 corrector step 中的 `v` 已经是：

```text
(predictor_v + corrector_v) / 2
```

所以它不是 99 个 call 的原始 denoiser输出。

这也是为什么 E2/E5 都需要自己写 call-level loop，而不是直接用 `return_v_trajs`。

---

## 18. 对 E5/E6/E7 的启发

### 18.1 E5

E5 的目标是估计：

```text
skip 某个 call 对最终图像的 causal damage
```

由于 Heun 有 predictor/corrector 两类 call，E5 必须保留 call kind：

```text
call_kind = predictor / corrector
```

否则会把两类机制混在一起，导致结论不清楚。

### 18.2 E6

如果做 perceptual-state cache，不能只按 step 设计 refresh。

因为：

```text
step-level 有 50 个位置
call-level 有 99 个位置
corrector 可能比 predictor 更危险
```

所以 E6 的 threshold / accumulator / max-skip 也应该明确是 call-level。

### 18.3 E7

如果做 prefix-probe cache，要考虑 probe 插在 denoiser call 内部。

对 predictor：

```text
probe 输入是当前 x_t, t_cur, condition
```

对 corrector：

```text
probe 输入是临时状态 x_tilde, t_next, condition
```

这两类输入分布不同。E5 结果如果持续显示 corrector 更危险，那么 E7 可能需要：

```text
1. 把 call_kind 作为 probe 输入特征；
2. 或者 predictor / corrector 使用不同 threshold；
3. 或者对 corrector 设置更保守的 refresh rule。
```

---

## 19. 最简代码阅读路线

如果你要从源码理解完整流程，建议按这个顺序读：

1. `configs_c2i/PixelGen_XL_without_CFG.yaml`
   - 看 sampler 配置。

2. `src/diffusion/base/sampling.py`
   - 看 `BaseSampler.forward` 如何调用 `_impl_sampling`。

3. `src/diffusion/flow_matching/sampling.py`
   - 看 `shift_respace_fn`
   - 看 `ode_step_fn`
   - 看 `HeunSamplerJiT.__init__`
   - 看 `HeunSamplerJiT._impl_sampling`

4. `src/diffusion/flow_matching/scheduling.py`
   - 看 `LinearScheduler`。

5. `src/diffusion/base/guidance.py`
   - 看 `simple_guidance_fn`。

6. `scripts/01_e1_online_cache.py`
   - 看 cache 版 call-level loop `heun_jit_e1_sampling`。

7. `scripts/07_e5_pis_single_skip.py`
   - 看 E5 如何复现 Heun call-level 结构，并做 single-skip intervention。

---

## 20. 总结

`HeunSamplerJiT` 可以概括成：

```text
一个针对 JiT x-pred 输出改造过的 Heun 二阶 ODE sampler。
```

它的关键点是：

1. 时间从 `t=0` 噪声端走到 `t=1` 图像端。
2. JiT denoiser 输出 clean image prediction，不直接输出 velocity。
3. 采样器用 `(xhat - x_t) / (1 - t)` 把 clean prediction 转成 velocity。
4. 每个 step 用 Heun predictor-corrector 二阶更新。
5. `exact_henu=true` 时，50 step 对应 99 个 denoiser calls。
6. cache / PIS 实验必须按 call-level 设计，而不是 step-level。
7. predictor 和 corrector 的语义不同；E5 main8 已显示 corrector skip 的 final perceptual damage 明显更大。

这也是后续 CPC / PIS / prefix probe 设计里最重要的采样器背景。
