# 07｜Online E6：xWPCache-v2 完整算法细节

这份文档讲真正 online cache 怎么写。前提是：你已经完成 E6-D0，选定了一个能预测 E5.5 window PIS 的 risk score。

---

## 1. Online 版本能用什么，不能用什么？

### 能用

```text
当前 x_t
当前 timestep t
当前 step size / solver coefficient h
当前 call_kind
上一次 refresh 保存的 cached_xhat
上一次 refresh 后的 accumulated residual
当前 cache_age
```

### 不能用

```text
当前 full xhat_i    # 因为算它就已经完整 forward 了
未来 final image
E5.5 的人工分段结论
oracle PIS
```

---

## 2. Online 状态变量

每个 sample 维护：

```python
cache = {
    "valid": False,
    "xhat": None,              # 上一次 refresh 的 clean prediction
    "accum_vec": None,         # 累计 vector residual
    "accum_norm": 0.0,          # vector relative L1 的归一化项
    "accum_scalar": 0.0,        # 可选 scalar risk
    "age": 0,                  # 已连续 skip 的 call 数
    "last_refresh_call": None,
}
```

如果 batch 里每个样本是否 refresh 不同，需要支持 per-sample cache。第一版如果代码复杂，可以先 batch size=1 做验证。

---

## 3. 每个 call 的流程

### Step 1：如果没有 cache，必须 refresh

```python
if not cache.valid:
    xhat = model(x_t, t, cond)
    cache = reset_cache(xhat, call_index)
    use xhat to update sampler
```

---

### Step 2：计算 cheap clean proxy

```python
xbar = wiener_clean_proxy(x_t, t, size=64 or 128)
```

这一步不能调用完整 denoiser。

---

### Step 3：计算感知表示

```python
z_cur = perceptual_phi(xbar, t)
z_cache = perceptual_phi(cache.xhat, t)
```

注意：

```text
z_cache 每个 timestep 都要用当前 t 的 W_t 重新过滤。
```

因为感知权重和 Wiener/SEA 类 filter 都是 timestep-dependent。

---

### Step 4：计算 solver residual

```python
coeff = abs(h) / max(1.0 - t, epsilon_clip)
residual = coeff * (z_cache - z_cur)
```

如果 E6-D0 发现 Heun corrector 需要特殊系数，再改成：

```python
coeff = effective_solver_coeff(call) / max(1.0 - t, epsilon_clip)
```

但第一版不要手工给 predictor/corrector 不同权重。

---

### Step 5：预测如果继续 skip 的累计风险

```python
accum_vec_next = cache.accum_vec + residual
accum_norm_next = cache.accum_norm + coeff * (l1(z_cache) + l1(z_cur))
risk_vec_next = 2 * l1(accum_vec_next) / (accum_norm_next + eps)
```

如果你使用 scalar risk：

```python
risk_scalar_next = cache.accum_scalar + coeff * rel_l1(z_cache, z_cur)
```

如果使用 uncertainty：

```python
risk_next = risk_vec_next + eta * uncertainty(t)
```

或者：

```python
risk_next = alpha * risk_vec_next + beta * risk_scalar_next + eta * uncertainty(t)
```

`alpha/beta/eta` 必须来自 E6-D0 validation，不要凭感觉。

---

### Step 6：决定 refresh 还是 skip

```python
if risk_next > delta:
    refresh
elif trust_region_penalty says unsafe:
    refresh
else:
    skip
```

refresh：

```python
xhat = model(x_t, t, cond)
cache.xhat = xhat.detach()
cache.accum_vec = zeros_like(z_cur)
cache.accum_norm = 0
cache.accum_scalar = 0
cache.age = 0
```

skip：

```python
xhat = cache.xhat
cache.accum_vec = accum_vec_next
cache.accum_norm = accum_norm_next
cache.accum_scalar = risk_scalar_next
cache.age += 1
```

然后把 `xhat` 交给 sampler update。

---

## 4. 伪代码

```python
def xwp_should_refresh(x_t, t, h, cond, cache, params):
    if not cache.valid:
        return True, None

    # cheap proxy
    xbar = wiener_clean_proxy(
        x_t,
        t,
        size=params.proxy_size,
        beta=params.beta,
        f0=params.f0,
    )

    # perceptual representations
    z_cur = perceptual_phi(
        xbar,
        t,
        lambda_d=params.lambda_d,
        lambda_l=params.lambda_l,
        tau=params.noise_gate_tau,
    )

    z_cache = perceptual_phi(
        downsample(cache.xhat, params.proxy_size),
        t,
        lambda_d=params.lambda_d,
        lambda_l=params.lambda_l,
        tau=params.noise_gate_tau,
    )

    # solver coefficient
    coeff = abs(h) / max(1.0 - float(t), params.epsilon_clip)

    residual = coeff * (z_cache - z_cur)
    accum_vec_next = cache.accum_vec + residual
    accum_norm_next = cache.accum_norm + coeff * (
        z_cache.abs().sum() + z_cur.abs().sum()
    )

    risk_vec = 2.0 * accum_vec_next.abs().sum() / (accum_norm_next + 1e-8)

    if params.use_uncertainty:
        risk = risk_vec + params.eta * uncertainty_score(t, params)
    else:
        risk = risk_vec

    if params.use_age_penalty:
        risk = risk + params.mu * (cache.age / params.age_M) ** params.age_p

    should = risk > params.delta

    return should, {
        "risk": float(risk),
        "risk_vec": float(risk_vec),
        "accum_vec_next": accum_vec_next,
        "accum_norm_next": accum_norm_next,
    }
```

---

## 5. sampler 集成方式

在原 sampler 的 denoiser call 位置：

```python
should_refresh, info = xwp_should_refresh(x_t, t, h, cond, cache, params)

if should_refresh:
    xhat = model(x_t, t, cond)
    cache.reset(xhat, call_index)
    refreshed = True
else:
    xhat = cache.xhat
    cache.update_skip(info)
    refreshed = False

v = convert_xhat_to_velocity(xhat, x_t, t)
x_next = sampler_update(x_t, v, ...)
```

不要改 sampler 的其他逻辑。

---

## 6. 日志必须记录什么？

每个 call 保存：

```text
sample_id
call_index
step_index
call_kind
t
h
refresh_or_skip
risk
risk_vec
uncertainty
age_before
age_after
last_refresh_call
```

输出：

```text
outputs/e6_online_xwp/main8_delta_xxx/schedule.csv
```

这个日志非常重要。它能帮助你检查：

```text
xWPCache 是否真的按 risk 决策？
online RR 是否符合预期？
失败样本是不是 risk 被低估？
```

---

## 7. online pilot 怎么跑？

第一批只跑 main8：

```text
samples = 8
thresholds = 来自 E6-D0 的 RR=0.5, 0.4, 0.3 三个 delta
baselines = no-cache, uniform, SEAInput-online, xWPCache-v2
```

输出：

```text
final image
schedule.csv
paired LPIPS / DINO / PSNR / SSIM vs no-cache
latency
refresh ratio
```

---

## 8. 成功标准

online main8 成功不等于论文成功，但至少要满足：

```text
1. xWPCache online RR 能通过 delta 控制。
2. 同 RR 下，xWPCache 的 paired LPIPS/DINO 优于 raw/uniform，最好优于 SEAInput。
3. schedule 不是简单固定分段，而是随样本有变化。
4. 失败样本能在 risk 日志中解释。
```

第 3 点非常重要。如果所有样本刷新模式几乎一样，reviewer 会怀疑它只是 time schedule。

---

## 9. 常见 bug

### Bug 1：风险爆炸

可能原因：

```text
1-t 没有 clip
z_cache / z_cur 没有归一化
FFT 后复数处理错误
```

### Bug 2：几乎每步 refresh

可能原因：

```text
delta 太小
Wiener proxy 和 cached_xhat 尺度不一致
uncertainty 太大
```

### Bug 3：几乎一直 skip

可能原因：

```text
delta 太大
risk 归一化分母太大
residual 没有累积
```

### Bug 4：结果比 uniform 差很多

可能原因：

```text
online trajectory drift 导致 full-trajectory diagnostic 失效
Wiener proxy 不能反映 skip 后的 x_t
需要加入 conservative trust-region 或改 proxy
```

---

## 10. 一句话总结

> **Online E6 的核心不是手工选择哪个 call skip，而是在每次可能 skip 前估计“继续复用当前 cached xhat 会让累计 solver error 增加多少”，超过阈值就 refresh。**
