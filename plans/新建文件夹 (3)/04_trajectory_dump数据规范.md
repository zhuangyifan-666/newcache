# 04｜trajectory dump 数据规范：E6-D0 需要保存什么？

E6-D0 需要 full trajectory 里的状态和输出。这个文档讲清楚要保存哪些字段、为什么保存、怎么避免磁盘爆炸。

---

## 1. 为什么需要 dump full trajectory？

E5.5 只告诉你：

```text
连续 skip window W 后最终图坏了多少。
```

但为了计算 xWPCache 的预测风险，你还需要知道 full trajectory 中每个 call 的：

```text
当前 x_t
当前 timestep t
当前 step size h
当前 full xhat
```

这样才能离线计算：

```text
如果在 window W 里连续复用 anchor xhat，
xWPCache 认为风险是多少？
```

---

## 2. 每个 sample 保存一个 `.pt`

推荐结构：

```text
outputs/e6_d0_fulltraj/main8/
  sample_000.pt
  sample_001.pt
  ...
```

每个 `.pt` 是一个 Python dict：

```python
{
    "meta": {
        "sample_id": 0,
        "class_id": 0,
        "seed": 0,
        "model": "PixelGen-XL/16",
        "sampler": "HeunSamplerJiT",
        "steps": 50,
        "num_calls": 99,
        "image_size": 256,
        "dtype_saved": "float16 or float32",
    },
    "final_image": Tensor,  # [3,H,W]
    "calls": [
        {
            "call_index": 0,
            "step_index": 0,
            "call_kind": "predictor" or "corrector",
            "t": float,
            "t_next": float,
            "h": float,
            "effective_solver_coeff": float,
            "x_t": Tensor,
            "xhat": Tensor,
        },
        ...
    ]
}
```

---

## 3. 字段解释

### 3.1 `call_index`

从 0 到 98。

PixelGen-XL / Heun 50 steps 通常有 99 个 denoiser call。

---

### 3.2 `step_index`

diffusion step 编号。

Heun 一个 step 可能对应 predictor 和 corrector 两个 call。

---

### 3.3 `call_kind`

建议保存：

```text
predictor
corrector
single
```

如果 sampler 不是 Heun，可以写 `single`。

---

### 3.4 `t`

当前 timestep。

注意确认 PixelGen 的方向：

```text
t = 0 接近噪声
t = 1 接近干净图
```

如果你的代码内部是反向顺序，保存时一定要统一成 PixelGen 论文里的含义。

---

### 3.5 `h`

采样更新步长。

最简单：

```python
h = abs(t_next - t)
```

但 Heun corrector 可能有不同有效系数。建议额外保存：

```python
effective_solver_coeff
```

如果暂时不知道，就先设：

```python
effective_solver_coeff = abs(t_next - t)
```

后面可以通过消融验证是否需要更精确。

---

### 3.6 `x_t`

denoiser forward 前的当前 sampler state。

这是计算 Wiener proxy 的输入。

---

### 3.7 `xhat`

完整 denoiser forward 后得到的 clean prediction。

它有两个作用：

```text
1. 作为 cache anchor：window start 前的 cached_xhat。
2. 作为 full-xhat oracle risk 的当前 xhat。
```

---

## 4. 保存 full resolution 还是 low resolution？

### 方案 A：保存 full resolution

优点：

```text
最灵活，后面可以改任何下采样尺寸和指标。
```

缺点：

```text
磁盘占用较大。
```

估算：

```text
8 samples × 99 calls × 2 tensors × 3×256×256 × fp16
≈ 600 MB 左右
```

实际还要加 pickle 开销，但 main8 可以接受。

---

### 方案 B：只保存 64/128 lowres

优点：

```text
省磁盘。
```

缺点：

```text
后面如果想比较 256 的 LPIPS-like 频率权重，需要重跑。
```

建议：

```text
main8 先保存 full resolution。
main64 如果磁盘紧张，可以保存 lowres + final image。
```

---

## 5. 保存 dtype

建议：

```text
x_t / xhat 保存 fp16
计算 risk 时转 fp32
```

但如果你担心数值误差，可以 main8 用 fp32 做一次对比。

E5.5 的 full rerun floor 基本为 0，说明你的采样过程在 FP32/no-autocast 下很稳定。trajectory dump 最好也先在同样设置下跑 smoke test。

---

## 6. 推荐 dump 脚本结构

文件：

```text
scripts/e6d0_dump_fulltraj.py
```

核心伪代码：

```python
@torch.no_grad()
def sample_with_trajectory(model, sampler, init_noise, cond):
    calls = []
    x_t = init_noise

    for call in sampler.iter_calls():
        t = call.t
        t_next = call.t_next
        h = abs(t_next - t)
        call_kind = call.kind

        # 保存 forward 前的状态
        x_in = x_t.detach()

        # full denoiser
        xhat = model(x_in, t, cond)

        # sampler update
        v = (xhat - x_in) / max(1.0 - t, epsilon_clip)
        x_t = call.update(x_in, v)

        calls.append({
            "call_index": call.index,
            "step_index": call.step_index,
            "call_kind": call_kind,
            "t": float(t),
            "t_next": float(t_next),
            "h": float(h),
            "effective_solver_coeff": float(h),
            "x_t": x_in.cpu().to(torch.float16),
            "xhat": xhat.detach().cpu().to(torch.float16),
        })

    return x_t, calls
```

注意：真实代码里 sampler update 可能比这复杂，尤其 Heun predictor/corrector。你要尽量调用原本 sampler 的更新逻辑，不要手写错。

---

## 7. Heun sampler 特别注意

Heun 通常包含：

```text
predictor: 用当前斜率预测下一状态
corrector: 再算一次斜率修正
```

E5 里早期 corrector 对最终图很敏感，所以你不能简单忽略 `call_kind`。

但新版不把 `call_kind` 写成经验规则，而是用于：

```text
1. 记录元数据。
2. 后处理分析 false negative 是否集中在 predictor/corrector。
3. 如果有必要，加入 solver-specific coefficient 消融。
```

---

## 8. 检查 trajectory 是否正确

dump 后做三个 sanity check。

### Check 1：用保存的 xhat 复现 full trajectory

不用重新 forward，只用保存的 xhat 走 sampler update，看 final 是否和原 full final 一样。

如果不同，说明你保存的 call 顺序、t、h、update 逻辑有问题。

---

### Check 2：xhat range

检查：

```text
xhat.min(), xhat.max(), xhat.mean(), xhat.std()
```

每个 call 画曲线。

如果某些 call 异常爆炸，说明 velocity conversion 或 dtype 有问题。

---

### Check 3：t 顺序

画：

```text
call_index vs t
call_index vs 1-t
call_index vs h
```

确认和 PixelGen 论文方向一致。

---

## 9. 输出小工具

建议写一个检查脚本：

```bash
python scripts/e6d0_check_fulltraj.py \
  --fulltraj outputs/e6_d0_fulltraj/main8 \
  --out outputs/e6_d0_fulltraj/main8_check
```

输出：

```text
check_summary.csv
t_curve.png
h_curve.png
xhat_norm_curve.png
replay_error.txt
```

---

## 10. 一句话总结

> **trajectory dump 是 E6-D0 的基础。如果 `x_t`、`xhat`、`t`、`h`、`call_kind` 保存错了，后面的数学 risk 再漂亮也没有意义。先把 full trajectory 保存和复现做对。**
