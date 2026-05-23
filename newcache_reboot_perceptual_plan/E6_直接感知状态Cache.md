# E6｜直接感知状态 Cache：先做一个无需训练的感知 online baseline

## 1. 这个实验要解决什么问题？

E5 得到的是“老师”：

```text
哪些 call skip 后会伤害最终图
```

E6 要做第一个 online 方法。

但这次不要再从 SEAInput 出发。

我们先做一个更直接的版本：

```text
直接用 DINO / LPIPS 看当前采样状态 x_t 的感知变化。
```

这叫：

```text
Perceptual-State Cache
中文：感知状态 Cache
```

---

## 2. 为什么看 x_t 比看 SEAInput 更靠谱？

SEAInput 是模型第一层里面的 token。

它不是图像。

它最多只能间接反映感知变化。

而 `x_t` 是当前采样状态。

在 PixelGen / flow matching 里，采样从噪声逐渐走向图像：

```text
早期 x_t 很像噪声
中后期 x_t 越来越像图片
最终 x_t 就是生成图
```

所以在中后期，直接对 `x_t` 做 DINO / LPIPS，比对 SEAInput 做距离更接近“感知空间”。

---

## 3. 重要限制：早期不要相信 DINO / LPIPS

早期 `x_t` 很吵，可能不像图片。

这时候 DINO / LPIPS 看到的是噪声，不是语义。

所以 E6 必须有 noise gate：

```text
前 30% call：不要用 DINO/LPIPS 判断，强制 refresh 或使用保守策略。
后 70% call：才使用感知状态距离。
```

这个设计和 PixelGen 的思路一致：PixelGen 论文里也指出，高噪声 timestep 上直接使用感知损失是危险的，因此只在后面低噪声阶段启用感知监督。

---

## 4. Online 判据怎么写？

每次 refresh 时，保存当前状态的感知特征：

```text
z_ref = DINO(x_t_at_refresh)
p_ref = LPIPS_feature(x_t_at_refresh)
```

当前 call 来了以后，计算：

```text
z_cur = DINO(x_t_current)
p_cur = LPIPS_feature(x_t_current)
```

然后算风险：

```text
risk = w_dino * distance(z_cur, z_ref)
     + w_lpips * distance(p_cur, p_ref)
```

如果：

```text
risk > threshold
```

就 refresh。

否则 skip。

---

## 5. 先从最简单版本开始

不要一开始就做复杂融合。

建议按下面顺序做：

### 版本 1：DINO-State

只用：

```text
risk = DINO_distance(x_t_current, x_t_at_last_refresh)
```

DINO 更偏语义和结构。

它适合看：

```text
物体布局有没有变
语义结构有没有变
```

---

### 版本 2：LPIPS-State

只用：

```text
risk = LPIPS(x_t_current, x_t_at_last_refresh)
```

LPIPS 更偏局部纹理和边缘。

它适合看：

```text
局部细节有没有变
边缘/纹理有没有明显变化
```

---

### 版本 3：DINO + LPIPS State

最后再融合：

```text
risk = 0.5 * normalized_DINO + 0.5 * normalized_LPIPS
```

先不要加 SEAInput。

这个实验的目的就是验证：

```text
直接看感知状态是否比 SEAInput 更靠谱。
```

---

## 6. threshold 怎么定？

为了公平比较，不能随便调。

你要和以前一样做 matched refresh ratio：

```text
目标 RR = 0.30 / 0.40 / 0.50
```

做法：

```text
1. 用少量 calibration samples，比如 8 或 16 张。
2. 在这些样本上扫 threshold。
3. 找到能达到目标 RR 的 threshold。
4. 固定 threshold，在 test samples 上跑。
```

---

## 7. E6 输出什么？

建议输出目录：

```text
outputs/e6_perceptual_state_cache/
```

里面保存：

```text
method_summary.csv
per_sample_metrics.csv
refresh_patterns.npy
runtime_profile.csv
```

`method_summary.csv` 至少包含：

```text
method
actual_rr
refresh_per_sample
speedup_vs_full
psnr
ssim
lpips
runtime_seconds
perceptual_encoder_overhead
```

---

## 8. 必须和哪些方法比较？

E6 至少比较这些：

```text
Full reference
Uniform RR0.3 / RR0.4 / RR0.5
SEAInput-online
DINO-State-online
LPIPS-State-online
DINO+LPIPS-State-online
```

RawInput 可以保留，但不是重点。

---

## 9. 这个方法可能失败在哪里？

### 失败点 1：DINO/LPIPS online 太慢

这很可能发生。

但它不是坏事。

因为 E6 的主要作用是建立一个“直接感知 online baseline”。

如果它质量很好但速度慢，那么 E7 就可以把它蒸馏成一个更快的前缀探针。

---

### 失败点 2：x_t 不是 xhat

对。

`x_t` 是当前采样状态，不是模型预测的 clean image `xhat`。

所以 E6 可能不如直接看 xhat。

但 E6 仍然比 SEAInput 更直接，因为它至少在图像空间里。

如果 E6 不够强，说明我们需要 E7：从模型前几层预测 `xhat` 的感知 embedding 或 PIS。

---

### 失败点 3：早期 DINO/LPIPS 被噪声骗了

所以一定要做 noise gate。

建议默认：

```text
call_fraction < 0.30：强制 refresh 或不用感知判据
call_fraction >= 0.30：启用 DINO/LPIPS-State
```

然后做消融：

```text
no gate
0.2 gate
0.3 gate
0.4 gate
```

---

## 10. 判断 E6 是否成功

如果 DINO/LPIPS-State 在相同 RR 下：

```text
final LPIPS 低于 SEAInput-online
或者 PSNR / SSIM 接近但 LPIPS 更好
或者 tail failure 更少
```

那说明直接感知状态有价值。

如果 E6 质量好但速度慢，也仍然成功，因为它可以作为 E7 的 teacher。

如果 E6 完全不如 SEAInput，也不代表项目失败。

那说明：

```text
当前 x_t 的感知变化不等于模型输出 xhat 的感知变化。
```

这会直接导向 E7。

