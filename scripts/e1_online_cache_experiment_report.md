# E1 Online RawInput / SEAInput Cache Baseline 实验报告

本文档记录在 PixelGen 主仓库中完成的 E1 阶段实验实现、运行方式、实验结果和初步结论。

实验代码位于：

- `scripts/01_e1_online_cache.py`
- `src/diffusion/flow_matching/e1_cache.py`

本次 64-sample pilot 输出位于：

- `outputs/e1_online_cache/e1_pilot_64_fp32/summary.json`
- `outputs/e1_online_cache/e1_pilot_64_fp32/method_summary.csv`

## 1. E1 阶段目标

Phase 1 的 E1 目标是验证一个关键前提：

```text
在 PixelGen 这种 x-pred pixel diffusion 中，SEAInput online cache 是否稳定优于 RawInput online cache 和 Uniform matched-RR cache？
```

如果 SEAInput online cache 不能稳定优于 RawInput 和 Uniform，那么后续做 PMA / perceptual metric oracle 的意义会明显下降，因为这说明 SeaCache-style 的基础移植、判据、hook 或 sampler 适配可能存在问题。

因此 E1 不是最终主实验，而是一个 gating experiment：

- 如果 E1 成立，可以继续做 E2 distance bank、E3 oracle schedule search、E4 oracle-schedule real cache rerun。
- 如果 E1 不成立，应优先 debug SeaCache 移植，而不是继续堆 DINO / LPIPS / PMA。

## 2. 本次实现的实验组

E1 当前实现了 4 类 baseline：

| Method | 是否缓存 | Refresh schedule 来源 | 是否 online | 说明 |
|---|---:|---|---:|---|
| Full reference | 否 | 每次 denoiser call 都刷新 | 是 | uncached PixelGen 完整采样，作为 paired fidelity reference |
| Uniform matched-RR | 是 | 固定均匀间隔 | 是 | 简单控制组，验证动态判据是否真的比均匀跳步好 |
| RawInput-online | 是 | 当前 cached trajectory 上的 raw first-block proxy distance | 是 | 不做 SEA filter，直接比较 first-block modulated proxy 的相对 L1 变化 |
| SEAInput-online | 是 | 当前 cached trajectory 上的 SEA-filtered first-block proxy distance | 是 | SeaCache-style baseline，对 proxy 做频谱滤波后再比较相对 L1 变化 |

注意：当前 E1 实现是 **output cache**，不是官方 SeaCache 的 transformer residual cache。也就是说：

- 当前脚本缓存的是 PixelGen denoiser 的 CFG xhat 输出。
- 官方 SeaCache 通常缓存 transformer block stack 的 residual，例如 `hidden_states - ori_hidden_states`。
- 因此当前方法更准确地说是 `Raw/SEA-gated denoiser output cache`。

采用 output cache 的原因是：

- E1 阶段主要验证 Raw vs SEA 判据是否在 PixelGen 上有效。
- PixelGen JiT 有 in-context tokens，直接做 block residual cache 会引入额外复杂度。
- output cache 侵入性低、容易 paired comparison、方便快速判断是否继续推进 PMA oracle。

## 3. 从 PixelGen-SeaCache fork 复用/改造的部分

参考路径：

```text
/mnt/iset/nfs-main/private/zhuangyifan/PixelGen-SeaCache
```

复用和改造的核心思想包括：

1. 使用 JiT 第一层 block 的 AdaLN-modulated token 作为 cache 判据 proxy。
2. 对 proxy 进行 SEA spectral filtering。
3. 使用 accumulated relative L1 distance 做 online refresh 判据。
4. warmup 阶段强制刷新。
5. 最后一次 denoiser opportunity 强制刷新。
6. 限制最大连续 cache hit 次数，避免长时间复用造成轨迹漂移。

主仓库中新增的实现为：

```text
src/diffusion/flow_matching/e1_cache.py
```

其中主要组件如下：

| 组件 | 作用 |
|---|---|
| `extract_jit_modulated_proxy` | 提取 JiT first block 的 AdaLN-modulated proxy |
| `apply_sea_filter` | 对 token grid 做 FFT -> SEA filter -> iFFT |
| `relative_l1_distance` | 计算当前 proxy 与上次 proxy 的 relative L1 |
| `AlwaysRefreshController` | Full reference，每次都跑 denoiser |
| `UniformCacheController` | 固定均匀 refresh schedule |
| `OnlineInputCacheController(metric="raw")` | RawInput online cache |
| `OnlineInputCacheController(metric="sea")` | SEAInput online cache |

## 4. JiT proxy 的定义

PixelGen 使用 JiT denoiser。对于输入图像状态 `x_t`、时间 `t`、条件 token `y`，当前 E1 使用第一层 JiT block 的 MSA 前 modulated hidden token 作为变化判据：

```python
t_emb = model.t_embedder(t)
y_emb = model.y_embedder(y)
c = t_emb + y_emb

tokens = model.x_embedder(x)
tokens = tokens + model.pos_embed

first_block = model.blocks[0]
shift_msa, scale_msa, _, _, _, _ = first_block.adaLN_modulation(c).chunk(6, dim=-1)
proxy = first_block.norm1(tokens)
proxy = proxy * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
```

这个 proxy 的意义是：

- 它位于 JiT denoiser 最早的内容处理阶段。
- 它已经融合了 time embedding 和 condition embedding。
- 相比 raw noisy input，它更接近模型实际看到的 feature space。
- 相比整层/整模型输出，它计算成本较低。

## 5. SEA filter 的定义

SEAInput-online 在 RawInput proxy 之上增加频域滤波。

流程：

1. 将 token 序列 `[B, N, C]` reshape 成 feature map `[B, C, H, W]`。
2. 对空间维度做 FFT。
3. 根据当前 PixelGen timestep `t` 构造 spectral filter。
4. 在频域乘以 filter。
5. iFFT 回到 token space。
6. 再计算 relative L1 distance。

当前 filter 使用：

```text
signal_coeff = t
noise_coeff = 1 - t
signal_prior = radius_sq ^ (-0.5 * beta)
filter = signal_power / (signal_power + noise_power + eps)
filter = filter / mean(filter)
```

默认参数：

```text
sea_filter_beta = 2.0
sea_filter_eps = 1e-6
```

它和官方 SeaCache 的 filter 不是完全逐行等价实现，但保留了核心思想：

- high-noise 阶段抑制噪声主导变化。
- 强调更可能对应内容/结构变化的频谱分量。
- 让 refresh 判据更少被 raw noisy feature 抖动误导。

## 6. Online refresh rule

对于 RawInput-online 和 SEAInput-online，每个 denoiser opportunity 都执行：

```text
1. 计算当前 proxy。
2. 如果是 SEAInput，则先对 proxy 做 SEA filter。
3. 计算 rel_l1(current_proxy, previous_proxy)。
4. 将 rel_l1 累积到 accumulator。
5. 如果满足 refresh 条件，执行 full denoiser forward 并缓存输出。
6. 否则复用 cached denoiser output。
```

Refresh 条件：

```text
refresh if:
  cached_output is None
  or call_index < warmup_calls
  or call_index == total_calls - 1
  or accumulated_distance >= delta
  or consecutive_hits >= max_skip_calls
```

本次实验默认：

```text
warmup_calls = 5
max_skip_calls = 4
```

这里的 `call_index` 是 denoiser opportunity 级别，而不是 sampler step 级别。

## 7. 为什么按 denoiser opportunity 计算 RR

PixelGen 当前配置使用：

```text
sampler = HeunSamplerJiT
num_steps = 50
exact_henu = true
```

Heun exact 采样中，每个 sample 的 denoiser opportunity 数量是：

```text
2 * num_steps - 1 = 99
```

原因：

- 每个 step 有 predictor call。
- 除最后一步外，每个 step 还有 corrector call。
- 因此 50 steps 对应 50 predictor calls + 49 corrector calls = 99 calls。

所以 E1 的实际 RR 定义为：

```text
actual RR = full denoiser refreshes / total denoiser opportunities
```

本次 64 samples 对应：

```text
64 * 99 = 6336 denoiser opportunities
```

## 8. 实验脚本

入口脚本：

```text
scripts/01_e1_online_cache.py
```

脚本功能：

1. 加载 PixelGen config。
2. 加载 PixelGen checkpoint。
3. 生成固定 class/seed pairs。
4. 跑 full reference。
5. 跑 Uniform matched-RR。
6. 对 Raw/SEA 做可选 delta calibration。
7. 跑 RawInput-online 和 SEAInput-online。
8. 保存 final image preview。
9. 计算 paired final-image PSNR / SSIM / LPIPS。
10. 输出 `method_summary.csv` 和 `summary.json`。

## 9. 本次 pilot 运行命令

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen

CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/01_e1_online_cache.py \
  --device cuda:0 \
  --num-samples 64 \
  --batch-size 1 \
  --methods uniform,raw,sea \
  --target-rrs 0.30,0.50 \
  --calibrate-online \
  --calib-samples 8 \
  --no-autocast \
  --run-id e1_pilot_64_fp32
```

关键设置：

| 参数 | 值 | 说明 |
|---|---:|---|
| `num_samples` | 64 | E1 pilot 样本数 |
| `batch_size` | 1 | 单卡稳定运行 |
| `target_rrs` | 0.30, 0.50 | 两个 refresh ratio 档位 |
| `calib_samples` | 8 | 用 8 个样本估计 Raw/SEA delta |
| `no_autocast` | true | fp32 推理，避免 bf16 非确定性干扰 |
| `LPIPS net` | alex | 默认 LPIPS-Alex |
| `CUDA_VISIBLE_DEVICES` | 0 | 单卡运行 |

## 10. Delta calibration 结果

RawInput-online calibration：

| delta | calibration RR |
|---:|---:|
| 0.03 | 0.5354 |
| 0.05 | 0.3573 |
| 0.08 | 0.2753 |
| 0.10 | 0.2614 |
| 0.15 | 0.2437 |
| 0.20 | 0.2424 |
| 0.30 | 0.2424 |
| 0.50 | 0.2424 |
| 0.80 | 0.2424 |

SEAInput-online calibration：

| delta | calibration RR |
|---:|---:|
| 0.03 | 0.5354 |
| 0.05 | 0.4697 |
| 0.08 | 0.3662 |
| 0.10 | 0.3384 |
| 0.15 | 0.2942 |
| 0.20 | 0.2828 |
| 0.30 | 0.2727 |
| 0.50 | 0.2525 |
| 0.80 | 0.2525 |

自动选择逻辑：

```text
选择 calibration RR 距离 target RR 最近的 delta。
```

因此本次选择为：

| Method | Target RR | Selected delta | Calibration RR |
|---|---:|---:|---:|
| RawInput-online | 0.30 | 0.08 | 0.2753 |
| RawInput-online | 0.50 | 0.03 | 0.5354 |
| SEAInput-online | 0.30 | 0.15 | 0.2942 |
| SEAInput-online | 0.50 | 0.05 | 0.4697 |

注意：

- `RR≈0.30` 档 Raw 和 SEA 的实际 RR 比较接近，可以相对公平比较。
- `RR≈0.50` 档 Raw 和 SEA 的实际 RR 没有完全对齐，Raw 实际 RR 更高，因此不能直接得出 Raw 优于 SEA 的结论。

## 11. 64-sample pilot 主结果

输出文件：

```text
outputs/e1_online_cache/e1_pilot_64_fp32/method_summary.csv
```

| Method | Actual RR | Refresh / sample | Time (s) | Speedup vs Full | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full reference | 1.0000 | 99.00 | 364.96 | 1.00x | - | - | - |
| Uniform RR0.3 | 0.3030 | 30.00 | 151.82 | 2.40x | 24.94 | 0.8891 | 0.1811 |
| Uniform RR0.5 | 0.5051 | 50.00 | 259.18 | 1.41x | 27.58 | 0.9181 | 0.1364 |
| RawInput RR0.3 | 0.2729 | 27.02 | 169.96 | 2.15x | 29.37 | 0.9360 | 0.1102 |
| RawInput RR0.5 | 0.5354 | 53.00 | 284.57 | 1.28x | 47.81 | 0.9950 | 0.0112 |
| SEAInput RR0.3 | 0.2932 | 29.03 | 167.22 | 2.18x | 35.19 | 0.9730 | 0.0461 |
| SEAInput RR0.5 | 0.4643 | 45.97 | 251.66 | 1.45x | 44.24 | 0.9941 | 0.0129 |

## 12. RR≈0.30 档分析

这个档位是本次最重要的结果，因为 RawInput 和 SEAInput 的 actual RR 比较接近：

```text
RawInput RR = 0.2729
SEAInput RR = 0.2932
Uniform RR = 0.3030
```

虽然 SEA 的 RR 比 Raw 高约 0.0204，但仍低于 Uniform 的 0.3030。因此比较有参考价值。

### 12.1 SEAInput vs RawInput

| Metric | SEAInput | RawInput | SEA - Raw |
|---|---:|---:|---:|
| Actual RR | 0.2932 | 0.2729 | +0.0204 |
| PSNR ↑ | 35.19 | 29.37 | +5.82 dB |
| SSIM ↑ | 0.9730 | 0.9360 | +0.0370 |
| LPIPS ↓ | 0.0461 | 0.1102 | -0.0642 |
| Time ↓ | 167.22s | 169.96s | -2.74s |

paired difference：

| Metric | Mean diff | 95% CI | SEA wins |
|---|---:|---:|---:|
| PSNR ↑ | +5.82 dB | ±0.85 | 63 / 64 |
| SSIM ↑ | +0.0370 | ±0.0140 | 55 / 64 |
| LPIPS ↓ | -0.0642 | ±0.0239 | 56 / 64 |

解释：

- SEAInput 在几乎所有样本上提高 PSNR。
- LPIPS 也明显下降，说明不是只优化像素误差。
- SEAInput 的 wall-clock 还略快于 RawInput，尽管它有 FFT 开销；原因是它实际 refresh 次数只比 Raw 多一点，且总体 denoiser 时间相近。

### 12.2 SEAInput vs Uniform

| Metric | SEAInput | Uniform | SEA - Uniform |
|---|---:|---:|---:|
| Actual RR | 0.2932 | 0.3030 | -0.0098 |
| PSNR ↑ | 35.19 | 24.94 | +10.24 dB |
| SSIM ↑ | 0.9730 | 0.8891 | +0.0839 |
| LPIPS ↓ | 0.0461 | 0.1811 | -0.1350 |
| Time ↓ | 167.22s | 151.82s | +15.39s |

paired difference：

| Metric | Mean diff | 95% CI | SEA wins |
|---|---:|---:|---:|
| PSNR ↑ | +10.24 dB | ±1.26 | 64 / 64 |
| SSIM ↑ | +0.0839 | ±0.0217 | 58 / 64 |
| LPIPS ↓ | -0.1350 | ±0.0341 | 62 / 64 |

解释：

- SEAInput 使用更少 refresh，却比 Uniform 质量显著更好。
- Uniform 更快，说明 Raw/SEA proxy 计算确实有 overhead。
- 但是从 fidelity-per-refresh 角度看，SEAInput 明显更有效。

### 12.3 RawInput vs Uniform

| Metric | RawInput | Uniform | Raw - Uniform |
|---|---:|---:|---:|
| Actual RR | 0.2729 | 0.3030 | -0.0301 |
| PSNR ↑ | 29.37 | 24.94 | +4.43 dB |
| SSIM ↑ | 0.9360 | 0.8891 | +0.0469 |
| LPIPS ↓ | 0.1102 | 0.1811 | -0.0709 |

paired difference：

| Metric | Mean diff | 95% CI | Raw wins |
|---|---:|---:|---:|
| PSNR ↑ | +4.43 dB | ±0.87 | 59 / 64 |
| SSIM ↑ | +0.0469 | ±0.0151 | 50 / 64 |
| LPIPS ↓ | -0.0709 | ±0.0249 | 54 / 64 |

解释：

- RawInput 已经强于 Uniform，说明动态 online 判据本身有价值。
- 但 SEAInput 进一步明显优于 RawInput，说明 SEA filter 的确改善了 proxy distance。

## 13. RR≈0.50 档分析

这个档位需要谨慎解读。

实际 RR：

```text
RawInput RR = 0.5354
SEAInput RR = 0.4643
Uniform RR = 0.5051
```

RawInput 比 SEAInput 多用了约 6.9% 的 denoiser refresh budget。因此 RawInput 的 PSNR 更高不能直接说明 Raw 判据更好。

### 13.1 RawInput vs SEAInput

| Metric | RawInput | SEAInput |
|---|---:|---:|
| Actual RR | 0.5354 | 0.4643 |
| Refresh / sample | 53.00 | 45.97 |
| PSNR ↑ | 47.81 | 44.24 |
| SSIM ↑ | 0.9950 | 0.9941 |
| LPIPS ↓ | 0.0112 | 0.0129 |
| Time ↓ | 284.57s | 251.66s |

paired difference `SEA - Raw`：

| Metric | Mean diff | 95% CI | SEA wins |
|---|---:|---:|---:|
| PSNR ↑ | -3.57 dB | ±0.87 | 10 / 64 |
| SSIM ↑ | -0.0009 | ±0.0004 | 10 / 64 |
| LPIPS ↓ | +0.0017 | ±0.0007 | 8 / 64 |

解释：

- RawInput 在 RR0.5 档看起来更好，但它用了更多 refresh。
- SEAInput 少用了约 7 个 refresh/sample，却仍达到 PSNR 44.24、LPIPS 0.0129。
- 这个档位需要更细的 delta sweep 或更大的 calibration set 才能公平比较。

### 13.2 与 Uniform 的比较

无论 RawInput 还是 SEAInput，都远好于 Uniform RR0.5：

| Method | Actual RR | PSNR ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| Uniform RR0.5 | 0.5051 | 27.58 | 0.1364 |
| RawInput RR0.5 | 0.5354 | 47.81 | 0.0112 |
| SEAInput RR0.5 | 0.4643 | 44.24 | 0.0129 |

这说明在中高 refresh budget 下，动态判据比固定间隔 schedule 强很多。

## 14. Worst-case 样本检查

每个方法的最差样本如下：

| Method | Worst PSNR sample | Worst PSNR | Worst LPIPS sample | Worst LPIPS |
|---|---|---:|---|---:|
| Uniform RR0.3 | idx 51, class 51, seed 51 | 17.86 | idx 11, class 11, seed 11 | 0.5770 |
| Uniform RR0.5 | idx 24, class 24, seed 24 | 19.15 | idx 11, class 11, seed 11 | 0.5541 |
| RawInput RR0.3 | idx 52, class 52, seed 52 | 23.24 | idx 52, class 52, seed 52 | 0.5631 |
| RawInput RR0.5 | idx 33, class 33, seed 33 | 29.17 | idx 33, class 33, seed 33 | 0.1632 |
| SEAInput RR0.3 | idx 39, class 39, seed 39 | 28.00 | idx 11, class 11, seed 11 | 0.1751 |
| SEAInput RR0.5 | idx 33, class 33, seed 33 | 29.31 | idx 33, class 33, seed 33 | 0.1593 |

观察：

- Uniform 的 worst-case 明显更差。
- RawInput RR0.3 在 idx 52 上有明显 failure，LPIPS 达到 0.5631。
- SEAInput RR0.3 的 worst LPIPS 只有 0.1751，说明 SEA 对 tail risk 有明显缓解。
- RR0.5 档 Raw/SEA 的 worst-case 都集中在 idx 33，且指标接近。

## 15. 实验输出文件说明

每次运行会生成：

```text
outputs/e1_online_cache/<run_id>/
├── class_seed_pairs.csv
├── summary.json
├── method_summary.csv
├── full_reference/
│   ├── final_images/
│   └── preview_grid.png
├── uniform_rr0p3/
│   ├── final_images/
│   ├── preview_grid.png
│   └── per_sample_metrics.csv
├── uniform_rr0p5/
│   └── ...
├── raw_online_rr...
│   └── per_sample_metrics.csv
└── sea_online_rr...
    └── per_sample_metrics.csv
```

关键文件：

| 文件 | 内容 |
|---|---|
| `class_seed_pairs.csv` | 本次实验的固定 class/seed pairs |
| `method_summary.csv` | 每个方法的聚合结果表 |
| `summary.json` | 完整 meta、calibration、method summary |
| `per_sample_metrics.csv` | 每个 sample 的 PSNR / SSIM / LPIPS |
| `preview_grid.png` | 前若干 sample 的预览图 |

## 16. 当前实现的局限

### 16.1 不是 official SeaCache residual cache

当前实现缓存的是 denoiser xhat 输出，而不是 transformer residual。它适合 E1 判据验证，但不能直接作为最终 SeaCache-like system claim。

后续如果要更贴近官方 SeaCache，需要考虑：

- 在 JiT block stack 内缓存 residual。
- 处理 in-context token 插入位置。
- 决定 residual cache 是覆盖 whole block stack 还是部分 block。
- 保证 cached residual 与 final layer / unpatchify 的关系正确。

### 16.2 Online proxy 有额外计算开销

SEAInput-online 每次 denoiser opportunity 都需要：

- first block proxy extraction。
- FFT / iFFT。
- relative L1 distance。

因此 wall-clock speedup 不等于 denoiser refresh reduction。比如 RR0.3：

```text
Full reference: 364.96s
Uniform RR0.3: 151.82s
RawInput RR0.3: 169.96s
SEAInput RR0.3: 167.22s
```

虽然 SEAInput 质量显著更好，但速度比 Uniform 慢。

这在 E1 阶段可以接受，因为 E1 主要验证判据质量；后续若做 deployable cache，需要优化 proxy 计算或改成 residual/blockwise cache。

### 16.3 RR0.5 calibration 还不够细

当前 delta grid：

```text
0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80
```

对 RR0.5 来说不够细：

- Raw delta 0.03 得到 RR 0.5354。
- SEA delta 0.05 得到 RR 0.4643。

两者相差较大，导致 RR0.5 档不能严格公平比较。

### 16.4 Calibration set 只有 8 samples

8 个样本足够做 smoke/pilot，但不足以稳定估计 delta -> RR 曲线。建议后续至少使用：

```text
calib_samples = 16 或 32
```

### 16.5 当前样本数是 64

64 samples 足够判断趋势，但还不是主实验规模。后续建议扩展到：

```text
128 samples 或 256 samples
```

并报告 confidence interval / paired difference。

## 17. 当前结论

本次 E1 pilot 的核心结论：

```text
在 RR≈0.30 档，SEAInput-online 明显优于 RawInput-online 和 Uniform matched-RR。
```

证据：

1. SEAInput RR0.3 的 actual RR 为 0.2932，低于 Uniform RR0.3 的 0.3030。
2. SEAInput RR0.3 的 PSNR 为 35.19，比 RawInput RR0.3 高 5.82 dB。
3. SEAInput RR0.3 的 LPIPS 为 0.0461，比 RawInput RR0.3 低 0.0642。
4. Paired PSNR 上 SEAInput 63/64 个样本优于 RawInput。
5. Paired LPIPS 上 SEAInput 56/64 个样本优于 RawInput。
6. SEAInput 对 worst-case tail risk 有明显缓解。

因此：

```text
E1 的继续条件基本满足，可以推进 E2 distance bank extraction 和 E3 oracle schedule search。
```

但是：

```text
RR≈0.50 档需要更细 delta sweep 后再做公平结论。
```

## 18. 建议的下一步实验

### 18.1 先补 RR0.5 细粒度 calibration

建议命令：

```bash
cd /mnt/iset/nfs-main/private/zhuangyifan/try/PixelGen

CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/01_e1_online_cache.py \
  --device cuda:0 \
  --num-samples 64 \
  --batch-size 1 \
  --methods raw,sea \
  --target-rrs 0.50 \
  --online-deltas 0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06,0.07 \
  --calibrate-online \
  --calib-samples 16 \
  --no-autocast \
  --run-id e1_pilot_64_fp32_rr05_refine
```

目的：

- 让 RawInput 和 SEAInput 的 actual RR 更接近。
- 判断 RR0.5 档到底是 Raw 真优，还是 refresh budget 不对齐造成的假象。

### 18.2 扩展到 128 或 256 samples

如果 RR0.5 refined 后仍然支持 SEAInput，那么建议跑：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n pixelgen python scripts/01_e1_online_cache.py \
  --device cuda:0 \
  --num-samples 128 \
  --batch-size 1 \
  --methods uniform,raw,sea \
  --target-rrs 0.30,0.40,0.50 \
  --calibrate-online \
  --calib-samples 32 \
  --no-autocast \
  --run-id e1_main_128_fp32
```

目的：

- 增加统计稳定性。
- 加入 RR0.4 中间档。
- 为 E2/E3 提供更可靠的 baseline。

### 18.3 进入 E2 distance bank

E2 应提取 full trajectory 上的：

```text
RawInput distance
SEAInput distance
DINO(xhat) distance
LPIPS(xhat) distance
xhat trajectory
```

E2 重点不是加速，而是生成 oracle schedule search 所需的 bank。

### 18.4 保留 E1 结果作为 sanity baseline

后续 E3/E4 中至少应继续报告：

```text
Uniform
RawInput-online
SEAInput-online
SEAInput-oracle
PMA-oracle
```

其中 E1 的 SEAInput-online 是 deployable baseline，E3/E4 的 SEAInput-oracle 是 metric upper-bound baseline。

## 19. 可引用的一句话总结

```text
In a 64-sample PixelGen-XL fp32 pilot with Heun-50 exact sampling, SEAInput-online cache at RR=0.293 achieved 35.19 dB PSNR and 0.046 LPIPS against full inference, substantially outperforming RawInput-online at RR=0.273 (29.37 dB, 0.110 LPIPS) and Uniform matched-RR at RR=0.303 (24.94 dB, 0.181 LPIPS).
```

中文总结：

```text
在 64 样本 PixelGen-XL fp32 pilot 中，SEAInput-online 在接近 0.30 的 refresh ratio 下显著优于 RawInput-online 和 Uniform matched-RR，说明 SEA-filtered input proxy 在 PixelGen 上确实是更可靠的 online cache refresh 判据。这支持继续推进 E2/E3/E4 的 PMA-oracle 实验。
```
