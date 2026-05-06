# 一种新的 cache 加速 pixel diffusion 推理的策略

本文档是 PixelGen cache 加速项目第一阶段 E0-E4 的完整实验记录与阶段收尾总结。实验主线来自
`pixeldiffusion_phase1_plan_revised.md`：在 PixelGen 这类 **x-pred pixel diffusion** 中，研究 cache
refresh 判据是否应该从传统的 noisy input / hidden feature space，扩展到 clean-image prediction 的
perceptual space。

当前状态更新于：2026-05-06。E0-E4 均已完成。

---

## 0. 核心问题

PixelGen 与许多 latent diffusion / DiT-style diffusion 的关键区别是：denoiser 每次调用直接预测 clean
image estimate：

```text
xhat_t = f_theta(x_t, t, c)
```

这使得 cache refresh 判据可以有两类设计：

```text
传统 cache 判据：
  noisy input / hidden feature 是否变化

PixelGen-specific 判据：
  clean-image prediction 在语义、纹理、感知空间中是否变化
```

本阶段研究的问题是：

```text
在 x-pred pixel diffusion 中，基于 clean-image perceptual drift 的 refresh schedule，
是否能在相同 denoiser refresh budget 下，比 RawInput / SEAInput schedule 更接近 uncached full trajectory？
```

需要注意：E2-E4 中的 oracle schedule 是 **离线 / cheating** 的，它们提前使用 full uncached trajectory
上已经计算好的距离序列。它们的作用是评估 metric / schedule 的上限价值，而不是直接声明一个可部署的在线方法。

---

## 1. 实验总览

| 阶段 | 名称 | 目的 | 状态 |
|---|---|---|---|
| E0 | Deterministic full-reference sanity | 确认 PixelGen full inference 可复现，可作为 paired reference | 已完成 |
| E1 | Online RawInput / SEAInput baseline | 验证 SEAInput-online 是否优于 RawInput-online 和 Uniform | 已完成 |
| E2 | Oracle distance bank extraction | 在 full trajectory 上提取 Raw / SEA / DINO / LPIPS distance bank | 已完成 |
| E3 | Schedule-level oracle analysis | 用 E2 bank 搜索 matched-RR oracle schedules，并分析 refresh pattern | 已完成 |
| E4 | Oracle-schedule real cache rerun | 用 E3 fixed schedules 真实 rerun cache，测 final-image fidelity | 已完成 |
| E4-ablation | PMA stage-aware weight candidates | 比较原 hard stage gate 与 soft stage-aware 权重 A/B/C | 已完成 |

核心脚本：

```text
scripts/00_sanity_full_reference.py
scripts/01_e1_online_cache.py
scripts/02_e2_extract_distance_bank.py
scripts/03_e3_schedule_oracle_analysis.py
scripts/04_e4_oracle_schedule_cache_rerun.py
scripts/05_e4_prepare_pma_weight_candidates.py
scripts/06_e4_compare_pma_weight_candidates.py
scripts/run_e4_pma_weight_candidates_parallel.sh
```

核心 cache 工具模块：

```text
src/diffusion/flow_matching/e1_cache.py
```

主要输出目录：

```text
outputs/e0_sanity/e0_32samples_fp32/
outputs/e1_online_cache/e1_pilot_64_fp32/
outputs/e2_distance_bank/e2_main_256_fp32/
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/
outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/
outputs/e4_pma_weight_candidates/e4_pma_weight_candidates_from_e2_fp32_calib64/
outputs/e4_pma_weight_candidates/comparison/compare_candidates_vs_main/
```

统一实验配置：

| 项目 | 设置 |
|---|---|
| Config | `configs_c2i/PixelGen_XL_without_CFG.yaml` |
| Checkpoint | `ckpts/PixelGen_XL_80ep.ckpt` |
| Model | PixelGen-XL / JiT |
| Sampler | `HeunSamplerJiT` |
| Sampling steps | 50 |
| Heun exact | true |
| Guidance | 1.0 |
| Timeshift | 2.0 |
| Precision | fp32, `--no-autocast` |
| SDPA | math SDPA, `--allow-fused-sdpa` false |
| GPU | NVIDIA GeForce RTX 3090 |
| Checkpoint weights | `ema_denoiser.` |

---

## 2. Call-Level 约定

PixelGen 当前 sampler 是 Heun exact：

```text
num_steps = 50
exact_henu = true  # repo 中该字段名表示 Heun exact
```

因此一个 sample 不是 50 次 denoiser evaluation，而是：

```text
50 predictor calls + 49 corrector calls = 99 denoiser opportunities
```

本项目所有 refresh ratio 都按 call-level 定义：

```text
RR = full denoiser refreshes / total denoiser opportunities
```

例如：

```text
64 samples  * 99 calls/sample = 6336 denoiser opportunities
192 samples * 99 calls/sample = 19008 denoiser opportunities
256 samples * 99 calls/sample = 25344 denoiser opportunities
```

这个约定贯穿 E1-E4。E2 中的 distance bank 长度是 98，因为 distance 是相邻 call 之间的变化：

```text
distance_count = 99 calls - 1 = 98 transitions
```

E3/E4 的 schedule length 是 99，而不是 50。所有 schedule 文件中的 `schedule` shape 为：

```text
[num_samples, 99]
```

---

## 3. Metric 定义

### 3.1 RawInput

RawInput 比较 JiT denoiser 第一层 block 的 timestep/condition-modulated proxy：

```text
Delta_raw = relative_L1(I_t, I_{t+1})
```

这里的 `I_t` 不是原始图像，而是 JiT first block MSA 前的 AdaLN-modulated token：

```python
t_emb = model.t_embedder(t)
y_emb = model.y_embedder(y)
c = t_emb + y_emb

tokens = model.x_embedder(x)
tokens = tokens + model.pos_embed

shift_msa, scale_msa, _, _, _, _ = first_block.adaLN_modulation(c).chunk(6, dim=-1)
proxy = first_block.norm1(tokens)
proxy = proxy * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
```

RawInput 代表模型早期 feature 在当前 timestep 和 condition 下的变化。

### 3.2 SEAInput

SEAInput 在 RawInput proxy 上加入 spectral filtering：

```text
Delta_sea = relative_L1(SEA(I_t), SEA(I_{t+1}))
```

实现流程：

```text
proxy tokens [B, N, C]
-> reshape to [B, C, H, W]
-> FFT
-> timestep-dependent spectral filter
-> iFFT
-> reshape back to [B, N, C]
-> relative L1
```

SEA filter 的动机是抑制 noisy feature 中与内容无关的频率成分，使 input-space distance 更接近内容变化。

当前参数：

```text
sea_filter_beta = 2.0
sea_filter_eps = 1e-6
```

### 3.3 DINO Clean-Image Drift

DINO 在 clean-image prediction space 中比较相邻 `xhat` 的语义变化：

```text
Delta_dino = 1 - cosine(DINO(xhat_t), DINO(xhat_{t+1}))
```

本阶段设置：

| 项目 | 设置 |
|---|---|
| Backbone | DINOv2 ViT-B/14 |
| Feature | last-layer patch tokens mean |
| Input size | 224 |
| Input range | `xhat [-1, 1] -> image [0, 1] -> ImageNet normalize` |

DINO 近似 global semantic drift，更关心物体结构、语义和布局。

### 3.4 LPIPS Clean-Image Drift

LPIPS 衡量相邻 `xhat` 的 perceptual / texture drift：

```text
Delta_lpips = LPIPS(xhat_t, xhat_{t+1})
```

本阶段设置：

| 项目 | 设置 |
|---|---|
| LPIPS net | Alex |
| Input size | 128 |
| Input range | `[-1, 1]` |

LPIPS 更偏局部纹理、边缘和人眼感知差异，与 DINO 互补。

### 3.5 PMA Fusion

E3/E4 中 PMA 使用归一化后的 SEA / DINO / LPIPS score：

```text
score = weighted_sum(normalized_SEA, normalized_DINO, normalized_LPIPS)
```

PMA-no-gate 使用固定权重：

```text
score = 0.4 * SEA + 0.3 * DINO + 0.3 * LPIPS
```

原始 PMA-stage-aware 使用分段 hard gate：

```text
early:
  score = 1.0 * SEA + 0.0 * DINO + 0.0 * LPIPS

middle:
  score = 0.5 * SEA + 0.5 * DINO + 0.0 * LPIPS

late:
  score = 0.25 * SEA + 0.35 * DINO + 0.40 * LPIPS
```

stage 划分：

```text
early  : call step fraction < 0.30
middle : 0.30 <= call step fraction < 0.70
late   : call step fraction >= 0.70
```

E4-ablation 额外测试了三组 soft stage-aware 权重：

```text
Candidate A:
early  = 0.75 SEA + 0.25 DINO + 0.00 LPIPS
middle = 0.45 SEA + 0.45 DINO + 0.10 LPIPS
late   = 0.25 SEA + 0.35 DINO + 0.40 LPIPS

Candidate B:
early  = 0.70 SEA + 0.20 DINO + 0.10 LPIPS
middle = 0.45 SEA + 0.35 DINO + 0.20 LPIPS
late   = 0.30 SEA + 0.30 DINO + 0.40 LPIPS

Candidate C:
early  = 0.85 SEA + 0.15 DINO + 0.00 LPIPS
middle = 0.55 SEA + 0.35 DINO + 0.10 LPIPS
late   = 0.35 SEA + 0.30 DINO + 0.35 LPIPS
```

---

## 4. E0: Deterministic Full-Reference Sanity

### 4.1 目的

E0 的目标是确认 full uncached PixelGen inference 在固定 class/seed pairs 下足够稳定。E1/E4 的 paired
PSNR / SSIM / LPIPS 都依赖这个 reference 可靠性。

### 4.2 实现与输出

```text
Script:
  scripts/00_sanity_full_reference.py

Output:
  outputs/e0_sanity/e0_32samples_fp32/
```

设置：

| 项目 | 值 |
|---|---|
| Samples | 32 |
| Batch size | 1 |
| Precision | fp32, `--no-autocast` |
| Fused SDPA | disabled |
| Denoiser calls | `32 * 99 = 3168` per run |

### 4.3 结果

| Metric | Value |
|---|---:|
| Samples | 32 |
| Max absolute difference | `2.32458e-06` |
| Mean max absolute difference | `7.26432e-08` |
| Min PSNR | `141.889 dB` |
| Mean PSNR | `141.889 dB` |
| uint8 equal count | 31 / 32 |
| uint8 equal fraction | 96.875% |

### 4.4 E0 结论

PixelGen full inference 在当前 fp32 + math SDPA + fixed class/seed pairs 设置下足够稳定，可以作为 paired
reference。E1/E4 final-image fidelity 差异远大于 E0 的数值误差。

---

## 5. E1: Online RawInput / SEAInput Baseline

### 5.1 目的

E1 测试真实 online cache 场景：

```text
当前 cached trajectory 上实时计算 RawInput / SEAInput distance
-> accumulated distance 超过 delta 则 refresh
-> 否则复用 cached denoiser output
```

比较对象：

| Method | 含义 |
|---|---|
| Full reference | 不使用 cache |
| Uniform | 固定间隔 refresh |
| RawInput-online | 当前 cached trajectory 上实时 RawInput distance |
| SEAInput-online | 当前 cached trajectory 上实时 SEAInput distance |

### 5.2 实现与输出

```text
Script:
  scripts/01_e1_online_cache.py

Output:
  outputs/e1_online_cache/e1_pilot_64_fp32/
```

设置：

| 项目 | 值 |
|---|---|
| Samples | 64 |
| Opportunities/sample | 99 |
| Total opportunities | 6336 |
| Calibration samples | 8 |
| LPIPS | Alex, loaded successfully |
| Precision | fp32, `--no-autocast` |

### 5.3 E1 主结果

| Method | Actual RR | Refresh/sample | Speedup vs full | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Uniform RR0.3 | 0.303 | 30.00 | 2.40x | 24.94 | 0.8891 | 0.1811 |
| Uniform RR0.5 | 0.505 | 50.00 | 1.41x | 27.58 | 0.9181 | 0.1364 |
| RawInput-online RR0.3 | 0.273 | 27.02 | 2.15x | 29.37 | 0.9360 | 0.1102 |
| RawInput-online RR0.5 | 0.535 | 53.00 | 1.28x | 47.81 | 0.9950 | 0.0112 |
| SEAInput-online RR0.3 | 0.293 | 29.03 | 2.18x | 35.19 | 0.9730 | 0.0461 |
| SEAInput-online RR0.5 | 0.464 | 45.97 | 1.45x | 44.24 | 0.9941 | 0.0129 |

### 5.4 E1 Paired 判断

RR≈0.30 是 E1 中最干净的比较，因为 RawInput-online 和 SEAInput-online 的 actual RR 接近：

```text
RawInput-online RR = 0.273
SEAInput-online RR = 0.293
```

SEAInput-online 相对 RawInput-online：

| Metric | SEA | Raw | Difference |
|---|---:|---:|---:|
| PSNR ↑ | 35.19 | 29.37 | +5.82 dB |
| SSIM ↑ | 0.9730 | 0.9360 | +0.0370 |
| LPIPS ↓ | 0.0461 | 0.1102 | -0.0642 |

SEAInput-online 相对 Uniform RR0.3：

| Metric | SEA | Uniform | Difference |
|---|---:|---:|---:|
| PSNR ↑ | 35.19 | 24.94 | +10.24 dB |
| SSIM ↑ | 0.9730 | 0.8891 | +0.0839 |
| LPIPS ↓ | 0.0461 | 0.1811 | -0.1350 |

### 5.5 E1 结论

E1 说明在 PixelGen 上，简单 RawInput-online 不是可靠 baseline。SEAInput-online 在低 refresh budget 下明显更稳，尤其显著降低 LPIPS 和 tail failure。这个结果支撑了本阶段从 spectral filtering 走向 perceptual geometry 的实验设计。

---

## 6. E2: Oracle Distance Bank Extraction

### 6.1 目的

E2 不做 cache rerun，而是在 full uncached trajectory 上提取每个 sample、每个相邻 call transition 的 distance：

```text
RawInput distance
SEAInput distance
DINO(xhat) distance
LPIPS(xhat) distance
```

E2 的输出是 E3 oracle schedule search 的输入。

### 6.2 实现与输出

```text
Script:
  scripts/02_e2_extract_distance_bank.py

Output:
  outputs/e2_distance_bank/e2_main_256_fp32/
```

设置：

| 项目 | 值 |
|---|---|
| Samples | 256 |
| Completed samples | 256 / 256 |
| Calls/sample | 99 |
| Distances/sample | 98 |
| Raw NaN | 0 |
| SEA NaN | 0 |
| DINO NaN | 0 |
| LPIPS NaN | 0 |
| DINO source | local DINOv2 cache |
| DINO feature | patch mean |
| DINO size | 224 |
| LPIPS | Alex |
| LPIPS size | 128 |
| Elapsed | 1255.42 s |

### 6.3 Metric Means

全 bank 平均距离：

| Metric | Mean |
|---|---:|
| Raw | 0.0217908 |
| SEA | 1.7937052 |
| DINO | 0.0033168 |
| LPIPS | 0.0034451 |

SEA mean 很大主要由第一个 transition 的巨大 early spike 主导，不能直接与 Raw / DINO / LPIPS 的原始尺度比较。

### 6.4 Stage-Wise Mean Distance

用 E3 同样的 stage 划分统计 E2 distance：

| Stage | Raw mean | SEA mean | DINO mean | LPIPS mean |
|---|---:|---:|---:|---:|
| Early | 0.021177 | 5.996945 | 0.009207 | 0.010085 |
| Middle | 0.018219 | 0.027143 | 0.001221 | 0.000960 |
| Late | 0.027331 | 0.027103 | 0.000318 | 0.000233 |

观察：

1. Raw late mean 最高，说明 RawInput 容易被 late numerical / feature drift 主导。
2. DINO / LPIPS 在 early 最大，中期仍有小的 perceptual bump，late 很低。
3. SEA 的 early spike 极强，但 middle / late 与 Raw 的形态不同。

### 6.5 Predictor / Corrector 结构

E2 显示 Heun exact 的 call-kind 结构非常强。相邻 transition 可以分成：

```text
predictor -> corrector
corrector -> next predictor
```

distance mass 几乎都集中在 predictor -> corrector：

| Metric | predictor -> corrector mass | corrector -> next predictor mass |
|---|---:|---:|
| Raw | 98.96% | 1.04% |
| SEA | 99.99% | 0.01% |
| DINO | 96.81% | 3.19% |
| LPIPS | 97.71% | 2.29% |

这解释了为什么 call-level schedule 大量 refresh 落在 corrector calls：E3 accumulator 用 `call c-1 -> call c` 的 distance 决定当前 call 是否 refresh，而 predictor -> corrector transition 往往最大。

### 6.6 Metric Correlation

`log1p` Pearson correlation：

| Pair | All transitions | Predictor -> corrector excluding idx0 |
|---|---:|---:|
| DINO - LPIPS | 0.748 | 0.693 |
| SEA - LPIPS | 0.649 | 0.813 |
| SEA - DINO | 0.501 | 0.582 |
| Raw - LPIPS | 0.257 | -0.091 |
| Raw - DINO | 0.219 | -0.078 |

结论：

1. SEA 与 DINO/LPIPS 的相关性明显高于 Raw。
2. 在主要变化的 predictor -> corrector transitions 中，Raw 与 perceptual metrics 接近无关甚至轻微负相关。
3. E2 从 trajectory 数据层面解释了 E1 中 SEAInput-online 优于 RawInput-online 的原因。

---

## 7. E3: Schedule-Level Oracle Analysis

### 7.1 目的

E3 使用 E2 distance bank 生成 matched-RR oracle refresh schedules。E3 不 rerun PixelGen cache，只分析 schedule 的 refresh pattern。

E3 的 oracle 含义：

```text
已知 full uncached trajectory 上的 Raw / SEA / DINO / LPIPS distances
-> 离线构造 accumulator schedule
-> 比较 metric 本身产生 refresh schedule 的上限形态
```

### 7.2 实现与输出

```text
Script:
  scripts/03_e3_schedule_oracle_analysis.py

Output:
  outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/
```

核心输出：

```text
schedule_summary.csv
stage_refresh_density.csv
stage_kind_refresh_density.csv
threshold_vs_rr.csv
matched_schedules/*.npz
```

其中 `matched_schedules/*.npz` 被 E4 fixed-schedule real cache rerun 使用。

设置：

| 项目 | 值 |
|---|---|
| Samples | 256 |
| Calibration samples | 64 |
| Test samples | 192 |
| Target RR | 0.30 / 0.40 / 0.50 |
| Warmup forced calls | 5 |
| Final call forced | true |
| Forced calls/sample | 6 |
| Stage split | early < 0.30, late >= 0.70 |

### 7.3 E3 Methods

| Method | Score source |
|---|---|
| Uniform | fixed interval schedule |
| RawInput-oracle | normalized RawInput distance |
| SEAInput-oracle | normalized SEAInput distance |
| DINO-oracle | normalized DINO clean-image drift |
| LPIPS-oracle | normalized LPIPS clean-image drift |
| PMA-no-gate-oracle | `0.4 SEA + 0.3 DINO + 0.3 LPIPS` |
| PMA-stage-aware-oracle | original stage-aware hard gate |

Normalization：

```text
SEA transform = log1p
Raw / DINO / LPIPS transform = identity
then robust median normalization on calibration set
then p99 clipping
```

本次 normalization 统计：

| Metric | Transform | Calibration median | p99 clip value |
|---|---|---:|---:|
| Raw | identity | 0.018181 | 3.523 |
| SEA | log1p | 0.020625 | 249.148 |
| DINO | identity | 0.000194 | 248.493 |
| LPIPS | identity | 0.000256 | 218.981 |

### 7.4 Matched-RR 结果

| Method | Target RR | Calibration RR | Test RR | All RR | Test refresh/sample |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.30 | 0.3030 | 0.3030 | 0.3030 | 30.00 |
| Uniform | 0.40 | 0.4040 | 0.4040 | 0.4040 | 40.00 |
| Uniform | 0.50 | 0.5051 | 0.5051 | 0.5051 | 50.00 |
| RawInput-oracle | 0.30 | 0.3000 | 0.3024 | 0.3018 | 29.94 |
| RawInput-oracle | 0.40 | 0.3999 | 0.4044 | 0.4033 | 40.04 |
| RawInput-oracle | 0.50 | 0.5000 | 0.5092 | 0.5069 | 50.41 |
| SEAInput-oracle | 0.30 | 0.3000 | 0.3010 | 0.3008 | 29.80 |
| SEAInput-oracle | 0.40 | 0.3999 | 0.4015 | 0.4011 | 39.75 |
| SEAInput-oracle | 0.50 | 0.5000 | 0.5060 | 0.5045 | 50.09 |
| DINO-oracle | 0.30 | 0.3000 | 0.3091 | 0.3068 | 30.60 |
| DINO-oracle | 0.40 | 0.3999 | 0.4129 | 0.4096 | 40.88 |
| DINO-oracle | 0.50 | 0.5000 | 0.5180 | 0.5135 | 51.28 |
| LPIPS-oracle | 0.30 | 0.3000 | 0.2872 | 0.2904 | 28.43 |
| LPIPS-oracle | 0.40 | 0.3999 | 0.3849 | 0.3887 | 38.10 |
| LPIPS-oracle | 0.50 | 0.5000 | 0.4904 | 0.4928 | 48.55 |
| PMA-no-gate-oracle | 0.30 | 0.3000 | 0.2979 | 0.2984 | 29.49 |
| PMA-no-gate-oracle | 0.40 | 0.3999 | 0.3986 | 0.3990 | 39.46 |
| PMA-no-gate-oracle | 0.50 | 0.5000 | 0.5026 | 0.5019 | 49.76 |
| PMA-stage-aware-oracle | 0.30 | 0.3000 | 0.3052 | 0.3039 | 30.22 |
| PMA-stage-aware-oracle | 0.40 | 0.3999 | 0.4045 | 0.4033 | 40.04 |
| PMA-stage-aware-oracle | 0.50 | 0.5000 | 0.5047 | 0.5036 | 49.97 |

PMA-no-gate 与 PMA-stage-aware 的 RR calibration 比单独 DINO / LPIPS 更稳。单独 DINO 在 test set 偏高，单独 LPIPS 偏低。

### 7.5 Stage Refresh Share

Test split 上的 stage refresh share：

RR≈0.30：

| Method | Early share | Middle share | Late share |
|---|---:|---:|---:|
| RawInput-oracle | 36.7% | 32.6% | 30.7% |
| SEAInput-oracle | 53.7% | 23.6% | 22.7% |
| PMA-no-gate-oracle | 57.3% | 26.3% | 16.4% |
| PMA-stage-aware-oracle | 49.6% | 31.5% | 18.9% |

RR≈0.40：

| Method | Early share | Middle share | Late share |
|---|---:|---:|---:|
| RawInput-oracle | 34.0% | 28.6% | 37.4% |
| SEAInput-oracle | 45.3% | 30.8% | 23.9% |
| PMA-no-gate-oracle | 46.6% | 32.3% | 21.0% |
| PMA-stage-aware-oracle | 45.0% | 33.5% | 21.5% |

RR≈0.50：

| Method | Early share | Middle share | Late share |
|---|---:|---:|---:|
| RawInput-oracle | 35.6% | 34.6% | 29.8% |
| SEAInput-oracle | 35.9% | 34.5% | 29.5% |
| PMA-no-gate-oracle | 38.5% | 35.3% | 26.2% |
| PMA-stage-aware-oracle | 36.0% | 37.3% | 26.7% |

E3 schedule-level 观察：

1. RawInput-oracle 在 RR0.40 下明显偏 late。
2. SEAInput-oracle 比 Raw 更稳定，但 RR0.30 下 early share 很高。
3. PMA-no-gate 受 early perceptual spike 影响，RR0.30 early share 达到 57.3%。
4. 原始 PMA-stage-aware 减少 no-gate 的 early share，并提高 middle share。
5. Schedule-level 上，stage-aware hard gate 确实改变了 budget 分布，而不是只改变 threshold。

---

## 8. E4: Oracle-Schedule Real Cache Rerun

### 8.1 目的

E4 是第一阶段主实验：把 E3 生成的 fixed oracle schedules 放回真实 PixelGen cache loop 中重新采样。

E4 与 E3 的区别：

```text
E3:
  只在 full trajectory distance bank 上离线分析 schedule。

E4:
  按 schedule 真实 rerun cache inference。
  schedule[i, c] = true  -> call c 正常 full denoiser refresh
  schedule[i, c] = false -> 复用 cached denoiser output
```

E4 直接测 cached final image 与 uncached full reference 的 paired fidelity：

```text
PSNR ↑
SSIM ↑
LPIPS ↓
paired delta vs SEA-oracle
bootstrap 95% CI
```

### 8.2 实现与输出

```text
Script:
  scripts/04_e4_oracle_schedule_cache_rerun.py

Output:
  outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/
```

设置：

| 项目 | 值 |
|---|---|
| Split | E3 test split |
| Samples | 192 |
| Sample indices | 64-255 |
| Opportunities/sample | 99 |
| Total opportunities | 19008 |
| Precision | fp32, `--no-autocast` |
| LPIPS | Alex, loaded successfully |

### 8.3 E4 主结果

| Method | Target RR | Actual RR | Refresh/sample | Speedup vs full | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 0.30 | 0.3030 | 30.00 | 3.24x | 31.61 | 0.9614 | 0.0684 |
| Uniform | 0.40 | 0.4040 | 40.00 | 2.45x | 33.78 | 0.9737 | 0.0478 |
| Uniform | 0.50 | 0.5051 | 50.00 | 1.97x | 38.54 | 0.9891 | 0.0216 |
| RawInput-oracle | 0.30 | 0.3024 | 29.94 | 3.25x | 29.92 | 0.9496 | 0.0834 |
| RawInput-oracle | 0.40 | 0.4044 | 40.04 | 2.45x | 36.70 | 0.9795 | 0.0381 |
| RawInput-oracle | 0.50 | 0.5092 | 50.41 | 1.96x | 45.13 | 0.9957 | 0.0094 |
| SEAInput-oracle | 0.30 | 0.3010 | 29.80 | 3.18x | 36.69 | 0.9800 | 0.0338 |
| SEAInput-oracle | 0.40 | 0.4015 | 39.75 | 2.45x | 41.37 | 0.9921 | 0.0152 |
| SEAInput-oracle | 0.50 | 0.5060 | 50.09 | 1.96x | 45.68 | 0.9959 | 0.0090 |
| PMA-no-gate-oracle | 0.30 | 0.2979 | 29.49 | 3.29x | 35.87 | 0.9736 | 0.0420 |
| PMA-no-gate-oracle | 0.40 | 0.3986 | 39.46 | 2.48x | 41.71 | 0.9929 | 0.0135 |
| PMA-no-gate-oracle | 0.50 | 0.5026 | 49.76 | 1.97x | 48.20 | 0.9981 | 0.0043 |
| PMA-stage-aware-oracle | 0.30 | 0.3052 | 30.22 | 3.22x | 36.26 | 0.9774 | 0.0369 |
| PMA-stage-aware-oracle | 0.40 | 0.4045 | 40.04 | 2.45x | 41.20 | 0.9915 | 0.0160 |
| PMA-stage-aware-oracle | 0.50 | 0.5047 | 49.97 | 1.97x | 46.02 | 0.9959 | 0.0090 |

### 8.4 Paired Delta vs SEAInput-Oracle

| Method | RR | Delta PSNR vs SEA | 95% CI | Delta LPIPS vs SEA | 95% CI |
|---|---:|---:|---|---:|---|
| Uniform | 0.30 | -5.078 | [-5.578, -4.562] | +0.03460 | [+0.02677, +0.04325] |
| Uniform | 0.40 | -7.593 | [-8.010, -7.162] | +0.03264 | [+0.02702, +0.03869] |
| Uniform | 0.50 | -7.147 | [-7.587, -6.707] | +0.01267 | [+0.01029, +0.01537] |
| RawInput-oracle | 0.30 | -6.766 | [-7.312, -6.221] | +0.04957 | [+0.04065, +0.05896] |
| RawInput-oracle | 0.40 | -4.668 | [-5.271, -4.071] | +0.02294 | [+0.01613, +0.03095] |
| RawInput-oracle | 0.50 | -0.557 | [-0.766, -0.367] | +0.00044 | [+0.00014, +0.00071] |
| PMA-no-gate-oracle | 0.30 | -0.820 | [-1.176, -0.463] | +0.00825 | [+0.00391, +0.01167] |
| PMA-no-gate-oracle | 0.40 | +0.345 | [-0.126, +0.863] | -0.00173 | [-0.00644, +0.00109] |
| PMA-no-gate-oracle | 0.50 | +2.515 | [+1.757, +3.323] | -0.00467 | [-0.00954, -0.00170] |
| PMA-stage-aware-oracle | 0.30 | -0.425 | [-0.669, -0.187] | +0.00313 | [+0.00128, +0.00501] |
| PMA-stage-aware-oracle | 0.40 | -0.165 | [-0.503, +0.178] | +0.00080 | [-0.00006, +0.00164] |
| PMA-stage-aware-oracle | 0.50 | +0.338 | [-0.063, +0.747] | -0.00000 | [-0.00043, +0.00035] |

### 8.5 E4 主实验结论

1. Uniform schedule 明显弱于 oracle schedules，说明 refresh 位置比单纯 refresh 数量更重要。
2. RawInput-oracle 明显弱于 SEAInput-oracle，尤其 RR0.30 / RR0.40。这说明 RawInput metric 本身不是可靠的 schedule signal。
3. SEAInput-oracle 在 RR0.30 是最强主方法：PSNR 36.69、LPIPS 0.0338。
4. PMA-no-gate-oracle 在 RR0.50 显著优于 SEAInput-oracle：Delta PSNR +2.515 dB，Delta LPIPS -0.00467。
5. 原始 PMA-stage-aware 在 E3 schedule-level 上改变了 refresh geometry，但 E4 real cache rerun 中没有稳定超过 SEAInput-oracle。
6. 原始 hard stage gate 过度压制 early perceptual branch 的迹象明显：RR0.40 / RR0.50 下 PMA-no-gate 明显强于原始 PMA-stage-aware。

---

## 9. E4-Ablation: PMA Stage-Aware Weight Candidates

### 9.1 目的

E4 主实验显示原始 hard stage-aware gate 没有稳定超过 no-gate。E4-ablation 固定其他设置不变，只改变 PMA-stage-aware 的 stage weights，比较 soft gate 是否改善真实 cache rerun fidelity。

### 9.2 实现与输出

```text
Schedule preparation:
  scripts/05_e4_prepare_pma_weight_candidates.py

Parallel rerun:
  scripts/run_e4_pma_weight_candidates_parallel.sh

Comparison:
  scripts/06_e4_compare_pma_weight_candidates.py

Outputs:
  outputs/e4_oracle_schedule_cache/e4_candidate_a_rr030_rr040_rr050_fp32/
  outputs/e4_oracle_schedule_cache/e4_candidate_b_rr030_rr040_rr050_fp32/
  outputs/e4_oracle_schedule_cache/e4_candidate_c_rr030_rr040_rr050_fp32/
  outputs/e4_pma_weight_candidates/comparison/compare_candidates_vs_main/
```

Reference tensor：

```text
outputs/e4_oracle_schedule_cache/e4_reference_test_fp32/reference_images.pt
```

A/B/C 三个 candidate rerun 复用了同一个 full reference tensor，因此没有重复跑 full reference。

### 9.3 Candidate Schedule RR

| Candidate | Target RR | Test RR | All RR |
|---|---:|---:|---:|
| A | 0.30 | 0.3052 | 0.3039 |
| A | 0.40 | 0.4045 | 0.4034 |
| A | 0.50 | 0.5049 | 0.5037 |
| B | 0.30 | 0.3020 | 0.3015 |
| B | 0.40 | 0.4007 | 0.4005 |
| B | 0.50 | 0.5018 | 0.5014 |
| C | 0.30 | 0.3033 | 0.3025 |
| C | 0.40 | 0.4030 | 0.4022 |
| C | 0.50 | 0.5051 | 0.5038 |

### 9.4 Candidate Real Rerun Results

| Candidate | RR | Actual RR | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| A | 0.30 | 0.3052 | 36.27 | 0.9759 | 0.0386 |
| A | 0.40 | 0.4045 | 41.56 | 0.9925 | 0.0141 |
| A | 0.50 | 0.5049 | 47.34 | 0.9976 | 0.0054 |
| B | 0.30 | 0.3020 | 36.09 | 0.9752 | 0.0400 |
| B | 0.40 | 0.4007 | 41.48 | 0.9926 | 0.0142 |
| B | 0.50 | 0.5018 | 47.31 | 0.9976 | 0.0057 |
| C | 0.30 | 0.3033 | 36.33 | 0.9768 | 0.0376 |
| C | 0.40 | 0.4030 | 41.49 | 0.9926 | 0.0141 |
| C | 0.50 | 0.5051 | 46.81 | 0.9971 | 0.0067 |

### 9.5 Candidate Paired Comparison

Candidate A：

| Baseline | RR | Delta PSNR | 95% CI | Delta LPIPS | 95% CI |
|---|---:|---:|---|---:|---|
| SEAInput-oracle | 0.30 | -0.414 | [-0.768, -0.088] | +0.00484 | [+0.00183, +0.00774] |
| PMA-no-gate | 0.30 | +0.406 | [+0.241, +0.541] | -0.00342 | [-0.00473, -0.00163] |
| Original stage-aware | 0.30 | +0.011 | [-0.162, +0.193] | +0.00171 | [-0.00048, +0.00332] |
| SEAInput-oracle | 0.40 | +0.193 | [-0.207, +0.602] | -0.00105 | [-0.00475, +0.00107] |
| PMA-no-gate | 0.40 | -0.152 | [-0.452, +0.094] | +0.00067 | [-0.00036, +0.00206] |
| Original stage-aware | 0.40 | +0.359 | [+0.147, +0.589] | -0.00185 | [-0.00520, -0.00018] |
| SEAInput-oracle | 0.50 | +1.655 | [+0.998, +2.380] | -0.00356 | [-0.00827, -0.00094] |
| PMA-no-gate | 0.50 | -0.860 | [-1.232, -0.537] | +0.00112 | [+0.00016, +0.00252] |
| Original stage-aware | 0.50 | +1.318 | [+0.768, +1.934] | -0.00355 | [-0.00824, -0.00096] |

Candidate B：

| Baseline | RR | Delta PSNR | 95% CI | Delta LPIPS | 95% CI |
|---|---:|---:|---|---:|---|
| SEAInput-oracle | 0.30 | -0.593 | [-0.918, -0.283] | +0.00616 | [+0.00326, +0.00893] |
| PMA-no-gate | 0.30 | +0.226 | [+0.071, +0.352] | -0.00209 | [-0.00338, -0.00032] |
| Original stage-aware | 0.30 | -0.168 | [-0.323, -0.007] | +0.00304 | [+0.00079, +0.00473] |
| SEAInput-oracle | 0.40 | +0.112 | [-0.309, +0.532] | -0.00103 | [-0.00500, +0.00138] |
| PMA-no-gate | 0.40 | -0.234 | [-0.476, -0.046] | +0.00070 | [-0.00008, +0.00176] |
| Original stage-aware | 0.40 | +0.277 | [+0.028, +0.560] | -0.00183 | [-0.00552, +0.00017] |
| SEAInput-oracle | 0.50 | +1.631 | [+0.968, +2.375] | -0.00324 | [-0.00796, -0.00065] |
| PMA-no-gate | 0.50 | -0.884 | [-1.231, -0.564] | +0.00144 | [+0.00052, +0.00282] |
| Original stage-aware | 0.50 | +1.293 | [+0.737, +1.902] | -0.00324 | [-0.00792, -0.00058] |

Candidate C：

| Baseline | RR | Delta PSNR | 95% CI | Delta LPIPS | 95% CI |
|---|---:|---:|---|---:|---|
| SEAInput-oracle | 0.30 | -0.359 | [-0.660, -0.080] | +0.00380 | [+0.00164, +0.00603] |
| PMA-no-gate | 0.30 | +0.460 | [+0.278, +0.618] | -0.00445 | [-0.00637, -0.00150] |
| Original stage-aware | 0.30 | +0.065 | [-0.061, +0.190] | +0.00068 | [-0.00037, +0.00167] |
| SEAInput-oracle | 0.40 | +0.125 | [-0.222, +0.467] | -0.00109 | [-0.00454, +0.00090] |
| PMA-no-gate | 0.40 | -0.220 | [-0.550, +0.053] | +0.00064 | [-0.00052, +0.00222] |
| Original stage-aware | 0.40 | +0.290 | [+0.109, +0.511] | -0.00189 | [-0.00502, -0.00029] |
| SEAInput-oracle | 0.50 | +1.125 | [+0.628, +1.653] | -0.00224 | [-0.00568, -0.00042] |
| PMA-no-gate | 0.50 | -1.391 | [-1.946, -0.905] | +0.00244 | [+0.00097, +0.00446] |
| Original stage-aware | 0.50 | +0.787 | [+0.431, +1.167] | -0.00223 | [-0.00581, -0.00035] |

### 9.6 E4-Ablation 结论

1. Soft stage-aware candidates A/B/C 在 RR0.40 和 RR0.50 都明显优于原始 hard stage-aware。
2. Candidate A 是三组 soft gate 中整体最强的版本：
   - RR0.40: PSNR 41.56，LPIPS 0.0141。
   - RR0.50: PSNR 47.34，LPIPS 0.0054。
3. RR0.30 下 SEAInput-oracle 仍然最强，A/B/C 都显著弱于 SEAInput-oracle。
4. RR0.50 下 A/B/C 都显著强于 SEAInput-oracle 和原始 stage-aware。
5. RR0.50 下 PMA-no-gate 仍然显著强于 A/B/C。
6. 这些结果表明：原始 hard gate 确实过度压制了 early perceptual signal；但 high-RR 条件下，no-gate 保留的 early DINO/LPIPS spike 中包含对 final fidelity 有用的信息。

---

## 10. 第一阶段完整结论

### 10.1 已确认的事实

1. **Full reference 可靠。**  
   E0 显示 fp32 + math SDPA 下 full inference 的 run-to-run 差异极小，paired comparison 可信。

2. **Call-level 是必要约定。**  
   Heun exact 产生 99 个 denoiser opportunities，而不是 50 个 step。E2/E3 进一步显示 predictor/corrector 结构对 distance 和 schedule 影响极大。

3. **SEAInput 明显优于 RawInput。**  
   E1 online cache 和 E4 oracle rerun 都说明 RawInput metric 不可靠。E2 correlation 也显示 Raw 与 DINO/LPIPS perceptual drift 的相关性弱。

4. **SEA 与 perceptual drift 的关系更强。**  
   E2 中 SEA-LPIPS / SEA-DINO correlation 明显高于 Raw-LPIPS / Raw-DINO。尤其在 predictor -> corrector 且去掉第一个巨大 spike 后，SEA-LPIPS correlation 达到 0.813。

5. **Refresh 位置比 refresh 数量更重要。**  
   E4 中 Uniform 在相同 RR 下显著落后于 oracle schedules，说明 matched-RR 本身不足以保证 fidelity。

6. **PMA perceptual branch 有上限价值，但依赖 budget。**  
   PMA-no-gate 在 RR0.50 显著优于 SEAInput-oracle；在 RR0.30 则显著弱于 SEAInput-oracle。

7. **原始 hard stage-aware gate 在 schedule-level 合理，但 real rerun 不够强。**  
   E3 中原始 PMA-stage-aware 确实把 refresh share 从 early 转向 middle；E4 中它没有稳定超过 SEAInput-oracle。

8. **Soft stage-aware 比原始 hard gate 更好。**  
   E4-ablation 中 A/B/C 在 RR0.40 / RR0.50 均提升原始 stage-aware，说明完全关闭 early perceptual branch 过于激进。

### 10.2 各 RR 档位的经验结论

RR≈0.30：

```text
SEAInput-oracle 最强。
PMA-stage-aware / soft candidates 接近但仍弱于 SEA。
PMA-no-gate 弱于 SEA，说明低 budget 下 early perceptual spike 容易浪费 refresh。
```

RR≈0.40：

```text
PMA-no-gate、SEAInput-oracle、soft stage-aware A/B/C 接近。
Candidate A/B/C 明显强于原始 hard stage-aware。
统计上没有一个 PMA-stage-aware candidate 稳定显著超过 SEA。
```

RR≈0.50：

```text
PMA-no-gate 最强。
Candidate A/B/C 显著超过 SEAInput-oracle 和原始 hard stage-aware。
Candidate A 是 soft stage-aware 中最强，但仍显著弱于 no-gate。
```

### 10.3 对核心问题的回答

第一阶段的答案是有条件成立的：

```text
Clean-image perceptual drift 确实包含 RawInput / SEAInput 之外的有效 refresh 信息。
这种信息在较高 refresh budget 下尤其有价值，PMA-no-gate RR0.50 明显超过 SEAInput-oracle。
```

同时，实验也给出了限制：

```text
低 refresh budget 下，SEAInput-oracle 更稳。
DINO/LPIPS early spike 不能被简单视作噪声，也不能不加控制地全部相信。
Hard stage gate 会过度压制 early perceptual signal。
Soft gate 能改善 hard gate，但当前最强 high-RR upper bound 仍然来自 no-gate。
```

因此，E0-E4 的完整实验结论可以概括为：

```text
PixelGen 的 x-pred trajectory 中确实存在有用的 perceptual drift structure。
SEA filtering 是 RawInput 到 perceptual geometry 的重要桥梁。
PMA clean-image perceptual score 在 oracle setting 下具有上限价值。
Stage-aware 设计需要 soft gating；hard early shutdown 在真实 cache rerun 中不是最优。
```

---

## 11. 输出文件索引

E0：

```text
outputs/e0_sanity/e0_32samples_fp32/summary.json
outputs/e0_sanity/e0_32samples_fp32/per_sample_determinism.csv
```

E1：

```text
outputs/e1_online_cache/e1_pilot_64_fp32/summary.json
outputs/e1_online_cache/e1_pilot_64_fp32/method_summary.csv
outputs/e1_online_cache/e1_pilot_64_fp32/*/per_sample_metrics.csv
```

E2：

```text
outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz
outputs/e2_distance_bank/e2_main_256_fp32/summary.json
outputs/e2_distance_bank/e2_main_256_fp32/per_call_average_curves.csv
```

E3：

```text
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/schedule_summary.csv
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/stage_refresh_density.csv
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/stage_kind_refresh_density.csv
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/matched_schedules/*.npz
```

E4 main：

```text
outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/summary.json
outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/method_summary.csv
outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/paired_delta_vs_baseline.csv
outputs/e4_oracle_schedule_cache/e4_test_rr030_rr040_rr050_fp32/*/per_sample_metrics.csv
```

E4 candidate ablation：

```text
outputs/e4_pma_weight_candidates/e4_pma_weight_candidates_from_e2_fp32_calib64/schedule_summary.csv
outputs/e4_oracle_schedule_cache/e4_candidate_a_rr030_rr040_rr050_fp32/method_summary.csv
outputs/e4_oracle_schedule_cache/e4_candidate_b_rr030_rr040_rr050_fp32/method_summary.csv
outputs/e4_oracle_schedule_cache/e4_candidate_c_rr030_rr040_rr050_fp32/method_summary.csv
outputs/e4_pma_weight_candidates/comparison/compare_candidates_vs_main/candidate_pairwise_comparison.csv
outputs/e4_pma_weight_candidates/comparison/compare_candidates_vs_main/summary.json
```
