# E3 Schedule-level Oracle Analysis 阶段性实验总结

本文总结 PixelGen cache acceleration 第一阶段中的 E3 实验：

```text
E3: Schedule-level oracle analysis
```

E3 的目标不是证明某个 cache 方法最终图像质量更好，而是利用 E2 已经提取好的 full-trajectory distance bank，在完全离线的 oracle 条件下回答：

1. 不同 metric 在 call-level Heun 轨迹上的 refresh pattern 是否合理；
2. accumulated-distance threshold 是否能稳定控制 refresh ratio；
3. PMA-stage-aware 是否产生了区别于 RawInput / SEAInput / PMA-no-gate 的 schedule；
4. 哪些现象值得进入 E4 的真实 cached inference rerun。

本次主结果使用：

```text
Distance bank:
outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz

E3 output:
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/

E3 script:
scripts/03_e3_schedule_oracle_analysis.py
```

---

## 1. 实验定位

### 1.1 E3 做什么

E3 只做离线 schedule search。它读取 E2 的 full trajectory distance bank：

```text
raw   : RawInput proxy adjacent-call distance
sea   : SEA-filtered input proxy adjacent-call distance
dino  : clean-image xhat DINO feature distance
lpips : clean-image xhat LPIPS distance
```

然后对每个 sample 生成不同方法的 refresh schedule：

```text
Uniform
RawInput-oracle
SEAInput-oracle
DINO-oracle
LPIPS-oracle
PMA-no-gate-oracle
PMA-stage-aware-oracle
```

这里所有 oracle schedule 都是 cheating schedule：它们提前看到了 full uncached trajectory 上的 metric 序列。因此 E3 比较的是 metric 的 schedule 上限价值，而不是可部署性。

### 1.2 E3 不做什么

E3 不重新跑 PixelGen denoiser，也不计算 cached final image fidelity。它不能直接证明：

```text
PMA-stage-aware cached image > SEA cached image
```

原因是 cache 一旦开始复用 denoiser output，后续 sampler trajectory 会偏离 full reference。一个离线 schedule 看起来合理，不代表真实 rerun 后 final image 一定更接近 full trajectory。

因此 E3 的结论必须进入 E4：

```text
用 E3 schedule 做真实 fixed-schedule cache rerun
```

---

## 2. Call-level 记号

PixelGen 当前使用 Heun exact sampler：

```text
num_steps = 50
exact_henu = true
```

所以每个 sample 有：

```text
50 predictor calls + 49 corrector calls = 99 denoiser opportunities
```

记：

```math
c \in \{0, 1, ..., 98\}
```

为 denoiser call index。

相邻 call distance 的 index 为：

```math
k \in \{0, 1, ..., 97\}
```

其中第 `k` 个 distance 描述：

```math
call_k \rightarrow call_{k+1}
```

因此：

```text
schedule length = 99
distance length = 98
```

这也是 E3 的基本单位。后续所有 RR、threshold search、refresh heatmap、stage density 都是 call-level，而不是 step-level。

---

## 3. Metric 定义

设第 `i` 个 sample 在 call `c` 的输入状态为 `x_{i,c}`，timestep 为 `t_c`。

E2 已经记录了每个相邻 call 的 metric：

```math
\Delta^m_{i,k}
```

其中：

```math
m \in \{raw, sea, dino, lpips\}
```

### 3.1 RawInput distance

RawInput 使用 JiT 第一层经过 timestep/class modulation 后的 token proxy：

```math
I_{i,c}
```

distance 定义为相对 L1：

```math
\Delta^{raw}_{i,k}
=
\frac{
  \| I_{i,k+1} - I_{i,k} \|_1
}{
  \| I_{i,k} \|_1 + \epsilon
}
```

RawInput 是最原始的 input/proxy-space 动态判据。

### 3.2 SEAInput distance

SEAInput 先对 token proxy 做频域 SEA filter：

```math
P(G(t), I_{i,c})
```

distance 为：

```math
\Delta^{sea}_{i,k}
=
\frac{
  \| P(G(t_{k+1}), I_{i,k+1}) - P(G(t_k), I_{i,k}) \|_1
}{
  \| P(G(t_k), I_{i,k}) \|_1 + \epsilon
}
```

SEAInput 仍然是 input/proxy-space 判据，但通过频域先验抑制高噪声成分，强调更像内容结构的成分。

### 3.3 DINO clean-image drift

PixelGen / JiT 是 clean-image prediction，也就是模型输出可以转成：

```math
\hat{x}_{i,c}
```

DINO distance 定义为：

```math
\Delta^{dino}_{i,k}
=
1 -
\cos(
  \phi_{DINO}(\hat{x}_{i,k}),
  \phi_{DINO}(\hat{x}_{i,k+1})
)
```

本实验中 E2 使用 DINOv2-B，并取 patch token mean feature。DINO 近似 global semantic drift。

### 3.4 LPIPS clean-image drift

LPIPS distance 定义为：

```math
\Delta^{lpips}_{i,k}
=
LPIPS(\hat{x}_{i,k}, \hat{x}_{i,k+1})
```

本实验中 E2 使用 AlexNet LPIPS，输入 resize 到 128。LPIPS 近似 local perceptual / texture drift。

---

## 4. Robust normalization

不同 metric 的数值尺度差异非常大。例如 E2 中 raw mean 约 `0.0218`，SEA mean 约 `1.7937`，DINO/LPIPS mean 约 `0.0033`。直接相加没有意义。

E3 使用 calibration set 上的 median 做 robust normalization：

```math
\tilde{\Delta}^{m}_{i,k}
=
\frac{
  T_m(\Delta^{m}_{i,k})
}{
  \operatorname{median}_{(i,k)\in C}(T_m(\Delta^{m}_{i,k})) + \epsilon
}
```

其中 `C` 是 calibration set。

本次主实验：

```text
total samples       = 256
calibration samples = 64
test samples        = 192
```

SEA 有极端 early spike，因此 SEA 先做：

```math
T_{sea}(\Delta) = \log(1+\Delta)
```

其他 metric：

```math
T_m(\Delta) = \Delta
```

之后再做 p99 clipping：

```math
\tilde{\Delta}^{m}_{i,k}
\leftarrow
\operatorname{clip}(
  \tilde{\Delta}^{m}_{i,k},
  0,
  q^{m}_{99}
)
```

本次实际 normalization 统计：

| Metric | Transform | Calibration median | p99 clip value |
|---|---:|---:|---:|
| Raw | identity | 0.018181 | 3.523 |
| SEA | log1p | 0.020625 | 249.148 |
| DINO | identity | 0.000194 | 248.493 |
| LPIPS | identity | 0.000256 | 218.981 |

解释：

- Raw 的 normalized p99 只有 `3.52`，说明 Raw 数值更连续、更少极端 outlier；
- SEA / DINO / LPIPS 的 p99 都在 `~200` 量级，说明它们存在非常尖锐的 early perceptual / filtered-feature jump；
- 这支持 E3 必须使用 robust normalization 和 clipping，而不能用 raw scale 直接加权。

---

## 5. PMA score 定义

### 5.1 PMA-no-gate

PMA-no-gate 不区分 denoising stage，固定融合：

```math
\Delta^{pma\_nogate}_{i,k}
=
0.4 \tilde{\Delta}^{sea}_{i,k}
+
0.3 \tilde{\Delta}^{dino}_{i,k}
+
0.3 \tilde{\Delta}^{lpips}_{i,k}
```

它的作用是消融：

```text
只加入 perceptual metrics 是否足够？
```

如果 no-gate 和 stage-aware 差不多，说明 stage-aware gate 可能没有必要。

### 5.2 PMA-stage-aware

PMA-stage-aware 按 denoising progress 分段。E3 当前实现用当前 call step 的 fraction 定义 stage：

```math
r_k = \frac{step(call_{k+1})}{49}
```

然后：

```text
early  : r_k < 0.30
middle : 0.30 <= r_k < 0.70
late   : r_k >= 0.70
```

本次 E3 的 transition stage counts：

```text
early  = 29 transitions
middle = 40 transitions
late   = 29 transitions
```

call stage counts：

```text
early  = 30 calls
middle = 40 calls
late   = 29 calls
```

stage-aware 权重：

Early stage：

```math
\Delta^{pma}_{i,k}
=
\tilde{\Delta}^{sea}_{i,k}
```

Middle stage：

```math
\Delta^{pma}_{i,k}
=
0.5\tilde{\Delta}^{sea}_{i,k}
+
0.5\tilde{\Delta}^{dino}_{i,k}
```

Late stage：

```math
\Delta^{pma}_{i,k}
=
0.25\tilde{\Delta}^{sea}_{i,k}
+
0.35\tilde{\Delta}^{dino}_{i,k}
+
0.40\tilde{\Delta}^{lpips}_{i,k}
```

设计动机：

- early 高噪声阶段，clean-image perceptual feature 不稳定，所以只信 SEA；
- middle 主体结构逐渐形成，DINO semantic drift 开始有意义；
- late 低噪声阶段，DINO/LPIPS 对语义一致性和局部纹理更有意义，SEA 只保留辅助权重。

---

## 6. Refresh rule

所有 oracle metric 共用 accumulated-distance refresh rule。

对 sample `i`，维护 accumulator：

```math
A_{i,c}
```

从 call `c=0` 到 `c=98` 依次判断。

强制 refresh：

```text
call 0
warmup calls: c < 5
final call: c = 98
```

非强制 call 的规则为：

```math
A_i \leftarrow A_i + \Delta_{i,c-1}
```

如果：

```math
A_i > \delta
```

则：

```text
refresh at call c
A_i = 0
```

否则：

```text
cache hit at call c
```

注意这里 `\Delta_{i,c-1}` 是：

```text
call c-1 -> call c
```

的 transition score。所以如果 predictor -> corrector distance 很大，它会触发当前 corrector call 的 refresh。这一点解释了为什么 E3 schedule 大量 refresh 落在 corrector call 上。

refresh ratio 定义为：

```math
RR
=
\frac{
  \sum_i \sum_c R_{i,c}
}{
  N \cdot 99
}
```

其中：

```math
R_{i,c} \in \{0,1\}
```

表示 sample `i` 的 call `c` 是否真的 full denoiser refresh。

每种 metric 的 threshold `\delta` 只在 calibration set 上搜索，使 calibration RR 接近目标：

```text
RR = 0.30 / 0.40 / 0.50
```

然后固定 threshold，在 test set 和 all samples 上检查 achieved RR。

---

## 7. 输出文件

E3 主结果目录：

```text
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/
```

关键文件：

```text
schedule_summary.csv
threshold_vs_rr.csv
average_metric_curves.csv
stage_refresh_density.csv
stage_kind_refresh_density.csv
matched_schedules/*.npz
refresh_heatmaps/*.csv
plots/*.svg
metadata.json
summary.json
```

其中 `matched_schedules/*.npz` 是 E4 最重要的输入。每个文件包含：

```text
schedule: bool array, shape [256, 99]
method
target_rr
selected_delta
actual_rr
call_timesteps
call_steps
call_kinds
calibration_indices
test_indices
forced_mask
score
```

示例：

```text
matched_schedules/pma_stageaware_oracle_rr0p30.npz
```

其 schedule shape 为：

```text
[256, 99]
```

---

## 8. Threshold 与 RR 控制

计划中要求 actual RR 误差控制在：

```text
target RR ± 0.02
```

本次使用 64 calibration samples 后，所有方法都满足该要求。

### 8.1 Achieved RR 总表

| Method | Target RR | Calibration RR | Test RR | All RR | All error |
|---|---:|---:|---:|---:|---:|
| Uniform | 0.30 | 0.3030 | 0.3030 | 0.3030 | +0.0030 |
| Uniform | 0.40 | 0.4040 | 0.4040 | 0.4040 | +0.0040 |
| Uniform | 0.50 | 0.5051 | 0.5051 | 0.5051 | +0.0051 |
| RawInput-oracle | 0.30 | 0.3000 | 0.3024 | 0.3018 | +0.0018 |
| RawInput-oracle | 0.40 | 0.3999 | 0.4044 | 0.4033 | +0.0033 |
| RawInput-oracle | 0.50 | 0.5000 | 0.5092 | 0.5069 | +0.0069 |
| SEAInput-oracle | 0.30 | 0.3000 | 0.3010 | 0.3008 | +0.0008 |
| SEAInput-oracle | 0.40 | 0.3999 | 0.4015 | 0.4011 | +0.0011 |
| SEAInput-oracle | 0.50 | 0.5000 | 0.5060 | 0.5045 | +0.0045 |
| DINO-oracle | 0.30 | 0.3000 | 0.3091 | 0.3068 | +0.0068 |
| DINO-oracle | 0.40 | 0.3999 | 0.4129 | 0.4096 | +0.0096 |
| DINO-oracle | 0.50 | 0.5000 | 0.5180 | 0.5135 | +0.0135 |
| LPIPS-oracle | 0.30 | 0.3000 | 0.2872 | 0.2904 | -0.0096 |
| LPIPS-oracle | 0.40 | 0.3999 | 0.3849 | 0.3887 | -0.0113 |
| LPIPS-oracle | 0.50 | 0.5000 | 0.4904 | 0.4928 | -0.0072 |
| PMA-no-gate-oracle | 0.30 | 0.3000 | 0.2979 | 0.2984 | -0.0016 |
| PMA-no-gate-oracle | 0.40 | 0.3999 | 0.3986 | 0.3990 | -0.0010 |
| PMA-no-gate-oracle | 0.50 | 0.5000 | 0.5026 | 0.5019 | +0.0019 |
| PMA-stage-aware-oracle | 0.30 | 0.3000 | 0.3052 | 0.3039 | +0.0039 |
| PMA-stage-aware-oracle | 0.40 | 0.3999 | 0.4045 | 0.4033 | +0.0033 |
| PMA-stage-aware-oracle | 0.50 | 0.5000 | 0.5047 | 0.5036 | +0.0036 |

### 8.2 RR 控制的解释

几个现象值得注意：

1. `SEAInput-oracle` 的 threshold generalization 很稳，test-calibration gap 很小。
2. `DINO-oracle` 在 test set 上 achieved RR 偏高，但仍在 `+0.02` 容差内。
3. `LPIPS-oracle` 在 test set 上 achieved RR 偏低，也在容差内。
4. `PMA-no-gate` 和 `PMA-stage-aware` 都比单独 DINO / LPIPS 更稳，说明 metric 融合对 threshold calibration 有稳定化作用。
5. 之前用 32 calibration samples 时，DINO 和 PMA-stage-aware 的 test/all RR 偏差更明显；64 calibration 后明显改善。因此后续 E4 推荐使用 `calib64` 结果。

---

## 9. Average metric curve 与峰值分析

### 9.1 RawInput 的峰值集中在 late stage

RawInput normalized score 的 top peaks：

| Rank | Distance idx | Stage | Transition | Mean normalized score |
|---:|---:|---|---|---:|
| 1 | 96 | late | step 48 predictor -> step 48 corrector | 3.403 |
| 2 | 94 | late | step 47 predictor -> step 47 corrector | 3.368 |
| 3 | 92 | late | step 46 predictor -> step 46 corrector | 3.343 |
| 4 | 90 | late | step 45 predictor -> step 45 corrector | 3.314 |
| 5 | 88 | late | step 44 predictor -> step 44 corrector | 3.273 |

这和 E2 观察一致：Raw distance 后期越来越大。

但 E2 同时显示 DINO/LPIPS 在 late stage 很小。因此 Raw 后期变大很可能不是 clean-image perceptual drift，而是 input/proxy space 的数值变化。这是 RawInput 可能浪费 refresh budget 的核心风险。

### 9.2 SEAInput 有极强 early spike

SEA normalized score 的 top peaks：

| Rank | Distance idx | Stage | Transition | Mean normalized score |
|---:|---:|---|---|---:|
| 1 | 0 | early | step 0 predictor -> step 0 corrector | 249.148 |
| 2 | 2 | early | step 1 predictor -> step 1 corrector | 19.582 |
| 3 | 4 | early | step 2 predictor -> step 2 corrector | 18.207 |
| 4 | 6 | early | step 3 predictor -> step 3 corrector | 15.627 |
| 5 | 8 | early | step 4 predictor -> step 4 corrector | 13.137 |

即使做了 `log1p` 和 p99 clipping，idx0 仍然非常大。这说明：

- early stage 的 SEA filter 对 timestep / noise composition 非常敏感；
- warmup forced refresh 是必要的；
- 不能用原始 SEA absolute scale 直接和 DINO/LPIPS 加权。

### 9.3 DINO / LPIPS 既有 early spike，也有 middle bump

DINO top peaks：

| Distance idx | Stage | Transition | Mean normalized score |
|---:|---|---|---:|
| 0 | early | step 0 predictor -> corrector | 200.309 |
| 2 | early | step 1 predictor -> corrector | 162.764 |
| 4 | early | step 2 predictor -> corrector | 134.212 |
| 44 | middle | step 22 predictor -> corrector | 126.756 |

LPIPS top peaks：

| Distance idx | Stage | Transition | Mean normalized score |
|---:|---|---|---:|
| 0 | early | step 0 predictor -> corrector | 197.280 |
| 2 | early | step 1 predictor -> corrector | 171.764 |
| 4 | early | step 2 predictor -> corrector | 141.219 |
| 44 | middle | step 22 predictor -> corrector | 69.821 |

这里最重要的是 idx44：

```text
idx44: t = 0.2821 -> 0.2987
```

这对应 E2 中观察到的 perceptual bump。它可能是 clean-image prediction 从粗语义向稳定结构过渡的关键阶段。

### 9.4 PMA-no-gate 的问题

PMA-no-gate top peaks：

| Distance idx | Stage | Transition | Mean normalized score |
|---:|---|---|---:|
| 0 | early | step 0 predictor -> corrector | 218.936 |
| 2 | early | step 1 predictor -> corrector | 108.191 |
| 4 | early | step 2 predictor -> corrector | 89.912 |
| 6 | early | step 3 predictor -> corrector | 71.772 |
| 44 | middle | step 22 predictor -> corrector | 59.983 |

no-gate 的 early perceptual spike 仍然很强，因为它在 early stage 也信 DINO/LPIPS。

这正是 stage-aware gate 要解决的问题：

```text
早期不要让 DINO/LPIPS 主导 refresh。
```

### 9.5 PMA-stage-aware 的效果

PMA-stage-aware top peaks：

| Distance idx | Stage | Transition | Mean normalized score |
|---:|---|---|---:|
| 0 | early | step 0 predictor -> corrector | 249.148 |
| 44 | middle | step 22 predictor -> corrector | 64.641 |
| 2 | early | step 1 predictor -> corrector | 19.582 |
| 4 | early | step 2 predictor -> corrector | 18.207 |
| 6 | early | step 3 predictor -> corrector | 15.627 |

stage-aware 的关键变化是：

- early stage 退回 SEA，因此 early DINO/LPIPS spike 不再被额外放大；
- middle stage 引入 DINO，使 idx44 perceptual bump 被显著抬高；
- late stage 引入 DINO/LPIPS，但降低 SEA 权重，避免 Raw/SEA late numerical drift 过度主导。

这说明 PMA-stage-aware 确实改变了 score geometry，而不是简单换了一个 threshold。

---

## 10. Stage-level refresh 分布

下面统计 all samples 上的 refresh share：

```text
early / middle / late 三段分别消耗了多少 refresh budget
```

### 10.1 RR = 0.30

| Method | Early share | Middle share | Late share | Early density | Middle density | Late density |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 40.0% | 33.3% | 26.7% | 0.400 | 0.250 | 0.276 |
| RawInput-oracle | 36.8% | 32.6% | 30.6% | 0.367 | 0.243 | 0.315 |
| SEAInput-oracle | 53.7% | 23.6% | 22.7% | 0.533 | 0.175 | 0.233 |
| DINO-oracle | 55.5% | 27.3% | 17.2% | 0.562 | 0.207 | 0.180 |
| LPIPS-oracle | 60.1% | 26.6% | 13.3% | 0.576 | 0.191 | 0.132 |
| PMA-no-gate-oracle | 57.3% | 26.3% | 16.4% | 0.564 | 0.194 | 0.167 |
| PMA-stage-aware-oracle | 49.9% | 31.2% | 19.0% | 0.500 | 0.234 | 0.197 |

RR0.30 是最关键的低-budget setting。几个结论：

1. RawInput-oracle 的 late share 最高，达到 `30.6%`，符合 Raw late peaks 的观察。
2. SEAInput-oracle / DINO-oracle / LPIPS-oracle 都明显偏 early。
3. PMA-no-gate 仍然强烈偏 early，early share `57.3%`。
4. PMA-stage-aware 把 no-gate 的 early share 从 `57.3%` 降到 `49.9%`，同时把 middle share 从 `26.3%` 提到 `31.2%`。
5. 与 SEAInput-oracle 相比，PMA-stage-aware 把 middle share 从 `23.6%` 提到 `31.2%`，减少了 early 和 late 的占比。

这正符合 PMA-stage-aware 的设计目标：

```text
不要被 early perceptual spike 支配；
把一部分 refresh budget 移到 middle semantic/perceptual transition；
避免 late Raw/SEA numerical drift 过度消耗 budget。
```

### 10.2 RR = 0.40

| Method | Early share | Middle share | Late share |
|---|---:|---:|---:|
| Uniform | 35.0% | 37.5% | 27.5% |
| RawInput-oracle | 34.1% | 28.5% | 37.5% |
| SEAInput-oracle | 45.3% | 30.8% | 23.9% |
| DINO-oracle | 46.2% | 32.3% | 21.5% |
| LPIPS-oracle | 48.3% | 33.6% | 18.1% |
| PMA-no-gate-oracle | 46.5% | 32.4% | 21.1% |
| PMA-stage-aware-oracle | 45.1% | 33.4% | 21.5% |

RR0.40 下 RawInput-oracle 的 late share 进一步升高到 `37.5%`。PMA-stage-aware 则维持了更均衡的分布：

```text
early 45.1%, middle 33.4%, late 21.5%
```

它比 SEAInput-oracle 多给 middle，一定程度上接近 DINO/LPIPS 对 middle bump 的响应，但没有像 LPIPS 一样过度 early-biased。

### 10.3 RR = 0.50

| Method | Early share | Middle share | Late share |
|---|---:|---:|---:|
| Uniform | 34.0% | 38.0% | 28.0% |
| RawInput-oracle | 35.8% | 34.4% | 29.9% |
| SEAInput-oracle | 36.0% | 34.4% | 29.6% |
| DINO-oracle | 39.8% | 34.7% | 25.5% |
| LPIPS-oracle | 40.5% | 36.0% | 23.5% |
| PMA-no-gate-oracle | 38.4% | 35.4% | 26.2% |
| PMA-stage-aware-oracle | 36.1% | 37.1% | 26.8% |

RR0.50 时 budget 更充足，各方法的分布差异减小。但 PMA-stage-aware 仍然有最高的 middle share：

```text
PMA-stage-aware middle share = 37.1%
```

这说明 stage-aware 在更高 refresh budget 下仍然保留了“中期结构/perceptual transition 更重要”的倾向。

---

## 11. Call-kind 分布：为什么大量 refresh 落在 corrector

E2 已经发现相邻 distance 有强烈的 Heun predictor/corrector 交替结构：

```text
predictor -> corrector: large distance
corrector -> next predictor: tiny distance
```

在 E3 refresh rule 中，distance `call c-1 -> call c` 触发当前 `call c` refresh。因此：

```text
predictor -> corrector 的大 distance
主要触发 corrector call refresh
```

这不是 bug，而是 call-level schedule 的自然结果。

### 11.1 RR0.30 corrector refresh density

| Method | Early corrector density | Middle corrector density | Late corrector density |
|---|---:|---:|---:|
| RawInput-oracle | 0.533 | 0.484 | 0.559 |
| SEAInput-oracle | 0.867 | 0.349 | 0.406 |
| DINO-oracle | 0.894 | 0.412 | 0.297 |
| LPIPS-oracle | 0.939 | 0.380 | 0.198 |
| PMA-no-gate-oracle | 0.917 | 0.387 | 0.272 |
| PMA-stage-aware-oracle | 0.800 | 0.466 | 0.331 |

关键解释：

1. LPIPS / DINO / PMA-no-gate 在 early corrector 上非常密集，说明 early perceptual drift 仍然主导 schedule。
2. PMA-stage-aware 把 early corrector density 从 no-gate 的 `0.917` 降到 `0.800`。
3. PMA-stage-aware 把 middle corrector density 从 no-gate 的 `0.387` 提到 `0.466`。
4. PMA-stage-aware 的 late corrector density `0.331`，低于 SEA 的 `0.406`，说明它减少了 late SEA-style refresh。

这正是 stage-aware gate 想要的行为。

### 11.2 RawInput 的 late corrector 饱和

RR0.40 时：

```text
RawInput-oracle late corrector density = 0.997
```

RR0.50 时：

```text
RawInput-oracle late corrector density = 1.000
```

这说明 RawInput 在中高 refresh budget 下几乎刷新所有 late corrector calls。

结合 E2 中的 late DINO/LPIPS 很小，这很可能是 RawInput 的后期数值漂移造成的 over-refresh，而不是 perceptual drift 真的很大。

---

## 12. Selected thresholds

E3 为每个 metric 和 target RR 搜索得到的 selected threshold：

| Method | δ @ RR0.30 | δ @ RR0.40 | δ @ RR0.50 |
|---|---:|---:|---:|
| RawInput-oracle | 3.4849 | 2.3210 | 1.8358 |
| SEAInput-oracle | 5.5628 | 2.9037 | 2.1570 |
| DINO-oracle | 8.4058 | 3.5743 | 1.4513 |
| LPIPS-oracle | 7.9757 | 3.3087 | 1.5058 |
| PMA-no-gate-oracle | 7.9270 | 3.6792 | 1.9766 |
| PMA-stage-aware-oracle | 6.1571 | 3.3868 | 1.6981 |

随着目标 RR 从 0.30 增加到 0.50，threshold 单调下降。这符合 accumulated-distance rule：

```text
threshold 越低 -> accumulator 更容易超过阈值 -> refresh 更多 -> RR 更高
```

---

## 13. E3 回答了哪些问题

### Q1. SEAInput-oracle 是否稳定优于 RawInput-oracle？

E3 还不能回答 final image fidelity 的“优于”，但可以回答 schedule geometry：

- RawInput-oracle 明显偏 late；
- SEAInput-oracle 明显避免了 Raw 的 late over-refresh；
- SEAInput-oracle 的 RR calibration 比 Raw 和 DINO/LPIPS 都稳；
- SEAInput-oracle 在 RR0.30 下延续了 E1/E2 的直觉：SEA 更关注 early/mid 的主要结构变化，而不是 late raw numerical drift。

因此从 schedule-level 看，SEAInput-oracle 是比 RawInput-oracle 更合理的 baseline。

### Q2. DINO / LPIPS 是否主要在 middle / late stage 起作用？

结果比较微妙：

- DINO / LPIPS 的最大 peaks 仍然在 early stage；
- 但二者都在 idx44 出现明显 middle bump；
- LPIPS late share 很低，说明 LPIPS 并不是简单 late-only texture metric，它在当前 xhat trajectory 中也被 early clean-image instability 强烈影响。

因此更准确的结论是：

```text
DINO / LPIPS 包含有价值的 middle perceptual signal，
但直接单独使用会被 early spike 主导。
```

这正支持 stage-aware gate，而不是支持纯 DINO/LPIPS oracle。

### Q3. PMA-stage-aware 是否产生了更合理的 schedule？

是的，E3 schedule pattern 支持这个判断。

相对 PMA-no-gate，在 RR0.30 下：

```text
Early share  : 57.3% -> 49.9%
Middle share : 26.3% -> 31.2%
Late share   : 16.4% -> 19.0%
```

相对 SEAInput-oracle，在 RR0.30 下：

```text
Early share  : 53.7% -> 49.9%
Middle share : 23.6% -> 31.2%
Late share   : 22.7% -> 19.0%
```

这说明 PMA-stage-aware 没有只是“换 threshold”，而是确实把 budget 分配到了不同位置，尤其是 middle perceptual transition。

### Q4. PMA 是否只是 early spike artifact？

PMA-no-gate 很可能受到 early spike artifact 影响。

证据：

```text
PMA-no-gate RR0.30 early share = 57.3%
PMA-no-gate RR0.30 early corrector density = 0.917
```

PMA-stage-aware 缓解了这个问题：

```text
PMA-stage-aware RR0.30 early share = 49.9%
PMA-stage-aware RR0.30 early corrector density = 0.800
```

同时它保留了 idx44 middle bump：

```text
PMA-stage-aware idx44 mean normalized score = 64.641
```

因此：

```text
no-gate 有 early artifact 风险；
stage-aware 不是完全消除 early spike，但显著降低其主导性。
```

### Q5. refresh heatmap 是否有样本级异常？

E3 已输出 full heatmap：

```text
refresh_heatmaps/*.csv
plots/*.svg
```

从 aggregate statistics 看，没有出现 RR 失控或某个方法完全异常的情况。需要注意的是，DINO / LPIPS 的 threshold generalization 有轻微偏移：

```text
DINO all RR: 目标上方
LPIPS all RR: 目标下方
```

但二者都在 `±0.02` 容差内。

---

## 14. 阶段性结论

E3 的核心结论如下。

### 14.1 Call-level 结构非常重要

E3 再次确认不能把 98 个 distance 当成普通 step-level 序列。Heun predictor/corrector 结构极强：

```text
large transition: predictor -> corrector
small transition: corrector -> next predictor
```

因此合理 schedule 大量 refresh corrector call 是正常现象。

后续 E4 必须继续使用 call-level schedule，而不是把 schedule 压缩成 50 个 step。

### 14.2 RawInput 有明显 late over-refresh 风险

RawInput 的最大 peaks 全部在 late stage，且 RR0.40 / RR0.50 几乎刷新所有 late corrector calls。

这和 E2 中 DINO/LPIPS late 很小的观察冲突，说明 RawInput 可能把 late numerical proxy drift 当成重要 perceptual drift。

### 14.3 SEAInput 是强 baseline

SEAInput-oracle 的 RR 控制稳定，schedule pattern 比 RawInput 更合理。它是 E4 中必须比较的最重要 baseline。

### 14.4 单独 DINO / LPIPS 不适合作为 naive online-style schedule

DINO / LPIPS 有价值的 middle bump，但也有强 early spike。直接作为 accumulated-distance schedule，会强烈偏 early。

因此 DINO / LPIPS 更适合作为 stage-aware PMA 的组成部分，而不是独立替代 SEA。

### 14.5 PMA-stage-aware 的 pattern 符合设计预期

PMA-stage-aware 相对 PMA-no-gate：

- 减少 early over-refresh；
- 增加 middle refresh；
- 降低 late SEA/Raw-style numerical refresh；
- 保留 idx44 perceptual bump；
- RR calibration 稳定。

这说明 stage-aware gate 在 schedule-level 是有作用的。

---

## 15. 局限性

### 15.1 E3 不是最终证据

E3 只是 schedule-level evidence。真正主结论必须来自 E4：

```text
same refresh budget 下，真实 cached rerun 后 final image 是否更接近 full reference。
```

### 15.2 Stage split 仍是手工规则

当前 stage split 是：

```text
call_step_fraction: 0-30%, 30-70%, 70-100%
```

它没有直接使用 SNR、sigma、或者 learned gate。第一阶段这样做足够清晰，但后续可考虑：

```text
SNR-aware gate
timestep continuous weights
learned surrogate gate
```

### 15.3 PMA 权重是手工设计

当前权重：

```text
early  = SEA
middle = 0.5 SEA + 0.5 DINO
late   = 0.25 SEA + 0.35 DINO + 0.40 LPIPS
```

没有在 validation set 上系统搜索。这样避免过拟合，但也意味着它不是最优 oracle。

### 15.4 Calibration split 有潜在偏置

当前主结果使用 first 64 samples 作为 calibration。它比 32 calibration 更稳定，但仍可能受 class/seed ordering 影响。

后续如果要更严谨，可以补：

```text
shuffle calibration
multiple calibration splits
k-fold threshold stability
```

---

## 16. 对 E4 的建议

E4 推荐直接使用：

```text
outputs/e3_schedule_oracle/e3_main_256_from_e2_fp32_calib64/matched_schedules/
```

优先 rerun：

```text
Uniform
RawInput-oracle
SEAInput-oracle
PMA-no-gate-oracle
PMA-stage-aware-oracle
```

RR 优先级：

```text
先做 RR0.30 和 RR0.50
如果趋势清晰，再补 RR0.40
```

最关键的 E4 比较：

```text
SEAInput-oracle vs PMA-stage-aware-oracle
PMA-no-gate-oracle vs PMA-stage-aware-oracle
RawInput-oracle vs SEAInput-oracle
Uniform vs oracle schedules
```

E4 需要报告：

```text
actual RR
refreshes/sample
PSNR vs full reference
SSIM vs full reference
LPIPS vs full reference
paired delta vs SEAInput-oracle
```

如果 E4 中 PMA-stage-aware 在 RR0.30 和 RR0.50 都优于 SEAInput-oracle，则第一阶段主假设获得强支持：

```text
clean-image perceptual drift 可以提供比 input/proxy drift 更好的 cache scheduling geometry。
```

如果 E4 中 PMA-stage-aware schedule-level 看起来合理但 rerun 不赢，说明问题可能不在 metric，而在：

```text
cache-induced trajectory drift accumulation
fixed schedule 对偏移后的 trajectory 不稳定
需要 online surrogate 或 cache correction
```

---

## 17. 一句话总结

E3 表明：在 PixelGen Heun exact call-level trajectory 上，RawInput 会明显偏向 late numerical drift，DINO/LPIPS 单独使用会被 early perceptual spike 主导；而 PMA-stage-aware 通过 early 关闭 perceptual branch、中期引入 DINO、后期引入 DINO/LPIPS，确实产生了更符合 PMA 假设的 matched-RR oracle schedule。这个结果足以支持进入 E4，用真实 cached rerun 检验最终图像 fidelity。
