# 03｜E6-D0 离线诊断：用 E5.5 检验 xWPCache 的数学 risk

这是新版方案中最重要的实验。

> **先不要做 online cache。**  
> 先做 E6-D0：在 full trajectory 上计算各种 risk score，看看它们能不能预测 E5.5 的连续 skip PIS。

如果 E6-D0 不通过，online E6 很可能只是碰运气。

---

## 1. 实验目标

E6-D0 回答 5 个问题：

```text
Q1: xWPCache risk 能否预测 E5.5 window PIS？
Q2: anchor-relative risk 是否强于 adjacent risk？
Q3: vector accumulated solver error 是否强于 scalar accumulated distance？
Q4: Wiener clean proxy 是否强于 raw x_t / SEA-like proxy？
Q5: ODE factor / perceptual weight / uncertainty 是否真的有贡献？
```

---

## 2. 输入文件

### 2.1 E5.5 window labels

路径建议：

```text
outputs/e6_d0_labels/e55_window_labels.csv
```

字段：

```text
window_id
sample_id
class_id
seed
start_call
end_call
window_len
pis_lpips
pis_dino
pis_psnr
pis_ssim
pis_total
```

`pis_total` 建议先做两种版本：

```text
pis_total_z = zscore(pis_lpips) + zscore(pis_dino)
pis_total_rank = ranknorm(pis_lpips) + ranknorm(pis_dino)
```

为什么需要两个版本？

因为 LPIPS 和 DINO 的数值尺度不同。`zscore` 适合做线性分析；`ranknorm` 对极端值更稳。

---

### 2.2 full trajectory dump

路径建议：

```text
outputs/e6_d0_fulltraj/main8/
  sample_000.pt
  sample_001.pt
  ...
```

每个 `.pt` 文件内容：

```python
{
    "sample_id": int,
    "class_id": int,
    "seed": int,
    "final_image": Tensor,        # [3,H,W]
    "calls": [
        {
            "call_index": int,
            "step_index": int,
            "call_kind": "predictor" or "corrector",
            "t": float,
            "t_next": float,
            "h": float,            # abs(t_next - t) or effective solver coefficient
            "x_t": Tensor,          # current sampler state before denoiser
            "xhat": Tensor,         # full denoiser clean prediction
        },
        ...
    ]
}
```

注意：

```text
x_t 和 xhat 可以保存 FP16 以节省空间，计算 diagnostic 时转 FP32。
如果磁盘紧张，可以先只保存 downsample 到 128 或 64 的版本。
```

---

## 3. 输出文件

主输出：

```text
outputs/e6_d0_xwp_diagnostic/
  window_risk_table.csv
  score_metrics.csv
  threshold_sweep.csv
  failure_false_negative.csv
  plots/
    scatter_pis_vs_risk_*.png
    auc_bar.png
    captured_pis_curve.png
    calibration_curve.png
    risk_heatmap_*.png
```

---

## 4. 每个 window 要计算什么？

对于 E5.5 的每个 window：

```text
W = [s, e]
anchor = s - 1
```

如果 `s == 0` 或没有 anchor，这个 window 暂时跳过。

真实标签：

```text
Y(W) = pis_total
Y_lpips(W) = pis_lpips
Y_dino(W) = pis_dino
```

然后计算多个预测分数：

---

### 4.1 Time-only baseline

这个 baseline 很重要，用来检查你的方法是不是只是学到了 timestep。

```text
R_time = a function of start_call, end_call, window_len, mean_t
```

最简单的版本：

```text
R_time_1 = window_len
R_time_2 = window_len / mean(t)
R_time_3 = sum_i 1 / max(1 - t_i, epsilon)
```

也可以训练一个很小的 ridge regression：

```text
features = [start_call, end_call, window_len, mean_t, min_t, max_t]
label = PIS
```

但训练版 baseline 要用 leave-one-sample-out，不能在同一批上报告训练集结果。

---

### 4.2 Raw x_t baseline

```text
R_raw_adj = sum rel_l1(x_t_i, x_t_{i-1})
R_raw_anchor = sum rel_l1(x_t_i, x_t_anchor)
```

这是最朴素的 pixel/state 距离。

---

### 4.3 SEA-like baseline

使用 SeaCache 风格的 timestep-dependent filter，但作用在 `x_t` 或第一层 input feature 上。

```text
R_sea_adj = sum rel_l1(SEA_t(x_t_i), SEA_{i-1}(x_t_{i-1}))
R_sea_anchor = sum rel_l1(SEA_t(x_t_i), SEA_t(x_t_anchor))
```

如果你当前代码已经有 SEAInput，则也要加入：

```text
R_seainput_adj
R_seainput_anchor
```

这是强 baseline。

---

### 4.4 xWiener proxy baseline

先用 Wiener filter 从 `x_t` 得到：

```text
xbar_i = WienerProxy(x_t_i, t_i)
```

然后比较：

```text
R_xw_adj = sum D(xbar_i, xbar_{i-1})
R_xw_anchor = sum D(xbar_i, xhat_anchor)
```

注意 anchor 应该用 `xhat_anchor`，不是 `xbar_anchor`。因为真实 cache 复用的是 denoiser 输出。

---

### 4.5 xWiener + perceptual weight

```text
z_i = Phi_t(xbar_i)
z_anchor = Phi_t(xhat_anchor)
R_xwp_anchor = sum D(z_i, z_anchor)
```

这里 `Phi_t` 包含 DINO-like / LPIPS-like 频率权重。

---

### 4.6 xWiener + perceptual + ODE factor

```text
c_i = abs(h_i) / max(1 - t_i, epsilon_clip)
R_xwp_ode_scalar = sum c_i * D(z_i, z_anchor)
```

---

### 4.7 xWiener + perceptual + vector accumulated solver error

这是新版最推荐重点测试的分数。

```text
residual_i = c_i * (z_anchor - z_i)
E = sum residual_i over i in window
R_xwp_ode_vector = 2 * l1(E) / accumulated_norm
```

其中：

```text
accumulated_norm = sum_i c_i * (l1(z_anchor) + l1(z_i))
```

这个分数更接近连续 cache 对 sampler state 的累计影响。

---

### 4.8 Full-xhat oracle upper bound

用 full trajectory 里的真实 xhat：

```text
z_i_oracle = Phi_t(xhat_i)
z_anchor = Phi_t(xhat_anchor)
```

计算：

```text
R_oracle_scalar
R_oracle_vector
```

它不是 online 方法，但很重要。

如果 `Full-xhat oracle` 都不能预测 E5.5 PIS，说明 “clean prediction drift -> final damage” 这个假设可能不充分。

如果 `Full-xhat oracle` 很好，但 `Wiener proxy` 不好，说明理论目标对，proxy 不够好。

---

## 5. window_risk_table.csv 字段

建议输出：

```text
window_id
sample_id
start_call
end_call
window_len
pis_lpips
pis_dino
pis_total_z
pis_total_rank

R_time_len
R_time_ode
R_raw_adj
R_raw_anchor
R_sea_adj
R_sea_anchor
R_xw_adj
R_xw_anchor
R_xwp_anchor
R_xwp_ode_scalar
R_xwp_ode_vector
R_xwp_ode_uncertainty
R_oracle_xhat_scalar
R_oracle_xhat_vector

mean_t
min_t
max_t
num_predictor
num_corrector
```

---

## 6. 评价指标

### 6.1 Spearman correlation

看风险分数排序是否和真实损伤排序一致：

```text
spearman(R, PIS)
```

分别报告：

```text
all windows
within each window_len
within each sample
mean over samples
```

为什么要分 window_len？

因为 window_len 本身会影响损伤。一个 risk 如果只是学到 “越长越危险”，在 all windows 上可能看起来不错，但不一定真的懂内容变化。

---

### 6.2 ROC-AUC / PR-AUC

定义危险 window：

```text
dangerous = PIS_total_rank >= top 20%
```

也可以做：

```text
dangerous_lpips = LPIPS top 20%
dangerous_dino = DINO top 20%
```

指标：

```text
ROC-AUC
PR-AUC
```

如果 dangerous 很少，PR-AUC 比 ROC-AUC 更有参考价值。

---

### 6.3 False negative rate

这个最关键。

假设某个阈值下，risk 低于阈值表示“可以 skip”，risk 高于阈值表示“应该 refresh”。

```text
False Negative = 真实 dangerous，但 risk 认为安全。
```

你最怕的就是 false negative。

因为它意味着：

```text
算法会跳过一个真实危险的 window。
```

---

### 6.4 Captured PIS at budget

选择一个 refresh budget，例如：

```text
refresh top 30% highest-risk windows
```

计算：

```text
CapturedPIS@30 = 被 refresh 的 windows 中包含的 PIS 总量 / 所有 PIS 总量
```

越高越好。

等价地也可以看：

```text
SkippedPIS@30 = 被 skip 的 PIS 总量 / 所有 PIS 总量
```

越低越好。

---

### 6.5 Controlled correlation

为了证明不是 time-only，做一个控制分析：

```text
先用 time-only features 预测 PIS，得到 residual_pis。
再看 xWPCache risk 是否能预测 residual_pis。
```

如果 xWPCache 还能预测 residual，说明它提供了超越 timestep/window_len 的信息。

---

## 7. 通过标准

不要只看一个指标。建议至少满足：

```text
1. xWPCache-v2 risk 的 Spearman 高于 Raw / SEA / time-only。
2. xWPCache-v2 risk 的 PR-AUC 高于 Raw / SEA / time-only。
3. 在 matched refresh budget 下，xWPCache-v2 的 SkippedPIS 更低。
4. Full-xhat oracle 是上限，xWiener proxy 应该接近但不一定超过。
5. 控制 window_len / start_call 后，xWPCache 仍然有额外预测力。
```

如果第 5 条不满足，说明方法可能还是经验时间规则。

---

## 8. 推荐命令模板

```bash
python scripts/e6d0_dump_fulltraj.py \
  --config configs_c2i/pixelgen_xl_256.yaml \
  --num-samples 8 \
  --seeds 0 1 2 3 4 5 6 7 \
  --save-dir outputs/e6_d0_fulltraj/main8

python scripts/e6d0_build_labels.py \
  --e55-dir outputs/e5_5_multi_skip_pis/e5_5_main8_windows70_fp32 \
  --out outputs/e6_d0_labels/e55_window_labels.csv

python scripts/e6d0_compute_window_risks.py \
  --labels outputs/e6_d0_labels/e55_window_labels.csv \
  --fulltraj outputs/e6_d0_fulltraj/main8 \
  --out outputs/e6_d0_xwp_diagnostic/main8

python scripts/e6d0_evaluate_scores.py \
  --risk-table outputs/e6_d0_xwp_diagnostic/main8/window_risk_table.csv \
  --out outputs/e6_d0_xwp_diagnostic/main8/eval
```

---

## 9. 最小 smoke test

先别跑大。

最小测试：

```text
2 samples
10 windows
只算 4 个 score：
    R_time_len
    R_raw_anchor
    R_xwp_ode_scalar
    R_xwp_ode_vector
```

确认代码不出错后，再跑 main8 全部 windows。

---

## 10. 一句话总结

> **E6-D0 是新版方法的门槛实验：只有当 xWPCache 的数学风险分数能预测 E5.5 的连续 skip 因果损伤时，才值得进入真正 online cache。**
