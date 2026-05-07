# 02. E5–E8 下一阶段实验路线

## 总览

你接下来应该把实验分成四个阶段。

| 阶段 | 名称 | 目的 | 产出 |
|---|---|---|---|
| E5 | Perceptual proxy fitting | 用便宜的 online features 预测 oracle perceptual drift | 一个可在线运行的 score predictor |
| E6 | Online PMA cache | 真实在线推理，不再偷看 full trajectory | 主方法结果表 |
| E7 | Solver-aware / forecast-aware extension | 利用 Heun predictor-corrector 结构和输出预测减少误差 | 增强版方法与 ablation |
| E8 | Paper-scale evaluation | 从小样本 paired fidelity 扩展到论文级评测 | 主表、FID 表、消融表、图 |

核心优先级：先 E5/E6，再 E7，最后 E8。不要一开始就跑很大的 FID，否则很容易花光算力却发现方法还没定型。

---

# E5. Perceptual proxy fitting

## E5 的问题

E4 已经证明 PMA perceptual score 在 oracle setting 下有价值，但真实推理时不能用 full trajectory 上的 DINO / LPIPS 来决定 refresh。E5 要回答：

> 能不能用便宜的 online features 预测 DINO / LPIPS / PMA 的 oracle score？

这里的“便宜”指不跑完整 denoiser、不跑 DINO、不跑 LPIPS，或者只在 refresh 后做很少量可选统计。

## E5.1 构建训练/校准数据

你已经有 E2 distance bank：Raw、SEA、DINO、LPIPS。建议在 E2 基础上额外保存每个 call transition 的 cheap features。

每个 transition `c-1 -> c` 建议保存：

| 特征名 | 含义 | 为什么有用 |
|---|---|---|
| `raw_rel_l1` | RawInput 相对 L1 | 传统 baseline |
| `sea_rel_l1` | SEAInput 相对 L1 | 已被证明强于 Raw |
| `log1p_sea` | `log(1+sea)` | SEA early spike 很大，log 更稳 |
| `t_prev`, `t_cur` | 当前 call 对应 timestep | diffusion 轨迹不同阶段重要性不同 |
| `dt` | Heun step 的步长 | 积分误差与步长相关 |
| `call_index_frac` | `call_index / 98` | stage 信息 |
| `is_predictor`, `is_corrector` | call kind | E2 显示 predictor→corrector mass 极强 |
| `transition_kind` | pc 或 cp | 不同 transition 的距离分布不同 |
| `cache_age_sim` | 假设连续 skip 的 age | 后续 online gate 要用 |
| `proxy_norm` | SEA proxy 的均值/范数 | 估计内容复杂度 |
| `proxy_delta_sign_stats` | token delta 的均值、方差、p95 | 单个均值可能丢失 tail risk |

标签可以设三种：

| 标签 | 定义 | 用途 |
|---|---|---|
| `y_dino` | normalized DINO drift | 语义变化预测 |
| `y_lpips` | normalized LPIPS drift | 纹理/局部变化预测 |
| `y_pma` | 你当前 PMA-no-gate 或 soft-stage score | 直接预测融合 score |

建议先预测 `y_pma`，因为最终 gate 用融合 score。然后再做 `y_dino` 和 `y_lpips` 的辅助分析。

## E5.2 三个由易到难的 proxy model

### Model 0：无学习的 heuristic proxy

这是最稳的 baseline：

```text
score = w_stage_kind_sea * norm(log1p(SEA))
      + b_stage_kind
      + age_penalty
```

其中 `w_stage_kind_sea` 和 `b_stage_kind` 用 calibration set 的相关性或网格搜索确定。它是 SEAInput 的增强版，优点是几乎不引入“训练”争议。

### Model 1：线性 / Ridge proxy

建议作为主方法候选之一：

```text
y_pma = Ridge([log1p_sea, raw, t, dt, call_kind, stage, proxy_norm, p95_delta])
```

Ridge regression 是带 L2 正则的线性回归。中文解释：它就是“给每个特征学一个权重”，同时避免权重太大导致过拟合。它计算非常便宜，64 个 calibration samples 就可以拟合。

### Model 2：分位数 / 不确定性 proxy

主方法更强的版本：预测 `mean + k * uncertainty`。

```text
score = mu(features) + k * sigma(features)
```

如果模型对某个 transition 不确定，就保守一点，更容易 refresh。这里 sigma 可以来自：

- bootstrap 多个 Ridge model；
- quantile regression 的 0.75 / 0.90 分位数；
- calibration residual 的 per-stage-kind 标准差。

这一步很适合写成“tail-failure protection”。cache 方法最怕少数样本崩坏，所以 uncertainty gate 是合理贡献。

## E5.3 数据划分

当前 E2 是 256 samples，E3 用 64 calibration + 192 test。建议 E5 先保持这个划分，方便和 E4 对齐。

然后做一个更严格划分：

| split | samples | 用途 |
|---|---:|---|
| calibration | 64 | fit proxy / threshold |
| validation | 64 | 选模型、选 k、选 max_skip |
| test | 128 | 只报最终结果 |

如果算力允许，扩展到 512 或 1024 samples：

| split | samples |
|---|---:|
| calibration | 128 |
| validation | 128 |
| test | 256 / 768 |

## E5.4 评价指标

E5 不需要跑真实采样，先评价 proxy 质量。

| 指标 | 解释 |
|---|---|
| Spearman correlation | 排名相关性；cache schedule 更关心“哪些 transition 更重要” |
| Pearson correlation | 数值相关性 |
| Top-k recall | oracle top 20% high-risk transitions 中，proxy 能找回多少 |
| stage-kind calibration error | 不同 stage / call kind 下预测是否偏高或偏低 |
| simulated RR error | 用 proxy score 做 schedule 后，actual RR 是否接近 target |
| oracle-schedule overlap | proxy schedule 与 PMA-oracle schedule 的 Jaccard overlap |

建议特别重视 Top-k recall，因为 cache 失败往往来自少数重要 transition 被误 skip。

## E5.5 成功标准

E5 通过的最低标准：

- proxy `y_pma` 的 Spearman correlation 明显高于 RawInput；
- proxy schedule 与 PMA-oracle schedule 的 overlap 高于 SEAInput schedule；
- 在 target RR 0.30 / 0.40 / 0.50 下，actual RR 偏差小于 ±2%；
- top-risk transition recall 在 RR 0.30 下优于 SEAInput。

如果 E5 不通过，不要急着跑 E6。先改 proxy features 或 normalization。

---

# E6. Online PMA cache

## E6 的问题

E6 要回答：

> 用 E5 得到的 proxy score，在真实 cached trajectory 上在线决策，能不能超过 SEAInput-online？

这是论文能否成立的关键。

## E6.1 主方法命名

建议临时命名：

- `PMA-ProxyCache`
- `PM-Cache`
- `PUMA-Cache`: Perceptual Manifold Aware Cache
- `xP-Cache`: x-prediction Perceptual Cache

文中可以先叫 `PMA-Cache`，最终投稿前再决定名字。

## E6.2 Online 决策规则

每个 call 执行：

```text
1. 计算 cheap proxy feature z_c
2. score_c = proxy_predictor(z_c)
3. accumulator += score_c
4. 如果满足以下任一条件，则 refresh：
   - call_index < warmup_calls
   - call_index == final_call
   - accumulator >= delta
   - consecutive_hits >= max_skip_calls
   - uncertainty >= uncertainty_threshold
5. 否则复用 cached denoiser output
```

注意：E6 在线时不能用当前 full denoiser 输出计算 DINO / LPIPS 作为 gate。可以在分析模式下记录它们，但主方法 speed 表必须关闭 DINO / LPIPS。

## E6.3 实验矩阵

先跑小矩阵：

| 方法 | RR target |
|---|---|
| Uniform | 0.30 / 0.40 / 0.50 |
| RawInput-online | 0.30 / 0.40 / 0.50 |
| SEAInput-online | 0.30 / 0.40 / 0.50 |
| PMA-ProxyCache heuristic | 0.30 / 0.40 / 0.50 |
| PMA-ProxyCache ridge | 0.30 / 0.40 / 0.50 |
| PMA-ProxyCache uncertainty | 0.30 / 0.40 / 0.50 |

然后选最强版本扩展：

| 方法 | RR target |
|---|---|
| SEAInput-online | 0.20 / 0.30 / 0.40 / 0.50 / 0.60 |
| Best PMA-ProxyCache | 0.20 / 0.30 / 0.40 / 0.50 / 0.60 |
| TeaCache-style | 0.30 / 0.50 |
| MagCache-style | 0.30 / 0.50 |

## E6.4 需要保存的输出

每个 method / RR 保存：

```text
method_summary.csv
per_sample_metrics.csv
per_call_decisions.npz
per_call_score_curves.csv
refresh_pattern_heatmap.png
worst_16_samples.png
reference_vs_cache_grid.png
latency_summary.json
```

`per_call_decisions.npz` 非常重要，因为论文图要展示你的方法在哪些 call refresh。

## E6.5 成功标准

最低成功标准：

- 在 RR≈0.30，PMA-ProxyCache 至少不低于 SEAInput-online；
- 在 RR≈0.40 / 0.50，PMA-ProxyCache 明显高于 SEAInput-online 或有更低 LPIPS；
- tail failure 少于 SEAInput-online，例如 per-sample LPIPS 的 p95 / p99 更低；
- actual wall-clock speedup 与 SEAInput-online 接近，不被 proxy 开销抵消。

更强成功标准：

- PMA-ProxyCache 在 0.30–0.60 整条曲线上 Pareto-dominates SEAInput-online；
- 在 1k samples 上 bootstrap 95% CI 显著；
- 在 FID quick eval 上没有明显质量退化。

---

# E7. Solver-aware / forecast-aware extension

E7 是增强阶段。如果 E6 已经能打，E7 用来提升 novelty 和曲线。如果 E6 很一般，E7 可能是救命点。

## E7.1 Predictor-corrector-aware cache

E2 已经发现 distance mass 几乎都集中在 predictor→corrector。下一步要把这个发现转成方法。

建议测试四种策略：

| 策略 | 说明 | 目的 |
|---|---|---|
| call-level accumulator | 当前做法 | baseline |
| step-pair accumulator | 一个 Heun step 的 predictor+corrector 视作一个 unit | 避免 predictor/corrector 不一致 |
| two-threshold accumulator | pc 和 cp 两类 transition 用不同 delta | 适配不同误差分布 |
| paired refresh | predictor refresh 时强制 corrector refresh，或反过来 | 保持 Heun correction 一致性 |

推荐主打 `two-threshold accumulator`，因为它最容易接到现有代码。

形式：

```text
if transition_kind == predictor_to_corrector:
    accumulator_pc += score / delta_pc
else:
    accumulator_cp += score / delta_cp
refresh if accumulator_pc + accumulator_cp >= 1
```

或者更简单：

```text
score = score / median_score[stage, transition_kind]
```

也就是 per-stage-kind normalization。

## E7.2 Forecast instead of reuse

现在 cache 的 skip 是直接复用 cached denoiser output。TaylorSeer 的启发是：有时候不应该只 reuse，而应该 forecast。

对 x-pred PixelGen，可以预测 clean image estimate 或 velocity：

```text
xhat_forecast_c = xhat_last + lambda(t) * (xhat_last - xhat_prev_refresh)
v_forecast_c    = v_last    + lambda(t) * (v_last    - v_prev_refresh)
```

先做简单线性预测即可：

| 方法 | 公式 |
|---|---|
| Reuse | `out_c = out_last` |
| Linear forecast | `out_c = out_last + alpha * (out_last - out_prev)` |
| Time-scaled forecast | `alpha = (t_c - t_last) / (t_last - t_prev)` |
| Damped forecast | `alpha = clamp(alpha, 0, alpha_max) * decay(age)` |

建议先预测 velocity，因为 sampler 真正使用的是 v；但 x-pred 模型输出 clean image，预测 xhat 后再转 v 也值得试。

关键 ablation：

| Gate | Skip action |
|---|---|
| SEAInput | reuse |
| SEAInput | forecast |
| PMA-Proxy | reuse |
| PMA-Proxy | forecast |

如果 forecast 能在 RR≤0.30 降低误差，你的论文贡献会更强：不仅判断何时 cache，还改进 skip 时如何使用 cache。

## E7.3 Magnitude-aware branch

MagCache 的启发是 residual magnitude ratio 在很多 diffusion trajectory 中有稳定规律。PixelGen 的 x-pred 可以定义：

```text
residual_norm = ||xhat_t - x_t||
velocity_norm = ||(xhat_t - x_t)/(1-t)||
ratio = residual_norm_t / residual_norm_prev
```

这只能在 refresh 时获得精确值，但可以作为 online state。你可以把它加入 PMA proxy features：

```text
score = proxy(features) + w_mag * mag_risk
```

先做分析：画出 PixelGen 上 residual magnitude ratio 是否单调、是否跨 class 稳定。如果稳定，就作为贡献的一部分；如果不稳定，就作为负结果写进 appendix，说明 x-pred pixel diffusion 不完全符合 video latent diffusion 的 magnitude law。

## E7.4 Token / spatial risk map：可选，不作为主线

ToCa 和 FreqCa 都说明不同 token / frequency band 的 cache 敏感性不同。但你当前已经有一条强主线，不建议过早转向复杂 token-wise cache。可以只做一个轻量分析：

- 把 SEA proxy delta reshape 成 spatial map；
- 看高 delta 区域是否对应 final error 热区；
- 如果明显，可以作为 future work 或 appendix。

---

# E8. Paper-scale evaluation

E8 是把方法做成论文证据链。

## E8.1 Paired full-reference fidelity

推荐最小规模：

| 设置 | 样本数 |
|---|---:|
| pilot | 192 |
| main paired | 1024 |
| stress paired | 2048，如果算力允许 |

指标：

- PSNR ↑
- SSIM ↑
- LPIPS ↓
- DINO distance ↓
- per-sample LPIPS p95 / p99 ↓
- actual RR
- wall-clock latency
- denoiser calls

## E8.2 Unpaired generation quality

paired fidelity 只能证明“接近 full model”，不能证明“生成质量没掉”。所以要补 ImageNet quality。

| 规模 | 用途 |
|---|---|
| 1k | debug，不写主表 |
| 5k | quick FID / 快速筛选 |
| 10k | 主文可用，如果算力紧 |
| 50k | 最标准，最好有 |

指标：

- FID ↓
- IS ↑
- Precision ↑
- Recall ↑

注意：如果你用 5k 或 10k，需要在论文中明确写清楚，与其他 paper 的 50k 不直接横比。你主要和自己的 full reference / cache baselines 对比。

## E8.3 Baseline 主表

主表建议这样设计：

| Method | RR | Speedup | PSNR ↑ | LPIPS ↓ | p95 LPIPS ↓ | FID ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Full | 1.00 | 1.00x | ∞ | 0 | 0 | x.xx |
| Uniform | 0.30 | ... | ... | ... | ... | ... |
| RawInput | 0.30 | ... | ... | ... | ... | ... |
| SEAInput | 0.30 | ... | ... | ... | ... | ... |
| TeaCache-style | 0.30 | ... | ... | ... | ... | ... |
| MagCache-style | 0.30 | ... | ... | ... | ... | ... |
| PMA-ProxyCache | 0.30 | ... | ... | ... | ... | ... |

同样表再做 RR 0.50，或者用一张 curve 展示 0.20–0.60。

## E8.4 论文图

建议至少准备 6 张核心图：

1. 方法图：x-pred clean-image trajectory + perceptual proxy gate。
2. E2 诊断图：Raw / SEA / DINO / LPIPS 的 per-call curves 与 correlation heatmap。
3. Refresh heatmap：SEAInput vs PMA-ProxyCache 的 refresh pattern。
4. Speed-quality curve：Latency / RR vs LPIPS / PSNR。
5. Worst-case samples：展示 tail failure 改善。
6. Ablation bar chart：去掉 DINO-proxy、LPIPS-proxy、call-kind normalization、uncertainty gate 后的性能。

## E8.5 成功线和投稿线

### 最小投稿线

- 在线方法超过 SEAInput-online，至少在 RR 0.40 / 0.50 显著；
- RR 0.30 不明显变差，最好 tail 更稳；
- 有 1k paired + 5k FID；
- 有 4 个以上强 baseline；
- 有明确 x-pred-specific 分析。

### 强投稿线

- 在线方法在 0.30–0.60 全曲线优于 SEAInput / TeaCache-style / MagCache-style；
- 10k 或 50k FID；
- 不同 checkpoint / resolution / maybe text-to-image 上泛化；
- 有 solver-aware / forecast-aware 的第二贡献。

