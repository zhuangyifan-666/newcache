# E5：先把“离线 PMA”变成一个便宜的在线预测器

> 一句话目标：以前 E4 的 PMA 是“看过答案以后再排日程”的 oracle，现在 E5 要做一个“小预测器”，让它在真正推理时、还没跑 denoiser 前，就能猜出“这一步是不是危险、该不该刷新 cache”。

---

## 0. 先用人话理解 E5 在做什么

你现在已经有 E0-E4 了。它们证明了一件事：

- SEAInput 比 RawInput 更靠谱。
- DINO / LPIPS 这些 clean-image 感知指标确实有用。
- PMA-no-gate 在 RR≈0.50 时很强。
- 但是 PMA 目前是 oracle，也就是它提前看到了 full trajectory 上的 `xhat`，这在真实推理时不能用。

E5 的任务就是解决这个尴尬：

> 我们不能在真实推理时提前知道 DINO / LPIPS 的变化，但能不能用一些很便宜的信息，提前猜出 DINO / LPIPS / PMA 大概什么时候会变大？

这里的“很便宜的信息”主要是：

- SEAInput distance：已经能在线算。
- RawInput distance：也能在线算。
- 当前是第几步：比如第 10 个 call 还是第 80 个 call。
- 当前 call 是 predictor 还是 corrector。
- 当前处于 early / middle / late 哪个阶段。

这个小预测器可以先非常简单，不要一开始就上深度学习。建议先用线性回归，也就是：

```text
预测风险 = a * SEA + b * Raw + c * step位置 + d * 是否corrector + ...
```

如果这个小模型能把 PMA 的大致形状预测出来，就进入 E6，把它放进真实 cache loop。

---

## 1. E5 最重要的概念小抄

### 1.1 oracle 是什么？

oracle 可以理解为“考试时提前看到了答案”。

E4 的 PMA-oracle 是这样来的：先完整跑一遍 PixelGen，拿到每一步的 `xhat`，再用 DINO / LPIPS 算每两步之间变化有多大。这样当然很准，但真实加速时不能这样，因为你完整跑一遍就已经没有加速意义了。

### 1.2 online 是什么？

online 就是“边推理边决定”。

真实 cache 方法必须这样：

```text
当前要进入第 c 次 denoiser call
还没有真正跑 denoiser
先看便宜信号
判断：刷新 or 复用上一次结果
```

所以 online 方法不能偷看未来，也不能偷看 full trajectory。

### 1.3 proxy 是什么？

proxy 就是“替身指标”。

DINO / LPIPS 很贵，而且需要 `xhat`。但是 SEAInput 很便宜，而且可以在 denoiser 前算。所以我们希望 SEAInput、step 位置、call 类型这些便宜东西，能成为 PMA 的替身。

### 1.4 label 是什么？

label 就是小模型要学习的“答案”。

E5 中，label 可以用 E2/E3 已经算好的 oracle PMA 分数：

```text
PMA-no-gate = 0.4 * SEA + 0.3 * DINO + 0.3 * LPIPS
```

注意：DINO / LPIPS 只在训练小预测器时用来当答案。到了 E6 真正在线推理时，不能再用 DINO / LPIPS。

---

## 2. E5 要回答的科学问题

E5 只回答一个问题：

> 只看便宜的 online 特征，能不能预测 oracle PMA 哪些 transition 比较危险？

这里的“危险”意思是：如果这一步不刷新，而是复用旧 cache，最后图像可能更容易偏离 full reference。

你不用在 E5 里生成新图片。E5 主要是在已有 E2 的 `distance_bank.npz` 上做表格实验。

---

## 3. E5 的输入和输出

### 3.1 输入文件

优先使用你已有的 E2 文件：

```text
outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz
```

它里面应该包含类似这些东西：

```text
raw distance
sea distance
dino distance
lpips distance
sample_id
transition_id
```

如果字段名不同，就按你实际代码里的名字改。

### 3.2 E5 输出文件

建议输出到：

```text
outputs/e5_proxy_fitting/e5_main_from_e2_fp32/
```

里面放：

```text
proxy_dataset.csv
proxy_model_weights.json
proxy_fit_summary.csv
proxy_fit_curves.png
topk_recall_summary.csv
```

每个文件的意思：

- `proxy_dataset.csv`：把每个 sample、每个 transition 都变成一行表格。
- `proxy_model_weights.json`：保存线性预测器的权重，E6 要读取它。
- `proxy_fit_summary.csv`：预测效果总结。
- `proxy_fit_curves.png`：画出 oracle PMA 和预测 PMA 随 call 变化的曲线。
- `topk_recall_summary.csv`：看模型有没有抓住最危险的那些步。

---

## 4. 具体怎么做

### 第一步：把 E2 distance bank 变成表格

一行代表一个 transition。

比如 PixelGen Heun exact 有 99 个 denoiser call，所以相邻变化有 98 个 transition。

对于 256 张图，就是：

```text
256 * 98 = 25088 行
```

每一行建议包含：

```text
sample_id
transition_id
call_fraction          # transition_id / 98，表示走到采样过程的百分之几
stage                  # early / middle / late
is_early
is_middle
is_late
call_kind              # predictor_to_corrector 或 corrector_to_predictor
is_pred_to_corr
is_corr_to_pred
raw
sea
log1p_sea
raw_norm
sea_norm
label_pma_no_gate
label_pma_candidate_a
label_perceptual_only
```

解释几个字段：

- `call_fraction`：第几步。第 0 步接近开始，第 98 步接近结束。
- `stage`：粗略阶段。可以沿用你 E3 的分法：前 30% 是 early，中间 40% 是 middle，最后 30% 是 late。
- `call_kind`：Heun 有 predictor 和 corrector。E2 已经说明 predictor→corrector 的变化特别大，所以这个字段很重要。
- `log1p_sea`：因为 SEA 的 early spike 很大，直接用原始 SEA 可能太夸张。`log1p(x)` 就是 `log(1+x)`，能把大数压小一点。

### 第二步：构造 label

最简单先做两个 label。

#### label 1：PMA-no-gate

```text
label_pma_no_gate = 0.4 * sea_norm + 0.3 * dino_norm + 0.3 * lpips_norm
```

这是 E4 高 RR 时最强的 oracle 分数。

#### label 2：PMA Candidate A

Candidate A 是 soft stage-aware 里面整体最强的一组。可以也做一个 label：

```text
early  = 0.75 * SEA + 0.25 * DINO + 0.00 * LPIPS
middle = 0.45 * SEA + 0.45 * DINO + 0.10 * LPIPS
late   = 0.25 * SEA + 0.35 * DINO + 0.40 * LPIPS
```

为什么做两个？

因为 E4 告诉你：

- RR≈0.50 时，PMA-no-gate 很强。
- RR≈0.40/0.50 时，soft stage-aware 比 hard gate 更好。
- RR≈0.30 时，SEAInput 仍然更稳。

所以 E5 可以先看看哪个 label 更容易被便宜特征预测。

### 第三步：切分训练集、验证集、测试集

注意：不要把同一个 sample 的 transition 随机分散到 train/test。这样会“泄漏”。

建议按 sample_id 切分：

```text
训练集：sample 0-63
验证集：sample 64-127
测试集：sample 128-255
```

如果你想沿用 E3 的 calibration / test 习惯，也可以：

```text
训练/校准：sample 0-63
测试：sample 64-255
```

但我更建议 E5 先用 train/val/test 三段，因为你能看到模型有没有过拟合。

### 第四步：训练最简单的线性预测器

先不要上神经网络。用 sklearn 的 Ridge 回归即可。

特征可以从简单到复杂分三组：

#### 版本 A：只用 SEA

```text
log1p_sea
sea_norm
```

这是最弱版本，用来当底线。

#### 版本 B：SEA + 时间信息

```text
log1p_sea
sea_norm
call_fraction
is_early
is_middle
is_late
```

这个版本开始知道自己在采样过程的哪个阶段。

#### 版本 C：SEA + 时间信息 + Heun call 类型

```text
log1p_sea
sea_norm
raw_norm
call_fraction
is_early
is_middle
is_late
is_pred_to_corr
is_corr_to_pred
```

这是 E5 主版本。因为 E2 已经说明 Heun predictor/corrector 结构非常强。

---

## 5. E5 要看哪些指标

### 5.1 Spearman 相关性

Spearman 可以理解为“排序像不像”。

我们关心的不一定是预测值完全等于 oracle PMA，而是：

> oracle 觉得危险的步骤，预测器是不是也觉得危险？

所以 Spearman 比普通 MSE 更有用。

目标建议：

```text
Spearman > 0.50：可以进入 E6
Spearman > 0.65：比较有希望
Spearman > 0.75：非常好
```

这只是经验线，不是硬标准。

### 5.2 Top-k Recall

这个指标很适合 cache。

比如目标 RR=0.30，意思是大约只刷新 30% 的 call。那你可以问：

> oracle PMA 认为最危险的 30% transition，预测器抓住了多少？

如果抓住很多，就说明它对 cache 有帮助。

建议看：

```text
Top-30% recall
Top-40% recall
Top-50% recall
```

目标建议：

```text
Top-30% recall > 60%：可以试 E6
Top-30% recall > 70%：比较好
Top-50% recall > 80%：比较好
```

### 5.3 按 stage 分开看

一定要分 early / middle / late 看。

因为你的 E4 已经说明：

- RR0.30 时 early perceptual spike 容易浪费 refresh。
- RR0.50 时 early perceptual signal 又可能有用。

所以 E5 不能只看整体平均，要分别输出：

```text
early Spearman
middle Spearman
late Spearman
early Top-k recall
middle Top-k recall
late Top-k recall
```

---

## 6. 推荐脚本安排

建议新增两个脚本。

### 脚本 1：构造表格

```text
scripts/07_e5_build_proxy_dataset.py
```

输入：

```text
--distance-bank outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz
```

输出：

```text
outputs/e5_proxy_fitting/e5_main_from_e2_fp32/proxy_dataset.csv
```

### 脚本 2：拟合预测器

```text
scripts/08_e5_fit_proxy.py
```

输入：

```text
--dataset outputs/e5_proxy_fitting/e5_main_from_e2_fp32/proxy_dataset.csv
--target label_pma_no_gate
--feature-set sea_time_kind
```

输出：

```text
proxy_model_weights.json
proxy_fit_summary.csv
topk_recall_summary.csv
```

---

## 7. 推荐命令模板

下面不是必须完全照抄，按你项目 argparse 改即可。

```bash
python scripts/07_e5_build_proxy_dataset.py \
  --distance-bank outputs/e2_distance_bank/e2_main_256_fp32/distance_bank.npz \
  --outdir outputs/e5_proxy_fitting/e5_main_from_e2_fp32
```

```bash
python scripts/08_e5_fit_proxy.py \
  --dataset outputs/e5_proxy_fitting/e5_main_from_e2_fp32/proxy_dataset.csv \
  --target label_pma_no_gate \
  --feature-set sea_time_kind \
  --outdir outputs/e5_proxy_fitting/e5_main_from_e2_fp32/no_gate_sea_time_kind
```

再跑 Candidate A：

```bash
python scripts/08_e5_fit_proxy.py \
  --dataset outputs/e5_proxy_fitting/e5_main_from_e2_fp32/proxy_dataset.csv \
  --target label_pma_candidate_a \
  --feature-set sea_time_kind \
  --outdir outputs/e5_proxy_fitting/e5_main_from_e2_fp32/candidate_a_sea_time_kind
```

---

## 8. 你应该得到什么结果表

`proxy_fit_summary.csv` 建议长这样：

```text
target, feature_set, split, spearman, pearson, mse, top30_recall, top40_recall, top50_recall
label_pma_no_gate, sea_only, test, ...
label_pma_no_gate, sea_time, test, ...
label_pma_no_gate, sea_time_kind, test, ...
label_pma_candidate_a, sea_only, test, ...
label_pma_candidate_a, sea_time, test, ...
label_pma_candidate_a, sea_time_kind, test, ...
```

你最希望看到的是：

```text
sea_time_kind > sea_time > sea_only
```

意思是：加入 timestep 和 call-kind 后，预测更准。

这会变成论文里很重要的一句话：

> 对 x-pred PixelGen 来说，感知风险不仅由 SEAInput 决定，还与采样阶段和 Heun call 结构有关。

---

## 9. E5 通过标准

E5 不需要完美。只要达到下面任意一种，就值得进入 E6：

### 情况 A：预测器明显比 SEA-only 好

例如：

```text
sea_only Spearman = 0.50
sea_time_kind Spearman = 0.65
```

说明“时间阶段 + call-kind”有用。

### 情况 B：整体相关性一般，但 top-k recall 好

例如：

```text
Spearman = 0.48
top30_recall = 0.72
```

这也可以，因为 cache 主要关心把最危险的步骤抓出来。

### 情况 C：Candidate A 比 no-gate 更容易预测

这也有价值。说明 soft stage-aware 更适合变成 online 方法。

---

## 10. 如果 E5 结果不好，怎么办？

### 问题 1：SEA early spike 太大，模型被第一步带偏

解决：

- 去掉 transition_id=0 再训练一个版本。
- 使用 `log1p_sea`，不要直接用原始 SEA。
- early / middle / late 分开训练三个小模型。

### 问题 2：整体 Spearman 很低

解决：

- 不要只看整体，分 stage 看。
- 加上 `is_pred_to_corr` 和 `is_corr_to_pred`。
- 把 label 改成二分类：危险=1，不危险=0。

二分类可以这样定义：

```text
oracle PMA 排名前 30% 的 transition = 1
其他 = 0
```

然后训练 logistic regression。

### 问题 3：模型在训练集好，测试集差

这叫过拟合。

解决：

- 降低特征数量。
- 使用 Ridge 正则。
- 按 sample_id 切分，确认没有数据泄漏。

---

## 11. E5 最小可运行版本

如果你今天就想开始，先做这个最小版本：

1. 从 `distance_bank.npz` 读出 raw、sea、dino、lpips。
2. 构造 `label_pma_no_gate`。
3. 特征只用：`log1p_sea`, `call_fraction`, `is_pred_to_corr`, `is_corr_to_pred`。
4. 用 sample 0-63 训练，64-255 测试。
5. 输出 Spearman 和 Top-30/40/50 recall。

这个版本一天内应该能完成，而且不需要重新生成图片。

---

## 12. E5 成功后，下一步是什么？

E5 成功后，你会得到一个文件：

```text
proxy_model_weights.json
```

E6 会把它读进去，真实在线推理时这样用：

```text
当前 call 的 SEA/Raw/step/call_kind
-> 小预测器输出 risk
-> risk 累加超过阈值
-> refresh denoiser
否则 skip，复用 cache
```

这就是从 oracle 走向真实方法的关键一步。

---

## 13. 本实验参考的论文思想

- PixelGen 的关键点：x-prediction 让模型直接预测 clean image，因此可以谈 clean-image perceptual drift；LPIPS 偏局部纹理，P-DINO/DINO 偏全局语义。
- SeaCache 的关键点：不要直接用 raw feature distance，而是在频谱对齐后的 SEA space 里衡量变化，并继续使用 accumulated distance refresh rule。

E5 就是把这两点合起来：用 SEA 这样的便宜 online signal，去预测 PixelGen clean-image 感知空间里的风险。
