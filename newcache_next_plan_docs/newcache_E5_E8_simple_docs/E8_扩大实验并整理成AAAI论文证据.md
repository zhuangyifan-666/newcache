# E8：扩大实验，并整理成 AAAI 论文证据

> 一句话目标：E5-E7 是把方法做出来，E8 是把方法证明到“像一篇论文”。

---

## 0. E8 是什么

E8 不是再随便加一个小实验。

E8 的目标是：

```text
把 E6/E7 里最好的在线 cache 方法，放到更公平、更完整、更像论文的实验设置里验证。
```

简单说：

- E6：方法第一次跑通。
- E7：证明哪些零件有用。
- E8：证明它不是偶然有效，整理成论文图表。

---

## 1. E8 要回答的科学问题

E8 回答三个更大的问题。

### 问题 1：方法是否稳定？

不是只在 64 或 192 张图上偶然有效，而是在更多样本上也有效。

### 问题 2：方法是否公平超过 baseline？

必须在 matched RR 或 matched latency 下比较。

不能这样比较：

```text
你的方法 RR=0.50
baseline RR=0.30
```

这不公平。

### 问题 3：方法故事是否能支撑论文？

你的论文故事应该是：

```text
x-prediction 模型直接预测 clean image，
因此 cache refresh 不应该只看 noisy feature，
还应该通过便宜 proxy 估计 clean-image perceptual drift。
```

E8 要准备能支撑这个故事的表格和图。

---

## 2. E8 的主实验设置

### 2.1 样本数量建议

分三档：

#### 最低档：算力很紧

```text
192 samples
```

这能完成论文初稿，但说服力一般。

#### 推荐档：比较适合本科生冲论文

```text
512 或 1024 samples
```

这会比 192 稳很多，paired fidelity 也更可信。

#### 理想档：如果算力允许

```text
5000 samples 或更多
```

这时可以尝试报告 FID / IS 之类分布指标。

但注意：你的任务是 cache acceleration，主要指标仍然应该是“cached output 和 full output 的 paired fidelity + speedup”。FID 可以作为补充，不一定是主指标。

### 2.2 RR 设置

建议固定：

```text
RR = 0.30 / 0.40 / 0.50
```

如果时间够，再加：

```text
RR = 0.20 / 0.60
```

这样可以画更完整的曲线。

### 2.3 统一设置

继续沿用你前面最稳定的设置：

```text
Config: configs_c2i/PixelGen_XL_without_CFG.yaml
Checkpoint: ckpts/PixelGen_XL_80ep.ckpt
Sampler: Heun exact
Calls/sample: 99
Precision: fp32, no autocast
Guidance: 1.0
Timeshift: 2.0
```

统一设置很重要，否则别人会怀疑差异来自别的地方。

---

## 3. E8 必须比较的 baseline

最少比较这些：

```text
Full reference
Uniform
RawInput-online
SEAInput-online
Your final ProxyPMA-online
```

如果能做，再加这些：

### 3.1 Step reduction baseline

比如直接少跑步：

```text
Vanilla fewer calls / fewer steps
```

意义：证明 cache 不是简单少跑几步，而是在更聪明地复用。

### 3.2 TeaCache-style baseline

TeaCache 的核心思想是用 timestep-modulated input 的变化做 accumulated cache 判断。

你已经有 RawInput / SEAInput，其实可以把 RawInput 看成类似 input-distance baseline，把 SEAInput 看成更强 baseline。

如果来不及完整复现 TeaCache，就不要硬写“我们复现了 TeaCache”。可以诚实写：

```text
We include RawInput-style and SEAInput-style online cache baselines following the input-distance accumulated rule.
```

### 3.3 MagCache-style baseline，可选

MagCache 思想是看 residual / magnitude 变化。

如果实现成本高，可以先不做。AAAI 投稿更重要的是把你的主线讲清楚。

---

## 4. E8 主表应该长什么样

### 4.1 主结果表

建议主表这样做：

```text
Method              RR target  Actual RR  Speedup  PSNR↑  SSIM↑  LPIPS↓  Worst10 LPIPS↓
Uniform             0.30       ...        ...      ...    ...    ...     ...
RawInput-online     0.30       ...        ...      ...    ...    ...     ...
SEAInput-online     0.30       ...        ...      ...    ...    ...     ...
ProxyPMA-online     0.30       ...        ...      ...    ...    ...     ...
Uniform             0.40       ...        ...      ...    ...    ...     ...
RawInput-online     0.40       ...        ...      ...    ...    ...     ...
SEAInput-online     0.40       ...        ...      ...    ...    ...     ...
ProxyPMA-online     0.40       ...        ...      ...    ...    ...     ...
Uniform             0.50       ...        ...      ...    ...    ...     ...
RawInput-online     0.50       ...        ...      ...    ...    ...     ...
SEAInput-online     0.50       ...        ...      ...    ...    ...     ...
ProxyPMA-online     0.50       ...        ...      ...    ...    ...     ...
```

注意：一定要报 Actual RR，不要只报 Target RR。

### 4.2 paired delta 表

还要直接比较你的方法和 SEAInput-online：

```text
RR  Delta PSNR vs SEA  95% CI       Delta LPIPS vs SEA  95% CI
0.30  ...              [...]        ...                 [...]
0.40  ...              [...]        ...                 [...]
0.50  ...              [...]        ...                 [...]
```

95% CI 是置信区间。简单理解：

> 如果重复抽样很多次，这个提升大概率会落在哪个范围里。

如果 CI 不跨 0，说明结果更有说服力。

---

## 5. E8 必须准备的图

### 图 1：Latency-quality 曲线

横轴：

```text
Actual RR 或 latency
```

纵轴：

```text
LPIPS 越低越好
或者 PSNR 越高越好
```

你希望图上显示：

```text
ProxyPMA-online 在同样 RR 下，比 SEAInput-online 更好。
```

### 图 2：refresh heatmap

横轴：

```text
call index 0-98
```

纵轴：

```text
Uniform / SEAInput / ProxyPMA
```

颜色：

```text
这一 call 被 refresh 的比例
```

这张图能说明：你的方法不是随便刷新，而是学到了不同阶段的风险。

### 图 3：E5 proxy 对齐图

画两条曲线：

```text
oracle PMA 平均曲线
E5 proxy 预测平均曲线
```

如果二者形状相似，就能说明 E5 的预测器确实在模仿感知风险。

### 图 4：视觉对比图

选 8-16 个样本，展示：

```text
Full reference | SEAInput-online | ProxyPMA-online
```

最好选：

- SEA 明显出错但 ProxyPMA 改善的样本。
- ProxyPMA 失败的样本也可以放 appendix，显得诚实。

---

## 6. E8 稳健性实验

稳健性就是证明方法不是只在一个小条件下有效。

建议做三类。

### 6.1 不同 sample split

比如：

```text
Split A: sample 64-255
Split B: sample 256-511
Split C: sample 512-767
```

如果你没有这么多预先生成的 sample，就至少换一批 seed。

### 6.2 不同 RR

至少：

```text
0.30 / 0.40 / 0.50
```

最好能覆盖：

```text
0.25 / 0.30 / 0.40 / 0.50 / 0.60
```

这样曲线更像论文。

### 6.3 不同精度或设置，可选

如果 fp32 已经很慢，可以最后试：

```text
fp16 / bf16 / autocast
```

但注意：前期不要混入精度变化。先把 fp32 主结果做扎实。

---

## 7. E8 的论文故事线

你的故事可以这样组织。

### 7.1 传统 cache 的问题

传统 cache 常看 noisy input 或 hidden feature 是否变化。

问题是：这些变化不一定等于人眼关心的图像变化。

### 7.2 PixelGen / x-pred 的机会

PixelGen 这类 x-pred 模型每一步直接预测 clean image estimate。

这给了你一个新机会：

```text
refresh 判据可以围绕 clean-image perceptual drift 设计。
```

### 7.3 你的第一阶段发现

E0-E4 证明：

```text
SEA 比 Raw 更接近 perceptual drift。
PMA oracle 在高 RR 下有上限价值。
Hard stage gate 不够好，soft / no-gate 更有希望。
```

### 7.4 你的方法

E5-E6 提出：

```text
用便宜 online proxy 预测 oracle perceptual risk，
再用 accumulated rule 做在线 cache refresh。
```

### 7.5 你的最终证据

E7-E8 证明：

```text
它真实在线有效，
不是 oracle cheating，
不是简单 SEA rescaling，
在多个 RR / samples 上稳定。
```

---

## 8. E8 通过标准

### 绿色信号：可以冲论文主线

如果你得到：

```text
RR0.50：ProxyPMA-online 明显优于 SEAInput-online
RR0.40：ProxyPMA-online 至少持平或略优
RR0.30：ProxyPMA-online 不明显崩
消融证明 timestep / call_kind / stage 至少有一个关键组件有效
```

这就是比较完整的故事。

### 黄色信号：需要调整方法

如果：

```text
只在 RR0.50 有效，RR0.30/0.40 都差
```

也不是不能写，但要把 claim 限制为：

```text
perceptual proxy is most beneficial under moderate-to-high refresh budgets
```

中文：

> 感知 proxy 在中高刷新预算下更有用。

### 红色信号：需要回到 E5/E6

如果：

```text
三个 RR 全部弱于 SEAInput-online
消融也解释不出原因
```

那就先不要扩大 E8，回去改 E5 预测器。

---

## 9. E8 推荐脚本

可以新建：

```text
scripts/11_e8_final_eval_online_cache.py
scripts/12_e8_make_paper_figures.py
```

第一个负责跑最终结果。

第二个负责把 csv 变成图。

命令模板：

```bash
python scripts/11_e8_final_eval_online_cache.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --checkpoint ckpts/PixelGen_XL_80ep.ckpt \
  --proxy-model outputs/e5_proxy_fitting/e5_main_from_e2_fp32/best/proxy_model_weights.json \
  --num-samples 1024 \
  --target-rr 0.30 0.40 0.50 \
  --methods uniform raw sea proxy_pma \
  --precision fp32 \
  --no-autocast \
  --outdir outputs/e8_final/e8_1024_fp32
```

画图：

```bash
python scripts/12_e8_make_paper_figures.py \
  --result-dir outputs/e8_final/e8_1024_fp32 \
  --outdir outputs/e8_final/e8_1024_fp32/figures
```

---

## 10. E8 最小可发表版本

如果时间很紧，至少完成：

```text
主结果：192 或 512 samples
RR：0.30 / 0.40 / 0.50
方法：Uniform / SEAInput-online / ProxyPMA-online
消融：ProxyPMA-full / ProxyPMA-sea-only / ProxyPMA-w/o-callkind
图：quality curve + refresh heatmap + visual grid
```

这比盲目追求 50000 images FID 更现实。

你的论文重点是 inference acceleration，不是重新证明 PixelGen 的生成质量。

---

## 11. 最终你应该整理出哪些文件

E8 完成后，建议有：

```text
outputs/e8_final/e8_1024_fp32/method_summary.csv
outputs/e8_final/e8_1024_fp32/paired_delta_vs_sea.csv
outputs/e8_final/e8_1024_fp32/stage_refresh_density.csv
outputs/e8_final/e8_1024_fp32/stage_kind_refresh_density.csv
outputs/e8_final/e8_1024_fp32/figures/rr_quality_curve.png
outputs/e8_final/e8_1024_fp32/figures/refresh_heatmap.png
outputs/e8_final/e8_1024_fp32/figures/proxy_alignment_curve.png
outputs/e8_final/e8_1024_fp32/figures/visual_grid.png
```

这些基本就是论文实验部分的素材。

---

## 12. 最后提醒

E8 最重要的不是“结果全赢”。

最重要的是你要讲清楚：

```text
为什么 x-prediction 让 perceptual cache 成为可能；
为什么 oracle PMA 有价值；
为什么 E5 proxy 能把一部分 oracle 价值搬到 online；
为什么这个方法在真实 cache rerun 中有效。
```

只要这条证据链完整，你的工作就会比“我又调了一个 cache 阈值”更像研究。
