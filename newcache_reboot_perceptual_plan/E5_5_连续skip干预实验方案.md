# E5.5 连续 Skip 干预实验方案

> 目标读者：刚开始做扩散模型推理加速实验的本科生。  
> 当前背景：你已经完成了 E5 单点 skip 干预实验。E5 说明，在当前 PixelGen-XL / Heun 50 steps / 99 denoiser calls 设置下，**单次 skip 的最终感知损伤主要集中在最早期 call，尤其是 call 1–7；call 16 以后单次 skip 几乎无损**。  
> E5.5 的核心任务：验证“单次 skip 安全”是否意味着“连续 skip 也安全”。

---

## 0. 一句话总结

E5 测的是：

```text
只 skip 第 k 个 call 一次，最终图会不会坏？
```

E5.5 要测的是：

```text
从第 s 个 call 开始，连续 skip L 个 call，最终图会不会坏？
```

为什么要做这个？因为真实 cache 不是只跳一次。真实 cache 经常是：

```text
refresh 一次
skip 1 次
skip 2 次
skip 3 次
...
直到风险太大才 refresh
```

所以 E5.5 直接决定 E6 里最重要的参数：

```text
不同阶段最多允许连续 skip 几次？
也就是 max_age 应该设多少？
```

---

## 1. E5 已经告诉了我们什么？

你的 E5 主实验设置是：

```text
模型：PixelGen-XL without CFG
采样器：HeunSamplerJiT
采样步数：50
每张图 denoiser call 数：99
样本数：8
干预方式：每次只 skip 一个 call，其余 call 正常 full compute
有效干预数：784
指标：LPIPS / DINO / PSNR / SSIM
```

E5 的关键发现如下：

| call 范围 | 占总 LPIPS 损伤 | 占总 DINO 损伤 | 解释 |
|---:|---:|---:|---|
| call 1–3 | 75.7% | 74.5% | 最早期极度危险 |
| call 1–5 | 89.7% | 89.6% | 早期 corrector / predictor 都重要 |
| call 1–7 | 94.2% | 94.4% | 前 7 个有效 call 几乎贡献了全部单点 skip 损伤 |
| call 1–15 | 98.1% | 98.9% | 前 15 个 call 是主要危险区 |
| call 16–89 | 1.1% | 0.85% | 单次 skip 基本无损 |
| call 90–98 | 0.83% | 0.29% | 末尾有轻微回升，但远小于早期 |

最危险的平均 LPIPS call 是：

| 排名 | call | step | t | 类型 | mean LPIPS | mean DINO |
|---:|---:|---:|---:|---|---:|---:|
| 1 | 1 | 0 | 0.0101 | corrector | 0.04999 | 0.03257 |
| 2 | 3 | 1 | 0.0204 | corrector | 0.02313 | 0.03890 |
| 3 | 2 | 1 | 0.0101 | predictor | 0.01242 | 0.02481 |
| 4 | 5 | 2 | 0.0309 | corrector | 0.01171 | 0.01377 |
| 5 | 4 | 2 | 0.0204 | predictor | 0.00418 | 0.00584 |
| 6 | 7 | 3 | 0.0417 | corrector | 0.00412 | 0.00557 |

这说明：

```text
E5 支持“早期必须强保护，中后期可能可以大胆 cache”。
```

但是 E5 还有一个很大的限制：

```text
E5 只测单次 skip。
E5 没有测连续 skip。
```

所以我们不能直接说：

```text
call 16 以后单次 skip 安全，所以 call 16 以后可以一直 skip。
```

真实情况可能有三种：

```text
情况 A：连续 skip 也安全。
    那 E6 可以大胆设置较大的 max_age。

情况 B：连续 skip 2 次安全，4 次开始坏。
    那 E6 要设置 max_age=2 或 3。

情况 C：单次 skip 安全，但连续 skip 很快崩。
    那 E6 必须引入更保守的 accumulated risk 或强制 refresh。
```

E5.5 就是为了区分这三种情况。

---

## 2. E5.5 的核心问题

E5.5 只回答一个问题：

> **在不同采样阶段，连续复用 cache 会不会造成最终图片的感知损伤累积？**

更具体地说，E5.5 要回答下面 6 个问题：

### 问题 1：call 16–89 单次 skip 几乎安全，那么连续 skip 2 / 4 / 8 / 12 次是否仍然安全？

这是最重要的问题。

如果答案是“仍然安全”，那么 E6 的中段可以大胆 skip。

如果答案是“不安全”，那么 E6 必须限制连续 skip 次数。

---

### 问题 2：call 8–15 是过渡区，连续 skip 几次会开始坏？

E5 显示 call 1–7 极危险，call 16 以后单点安全。

那么 call 8–15 就是中间过渡区。

E5.5 要测：

```text
call 8–15 能不能 skip？
最多能连续 skip 几次？
```

这个区域可能决定 E6 的“早期保护长度”。

---

### 问题 3：末尾 call 90–98 有轻微风险回升，连续 skip 会不会导致纹理变差？

E5 显示末尾单次 skip 有一点小 bump。

这可能是因为最后阶段在修细节和纹理。

E5.5 要测：

```text
最后阶段连续 skip 2 / 4 次，会不会导致细节损失？
```

如果会，E6 应该在最后几个 call 做轻保护。

---

### 问题 4：连续 skip 的损伤是否等于单次 skip 损伤的简单相加？

比如：

```text
单独 skip call 16：LPIPS = 0.0001
单独 skip call 17：LPIPS = 0.0001
单独 skip call 18：LPIPS = 0.0001
单独 skip call 19：LPIPS = 0.0001
```

那么连续 skip call 16–19 的 LPIPS 会是多少？

有三种可能：

```text
约等于 0.0004：近似相加
小于 0.0004：后续过程能修复误差
大于 0.0004：误差出现非线性放大
```

这个非常重要，因为真实 cache 的风险往往来自误差累积。

---

### 问题 5：predictor 和 corrector 连续 skip 的风险是否不同？

E5 显示早期 corrector 更危险。

E5.5 要进一步看：

```text
连续 skip 的窗口里，如果 corrector 多，是不是更危险？
连续 skip 的窗口里，如果 predictor 多，是不是更安全？
```

这会指导 E6 是否要给 predictor / corrector 设置不同的 max_age 或阈值。

---

### 问题 6：E6 最应该采用怎样的阶段性 cache 规则？

最终 E5.5 要输出下面这类结论：

```text
call 1–7：绝对不要 skip
call 8–15：最多连续 skip 1 次
call 16–31：最多连续 skip 2 或 4 次
call 32–79：最多连续 skip 4 或 8 次
call 80–89：最多连续 skip 4 次
call 90–98：最多连续 skip 1 或 2 次
```

这就是 E6 的雏形。

---

## 3. 重要概念解释

### 3.1 denoiser call 是什么？

一次 denoiser call 就是模型完整前向一次：

```text
xhat = net(x_t, t, condition)
```

在你的 PixelGen / Heun 50 steps 设置里，一张图大约有 99 个 denoiser call。

这 99 个 call 不是 50 个，因为 Heun sampler 里很多 step 有 predictor 和 corrector 两次模型调用。

你可以简单理解为：

```text
一个采样 step ≈ predictor call + corrector call
```

---

### 3.2 skip 是什么？

skip 指的是：

```text
当前 call 不重新跑模型，而是复用上一次 refresh 保存下来的模型输出。
```

例如：

```text
call 15：refresh，正常跑模型，保存输出 out_15
call 16：skip，不跑模型，直接复用 out_15
call 17：skip，不跑模型，继续复用 out_15
call 18：refresh，重新跑模型，保存 out_18
```

---

### 3.3 cache age 是什么？

cache age 指连续 skip 了多少次。

例如：

```text
call 15：refresh，age = 0
call 16：skip，age = 1
call 17：skip，age = 2
call 18：skip，age = 3
call 19：refresh，age = 0
```

E5.5 的目标之一就是找到不同阶段合理的 `max_age`。

---

### 3.4 skip window 是什么？

E5.5 里的一个 skip window 由两个数字决定：

```text
start_call = s
window_len = L
```

意思是：

```text
从 call s 开始，连续 skip L 个 call。
```

也就是 skip：

```text
call s, call s+1, ..., call s+L-1
```

例如：

```text
start_call = 16
window_len = 4
```

表示：

```text
skip call 16, 17, 18, 19
```

---

### 3.5 PIS_window 是什么？

PIS 是 Perceptual Intervention Score，中文可以叫：

```text
感知干预分数
```

E5 的 PIS 是：

```text
单独 skip 一个 call 后，最终图和 full reference 的差异。
```

E5.5 的 PIS_window 是：

```text
连续 skip 一个窗口后，最终图和 full reference 的差异。
```

公式可以写成：

```text
PIS_window(s, L) = distance(final_skip_window(s,L), final_full)
```

其中 distance 可以由多个指标组成：

```text
LPIPS：更偏局部纹理、边缘、细节
DINO distance：更偏全局语义、物体结构
PSNR：更偏像素级一致性
SSIM：更偏结构相似性
```

最小版本建议重点看：

```text
LPIPS + DINO
```

---

## 4. E5.5 和 E6 的关系

E5.5 不是为了直接加速。

E5.5 是为了给 E6 提供三个关键参数：

```text
1. 哪些阶段必须强制 refresh？
2. 哪些阶段可以连续 skip？
3. 每个阶段最多允许连续 skip 几次，也就是 max_age？
```

可以这样理解：

```text
E5：告诉你单次 skip 哪里危险。
E5.5：告诉你连续 skip 哪里危险。
E6：根据 E5 和 E5.5 的危险地图，写出真正的 online cache 策略。
```

如果没有 E5.5，E6 很容易出现这种问题：

```text
E5 说 call 16 以后单点安全。
于是 E6 在 call 16 以后连续 skip 很多次。
结果最终图崩了。
```

这时问题不在 E5，而在于：

```text
单点安全 ≠ 连续安全。
```

---

## 5. 实验总设计

E5.5 的基本流程如下：

```text
对每个 sample：
    1. 先跑一次 full reference，得到 final_full。
    2. 对多个 skip window 逐个做干预。
    3. 每个 skip window 都生成一张 final_skip_window。
    4. 计算 final_skip_window 和 final_full 的 LPIPS / DINO / PSNR / SSIM。
    5. 保存所有结果到 CSV。
```

更具体地说，一个窗口实验是：

```text
输入：sample i，start_call=s，window_len=L

正常轨迹：
    call 0, 1, 2, ..., 98 全部 full compute
    得到 final_full_i

干预轨迹：
    call 0 到 call s-1：正常 full compute
    call s 到 call s+L-1：skip，复用上一次 cache
    call s+L 到 call 98：恢复 full compute
    得到 final_skip_i_s_L

比较：
    LPIPS(final_skip_i_s_L, final_full_i)
    DINO(final_skip_i_s_L, final_full_i)
    PSNR(final_skip_i_s_L, final_full_i)
    SSIM(final_skip_i_s_L, final_full_i)
```

---

## 6. 最重要的 sanity check

E5.5 必须先做两个 sanity check。

### 6.1 Sanity check 1：L=1 必须复现 E5

E5.5 里 `window_len=1` 的情况，本质上就是 E5。

例如：

```text
E5.5: start_call=16, window_len=1
```

等价于：

```text
E5: skip call 16
```

所以你必须检查：

```text
E5.5 的 L=1 结果 ≈ E5 的单点 skip 结果
```

建议选这些 call 做 L=1 复现：

```text
call 1, 2, 3, 5, 7, 8, 16, 32, 64, 90, 98
```

如果 L=1 对不上 E5，说明 E5.5 的 skip 逻辑、采样器状态、指标计算或随机性有 bug。

不要继续跑大实验。

---

### 6.2 Sanity check 2：full rerun floor

你还需要跑一个“没有任何 skip 的重复生成”：

```text
full reference 第一次：final_full_A
full reference 第二次：final_full_B
比较 final_full_A 和 final_full_B
```

理论上，如果完全确定性，差异应该接近 0。

但是由于 GPU 算子、attention 实现、bf16/fp32、TF32、fused SDPA 等原因，可能会有一点点数值差异。

这个差异叫：

```text
numerical floor
数值误差底线
```

你要记录：

```text
floor_lpips = LPIPS(final_full_A, final_full_B)
floor_dino  = DINO(final_full_A, final_full_B)
```

后面判断一个 window 是否真的有损伤时，不能只看它是不是大于 0，而要看它是不是明显大于 floor。

推荐设置：

```text
真正有意义的损伤 > 5 × numerical floor
```

如果 floor 很大，说明实验不够确定性，需要先修确定性设置。

---

## 7. 推荐实验分阶段执行

不要一上来跑最大版本。

建议分四步：

```text
E5.5-A：1 个 sample 的 smoke test
E5.5-B：8 个 sample 的主 pilot
E5.5-C：32 或 64 个 sample 的扩展版
E5.5-D：可选的随机窗口补充实验
```

---

## 8. E5.5-A：smoke test

### 8.1 目的

确认代码能跑，确认 L=1 能复现 E5，确认连续 skip 逻辑没有明显 bug。

### 8.2 样本

只用 1 张图：

```text
sample_index = 0
class_id = 0
seed = 0
```

和 E5 保持一致。

### 8.3 窗口

建议窗口：

| start_call | window_len | 目的 |
|---:|---:|---|
| 1 | 1 | 复现 E5 早期危险 call |
| 3 | 1 | 复现 E5 早期危险 call |
| 7 | 1 | 复现 E5 早期危险 call |
| 16 | 1 | 复现 E5 中期安全 call |
| 64 | 1 | 复现 E5 中后期安全 call |
| 96 | 1 | 复现 E5 末尾 call |
| 8 | 2 | 过渡区连续 skip |
| 16 | 2 | 中期连续 skip 2 次 |
| 16 | 4 | 中期连续 skip 4 次 |
| 32 | 4 | 中期连续 skip 4 次 |
| 64 | 8 | 中后期连续 skip 8 次 |
| 90 | 2 | 末尾连续 skip 2 次 |
| 94 | 4 | 末尾连续 skip 4 次 |

### 8.4 通过标准

满足下面三条就算通过：

```text
1. 程序能完整跑完。
2. L=1 的 LPIPS / DINO 和 E5 对应 call 基本一致。
3. 输出 CSV、图片、summary.json 都正常生成。
```

如果 L=1 和 E5 不一致，优先检查：

```text
是否使用了同一个 checkpoint？
是否使用同一套 fp32 / autocast 设置？
是否关闭了 fused attention？
是否保存/恢复了正确的 sampler state？
skip 时复用的是不是“上一次 refresh 的模型输出”？
```

---

## 9. E5.5-B：8 个 sample 主 pilot

这是你现在最应该跑的版本。

### 9.1 目的

用和 E5 相同的 8 个 sample，测连续 skip 是否会产生误差累积。

### 9.2 样本

保持和 E5 一样：

```text
num_samples = 8
classes = 0,1,2,3,4,5,6,7
seeds = 0,1,2,3,4,5,6,7
```

这样做的好处是：

```text
可以直接和 E5 的单点 PIS 表对比。
```

---

## 10. E5.5-B 的窗口设计

窗口设计非常关键。

你的窗口要覆盖四个区域：

```text
1. 极危险早期：call 1–7
2. 过渡区：call 8–15
3. 中期主安全区：call 16–89
4. 末尾细节区：call 90–98
```

---

### 10.1 L=1 sanity windows

这些窗口用于复现 E5：

| start_call | window_len | 目的 |
|---:|---:|---|
| 1 | 1 | E5 最危险 call |
| 2 | 1 | 早期 predictor |
| 3 | 1 | 早期 corrector |
| 5 | 1 | 早期 corrector |
| 7 | 1 | 早期 corrector |
| 8 | 1 | 过渡区开始 |
| 16 | 1 | 中期安全区开始 |
| 32 | 1 | 中期 |
| 64 | 1 | 中后期 |
| 90 | 1 | 末尾开始 |
| 98 | 1 | 最后 call |

---

### 10.2 早期危险窗口

早期预计非常危险，所以不要跑太多，主要是为了证明：

```text
连续 skip 早期 call 会明显破坏最终图。
```

建议：

| start_call | window_len | skip 范围 | 目的 |
|---:|---:|---|---|
| 1 | 2 | 1–2 | 最危险开头，验证不可 skip |
| 3 | 2 | 3–4 | 早期 corrector + predictor |
| 5 | 2 | 5–6 | 早期小窗口 |
| 7 | 2 | 7–8 | 危险区到过渡区 |
| 1 | 4 | 1–4 | 极端早期连续 skip |
| 3 | 4 | 3–6 | 早期连续 skip |

这些窗口大概率会很差，但它们有论文价值：它们能证明“早期强制 refresh”不是拍脑袋。

---

### 10.3 过渡区窗口：call 8–15

这是很重要的区域，因为 E6 可能需要在这里从强制 refresh 变成允许 cache。

建议：

| start_call | window_len | skip 范围 | 目的 |
|---:|---:|---|---|
| 8 | 2 | 8–9 | 过渡区短 skip |
| 8 | 4 | 8–11 | 过渡区连续 skip |
| 10 | 2 | 10–11 | 过渡区短 skip |
| 10 | 4 | 10–13 | 过渡区连续 skip |
| 12 | 2 | 12–13 | 靠近中期 |
| 12 | 4 | 12–15 | 过渡区末尾 |
| 14 | 2 | 14–15 | 过渡区末尾短 skip |
| 14 | 4 | 14–17 | 跨入中期安全区 |

你重点看：

```text
从哪个 start_call 开始，L=2 变安全？
从哪个 start_call 开始，L=4 变安全？
```

---

### 10.4 中期主窗口：call 16–89

这是 E5.5 的核心。

E5 显示 call 16–89 单点 skip 几乎无损。

现在要测连续 skip。

建议窗口：

| start_call | window_len 候选 | 目的 |
|---:|---|---|
| 16 | 2, 4, 8, 12, 16 | 中期安全区刚开始，重点测 |
| 20 | 2, 4, 8 | 稍微离开早期 |
| 24 | 2, 4, 8 | 中前期 |
| 32 | 2, 4, 8, 12 | 中期 |
| 40 | 2, 4, 8, 16 | 中期 |
| 50 | 2, 4, 8, 12 | 中期 |
| 60 | 2, 4, 8 | 中后期 |
| 70 | 2, 4, 8, 12 | 中后期 |
| 80 | 2, 4, 8 | 接近末尾 |

这个区域应该产出 E6 最关键的结论。

例如：

```text
如果 call 32/40/50 的 L=8 都安全，E6 中段 max_age 可以从 2 提高到 4 或 6。

如果 L=4 安全但 L=8 不安全，E6 中段 max_age 应该设 3 或 4。

如果 L=2 都不安全，说明 E5 单点安全不能推广到连续 cache，E6 必须很保守。
```

---

### 10.5 末尾窗口：call 88–98

末尾区域可能负责修细节。E5 显示末尾有轻微风险回升。

建议：

| start_call | window_len | skip 范围 | 目的 |
|---:|---:|---|---|
| 88 | 2 | 88–89 | 进入尾部前 |
| 88 | 4 | 88–91 | 尾部前连续 skip |
| 88 | 6 | 88–93 | 长一点 |
| 90 | 2 | 90–91 | 尾部开始 |
| 90 | 4 | 90–93 | 尾部连续 skip |
| 90 | 6 | 90–95 | 尾部长 skip |
| 92 | 2 | 92–93 | 尾部 |
| 92 | 4 | 92–95 | 尾部 |
| 92 | 6 | 92–97 | 尾部较长 |
| 94 | 2 | 94–95 | 最后几步 |
| 94 | 4 | 94–97 | 最后几步 |
| 96 | 2 | 96–97 | 最后两次附近 |

注意：

```text
end_call = start_call + window_len - 1 必须 <= 98
```

所以 `start_call=96, window_len=4` 不合法，因为会超过 98。

---

## 11. 推荐窗口总数

E5.5-B 大约每个 sample 70 个窗口。

8 个 sample 就是：

```text
8 × 70 = 560 个 intervention trajectories
```

你的 E5 是 784 个有效干预，用 3090 跑了大约 25 分钟。

E5.5-B 的量级应该可以接受。

如果你的实现每个窗口都从头跑，会慢一些；如果像 E5 一样利用 prefix / suffix，可以更快。

---

## 12. 如果算力不够，先跑最小版本

如果你想更省算力，可以先跑这个 minimal set。

### 12.1 Minimal sanity windows

```text
L=1: start_call = [1, 3, 5, 7, 16, 48, 80, 96]
```

### 12.2 Minimal early windows

```text
[1,2], [3,4], [5,6], [7,8]
```

也就是：

```text
(start=1, L=2)
(start=3, L=2)
(start=5, L=2)
(start=7, L=2)
```

### 12.3 Minimal mid windows

```text
start_call = [16, 32, 48, 64, 80]
window_len = [2, 4, 8]
```

共 15 个窗口。

### 12.4 Minimal tail windows

```text
(start=90, L=2)
(start=90, L=4)
(start=94, L=2)
(start=94, L=4)
(start=96, L=2)
```

这个 minimal set 大约：

```text
8 + 4 + 15 + 5 = 32 个窗口 / sample
8 samples = 256 个 intervention trajectories
```

这是非常合适的第一版。

---

## 13. E5.5-C：扩展版

当 E5.5-B 跑通后，再做扩展版。

### 13.1 为什么要扩展？

E5 和 E5.5-B 只有 8 张图，而且 class 是 0–7。

这适合 pilot，不适合论文主表。

论文里最好至少有：

```text
32 或 64 个 samples
随机 class
随机 seed
```

### 13.2 推荐扩展设置

```text
num_samples = 64
classes = 从 ImageNet 1000 类里随机抽
seeds = 10000 到 10063，或其他固定随机种子
```

### 13.3 扩展版不要跑太多窗口

扩展版不需要每个 sample 跑 70 个窗口。

根据 E5.5-B 结果，选最关键的窗口。

推荐扩展窗口：

```text
L=1 sanity: start = [1, 3, 7, 16, 64, 98]
早期危险: (1,2), (3,2), (5,2), (1,4)
过渡区: start=[8,10,12,14], L=[2,4]
中期: start=[16,32,50,70,80], L=[2,4,8]
末尾: start=[90,94,96], L=[2,4]
```

大约：

```text
6 + 4 + 8 + 15 + 5 = 38 个窗口 / sample
64 samples = 2432 个 trajectories
```

这个就比较像论文级分析了。

如果算力不够，先做 32 samples。

---

## 14. E5.5-D：可选随机窗口实验

如果你想让结论更稳，可以补一个随机窗口实验。

### 14.1 目的

避免 reviewer 说：

```text
你只挑了几个窗口，可能有选择偏差。
```

### 14.2 做法

对每个 sample，随机选一些窗口：

```text
stage = mid，即 call 16–89
window_len 从 [2, 4, 8] 中随机选
start_call 随机选，保证 end_call <= 98
每个 sample 随机选 16 个窗口
```

这样可以检查：

```text
中期随机位置的连续 skip 是否整体安全。
```

### 14.3 输出

可以单独输出：

```text
e5_5_random_mid_windows.csv
```

论文里可以作为 appendix 或补充实验。

---

## 15. 输出文件设计

建议 E5.5 输出目录：

```text
outputs/e5_5_multi_skip_pis/
    e5_5_smoke_1sample/
    e5_5_main8/
    e5_5_main64/
```

每个 run 下面建议有：

```text
summary.json
windows.csv
pis_window_summary.csv
pis_window_aggregate.csv
pis_safe_age_by_stage.csv
pis_synergy_with_e5.csv
heatmap_lpips_mean.png
heatmap_lpips_q95.png
heatmap_dino_mean.png
heatmap_dino_q95.png
safe_max_age_by_stage.png
synergy_ratio_heatmap.png
visual_top_failures.png
visual_safe_long_skips.png
full_reference_images/
intervention_images/
```

---

## 16. `windows.csv` 应该长什么样？

`windows.csv` 是实验输入表。

每一行代表一个 skip window。

建议列：

| 字段 | 含义 |
|---|---|
| window_id | 窗口编号 |
| stage | early / transition / mid / tail |
| start_call | 开始 skip 的 call index |
| window_len | 连续 skip 长度 |
| end_call | 最后一个 skip call |
| note | 备注 |

例子：

```csv
window_id,stage,start_call,window_len,end_call,note
0,sanity,1,1,1,match_e5_call1
1,sanity,16,1,16,match_e5_call16
2,early,1,2,2,early_danger_short
3,transition,8,4,11,transition_len4
4,mid,32,8,39,mid_len8
5,tail,90,4,93,tail_len4
```

---

## 17. `pis_window_summary.csv` 应该长什么样？

这是最重要的输出表。

每一行代表：

```text
一个 sample + 一个 skip window
```

建议列：

| 字段 | 含义 |
|---|---|
| sample_index | 样本编号 |
| class_id | ImageNet 类别 |
| seed | 随机种子 |
| window_id | 窗口编号 |
| stage | 阶段 |
| start_call | 起始 call |
| end_call | 结束 call |
| window_len | 连续 skip 长度 |
| start_step | 起始采样 step |
| end_step | 结束采样 step |
| start_t | 起始 timestep |
| end_t | 结束 timestep |
| start_call_kind | predictor / corrector |
| num_predictor | 窗口内 predictor 个数 |
| num_corrector | 窗口内 corrector 个数 |
| pis_lpips | 最终图 LPIPS 差异 |
| pis_dino | 最终图 DINO 差异 |
| pis_psnr | 最终图 PSNR |
| pis_ssim | 最终图 SSIM |
| elapsed_sec | 该窗口耗时 |
| reference_image | full reference 图片路径 |
| intervention_image | skip window 后图片路径 |

例子：

```csv
sample_index,class_id,seed,window_id,stage,start_call,end_call,window_len,pis_lpips,pis_dino,pis_psnr,pis_ssim
0,0,0,4,mid,32,39,8,0.0008,0.0004,54.2,0.9989
```

---

## 18. 聚合表应该怎么做？

### 18.1 `pis_window_aggregate.csv`

按窗口聚合所有 samples。

每一行是一个 window。

建议列：

```text
window_id
stage
start_call
end_call
window_len
mean_lpips
median_lpips
q90_lpips
q95_lpips
max_lpips
mean_dino
median_dino
q90_dino
q95_dino
max_dino
mean_psnr
q05_psnr
failure_rate_lpips_0p005
failure_rate_lpips_0p01
failure_rate_dino_0p005
failure_rate_dino_0p01
```

为什么要看 q95 和 max？

因为 cache 方法最怕：

```text
平均很好，但偶尔一张图严重崩。
```

所以不能只看 mean。

---

### 18.2 `pis_safe_age_by_stage.csv`

这个表直接服务 E6。

每一行是一个 stage。

建议列：

```text
stage
start_range
safe_len_mean
safe_len_q95
recommended_max_age_for_e6
reason
```

例子：

```csv
stage,start_range,safe_len_mean,safe_len_q95,recommended_max_age_for_e6,reason
early,1-7,0,0,0,E5 and E5.5 both dangerous
transition,8-15,2,1,1,L=2 borderline, use conservative max_age=1
mid,16-79,8,4,4,L=8 mean safe but q95 maybe high, use max_age=4
late,80-89,4,2,2,tail approaching, use max_age=2
tail,90-98,2,1,1,detail sensitive tail
```

---

### 18.3 `pis_synergy_with_e5.csv`

这个表用来分析连续 skip 是否只是单点损伤的简单相加。

对每个 window：

```text
single_sum_lpips = sum(E5 single-call LPIPS for all calls in the window)
window_lpips = E5.5 measured LPIPS for this continuous window
ratio_lpips = window_lpips / single_sum_lpips
```

DINO 同理。

建议列：

```text
sample_index
window_id
start_call
end_call
window_len
single_sum_lpips
window_lpips
ratio_lpips
single_sum_dino
window_dino
ratio_dino
```

解释：

| ratio | 含义 |
|---:|---|
| ratio < 0.5 | 连续 skip 损伤比单点求和小，后续修复能力强 |
| ratio ≈ 1 | 损伤大致相加 |
| ratio > 1.5 | 连续 skip 出现非线性放大，需要非常小心 |
| ratio 很大但 single_sum 很小 | 说明单点几乎无损，但连续 skip 突然有损，这非常重要 |

注意：如果 `single_sum_lpips` 非常接近 0，ratio 会不稳定。可以加一个小常数：

```text
ratio_lpips = window_lpips / (single_sum_lpips + 1e-8)
```

---

## 19. 推荐的安全阈值

阈值不要一开始写死成论文结论。

建议先用下面的经验阈值做分析：

### 19.1 LPIPS 阈值

| LPIPS | 粗略解释 |
|---:|---|
| < 0.001 | 几乎无感知差异，通常安全 |
| 0.001–0.005 | 很小差异，大概率可接受 |
| 0.005–0.01 | 可能有轻微可见变化，需要看图 |
| 0.01–0.02 | 有明显风险 |
| > 0.02 | 危险，通常不应 skip |

### 19.2 DINO 阈值

| DINO distance | 粗略解释 |
|---:|---|
| < 0.001 | 语义几乎不变 |
| 0.001–0.005 | 很小语义变化 |
| 0.005–0.01 | 可能有结构变化 |
| > 0.01 | 需要警惕 |
| > 0.02 | 危险 |

### 19.3 最终判定不要只靠阈值

建议同时看：

```text
mean
median
q95
max
visual examples
```

特别是 q95。

一个 cache 策略不能只平均好，它还要避免少数样本崩坏。

---

## 20. 判断一个窗口是否安全的推荐规则

你可以先用下面这个规则：

```text
一个 window 被认为 safe，当且仅当：

1. mean_lpips <= 0.002
2. q95_lpips <= 0.005
3. mean_dino <= 0.002
4. q95_dino <= 0.005
5. 没有明显视觉崩坏样例
```

一个 window 被认为 dangerous，如果满足任意一条：

```text
1. mean_lpips > 0.01
2. q95_lpips > 0.02
3. mean_dino > 0.01
4. q95_dino > 0.02
5. top failure 图肉眼明显坏
```

中间的就叫 borderline。

```text
borderline = 可能能用，但 E6 要保守，例如 max_age 减半。
```

---

## 21. 从 E5.5 到 E6 的具体转换方法

这是最重要的部分。

E5.5 跑完以后，不要只说“中期安全”。

你要把结果变成 E6 的参数。

---

### 21.1 第一步：确定 early force-refresh 区域

看 E5.5 早期窗口：

```text
(start=1,L=2)
(start=3,L=2)
(start=5,L=2)
(start=7,L=2)
```

如果它们都很危险，那么 E6 直接设置：

```python
if call_index <= 7:
    refresh()
```

如果 `(start=7,L=2)` 其实已经安全，可以考虑把强制 refresh 改成：

```python
if call_index <= 5:
    refresh()
```

但我建议第一版 E6 仍然保守：

```python
call 1–7 强制 refresh
```

因为 E5 已经显示 call 1–7 占了绝大多数单点损伤。

---

### 21.2 第二步：确定 transition 区域 max_age

看 call 8–15 的窗口。

例如结果可能是：

```text
L=2 安全
L=4 borderline
```

那 E6 设置：

```python
if 8 <= call_index <= 15:
    max_age = 1 或 2
```

为什么不是直接 max_age=4？

因为 E5.5 是离线单窗口测试，而真实 E6 会在一条轨迹里多次 skip，风险可能叠加。

所以建议保守一点：

```text
E5.5 证明 L=4 安全，E6 先用 max_age=2。
E5.5 证明 L=2 安全，E6 先用 max_age=1。
```

---

### 21.3 第三步：确定 mid 区域 max_age

中期是加速的主战场。

看这些窗口：

```text
start=16,32,40,50,60,70,80
L=2,4,8,12,16
```

假设 E5.5 发现：

```text
L=2 全部安全
L=4 全部安全
L=8 大部分安全，但少数 q95 较高
L=12 开始 borderline
L=16 明显危险
```

那么 E6 可以设置：

```python
if 16 <= call_index <= 79:
    max_age = 4
```

如果结果更强：

```text
L=12 也安全
```

可以试：

```python
max_age = 6 或 8
```

如果结果很弱：

```text
L=4 就开始危险
```

那 E6 应该设置：

```python
max_age = 2
```

---

### 21.4 第四步：确定 tail 区域 max_age

看 call 90–98。

如果 E5.5 发现：

```text
tail L=2 安全，L=4 不安全
```

E6 可以设置：

```python
if call_index >= 90:
    max_age = 1 或 2
```

如果 tail 连 L=2 都危险，那设置：

```python
if call_index >= 90:
    refresh()
```

如果 tail L=4 也安全，可以设置：

```python
max_age = 2 或 3
```

注意：末尾 call 数很少，对总加速贡献不大，所以没必要为了多省一两个 call 冒明显风险。

---

### 21.5 第五步：形成 E6-v1 的规则

E5.5 后，你的 E6-v1 可以长这样：

```python
if call_index == 0:
    refresh()

elif call_index <= 7:
    # E5 + E5.5 都证明早期危险
    refresh()

elif call_index <= 15:
    # 过渡区，保守
    max_age = 1
    cache_by_age(max_age)

elif call_index <= 79:
    # 中期主加速区，根据 E5.5 设置
    max_age = 4
    cache_by_age(max_age)

elif call_index <= 89:
    # 接近尾部，稍微保守
    max_age = 2
    cache_by_age(max_age)

else:
    # 末尾细节区
    max_age = 1
    cache_by_age(max_age)
```

其中：

```python
def cache_by_age(max_age):
    if cache_age >= max_age:
        refresh()
    else:
        skip()
```

这就是最简单、最可靠的 E6 起点。

---

## 22. 实现建议

### 22.1 不要重新发明 E5 代码

最稳的方式是：

```text
复制你现有的 E5 单点干预脚本，改成 E5.5 多点连续干预脚本。
```

例如：

```text
scripts/e5_pis_intervention.py
复制为：
scripts/e5_5_multi_skip_pis.py
```

E5 原来可能有类似逻辑：

```python
if call_index == skip_call_index:
    use_cache = True
else:
    use_cache = False
```

E5.5 改成：

```python
skip_set = set(range(start_call, end_call + 1))

if call_index in skip_set:
    use_cache = True
else:
    use_cache = False
```

这就是核心变化。

---

### 22.2 伪代码

下面是最简单的伪代码。

注意：这不是让你完全照抄，而是说明逻辑。

```python
def run_intervention_window(sample, start_call, window_len):
    end_call = start_call + window_len - 1
    skip_set = set(range(start_call, end_call + 1))

    cache_output = None
    x_t = init_noise(sample.seed)

    for call_index, call_info in enumerate(denoiser_call_schedule):
        t = call_info.t
        call_kind = call_info.kind  # predictor or corrector

        if call_index in skip_set:
            assert cache_output is not None
            model_output = cache_output
            did_refresh = False
        else:
            model_output = denoiser(x_t, t, sample.condition)
            cache_output = detach(model_output)
            did_refresh = True

        x_t = sampler_update(x_t, model_output, call_info)

    final_image = decode_or_postprocess(x_t)
    return final_image
```

---

### 22.3 更推荐的高效实现：从 start_call 前的状态恢复

如果每个窗口都从 call 0 跑到 call 98，会浪费很多前缀计算。

E5 应该已经保存或利用了 suffix rerun 的思想。

E5.5 也可以这么做：

```text
1. full reference 时，保存每个 call 之前的 sampler state。
2. 对窗口 (s,L)，直接恢复 call s 之前的 state。
3. 设置 cache_output = full reference 中 call s-1 的模型输出。
4. 连续 skip s 到 s+L-1。
5. 从 s+L 开始 full compute 到结束。
```

这样窗口越靠后，重跑越少。

但是要注意 Heun sampler 的状态可能不只是 `x_t`。

可能还包括：

```text
当前 x_t
当前 t
上一个 predictor 的 velocity / xhat
corrector 所需的临时状态
cache_output
call_kind
```

如果你不确定 sampler state 是否完整，先用慢但可靠的方式：

```text
每个窗口都从头跑。
```

等结果正确后，再优化速度。

---

### 22.4 Heun sampler 里的一个重要提醒

Heun 有 predictor 和 corrector。

所以 skip 一个 call 时，不能只考虑“这个 call 的 t”。

你应该复用 E5 里已经验证过的 skip 逻辑。

不要自己重新写一套 Heun 更新公式。

最安全的做法是：

```text
E5 原来怎么 skip 单个 call，E5.5 就怎么 skip 多个 call。
唯一变化是：skip_call_index 变成 skip_set。
```

---

## 23. 窗口生成代码模板

你可以写一个小脚本生成 `windows.csv`。

```python
import pandas as pd

rows = []
wid = 0

def add(stage, start, length, note=""):
    global wid
    end = start + length - 1
    if end > 98:
        return
    rows.append({
        "window_id": wid,
        "stage": stage,
        "start_call": start,
        "window_len": length,
        "end_call": end,
        "note": note,
    })
    wid += 1

# 1. L=1 sanity
for s in [1, 2, 3, 5, 7, 8, 16, 32, 64, 90, 98]:
    add("sanity", s, 1, f"match_e5_call{s}")

# 2. early danger
for s in [1, 3, 5, 7]:
    add("early", s, 2, "early_len2")
for s in [1, 3]:
    add("early", s, 4, "early_len4")

# 3. transition 8-15
for s in [8, 10, 12, 14]:
    for L in [2, 4]:
        add("transition", s, L, f"transition_len{L}")

# 4. mid main region
mid_specs = {
    16: [2, 4, 8, 12, 16],
    20: [2, 4, 8],
    24: [2, 4, 8],
    32: [2, 4, 8, 12],
    40: [2, 4, 8, 16],
    50: [2, 4, 8, 12],
    60: [2, 4, 8],
    70: [2, 4, 8, 12],
    80: [2, 4, 8],
}
for s, lens in mid_specs.items():
    for L in lens:
        add("mid", s, L, f"mid_len{L}")

# 5. tail region
for s in [88, 90, 92, 94, 96]:
    for L in [2, 4, 6]:
        add("tail", s, L, f"tail_len{L}")

windows = pd.DataFrame(rows)
windows.to_csv("windows_e5_5_main8.csv", index=False)
print(windows)
print("num windows:", len(windows))
```

---

## 24. 分析代码模板

E5.5 跑完后，可以用下面思路分析。

```python
import pandas as pd
import numpy as np

summary = pd.read_csv("pis_window_summary.csv")

# 聚合
agg = summary.groupby(["window_id", "stage", "start_call", "end_call", "window_len"]).agg(
    mean_lpips=("pis_lpips", "mean"),
    median_lpips=("pis_lpips", "median"),
    q90_lpips=("pis_lpips", lambda x: np.quantile(x, 0.90)),
    q95_lpips=("pis_lpips", lambda x: np.quantile(x, 0.95)),
    max_lpips=("pis_lpips", "max"),
    mean_dino=("pis_dino", "mean"),
    median_dino=("pis_dino", "median"),
    q90_dino=("pis_dino", lambda x: np.quantile(x, 0.90)),
    q95_dino=("pis_dino", lambda x: np.quantile(x, 0.95)),
    max_dino=("pis_dino", "max"),
    mean_psnr=("pis_psnr", "mean"),
    q05_psnr=("pis_psnr", lambda x: np.quantile(x, 0.05)),
).reset_index()

# failure rate
for thr in [0.001, 0.005, 0.01, 0.02]:
    fr = summary.groupby("window_id").apply(lambda g: (g["pis_lpips"] > thr).mean())
    agg[f"failure_rate_lpips_gt_{thr}"] = agg["window_id"].map(fr)

agg.to_csv("pis_window_aggregate.csv", index=False)
```

---

## 25. 和 E5 单点结果比较的分析模板

假设你有 E5 的 `pis_summary.csv` 和 E5.5 的 `pis_window_summary.csv`。

```python
import pandas as pd
import numpy as np

# E5 single-call results
e5 = pd.read_csv("e5_main8_fullcalls_fp32/pis_summary.csv")
e5 = e5.dropna(subset=["pis_lpips"])

# E5.5 window results
e55 = pd.read_csv("pis_window_summary.csv")

rows = []
for _, r in e55.iterrows():
    sample = r["sample_index"]
    start = int(r["start_call"])
    end = int(r["end_call"])

    single = e5[
        (e5["sample_index"] == sample) &
        (e5["call_index"] >= start) &
        (e5["call_index"] <= end)
    ]

    single_sum_lpips = single["pis_lpips"].sum()
    single_sum_dino = single["pis_dino"].sum()

    rows.append({
        "sample_index": sample,
        "window_id": r["window_id"],
        "start_call": start,
        "end_call": end,
        "window_len": r["window_len"],
        "window_lpips": r["pis_lpips"],
        "single_sum_lpips": single_sum_lpips,
        "ratio_lpips": r["pis_lpips"] / (single_sum_lpips + 1e-8),
        "window_dino": r["pis_dino"],
        "single_sum_dino": single_sum_dino,
        "ratio_dino": r["pis_dino"] / (single_sum_dino + 1e-8),
    })

synergy = pd.DataFrame(rows)
synergy.to_csv("pis_synergy_with_e5.csv", index=False)
```

---

## 26. 必须画的图

### 26.1 LPIPS heatmap

横轴：start_call  
纵轴：window_len  
颜色：mean LPIPS 或 q95 LPIPS

这个图回答：

```text
从哪里开始连续 skip 会坏？
连续 skip 多长会坏？
```

建议画两张：

```text
heatmap_lpips_mean.png
heatmap_lpips_q95.png
```

---

### 26.2 DINO heatmap

和 LPIPS 一样，但颜色换成 DINO。

```text
heatmap_dino_mean.png
heatmap_dino_q95.png
```

LPIPS 和 DINO 的区别：

```text
LPIPS 更关注局部纹理和细节。
DINO 更关注全局语义和结构。
```

如果一个窗口 LPIPS 高但 DINO 低，可能是纹理细节变了。

如果 DINO 高，说明结构或语义更可能变了，要更警惕。

---

### 26.3 safe max_age curve

横轴：call stage 或 start_call  
纵轴：safe window_len

这个图可以直接转成 E6 的 max_age 设置。

例子：

```text
call 1–7: safe_len = 0
call 8–15: safe_len = 1
call 16–79: safe_len = 8
call 80–89: safe_len = 4
call 90–98: safe_len = 2
```

---

### 26.4 synergy ratio heatmap

颜色：

```text
ratio = window_pis / sum(single_call_pis)
```

这个图回答：

```text
连续 skip 的误差是否比单点误差求和更严重？
```

特别注意：

```text
中期单点 PIS 很小，如果连续 skip ratio 很高，并且绝对 LPIPS 也高，说明误差累积是真问题。
```

---

### 26.5 top failure visual grid

保存最坏的 32 个窗口可视化。

每一组最好包含：

```text
full reference
skip window result
absolute diff map，可选
文字：sample_id, class_id, start_call, L, LPIPS, DINO
```

这个图非常适合放论文或 README。

---

### 26.6 safe long skip visual grid

还要保存一些“看起来安全的长 skip”。

例如：

```text
call 32 开始连续 skip 8 次，LPIPS 很低
call 50 开始连续 skip 12 次，LPIPS 很低
```

这个图可以证明：

```text
中期确实存在可利用的冗余。
```

---

## 27. 画图代码示意

```python
import pandas as pd
import matplotlib.pyplot as plt

agg = pd.read_csv("pis_window_aggregate.csv")

# 只画 mid + transition + tail，不画 sanity
plot_df = agg[agg["stage"].isin(["transition", "mid", "tail"])]

pivot = plot_df.pivot_table(
    index="window_len",
    columns="start_call",
    values="mean_lpips",
    aggfunc="mean",
)

plt.figure(figsize=(14, 5))
plt.imshow(pivot.values, aspect="auto")
plt.colorbar(label="mean LPIPS")
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90)
plt.xlabel("start_call")
plt.ylabel("window_len")
plt.title("E5.5 Continuous Skip PIS: mean LPIPS")
plt.tight_layout()
plt.savefig("heatmap_lpips_mean.png", dpi=200)
```

注意：如果论文图，建议用更清楚的配色和标注；现在先能看懂即可。

---

## 28. 命令行参数建议

你的 E5.5 脚本建议支持这些参数：

```text
--config
--ckpt
--output-dir
--run-id
--num-samples
--classes
--seeds
--seed-start
--windows-csv
--device
--no-autocast
--allow-fused-sdpa
--allow-tf32
--skip-lpips
--skip-dino
--save-image-count
--resume
--shard-id
--num-shards
```

其中 `--shard-id` 和 `--num-shards` 很有用。

因为每个窗口是独立的，可以多 GPU 并行。

例如：

```bash
python scripts/e5_5_multi_skip_pis.py \
  --config configs_c2i/PixelGen_XL_without_CFG.yaml \
  --ckpt ckpts/PixelGen_XL_80ep.ckpt \
  --output-dir outputs/e5_5_multi_skip_pis \
  --run-id e5_5_main8_shard0 \
  --num-samples 8 \
  --seed-start 0 \
  --windows-csv windows_e5_5_main8.csv \
  --device cuda:0 \
  --no-autocast \
  --allow-fused-sdpa false \
  --shard-id 0 \
  --num-shards 4
```

然后另外三张卡跑：

```bash
--shard-id 1 --num-shards 4
--shard-id 2 --num-shards 4
--shard-id 3 --num-shards 4
```

最后合并 CSV。

---

## 29. 推荐 run 命名

建议命名清楚：

```text
e5_5_smoke_s1_fp32
e5_5_main8_windows70_fp32
e5_5_main32_windows38_fp32
e5_5_main64_windows38_fp32
e5_5_random_mid64_fp32
```

不要用：

```text
run1
run2
test_new
final_final
```

以后整理论文时会崩溃。

---

## 30. 结果解释模板

跑完 E5.5 后，可以按下面模板写结果。

### 30.1 如果中期连续 skip 很安全

你可以写：

```text
E5.5 shows that the single-call robustness observed in E5 extends to multi-call cache reuse in the middle trajectory. In particular, windows starting after call 16 remain perceptually stable up to length 4/8, while early windows remain highly sensitive. This supports a stage-aware cache schedule with forced early refresh and aggressive middle-stage reuse.
```

中文意思：

```text
E5.5 证明 E5 发现的中期安全性不只是单点现象，连续复用也仍然安全。
因此 E6 可以采用早期强制 refresh、中期大胆复用的策略。
```

---

### 30.2 如果中期连续 skip 不安全

你可以写：

```text
Although E5 indicates that isolated mid-stage skips have negligible final perceptual damage, E5.5 reveals that consecutive skips can accumulate errors nonlinearly. This motivates an explicit max-age constraint in the online cache policy.
```

中文意思：

```text
虽然单点 skip 安全，但连续 skip 会出现误差累积。
所以 E6 不能只根据 call_index 判断，还必须限制 max_age。
```

这也是很有价值的结果。

---

### 30.3 如果末尾连续 skip 不安全

你可以写：

```text
Late-stage windows show a small but non-negligible increase in LPIPS, suggesting that the tail of the sampling trajectory refines local perceptual details. We therefore adopt a conservative tail policy in E6.
```

中文意思：

```text
末尾阶段可能在修局部细节，所以 E6 最后几个 call 要保守。
```

---

## 31. E5.5 可能出现的结果与下一步

### 31.1 结果 A：早期危险，中期 L=8 仍安全，末尾 L=2 安全

这是最理想的结果。

E6-v1：

```python
call 1-7: refresh always
call 8-15: max_age = 1
call 16-79: max_age = 4 or 6
call 80-89: max_age = 2 or 4
call 90-98: max_age = 1 or 2
```

然后直接跑 E6。

---

### 31.2 结果 B：中期 L=4 安全，L=8 不安全

这是也很正常。

E6-v1：

```python
call 16-79: max_age = 3 or 4
```

不要强行设置 max_age=8。

---

### 31.3 结果 C：中期 L=2 都不稳定

说明真实 cache 比 E5 单点实验更难。

这时 E6 不能只靠 stage-aware max_age。

你需要引入 accumulated risk：

```python
risk_accum += cheap_distance
if risk_accum > threshold:
    refresh()
else:
    skip()
```

这里的 cheap_distance 可以先用简单的：

```text
x_t 的像素差 / 低分辨率差
SEAInput distance
模型输入 norm 变化
```

但这一步要建立在 E5.5 之后，不要提前假设某个 proxy 一定靠谱。

---

### 31.4 结果 D：不同 sample 差异很大

如果平均安全但有少数样本崩，说明需要内容自适应。

E6 可以加一个轻量规则：

```text
如果当前 x_t 和上次 refresh 的低频差异过大，就提前 refresh。
```

或者：

```text
在 max_age 之外再加 threshold gate。
```

例如：

```python
if cache_age >= max_age:
    refresh()
elif cheap_distance > threshold:
    refresh()
else:
    skip()
```

---

## 32. 实验注意事项

### 32.1 保持和 E5 完全相同的设置

E5.5 要和 E5 对齐：

```text
同一个 config
同一个 checkpoint
同一个 sampler
同一个 num_steps
同一个 seed
同一个 class
同一个 fp32 / autocast 设置
同一个 LPIPS / DINO 计算方式
```

否则不能和 E5 比。

---

### 32.2 不要一开始用 bf16

E5 主实验用了 fp32 / no_autocast。

E5.5 也先用 fp32。

等结论稳定后，再测 bf16 对结果是否有影响。

---

### 32.3 注意 call 0 不能 skip

E5 里 call 0 是无效的，因为没有 previous cache。

E5.5 也一样：

```text
start_call 不能是 0。
```

---

### 32.4 保存 top failure 图片

不要只保存 sample 0。

建议自动保存：

```text
LPIPS 最大的 top 32 windows
DINO 最大的 top 32 windows
中期最长且仍安全的 top 16 windows
```

这样你能看出：

```text
坏的时候到底坏在哪里？
安全的时候是否真的肉眼无差？
```

---

### 32.5 关注 q95，不要只看 mean

比如一个窗口：

```text
mean_lpips = 0.001
q95_lpips = 0.02
```

这说明大部分样本没事，但少数样本可能明显坏。

真实论文里，这种情况不能简单说 safe。

---

### 32.6 E5.5 是 whole-denoiser skip，不一定等于 block cache

你现在 E5 / E5.5 测的是：

```text
整个 denoiser call 的输出复用
```

如果你后续 E6 做的是 block-level cache，例如只缓存 transformer block 输出，那还要补一个对应版本：

```text
E5.5-block：连续复用 block cache 的干预实验
```

但第一阶段先做 whole-denoiser skip 是合理的，因为它能快速画出采样轨迹的敏感性地图。

---

## 33. 推荐最终 README 展示方式

E5.5 完成后，你可以在 README 里增加一节：

```markdown
## E5.5: Multi-Skip Perceptual Intervention
```

建议放这 5 个内容：

### 33.1 实验设置

```text
PixelGen-XL, ImageNet-256, Heun 50 steps, 99 calls, 8/64 samples.
We intervene by consecutively skipping L denoiser calls from start call s and compare the final image with the full-compute reference using LPIPS, DINO, PSNR and SSIM.
```

### 33.2 主要 heatmap

放：

```text
heatmap_lpips_q95.png
heatmap_dino_q95.png
```

### 33.3 safe max_age 表

放：

| Stage | Call range | safe L | E6 max_age |
|---|---:|---:|---:|
| early | 1–7 | 0 | 0 |
| transition | 8–15 | ? | ? |
| mid | 16–79 | ? | ? |
| late | 80–89 | ? | ? |
| tail | 90–98 | ? | ? |

### 33.4 视觉样例

放两组：

```text
Top failures：证明早期/危险窗口真的会坏。
Safe long skips：证明中期连续 skip 真的可以省算力。
```

### 33.5 给 E6 的结论

最后写：

```text
Based on E5.5, we set the E6 stage-aware max-age schedule as ...
```

---

## 34. E5.5 完成后的决策树

跑完 E5.5 后，按这个顺序决策：

```text
1. L=1 是否复现 E5？
   否：修代码。
   是：继续。

2. full rerun floor 是否很低？
   否：修确定性设置。
   是：继续。

3. call 1–7 连续 skip 是否危险？
   大概率是。E6 强制 refresh。

4. call 8–15 的安全长度是多少？
   决定 transition max_age。

5. call 16–79 的安全长度是多少？
   决定主要加速区 max_age。

6. call 90–98 的安全长度是多少？
   决定 tail 是否保守。

7. 是否存在明显 sample-specific failure？
   如果有，E6 需要加 cheap adaptive gate。
   如果没有，E6-v1 可以先用 stage-aware max_age。
```

---

## 35. 你现在最应该跑的版本

我建议你现在立刻跑：

```text
E5.5-B main8 minimal 或 full window set
```

如果你想稳一点，先跑 minimal：

```text
8 samples × 32 windows ≈ 256 trajectories
```

如果顺利，再跑 full：

```text
8 samples × 约 70 windows ≈ 560 trajectories
```

推荐顺序：

```text
第 1 步：smoke 1 sample
第 2 步：main8 minimal
第 3 步：main8 full
第 4 步：根据结果设计 E6-v1
第 5 步：main32/main64 扩展验证
```

---

## 36. 最终你希望从 E5.5 得到什么结论？

最理想的结论不是“所有中期都能随便 skip”。

更严谨的结论应该是：

```text
E5.5 shows that cache sensitivity in PixelGen is strongly stage-dependent not only for isolated skips but also for consecutive cache reuse. Early calls are highly sensitive and require full refresh, while middle calls tolerate multi-call reuse up to a bounded cache age. This motivates a simple stage-aware max-age cache policy for E6.
```

中文：

```text
E5.5 证明 PixelGen 的 cache 敏感性不仅在单点 skip 上有阶段性，在连续复用上也有阶段性。早期 call 高度敏感，必须 refresh；中期 call 可以容忍一定长度的连续复用。因此 E6 应该采用阶段感知的 max-age cache 策略。
```

这条线比“用 SEAInput 预测感知变化”更稳，因为它直接从 causal intervention 得到 E6 规则。

---

## 37. 和论文背景的关系

PixelGen 的 x-prediction 会直接预测 clean image，并用 LPIPS / P-DINO 这类感知监督让模型更关注 perceptual manifold。因此，用 LPIPS 和 DINO 来评估 skip 对最终图的感知损伤，是和 PixelGen 设计目标一致的。

SeaCache / TeaCache 这类方法提醒我们：真实 cache 通常不是孤立 skip，而是基于累计距离或连续复用的动态策略。因此，你的 E5 只能回答“单点敏感性”，E5.5 才能回答“连续复用敏感性”。

换句话说：

```text
E5：感知因果敏感性地图
E5.5：连续复用感知风险地图
E6：根据这两张地图设计 online cache
```

---

## 38. 最小可交付 checklist

E5.5 最小完成标准：

```text
[ ] 生成 windows_e5_5_main8.csv
[ ] 跑通 smoke 1 sample
[ ] L=1 能复现 E5
[ ] 跑通 main8 minimal 或 main8 full
[ ] 输出 pis_window_summary.csv
[ ] 输出 pis_window_aggregate.csv
[ ] 输出 LPIPS / DINO heatmap
[ ] 输出 safe_max_age_by_stage.csv
[ ] 输出 top failure visual grid
[ ] 输出 safe long skip visual grid
[ ] 写 README E5.5 小结
[ ] 给出 E6-v1 的 max_age schedule
```

---

## 39. E5.5 的一句话结论模板

等你跑完后，可以把下面这句话补完整：

```text
Based on E5.5, we find that consecutive skip windows after call ___ remain perceptually stable up to length ___, while windows before call ___ cause large LPIPS/DINO degradation. Therefore, E6 uses forced refresh for call ___, max_age=___ for transition, max_age=___ for middle stage, and max_age=___ for the tail stage.
```

中文版本：

```text
根据 E5.5，我们发现从 call ___ 之后开始的连续 skip 窗口在长度不超过 ___ 时仍然保持感知稳定；而 call ___ 之前的窗口会造成明显 LPIPS/DINO 损伤。因此，E6 对 call ___ 采用强制 refresh，对过渡区设置 max_age=___，对中期设置 max_age=___，对末尾设置 max_age=___。
```

这就是 E5.5 最终要交给 E6 的东西。

---

## 40. 最后提醒

E5.5 的重点不是把实验搞得很复杂。

重点是回答一个非常朴素但关键的问题：

```text
连续复用 cache 到底会不会把图搞坏？
```

只要你把这个问题回答清楚，E6 就会自然很多。

不要急着训练 proxy，不要急着做复杂判据。

先把下面这张表跑出来：

| Stage | Call range | L=2 | L=4 | L=8 | L=12 | 推荐 E6 max_age |
|---|---:|---|---|---|---|---:|
| early | 1–7 | dangerous | dangerous | - | - | 0 |
| transition | 8–15 | ? | ? | - | - | ? |
| mid | 16–79 | ? | ? | ? | ? | ? |
| late | 80–89 | ? | ? | ? | - | ? |
| tail | 90–98 | ? | ? | - | - | ? |

这张表就是你从 E5 走向 E6 的桥。
