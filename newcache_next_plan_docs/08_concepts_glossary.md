# 08. 关键概念中文小词典

这份词典尽量用本科生友好的语言解释你后续会用到的概念。

---

## 1. Diffusion / Flow Matching

扩散模型可以理解为：先把真实图片逐渐加噪变成纯噪声，再训练一个模型学会反向一步步去噪。采样时从随机噪声出发，经过很多步 denoising，最后得到图像。

Flow Matching / Rectified Flow 是类似的一类生成方法。它常把从噪声到图像的过程看成一条连续路径，模型预测沿着这条路径应该走的速度 `v`。

在你的代码里，采样器每一步需要一个 velocity，用来更新当前图像状态。

---

## 2. x-prediction

x-prediction 指模型直接预测干净图像 `x` 或 clean image estimate：

```text
xhat_t = model(x_t, t, condition)
```

这里：

- `x_t`：当前带噪状态；
- `t`：当前时间步；
- `condition`：类别标签或文本；
- `xhat_t`：模型认为最终干净图像大概长什么样。

x-prediction 的好处是预测目标更直观，特别适合直接在 `xhat_t` 上加感知损失或做感知分析。

---

## 3. Velocity conversion

虽然 x-pred 模型输出 `xhat_t`，采样器通常需要 velocity。因此要转换：

```text
v_t = (xhat_t - x_t) / (1 - t)
```

直觉：

- `x_t` 是当前位置；
- `xhat_t` 是目标干净图像；
- 两者差值表示“往干净图像方向走”；
- 除以 `(1-t)` 是根据当前时间尺度做归一化。

注意：当 `t` 接近 1 时，`1-t` 很小，所以代码里通常会 clamp，避免数值爆炸。

---

## 4. Cache acceleration

cache 加速就是：相邻 timestep 的模型输出很相似时，不重新跑完整 denoiser，而是复用上一次的输出。

完整计算：

```text
call 1: run model
call 2: run model
call 3: run model
```

cache：

```text
call 1: run model, save output
call 2: reuse output
call 3: run model, update cache
```

这样能减少 forward 次数，从而加速。

---

## 5. Refresh / Skip

- **Refresh**：重新运行完整 denoiser，并更新 cache。
- **Skip / Reuse**：不运行完整 denoiser，直接复用缓存结果。

cache 方法的核心问题是：什么时候 refresh，什么时候 skip。

---

## 6. Refresh Ratio, RR

Refresh Ratio 是刷新比例：

```text
RR = refresh 次数 / 可刷新机会总数
```

你的 Heun exact 设置里，可刷新机会是 99 次：

```text
50 predictor + 49 corrector = 99 calls
```

所以 RR=0.30 表示大约只跑 30% denoiser calls，剩下 70% 复用。

---

## 7. Online vs Oracle

### Online

真实推理时能用的信息。比如：

- 当前 `x_t`；
- 当前 timestep；
- 条件 embedding；
- 第一个 block 前的 cheap proxy；
- 过去 refresh 得到的输出。

Online 方法可以真正加速。

### Oracle

提前知道完整轨迹的信息。比如：

- full uncached trajectory 里每一步的 `xhat_t`；
- 相邻 `xhat_t` 的 DINO / LPIPS 距离；
- 最终 full sample。

Oracle 只能用于分析上限，不能直接作为加速方法。

---

## 8. 感知流形 Perceptual Manifold

“流形”在数学里是很严格的概念，但你的论文里可以用工程化定义：

> 感知流形是由人眼和语义模型认为重要的图像变化组成的空间。

比如两张图 RGB 像素差别很大，但人眼看起来差不多，那它们在感知流形上距离可能很小；反过来，两张图像素差别不大，但物体类别或结构变了，那感知距离可能很大。

你可以用三个信号近似它：

- SEA：频谱上的内容信号；
- DINO：语义结构；
- LPIPS：局部视觉相似度。

---

## 9. LPIPS

LPIPS 是一种感知距离。它不是直接比较像素，而是把图片送进一个预训练视觉网络，比较中间特征。

直觉：

- 像素 L2：关心每个像素是否一样；
- LPIPS：更关心人眼看起来是否一样。

LPIPS 通常对纹理、边缘、局部细节比较敏感。

---

## 10. DINO / DINOv2

DINO 是自监督视觉模型。它不需要人工标签，也能学到强语义特征。

在你的项目里，DINO feature 可以用来判断：

- 物体结构是否变了；
- 场景布局是否变了；
- 语义是否一致。

DINO drift 比 LPIPS 更偏全局语义，LPIPS 更偏局部视觉。

---

## 11. SEA filter

SEA filter 是 SeaCache 的核心思想之一。它在频域里过滤特征，让距离计算更关注内容信号，而不是随机噪声。

频域可以理解为：

- 低频：大轮廓、大结构、平滑区域；
- 高频：边缘、纹理、噪声、细节。

扩散采样有一个常见现象：早期先形成低频结构，后期再补高频细节。SEA filter 根据 timestep 调整频率权重，让 cache 判据更符合这个过程。

---

## 12. FFT / iFFT

FFT 是快速傅里叶变换，把空间图像/特征转到频率空间。

iFFT 是反变换，把频率空间转回原空间。

SeaCache / SEAInput 的做法大致是：

```text
feature -> FFT -> 乘 SEA filter -> iFFT -> filtered feature
```

然后在 filtered feature 上算相邻距离。

---

## 13. Heun sampler

Heun 是一种数值积分方法。简单说，它每一步会先做一个预测，再做一个修正：

```text
predictor: 先估计往哪走
corrector: 再修正这个方向
```

所以 50 个 Heun steps 不是 50 次 denoiser call，而是接近 2 倍：

```text
50 predictor + 49 corrector = 99 calls
```

这就是你仓库里 call-level convention 的来源。

---

## 14. Predictor / Corrector transition

在 Heun 里，相邻 call 可能是：

```text
predictor -> corrector
corrector -> next predictor
```

这两类 transition 的统计性质不同。你的 E2 已经发现 predictor→corrector 的 distance mass 很强，所以后续 cache gate 最好区分它们。

---

## 15. Accumulated-distance rule

很多动态 cache 方法不是看单步距离，而是累积距离：

```text
acc += distance
if acc > threshold:
    refresh
    acc = 0
else:
    skip
```

直觉：一次小变化可以忍，但连续很多小变化累积起来也可能出问题。

---

## 16. Calibration

Calibration 是校准，不是训练扩散模型。它用少量样本确定：

- score 的归一化参数；
- 阈值 delta；
- ridge proxy 的小权重；
- target RR 对应的设置。

这比训练一个生成模型便宜很多。

---

## 17. Ridge regression

Ridge regression 是带 L2 正则的线性回归。它学一个简单公式：

```text
score = w1*x1 + w2*x2 + ... + b
```

L2 正则会限制权重不要太大，减少过拟合。

它适合你的原因：

- 数据量小也能用；
- 速度极快；
- 可解释；
- reviewer 不会觉得是复杂黑盒。

---

## 18. Uncertainty gate

不确定性 gate 是：当 proxy 不确定时，宁可 refresh。

直觉：cache 加速里，误 skip 的代价可能很大，所以风险高或不确定时保守一点。

公式：

```text
score = predicted_mean + k * predicted_uncertainty
```

---

## 19. PSNR

PSNR 是像素级相似指标。越高表示两张图像越接近。

缺点：它对人眼感知不一定准确。两张图片 PSNR 高，不一定看起来最自然。

---

## 20. SSIM

SSIM 比较图像结构相似性，比如亮度、对比度、结构。比 PSNR 更符合视觉结构，但仍然不是完美感知指标。

---

## 21. FID

FID 衡量生成图像分布和真实图像分布的距离。越低越好。

它不是 paired metric，不要求每张生成图和某张 reference 一一对应。它看整体分布质量。

---

## 22. Precision / Recall for generative models

生成模型里的 Precision / Recall 可以粗略理解为：

- Precision：生成图是否真实、质量是否高；
- Recall：生成结果是否覆盖多样性。

cache 如果过度复用，可能提高某些样本的稳定性，但降低多样性，所以 Recall 也要看。

---

## 23. Pareto curve

Pareto curve 是速度和质量的折中曲线。

比如横轴是 latency，纵轴是 LPIPS。如果 A 方法在同样 latency 下 LPIPS 更低，或者同样 LPIPS 下 latency 更低，就说明 A 更好。

cache 论文不要只报一个点，最好画曲线。

---

## 24. Tail failure

Tail failure 指少数样本严重失败。平均 LPIPS 可能不错，但 p95 / p99 很差。

cache 方法尤其要关注 tail failure，因为错误复用可能让某些样本出现明显 artifacts。

---

## 25. Bootstrap confidence interval

Bootstrap 是一种估计指标置信区间的方法。做法是从测试样本中有放回地重复抽样，计算很多次平均指标，看波动范围。

它可以帮助你判断：两个方法差距是真实稳定，还是随机波动。

