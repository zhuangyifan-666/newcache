# CHANGELOG v2：相对旧版 xWPCache 方案改了什么？

## 1. 删除经验分段主方法

旧版中出现过类似：

```text
early refresh
middle skip
tail safe cap
```

v2 中不再把这作为主方法。

现在 E5.5 的结果只用作：

```text
window-level causal labels
```

不直接变成 refresh schedule。

---

## 2. 把 E6-D0 提升为必要步骤

旧版中 E6-D0 是推荐诊断。

v2 中它是进入 online E6 前的必要门槛。

如果 E6-D0 不能证明 xWPCache risk 预测 E5.5 PIS，就不做 online。

---

## 3. 新增 anchor-relative 风险

v2 强调：

```text
真实 cache 复用的是上一次 refresh 的 xhat。
```

所以风险要相对于 anchor，而不是只看 adjacent difference。

---

## 4. 新增 vector accumulated solver error

旧版主要是 scalar accumulated distance。

v2 新增：

```text
先累积带方向的 solver residual，再取 norm。
```

这更符合 ODE update 的误差形式。

---

## 5. 新增 time-only controlled baseline

v2 要证明 xWPCache 不是 timestep 经验规则。

因此必须加入：

```text
time-only baseline
controlled correlation
leave-one-sample-out calibration
```

---

## 6. trust-region 从经验分段改成全局校准

旧版可能按 call 区间设置 max_age。

v2 改成：

```text
global max_age 或 risk-based age penalty
通过 E5.5 validation 选择
```

不按具体 call 段手工设置。

---

## 7. 失败诊断更清晰

v2 把失败分成：

```text
目标错
proxy 错
online drift
threshold 错
time-only 混淆
```

避免盲目调参。
