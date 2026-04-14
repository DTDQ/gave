# Score 计算方法 - CPA 违反惩罚设计

## 现有公式

```
cpa_t = cost_t / (reward_t + epsilon)
coef_t = cpa_cons / (cpa_t + epsilon)
penalty_t = coef_t^beta,  若 > 1.0 则截断为 1.0；否则 penalty = coef_t
score_t = penalty_t * reward_t
```

其中 `beta = 2`, `cost_t = budget * (1 - remaining_ratio_t)`。

**问题**：当 CPA 超标时，penalty = coef = cpa_cons / cpa_actual，是线性衰减，惩罚力度不够。

---

## 加强惩罚方案

### 方案1：指数惩罚（推荐）

```
penalty_t =
    1.0                                              if cpa_t <= cpa_cons
    exp(-alpha * (cpa_t - cpa_cons) / cpa_cons)      if cpa_t > cpa_cons

score_t = penalty_t * reward_t
```

其中 `alpha` 控制衰减速度（建议 `alpha in [3, 10]`）。CPA 超标比例越大，score 指数级下降。例如 `alpha=5` 时，CPA 超标 20%，penalty ≈ 0.37；超标 50%，penalty ≈ 0.08。

---

### 方案2：幂次惩罚（更简单）

```
penalty_t =
    1.0                            if cpa_t <= cpa_cons
    (cpa_cons / cpa_t) ^ gamma     if cpa_t > cpa_cons

score_t = penalty_t * reward_t
```

把现有的 `gamma = 1` 提高到 `gamma in [3, 5]`。当 cpa_actual = 2 * cpa_cons 时：`gamma=1` -> penalty=0.5，`gamma=4` -> penalty=0.0625。

---

### 方案3：阈值 + 悬崖式惩罚（最激进）

```
violation_ratio_t = (cpa_t - cpa_cons) / cpa_cons

penalty_t =
    1.0                                          if violation_ratio_t <= 0
    max(0, 1 - k * violation_ratio_t ^ 2)        if violation_ratio_t > 0

score_t = penalty_t * reward_t
```

其中 `k` 控制悬崖陡度（建议 `k in [5, 20]`）。`k=10` 时，超标 10% -> penalty=0.9，超标 30% -> penalty=0.1，超标 32%+ -> penalty=0（score 归零）。

---

## 对比总结

| 超标比例 | 现有(线性) | 方案1(alpha=5) | 方案2(gamma=4) | 方案3(k=10) |
|---------|-----------|---------------|---------------|------------|
| 10%     | 0.91      | 0.61          | 0.68          | 0.90       |
| 20%     | 0.83      | 0.37          | 0.48          | 0.60       |
| 50%     | 0.67      | 0.08          | 0.20          | 0 (截断)    |
| 100%    | 0.50      | 0.007         | 0.06          | 0 (截断)    |

推荐 **方案1（指数惩罚）**，平滑且可控，`alpha` 越大惩罚越狠。
