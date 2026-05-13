# B 线 Regime Guardrail 设计

日期：2026-05-13

## 1. 目标

本 spec 专门定义 B 线 `regime` 的职责、输入证据、merge/split/boundary 规则和 implementation plan guardrails。

它服务于 PKA 与 B/C 线结合方法中的一个关键风险：`regime` 容易被实现成简单数值区间、kernel name 分类、operator label，或 PKA cluster 的重命名。这样的实现会让 B 线偏离 simulator-side decision layer 的职责。

本 spec 的核心定义是：

```text
regime = family 内部的最小 C-line validation target
```

它只回答一个问题：

```text
在同一个 family 里，哪些 anchors 应该一起被验证，哪些必须拆开？
```

## 2. Regime 不是什么

Regime 不是：

- PKA cluster；
- family 的别名；
- lane 的别名；
- kernel name 或 operator name 分类；
- 单纯按数值阈值切出的执行区间；
- 网络模块名称，例如 attention、FFN、embedding；
- backend parameter scenario。

这些对象都可以为 regime 提供证据，但不能替代 regime。

## 3. Regime 的职责

Regime 是 B 线交给 C 线的最小可验证对象。它必须同时具备：

1. 明确所属 `family`；
2. 明确来源 anchors；
3. 可解释的硬件执行证据；
4. 可解释的 network / algorithm role；
5. 时间重要性来源；
6. 明确 validation role；
7. 明确 boundary status。

Regime 的职责不是“多分一点组”，而是让 C 线可以生成一个清晰问题：

```text
这个对象作为一个 validation target，是否能帮助 backend 更快找到有效 tuning 或约束？
```

## 4. 允许的输入证据

Regime builder 只能使用以下证据：

| 证据 | 来源 | 用途 |
|---|---|---|
| `family_id` | B 线 family builder | 限定 regime 只在同 family 内 merge/split |
| `source_anchor_ids` | RepresentativeAnchorTable | 追踪 regime 覆盖哪些 anchors |
| `member_record_ids` | RepresentativeAnchorTable | 汇总成员级 evidence |
| `coverage_weight` | RepresentativeAnchorTable | 衡量覆盖贡献 |
| `hardware_axis_weights` | RepresentationWeightSummary | 判断硬件行为是否相容 |
| `dominant_hardware_axis` | RepresentationWeightSummary | 判断主硬件压力 |
| `algorithmic_weight` | RepresentationWeightSummary | 提供 measured/proxy timing 权重 |
| `network_structure_prior` | NetworkStructureContext | 提供网络结构位置和时间重要性先验 |
| `boundary_evidence` | join / consistency checks | 防止错误 stable grouping |

禁止作为主证据：

- kernel name；
- raw cluster id；
- workload label；
- trace order；
- grid/block shape 的孤立值；
- 未带 provenance 的人工判断；
- fixture-only artifact。

这些字段可以进入 debug note 或 provenance，但不能单独决定 regime。

## 5. Regime 的最小输出字段

每个 regime row 至少包含：

| 字段 | 要求 |
|---|---|
| `regime_id` | 稳定、可复现 |
| `family_id` | 所属 family |
| `source_anchor_ids` | 来源 anchors |
| `source_record_ids` | 来源 member records |
| `hardware_pattern_summary` | dominant axis、secondary axes、axis confidence |
| `network_role_summary` | network role、scope、confidence、source |
| `temporal_importance_summary` | measured timing、network prior、importance provenance |
| `validation_role` | `main-object`、`review-object` 或 `constraint-object` |
| `primary_lane_id` | C 线主要验证方向 |
| `secondary_lane_hints` | 可选辅助方向 |
| `merge_reason` | 为什么这些 anchors 可以合并 |
| `split_reason` | 为什么没有和相邻候选合并 |
| `boundary_status` | `stable`、`provisional`、`boundary` 或 `blocked` |
| `evidence_status` | `claim_bearing`、`fixture_non_claim_bearing` 或 blocker |

`merge_reason` 和 `split_reason` 都必须存在。没有可解释原因的 regime 不允许标记为 `stable`。

## 6. Merge Rules

多个 anchors 可以合并为同一个 regime，必须同时满足以下条件：

1. **Same family**  
   Anchors 必须属于同一个 family。不同 family 的 anchors 不允许合并为同一个 regime。

2. **Compatible hardware pattern**  
   Dominant hardware axis 相同，或 axis distribution 明确相容。例如 compute-heavy 和 compute-memory-mixed 可以相容，但 compute-heavy 和 irregular-control-heavy 不应直接合并。

3. **Compatible network role**  
   Network role 相同或在同一算法阶段中承担相近功能。例如多个 projection anchors 可以合并；projection 和 softmax 不应合并。

4. **Similar temporal importance class**  
   Measured timing weight 或 network prior 的重要性等级不能强烈冲突。例如 high-time main object 不应和 low-time constraint object 合并。

5. **Same validation role**  
   `main-object`、`review-object`、`constraint-object` 不应混合进同一个 stable regime。

6. **Representative consistency**  
   Representative record 的硬件轴和 member summary 不能明显冲突。

合并后的 regime 必须记录：

```text
merge_reason = same family + compatible hardware pattern + compatible network role + similar temporal importance + same validation role
```

如果某一项只是弱相容，应标记 `provisional`，不能标记 `stable`。

## 7. Split Rules

同一个 family 内出现以下情况时，必须拆成不同 regimes：

1. **Network role 不同且会改变验证意义**  
   例如 dense family 内的 QKV projection、attention score、FFN expand 虽然硬件机制相近，但网络位置和时间意义不同，应拆开。

2. **Temporal importance 强差异**  
   高时间贡献对象不应和低时间贡献对象合并。这里的时间重要性可以来自 measured timing、cycle proxy 或带 provenance 的 network prior。

3. **Hardware pattern 强差异**  
   同 family 内如果一个 anchor 明显 compute-heavy，另一个明显 memory-mixed 或 shared-memory-coupled，应拆开。

4. **Validation role 不同**  
   主调参对象、review object 和 constraint/regression object 必须拆开。

5. **Member evidence 显示混合行为**  
   如果同一个 PKA cluster 的 members 在硬件轴或 network role 上分裂，B 线必须拆 regime 或标 boundary，不能照搬 cluster。

6. **Lane mapping 不同**  
   如果 anchors 需要不同 primary lane，通常应拆 regime。第一版不允许一个 stable regime 有多个互相竞争的 primary lanes。

拆分后的 regime 必须记录 `split_reason`。例如：

```text
split_reason = same dense family, but attention-score role has higher temporal prior and different validation role from projection anchors
```

## 8. Boundary Rules

出现以下情况时，不允许输出 stable regime：

| 情况 | 状态 |
|---|---|
| 缺 anchor membership | `blocked_missing_anchor_membership` |
| 缺完整 12D measured representation | `blocked_missing_12d_representation` |
| Anchor members 无法 join representation rows | `blocked_unjoined_members` |
| Representative record 与 member summary 明显冲突 | `boundary_representative_mismatch` |
| Hardware axis 分布过度混合 | `boundary_mixed_hardware_pattern` |
| Network prior 与 measured timing 强冲突 | `boundary_prior_timing_conflict` |
| 无法确定 primary lane | `pending_lane_mapping` |
| 只有 fixture evidence | `fixture_non_claim_bearing` |

Boundary regime 可以输出给 review 和测试，但不能进入 claim-bearing C-line formal validation。

## 9. Regime 与 Family / Lane 的关系

Family 回答：

```text
这个对象属于哪类硬件执行机制？
```

Regime 回答：

```text
在这个 family 内，它是否是一个独立 C-line validation target？
```

Lane 回答：

```text
这个 regime 应该沿哪个 backend 方向验证？
```

因此：

- family 是硬件机制大类；
- regime 是验证对象；
- lane 是验证方向。

一个 regime 必须有一个 primary lane。可以有 secondary lane hints，但第一版 C 线只强制消费 primary lane，避免 scenario matrix 爆炸。

## 10. 例子

### 10.1 Dense Family 内必须拆开的 regimes

以下 anchors 可能都属于 `dense_compute` family：

- QKV projection；
- attention score matmul；
- output projection；
- FFN expand；
- FFN contract。

它们不能因为同属 dense family 就合并成一个 regime。合理拆分原因包括：

- network role 不同；
- temporal importance 不同；
- shape / launch scale 不同；
- validation role 不同。

### 10.2 Reduction Family 内的 stable regime

多个 softmax anchors 可以合并为同一个 regime，当它们满足：

- 同属 `reduction_normalization`；
- shared / reduction path 证据相容；
- network role 都是 softmax；
- timing importance 等级相近；
- primary lane 都是 `reduction_path_sensitive`。

### 10.3 PKA Cluster 不能直接变 regime

如果一个 PKA cluster 覆盖：

- 一个 attention score record；
- 一个 FFN dense record；
- 一个 low-time bookkeeping record；

即使 PKA 认为它们 feature 距离接近，B 线也不能直接生成一个 stable regime。它必须拆分或标 boundary。

## 11. Implementation Plan Guardrails

任何 implementation plan 如果涉及 regime builder，必须包含以下检查项：

1. Regime builder 不直接按 `source_cluster_id` 建 regime；
2. Regime builder 不直接按 kernel name / operator name 建 regime；
3. Regime builder 先约束 family，再在 family 内 merge/split；
4. Regime builder join anchor membership 和 12D representation；
5. Regime builder 支持 network_structure_prior，但标明 provenance；
6. Regime builder 输出 merge_reason 和 split_reason；
7. Regime builder 输出 boundary_status；
8. Regime builder 不把 fixture evidence 标成 claim-bearing；
9. C-line adapter 只消费 stable/provisional regimes，blocked regimes 只进入 report；
10. Tests 覆盖 merge、split、boundary 三类行为。

如果 plan 缺少这些检查项，应视为偏离本 spec。

## 12. Acceptance Criteria

### AC-1: Regime 定义清晰

Spec 和实现必须把 regime 定义为 family 内部的最小 C-line validation target。

### AC-2: Regime 不等于 cluster / kernel / operator

任何直接由 cluster id、kernel name 或 operator name 生成 stable regime 的实现都不合格。

### AC-3: Merge 有证据

合并 anchors 必须满足 same family、hardware pattern 相容、network role 相容、temporal importance 相近、validation role 一致。

### AC-4: Split 有规则

同 family 内 network role、temporal importance、hardware pattern、validation role 或 primary lane 明显不同，必须拆开。

### AC-5: Boundary 不强行解释

证据缺失、冲突或无法映射 lane 时，必须输出 blocker 或 boundary status，不能输出 stable regime。

### AC-6: Regime 可被 C 线消费

每个 stable/provisional regime 必须有 primary lane、validation role、importance provenance 和清晰 source anchors。

### AC-7: Plan 不偏离

后续 implementation plan 必须显式引用本 spec 的 merge/split/boundary guardrails。
