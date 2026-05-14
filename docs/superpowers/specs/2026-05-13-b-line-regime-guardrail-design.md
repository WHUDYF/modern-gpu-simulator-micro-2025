# B 线 Regime Guardrail 设计

日期：2026-05-13

## 1. 目标

本 spec 专门定义 B 线 `regime` 的职责、输入证据、提取顺序、merge/split/boundary 规则和 implementation plan guardrails。

这份文档修正一个容易出错的理解：regime 不是“反算法拆分”。原 B 线方法一直允许并且需要按算法路径结构拆分，但使用的概念是 `Route Primitive`、`phase context` 和 `context_scope`，不是直接用 kernel name、operator string 或网络模块名称做分类。

本 spec 的核心定义是：

```text
regime = 同一 phase 内，由 family + route primitive + hardware template + shape/size regime + resource signature
         共同确定的 representative execution regime
```

它回答的问题是：

```text
在 squash 提取出的同一个 phase 里，哪些 anchors / records 属于同一段可复用的执行工作区间，
哪些必须拆成独立的 C-line validation target？
```

## 2. Regime 不是什么

Regime 不是：

- PKA cluster；
- family 的别名；
- lane 的别名；
- kernel name 或 operator name 分类；
- 只按单个数值阈值切出的区间；
- attention、FFN、embedding 这类网络模块名称；
- backend parameter scenario。

这些信息可以提供 evidence 或 provenance，但不能单独定义 stable regime。

## 3. Regime 的职责

Regime 是 B 线送进 C 线的代表执行工作区间。它比 family 更细，比单个 representative kernel 更稳。

Squash 先负责说明：

```text
这些对象属于 workload 时间轴上的哪个稳定 phase？
```

Family 再负责说明：

```text
在这个 phase 内，这些对象是否共享硬件执行机制？
```

Regime 继续说明：

```text
在这个 phase 和 family 内，它们是否共享同一段 route / template / shape / resource 组合，
从而可以作为同一个 simulator validation target？
```

Regime 必须同时表达：

1. 所属 `phase` 或 phase context；
2. 所属 `family`；
3. 算法路径角色，即 `Route Primitive`；
4. GPU 执行骨架，即 `Hardware Execution Template`；
5. shape / size 区间；
6. resource signature；
7. coverage / timing / decision weights；
8. validation role 和 lane advice；
9. boundary status。

## 4. 允许的输入证据

Regime builder 允许使用以下证据：

| 证据 | 来源 | 用途 |
|---|---|---|
| `family_id` | B 线 family builder | 在同一 phase 内限定 family 边界 |
| `phase_id` / `phase_context` | squash / anchor context / network context | 保留时间结构，防止跨 phase 误并 |
| `route_primitive` | route primitive table / network structure context | 表达算法路径角色，是 regime 主拆分维度之一 |
| `hardware_template` | family / route-template mapping / 12D hardware axes | 表达 GPU 执行骨架 |
| `shape_size_signature` | launch metadata / model shape / anchor context | 形成 shape / size regime |
| `resource_signature` | 12D hardware-axis weights / profiling summary | 检查资源敏感性是否相容 |
| `source_anchor_ids` | RepresentativeAnchorTable | 追踪 regime 覆盖哪些 anchors |
| `member_record_ids` | RepresentativeAnchorTable | 汇总成员级 evidence |
| `coverage_weight` | RepresentativeAnchorTable | 衡量覆盖贡献 |
| `algorithmic_weight` | RepresentationWeightSummary | 提供 measured/proxy timing 权重 |
| `network_structure_prior` | NetworkStructureContext | 提供 phase、route、context、时间重要性先验 |
| `boundary_evidence` | join / consistency checks | 防止错误 stable grouping |

禁止作为主证据：

- raw `source_cluster_id`；
- kernel name；
- operator string；
- workload label；
- trace order 的孤立值；
- grid/block shape 的孤立值；
- 未带 provenance 的人工判断；
- fixture-only artifact。

这些禁止项可以进入 debug note 或 provenance，但不能单独生成 stable regime。

## 5. Regime 的最小输出字段

每个 regime row 至少包含：

| 字段 | 要求 |
|---|---|
| `regime_id` | 稳定、可复现 |
| `family_id` | 所属 family |
| `source_anchor_ids` | 来源 anchors |
| `source_record_ids` | 来源 member records |
| `phase_id` / `phase_context` | 所属稳定 phase 或过渡 phase |
| `route_primitive` | 算法路径角色 |
| `hardware_template` | GPU 执行骨架 |
| `shape_size_regime` | shape / size 区间，不是孤立 shape |
| `context_scope` | workload / network 路线中的上下文 |
| `resource_signature` | register、occupancy、shared、DRAM、cache、locality、reduction 等资源签名 |
| `coverage_weight` | 覆盖权重 |
| `time_weight` | measured timing / cycle proxy / provenance-tagged prior |
| `decision_weight` | 本地决策权重 |
| `validation_role` | `main-object`、`review-object` 或 `constraint-object` |
| `primary_lane_id` | C 线主要验证方向 |
| `secondary_lane_hints` | 可选辅助方向 |
| `merge_reason` | 为什么这些 anchors 可以合并 |
| `split_reason` | 为什么没有和相邻候选合并 |
| `boundary_status` | `stable`、`weak-share`、`provisional`、`boundary` 或 `blocked` |
| `evidence_status` | `claim_bearing`、`fixture_non_claim_bearing` 或 blocker |

没有 `phase_id`、`route_primitive`、`hardware_template`、`shape_size_regime` 和 `resource_signature` 的 regime 不能标记为 `stable`。

## 6. Regime 提取顺序

Regime builder 必须按下面顺序提取对象。

### Step 1: 先按 phase context 切开

不同稳定 phase 默认不直接合并 regime。过渡 phase 中的对象优先单独观察或标 `provisional`。

原因：phase 表示 workload 时间结构。即使两个 kernel 名字或硬件模板相似，如果它们出现在不同 phase，其 simulator reasoning lane 也可能不同。

### Step 2: 在同一 phase 内先形成 family 候选

Family 不应在全 workload 上脱离 phase 直接展开。更稳的做法是：

```text
squash 提取 phase
  -> 在每个 phase 内做 family / shared mechanism 判断
  -> 再在 phase + family 内提取 regime
```

原因：family 是共享机制组织层，但共享机制是否能复用通常依赖 phase 上下文。同一个硬件模板跨 phase 出现时，可以记录 family-level reuse，但 regime 层默认仍保留 phase 边界。

### Step 3: 在同一 phase + family 内按 Route Primitive 切第一刀

`Route Primitive` 回答：

```text
这个对象在 workload 主计算路径中扮演什么算法角色？
```

这是 regime 的正式拆分维度之一。它不是 operator string，而是归一化后的算法路径角色。

示例 primitive 包括：

- `Dense Projection/Transform`
- `Pairwise Score`
- `Reduction / Normalize`
- `Weighted Aggregation`
- `Elementwise Fusion`

在同一 phase 中，primitive 不同的对象默认不生成同一个 stable regime。它们可以进入 family-level weak sharing，但 regime 层必须保留边界。

### Step 4: 在同一 phase + family + Route Primitive 内校验 Hardware Execution Template 相容性

`Hardware Execution Template` 回答：

```text
这个对象在 GPU 上主要通过什么执行骨架实现？
```

示例 template 包括：

- `Dense Tiled Compute`
- `Reduction Template`
- `Streaming Aggregation Template`
- `Elementwise Template`

这一步不是重新定义 family，也不是在 family 之后再引入一个完全独立的分类体系。它的职责是做 template compatibility check，从而保证 `family` 和 `regime` 保持相对独立：

```text
family = phase 内可共享 simulator reasoning 的机制组
hardware template = 当前 route primitive 的具体 GPU 执行骨架
regime = family 内可以送入 C 线验证的具体执行工作区间
```

判断规则：

1. 如果某个 family 已经唯一约束一个 template，Step 4 只确认 template 字段，不额外拆分。
2. 如果同一 family 内允许多个 template，Step 4 必须按 template 相容性拆分或标 boundary。
3. 如果 template evidence 缺失或冲突，regime 不能标记为 `stable`。
4. 同 primitive 内如果 template 明显不同，默认拆成不同 regimes。

这样做的目的不是增加概念，而是防止两种错误：

- 把 `family` 做成 `Hardware Execution Template` 的别名，导致 family 失去共享机制组织层的作用；
- 只看 family，不检查具体 template，导致同一 family 内不同 GPU 执行骨架被误合并为一个 stable regime。

### Step 5: 在同一 primitive + template 内形成 shape / size regime

Regime 不是单个孤立 shape，而是一段 shape / size 空间里的典型工作方式。

实现时应把 shape 处理成区间或类别，例如：

- small / medium / large sequence length；
- projection-like dense region；
- expansion-like dense region；
- row-wise normalization region；
- streaming aggregation region。

Grid/block shape、M/N/K、sequence length、head dimension、batch size 等不能单独决定 regime；只有当它们共同定义了可复用的 shape / size 区间时，才作为 regime 字段。

### Step 6: 用 resource signature 做最后检查

即使对象已经同 phase、同 route primitive、同 hardware template、shape/size 相近，如果 resource signature 明显不同，也必须继续拆分或标 boundary。

常见 resource signature 包括：

- register / occupancy sensitive；
- shared-memory coupled；
- DRAM bandwidth / DRAM pressure；
- cache-capacity sensitive；
- locality / L1-resident；
- reduction / synchronization sensitive；
- irregular control / atomic sensitive。

这一步防止把“算法路径相近但后端响应不同”的对象强行并成一个 regime。

## 7. Merge Rules

多个 anchors 可以合并为同一个 stable regime，必须同时满足：

1. **Compatible phase context**
   同一稳定 phase，或有明确证据说明跨 phase 可以共享同一 execution regime。

2. **Same family within that phase**
   在同一 phase 内属于同一个 family。不同 family 不合并为同一个 regime。

3. **Same or compatible Route Primitive**
   Primitive 相同最稳。Primitive 不同只能进入 weak-share 或 boundary，不能默认 stable merge。

4. **Same or compatible Hardware Execution Template**
   Template 是 compatibility check。若 family 已唯一约束 template，这一项只做确认；若同一 family 内存在多个 template，template 不同默认拆分或标 boundary。

5. **Similar shape / size regime**
   对象落在相近 shape / size 区间，而不是只碰巧有相同 kernel name。

6. **Compatible resource signature**
   主导资源压力和敏感性相容。

7. **Similar weight / decision role**
   coverage、time、decision role 不能强烈冲突。

8. **Representative consistency**
   representative record 与 member summary 不能明显冲突。

合并后的 regime 必须记录：

```text
merge_reason = compatible phase + same family within phase + compatible route primitive
             + compatible hardware template + similar shape/size regime
             + compatible resource signature + similar decision role
```

如果某一项只是弱相容，应标记 `weak-share` 或 `provisional`，不能标记 `stable`。

## 8. Split Rules

同一 family 内出现以下情况时，必须拆成不同 regimes 或标 boundary：

1. **Phase context 不同**
   不同主 phase 默认不合并。过渡 phase 优先单独观察。

2. **Route Primitive 不同**
   Primitive 是算法路径层面的正式拆分因素。不同 primitive 不应直接生成同一个 stable regime。

3. **Hardware Execution Template 不同**
   Template 不同表示 GPU 执行骨架不同，必须拆分。

4. **Shape / size regime 不同**
   同 primitive + template 内，如果 shape / size 区间不同到会改变后端响应，应拆分。

5. **Resource signature 不同**
   例如一个 register-limited，另一个 shared-memory-coupled；一个 locality-dominated，另一个 DRAM-pressure-dominated，应拆分。

6. **Time / decision role 强差异**
   high-time main object 不应和 low-time constraint object 合并。

7. **Primary lane 不同**
   第一版不允许一个 stable regime 有多个互相竞争的 primary lanes。

8. **Member evidence 显示混合行为**
   如果同一个 PKA cluster 的 members 在 phase、route、template、shape 或 resource signature 上分裂，B 线必须拆 regime 或标 boundary，不能照搬 cluster。

9. **Representative 不能代表 member summary**
   如果 representative record 的 phase/route/template/resource/timing 与 member summary 明显不一致，不能生成 stable merged regime。

拆分后的 regime 必须记录 `split_reason`，例如：

```text
split_reason = same family, but route primitive and resource signature differ;
               objects should share family-level reasoning only, not one stable regime
```

## 9. Boundary Rules

出现以下情况时，不允许输出 stable regime：

| 情况 | 状态 |
|---|---|
| 缺 anchor membership | `blocked_missing_anchor_membership` |
| 缺完整 12D measured representation | `blocked_missing_12d_representation` |
| Anchor members 无法 join representation rows | `blocked_unjoined_members` |
| 缺 phase / route / template / shape / resource 任一关键字段 | `blocked_missing_regime_basis` |
| Representative record 与 member summary 明显冲突 | `boundary_representative_mismatch` |
| PKA cluster 内部 phase/route/template 混合 | `boundary_mixed_structural_context` |
| Hardware axis 或 resource signature 过度混合 | `boundary_mixed_resource_signature` |
| Network prior 与 measured timing 强冲突 | `boundary_prior_timing_conflict` |
| 无法确定 primary lane | `pending_lane_mapping` |
| 只有 fixture evidence | `fixture_non_claim_bearing` |

Boundary regime 可以输出给 review 和测试，但不能进入 claim-bearing C-line formal validation。

## 10. Regime 与 Family / Lane 的关系

Squash / phase 回答：

```text
这个对象出现在 workload 时间轴上的哪个稳定上下文里？
```

Family 回答：

```text
在这个 phase 内，这些对象是否共享硬件执行机制和 simulator reasoning family？
```

Regime 回答：

```text
在这个 phase 和 family 内，它们是否共享同一段 route / template / shape / resource 执行工作区间？
```

Lane 回答：

```text
这个 regime 应该沿哪个 backend 参数方向验证？
```

因此：

- phase 是时间结构层；
- family 是 phase 内的共享机制组织层；
- regime 是代表执行工作区间；
- lane 是验证方向。

一个 regime 必须有一个 primary lane。可以有 secondary lane hints，但第一版 C 线只强制消费 primary lane，避免 scenario matrix 爆炸。

## 11. 例子

### 11.1 同 family 内仍然必须保留 route / shape / resource regimes

同属 `dense_compute` 或 `dense_tiled_backbone` family 的 anchors 可以共享 dense tiled execution template，但仍可能需要多个 regimes：

| Regime evidence | 为什么独立 |
|---|---|
| `Phase A + Dense Projection/Transform + Dense Tiled Compute + projection-like shape + register-limited` | front projection path，shape 和 phase 稳定 |
| `Phase B + Pairwise Score + Dense Tiled Compute + pairwise-score shape + shared-memory-coupled` | route primitive 和 resource signature 与 generic projection 不同 |
| `Phase C + Dense Projection/Transform + Dense Tiled Compute + expansion-like shape + register-sensitive` | route primitive 可相近，但 shape/size 和 time/decision role 不同 |

这里不是按 kernel name 拆，也不是按网络模块名拆，而是按 phase、route primitive、hardware template、shape/size 和 resource signature 拆。

### 11.2 同 attention 路线内不一定同 regime

同属 attention 上层路线的对象可能需要拆开：

| Route Primitive | Hardware Template | 结论 |
|---|---|---|
| `Pairwise Score` | `Dense Tiled Compute` | dense score regime |
| `Reduction / Normalize` | `Reduction Template` | reduction regime |
| `Weighted Aggregation` | `Streaming Aggregation Template` | streaming aggregation regime |

原因是它们 route primitive 和 hardware template 都不同。上层同属 attention 不能作为合并 regime 的理由。

### 11.3 PKA Cluster 不能直接变 regime

如果一个 PKA cluster 覆盖：

- 一个 `Dense Projection/Transform` record；
- 一个 `Pairwise Score` record；
- 一个 low-time constraint record；

即使 PKA 认为它们 feature 距离接近，B 线也不能直接生成一个 stable regime。它必须按 phase / route / template / shape / resource 拆分，或标 boundary。

## 12. Implementation Plan Guardrails

任何 implementation plan 如果涉及 regime builder，必须包含以下检查项：

1. Regime builder 不直接按 `source_cluster_id` 建 regime；
2. Regime builder 不直接按 kernel name / operator name 建 regime；
3. Regime builder 先使用 squash/phase context，再在每个 phase 内展开 family / regime / lane；
4. Regime builder join anchor membership 和 12D representation；
5. Regime builder 支持 NetworkStructureContext，并将其归一化为 phase、route primitive、context scope 和 temporal prior；
6. Regime builder 输出 merge_reason 和 split_reason；
7. Regime builder 输出 boundary_status；
8. Regime builder 不把 fixture evidence 标成 claim-bearing；
9. C-line adapter 只消费 stable/provisional regimes，blocked regimes 只进入 report；
10. Tests 覆盖 phase split、route split、template split、shape/resource split、merge、boundary 六类行为。

如果 plan 缺少这些检查项，应视为偏离本 spec。

## 13. Acceptance Criteria

### AC-1: Regime 定义清晰

Spec 和实现必须把 regime 定义为 phase 内部由 family、Route Primitive、Hardware Execution Template、shape/size regime 和 resource signature 共同确定的 representative execution regime。

### AC-2: Regime 不等于 cluster / kernel / operator

任何直接由 cluster id、kernel name 或 operator name 生成 stable regime 的实现都不合格。

### AC-3: Route Primitive 是正式拆分维度

实现必须支持按 Route Primitive 在同一 phase + family 内拆 regime。Route Primitive 是算法路径结构，不是 raw operator string。

### AC-4: Hardware Template 和 Resource Signature 共同约束 regime

同 primitive 内 template 或 resource signature 明显不同，必须拆分或标 boundary。

### AC-5: Shape / Size 是区间，不是孤立值

实现必须把 shape / size 作为 regime 区间或类别处理，不能只按单个 grid/block 数值做硬切分。

### AC-6: Boundary 不强行解释

证据缺失、冲突或无法映射 lane 时，必须输出 blocker 或 boundary status，不能输出 stable regime。

### AC-7: Regime 可被 C 线消费

每个 stable/provisional regime 必须有 primary lane、validation role、importance provenance 和清晰 source anchors。

### AC-8: Plan 不偏离

后续 implementation plan 必须显式引用本 spec 的 squash/phase-first 以及 family / route / template / shape / resource guardrails。
