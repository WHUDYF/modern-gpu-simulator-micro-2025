# A 线 L2 RLCR Spec

日期：2026-04-27

## 1. 目标

这份 spec 定义 A 线第二轮 `L2 RLCR` 的任务边界、数据集分层、stage-gate、输出产物和验收标准。

L2 的核心目标是：

**在继承 L1 measured-only PKA 输入契约的前提下，把 A 线从小规模正确性闭环扩展到真实 benchmark kernel 和 compression evaluation。**

L2 不是重新定义 PKA baseline。
L2 只做三件事：

1. 扩大可采集 kernel / invocation 数量；
2. 在更大数据集上运行 PKA baseline selector；
3. 评估 compression quality，并为后续 A-line extension 对比准备基线。

---

## 2. 与 L1 的继承关系

L2 必须继承 L1 已固定的硬约束：

- PKA feature space 仍然是 12 维；
- 进入 selector 的每个 feature status 必须是 `measured`；
- `kernel_name`、`grid_dim`、`block_dim`、`source_path`、`expected_behavior_axis` 不得进入 selector；
- 任一 invocation 无法获得完整 12 维 measured feature 时，不得生成可供 selector 消费的 `PkaFeatureRecord`；
- 任一 P0 acquisition gap 未解决时，不得进入 anchor selection 和 compression evaluation。

L2 可以扩大输入规模和增加 compression metrics，
但不能放宽 L1 的输入规则。

L2 的基本数据流固定为：

```text
benchmark source
  -> build / run
  -> trace / NCU acquisition
  -> 12 measured PKA features
  -> PkaFeatureTable
  -> pka_baseline selector
  -> RepresentativeAnchorTable
  -> compression evaluation
```

---

## 3. L2 数据集分层

L2 数据集不是静态数据文件，
而是可以编译、运行、profile 并转成 PKA 12 维 measured feature 的 benchmark / kernel 来源。

### 3.1 Tier 1：Core Benchmark Set

第一阶段主集合：

- `Rodinia`
- `Altis`

角色：

- 作为 L2 的第一批真实 benchmark kernel；
- 验证 L1 pipeline 在非 microbench kernel 上是否仍然稳定；
- 形成第一版 compression evaluation 的主干。

推荐起步对象：

- Rodinia: `nn`, `backprop`, `bfs`, `lud`, `nw`
- Altis: 选择 `3 ~ 5` 个能稳定 build / run / NCU capture 的 benchmark

目标规模：

- `15 ~ 30` 个 measured kernel / invocation objects

### 3.2 Tier 2：Supplement Benchmark Set

第二阶段补充集合：

- `Parboil`
- `PolyBench/GPU`

角色：

- 补充 regular numeric、scientific kernel、stencil、sparse / medium-complexity 行为；
- 增强 behavior space 覆盖；
- 检查 selector 是否只对 Rodinia / Altis 稳定。

目标规模：

- `15 ~ 30` 个 measured kernel / invocation objects

### 3.3 Tier 3：Scale-up Set

第三阶段扩样本集合：

- `CUTLASS Profiler`

角色：

- 通过参数 sweep 快速扩展 dense compute 样本；
- 检查 compression robustness；
- 支撑 `100 ~ 300` 个 compression objects 规模的实验。

注意：

- CUTLASS 适合扩样本；
- CUTLASS 不应替代 Rodinia / Altis / Parboil / PolyBench 的行为覆盖角色。

### 3.4 Tier 4：Generalization Set

第四阶段泛化集合：

- `HeCBench`

角色：

- 在更大、更杂的 benchmark collection 上检查泛化；
- 用于 L2 后期或 L3 风格扩展验证；
- 不建议作为 L2 第一入口。

---

## 4. L2 Stage-gate

L2 必须按 stage-gate 执行。

### Gate 1：Benchmark Acquisition

每个 benchmark entry 必须记录：

- source origin；
- build command；
- run command；
- input size / arguments；
- binary path；
- output path；
- run status。

通过条件：

- build 成功；
- run 成功；
- kernel invocation 可被稳定定位；
- 运行结果可重复。

未通过时：

- 该 entry 留在 acquisition backlog；
- 不进入 NCU acquisition。

### Gate 2：Trace / NCU Acquisition

每个通过 Gate 1 的 invocation 必须采集 PKA 12 维 measured feature。

通过条件：

- 12 个 PKA feature 均有 measured value；
- 每个 feature 都记录 source metric；
- `num_thread_blocks` 来自 profiler / launch metadata record；
- 同一 invocation 的多次采集结果在可接受范围内稳定。

未通过时：

- 输出到 `pka_acquisition_gap_l2.json`；
- 回到 trace / NCU acquisition 继续迭代；
- 不进入 `PkaFeatureTable`。

### Gate 3：PkaFeatureTable Completeness

通过 Gate 2 的 invocation 才能进入 `pka_feature_table_l2.json`。

通过条件：

- 每行都有 `record_id`；
- 每行都有 `kernel_invocation_id`；
- 每行都有 `feature_mode`；
- 每行都有完整 12 维 `features`；
- 每个 feature 的 status 都是 `measured`。

未通过时：

- 停止 selector；
- 回到 feature extraction / acquisition 修复。

### Gate 4：PKA Baseline Selector

通过 Gate 3 后，才能运行 `pka_baseline` selector。

通过条件：

- selector 只读取 `record_id`、`kernel_invocation_id`、`features`、`feature_mode`；
- selector 不读取 metadata；
- 每个 anchor 有 representative；
- 每个 anchor 有 explicit membership；
- 每个 invocation 至少被一个 anchor 覆盖或被明确标记为 excluded。

未通过时：

- 不进入 compression evaluation；
- 回到 selector implementation / feature normalization 检查。

### Gate 5：Compression Evaluation

通过 Gate 4 后，才能计算 compression metrics。

通过条件：

- compression ratio 可计算；
- coverage count 可计算；
- weighted coverage 可计算；
- cluster feature variance 可计算；
- top-k coverage 可计算；
- evaluation report 能追溯到对应 anchor table。

---

## 5. L2 输出产物

L2 产物建议统一放在：

- `artifacts/a_line/l2/`

### 5.1 `L2KernelValidationManifest`

建议路径：

- `artifacts/a_line/l2/kernel_validation_manifest_l2.json`

用途：

- 记录所有 L2 候选 benchmark / kernel；
- 记录 build / run / acquisition 状态；
- 记录 benchmark 来源和运行参数；
- 区分 accepted、blocked、backlog。

### 5.2 `L2PkaFeatureTable`

建议路径：

- `artifacts/a_line/l2/pka_feature_table_l2.json`

用途：

- 保存通过 Gate 2 / Gate 3 的 12 维 measured PKA records；
- 作为 PKA baseline selector 的唯一正式输入。

### 5.3 `L2PkaFeatureAudit`

建议路径：

- `artifacts/a_line/l2/pka_feature_audit_l2.json`
- `artifacts/a_line/l2/pka_feature_audit_l2.md`
- `artifacts/a_line/l2/pka_acquisition_gap_l2.json`

用途：

- 记录每个候选 invocation 是否采齐 12 维 measured feature；
- 记录缺失的 metric、source path、benchmark、run command；
- 为 trace / NCU acquisition 迭代提供任务清单。

### 5.4 `L2PkaBaselineAnchorTable`

建议路径：

- `artifacts/a_line/l2/pka_baseline_anchor_table_l2.json`

用途：

- 保存 PKA baseline selector 的 representative anchors；
- 记录 anchor membership；
- 记录 coverage count / weight；
- 作为 compression evaluation 的输入。

### 5.5 `L2CompressionEvaluation`

建议路径：

- `artifacts/a_line/l2/compression_eval_l2.json`
- `artifacts/a_line/l2/compression_eval_l2.md`

用途：

- 汇总 L2 compression quality；
- 记录 baseline selector 在 L2 数据集上的压缩表现；
- 为后续 A-line extension 对比提供基线。

---

## 6. Compression Metrics

L2 第一版至少计算下面指标。

### 6.1 Compression ratio

定义：

```text
compression_ratio = original_invocation_count / selected_anchor_count
```

用途：

- 衡量 selector 将 workload 压缩到多少 representative anchors。

### 6.2 Coverage count

定义：

```text
coverage_count(anchor) = number of member invocations covered by anchor
```

用途：

- 衡量每个 anchor 覆盖多少 invocation。

### 6.3 Weighted coverage

定义：

```text
weighted_coverage(anchor) = sum(observed_time_or_cycle of member invocations)
```

用途：

- 避免只看 invocation 数量；
- 让长时间 kernel 对 compression evaluation 有更合理的权重。

### 6.4 Cluster feature variance

定义：

```text
cluster_feature_variance(anchor) = variance of normalized 12-dim PKA features inside anchor membership
```

用途：

- 检查一个 anchor 覆盖的 members 是否真的行为相近；
- 防止 compression ratio 高但 cluster 内部非常分散。

### 6.5 Top-k coverage

定义：

```text
top_k_coverage(k) = weighted coverage of top-k anchors / total weighted workload
```

用途：

- 衡量少数最重要 anchors 能覆盖多少 workload；
- 支撑后续 simulator budget 选择。

### 6.6 Acquisition success rate

定义：

```text
acquisition_success_rate = accepted_measured_records / candidate_invocations
```

用途：

- 量化 L2 数据集 bring-up 成本；
- 判断当前瓶颈在 benchmark acquisition 还是 selector / compression。

---

## 7. L2 工作包

### Task 1：L2 manifest builder

目标：

- 将 L2 候选 benchmark / kernel 写成机器可读 manifest。

输入：

- `docs/a-line-l2-acquisition-list-2026-04-26.md`
- benchmark source path
- build / run configuration

输出：

- `kernel_validation_manifest_l2.json`

验收：

- 每个 candidate entry 有 suite、benchmark、run command、priority、status；
- P0 entries 至少覆盖 Rodinia + Altis；
- manifest 能被 acquisition runner 读取。

### Task 2：L2 acquisition runner

目标：

- 对 manifest 中的 P0 / P1 entries 执行 build、run、NCU capture。

输出：

- raw NCU CSV / profile reports；
- run logs；
- acquisition status table。

验收：

- 每个 P0 entry 都有 build / run / NCU status；
- 失败 entry 有明确 failure reason；
- 成功 entry 能定位 kernel invocation。

### Task 3：L2 PKA feature extractor

目标：

- 将 raw NCU output 转成 12 维 measured `PkaFeatureRecord`。

输出：

- `pka_feature_table_l2.json`
- `pka_feature_audit_l2.json`
- `pka_feature_audit_l2.md`
- `pka_acquisition_gap_l2.json`

验收：

- 只有 12 维 measured feature 全齐的 invocation 才能进入 feature table；
- 任何 P0 acquisition gap 未解决时，不得执行 Task 4；
- 每个 feature 记录 value、status=`measured`、source metric。

### Task 4：PKA baseline selector on L2

目标：

- 在 L2 measured feature table 上运行 `pka_baseline` selector。

前置条件：

- Task 3 通过；
- P0 acquisition gap 清零；
- feature table 完整性检查通过。

输出：

- `pka_baseline_anchor_table_l2.json`

验收：

- selector 不读取 metadata；
- anchor membership 完整；
- selected anchors 可追溯到 input records。

### Task 5：L2 compression evaluation

目标：

- 在 PKA baseline anchor table 上计算 compression metrics。

前置条件：

- Task 4 通过；
- anchor table schema 完整。

输出：

- `compression_eval_l2.json`
- `compression_eval_l2.md`

验收：

- 至少输出 compression ratio、coverage count、weighted coverage、cluster feature variance、top-k coverage、acquisition success rate；
- report 能说明当前 L2 数据集规模、accepted records 数量、blocked acquisition 数量。

### Task 6：Extension comparison preparation

目标：

- 为后续 A-line extension selector 对比预留接口。

第一版不要求实现 extension comparison，
但输出 schema 应避免把 PKA baseline anchor table 写死成唯一格式。

后续新增产物：

- `extension_anchor_table_l2.json`
- `baseline_vs_extension_l2.json`
- `baseline_vs_extension_l2.md`

---

## 8. 验收标准

### AC-1：L2 manifest 可机器读取

至少 P0 entries 覆盖：

- Rodinia core；
- Altis core。

每个 entry 都有 build / run / acquisition 状态字段。

### AC-2：L2 acquisition gate 可执行

每个 P0 entry 都能被 runner 尝试执行，
并得到明确状态：

- accepted；
- blocked on build；
- blocked on run；
- blocked on NCU acquisition；
- blocked on feature completeness。

### AC-3：L2 feature table 严格 measured-only

进入 `pka_feature_table_l2.json` 的 records 必须全部满足：

- 12 维 PKA features 完整；
- 每个 feature status 为 `measured`;
- 每个 feature 有 source metric；
- 每个 record 有 source invocation。

### AC-4：Stage-gate 不可跳过

如果 P0 entry 仍存在 acquisition gap：

- 不运行 selector；
- 不输出 anchor table；
- 不输出 compression evaluation；
- 必须回到 trace / NCU acquisition 继续迭代。

### AC-5：PKA baseline selector 可在 L2 上运行

selector 输出必须包含：

- anchor id；
- representative record id；
- member record ids；
- coverage count；
- coverage weight；
- selection rule。

### AC-6：Compression evaluation 可解释

compression report 至少说明：

- candidate invocation count；
- accepted measured record count；
- selected anchor count；
- compression ratio；
- top-k weighted coverage；
- cluster variance summary；
- acquisition success rate。

---

## 9. 推荐执行顺序

第一版 L2 RLCR 推荐顺序：

1. `L2 manifest builder`
2. `Rodinia P0 acquisition`
3. `Altis P0 acquisition`
4. `Gate: 12 measured PKA features complete`
5. `L2 PKA feature table export`
6. `PKA baseline selector on L2`
7. `L2 compression evaluation`
8. `Parboil / PolyBench supplement`
9. `CUTLASS scale-up`
10. `HeCBench generalization`

如果第 4 步失败，
必须回到第 2 或第 3 步继续 trace / NCU acquisition，
不能继续第 6 步。

---

## 10. 简短结论

L2 的目标可以压成一句话：

**在 L1 已经固定 measured-only PKA 输入契约后，用 Rodinia / Altis 起步，把真实 benchmark kernel 扩展成可采集、可压缩、可评估的 A 线 compression dataset。**

L2 成功的标志不是下载了多少 benchmark，
而是有多少 invocation 能通过完整 stage-gate，
最终进入 measured-only PKA feature table 并支撑 compression evaluation。
