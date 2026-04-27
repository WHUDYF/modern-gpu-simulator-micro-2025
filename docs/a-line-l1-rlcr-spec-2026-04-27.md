# A 线 L1 RLCR Spec

日期：2026-04-27

## 1. 目标

这份 spec 定义 A 线第一轮 `L1 RLCR` 的任务边界、输入、输出和验收标准。

当前 L1 的核心目标不是证明 A 线 compression 效果最好，
而是先证明：

**PKA baseline 的输入、特征、anchor 输出，以及 B 线消费接口能够在一组小而可解释的 kernel 上稳定闭环。**

因此 L1 是：

- correctness gate
- feature sanity gate
- downstream interface gate

而不是：

- compression quality gate
- large-scale benchmark evaluation
- extension superiority proof

---

## 2. 为什么 L1 要独立跑 RLCR

L1 和 L2 关注的问题不同。

L1 关注：

- 输入字段是否可信；
- PKA 12 维特征能否被稳定抽取；
- baseline selector 是否错误依赖 `kernel_name` / `grid_dim` / `block_dim`；
- `Representative Anchor Table` 是否能被 B 线消费；
- B 线 family / regime / writeback 是否能在小集合上跑通。

L2 关注：

- 压缩率；
- top-k coverage；
- cluster 内方差；
- 大样本稳定性；
- baseline 与 extension 的统计差异。

如果 L1 和 L2 混在一轮 RLCR 中，
很容易出现两个问题：

1. L1 的接口问题被 L2 的数据规模问题掩盖；
2. L2 的 compression 结果不稳定时，很难判断根因来自 A 线特征、selector、数据采集，还是 B 线消费逻辑。

因此，当前推荐顺序是：

`L1 RLCR -> L2 RLCR`

其中 L2 必须继承 L1 的 schema、feature extractor 和 anchor 输出契约。

---

## 3. L1 的输入范围

L1 使用已有的 `L1 基础验证集 Manifest` 作为输入清单：

- `L1_MB_*`: canonical microbench
- `L1_RD_*`: 少量 Rodinia / benchmark kernel
- `L1_AI_*`: mini-transformer target kernels

参考文件：

- `docs/a-line-l1-validation-manifest-2026-04-26.md`
- `experiments/baseline_diagnosis/schemas/kernel_validation_manifest_schema.json`

### 3.1 第一批必须接入对象

第一批只要求接入 P0 对象：

- `l1_bw_32f`
- `l2_bw_32f`
- `mem_bw`
- `mem_lat`
- `shared_bw`
- `MaxFlops`
- Rodinia `nn`
- mini-transformer `gemm_tiled`
- mini-transformer `attention_score`
- mini-transformer `softmax_kernel`

### 3.2 第二批可选对象

在第一批跑通后，再接入：

- `shared_lat`
- `atomic_add_bw`
- `atomic_add_lat`
- Rodinia `backprop`
- mini-transformer `context_mul`
- mini-transformer `layernorm_kernel`
- mini-transformer `residual_add`

### 3.3 输入约束

L1 允许使用已有仓库结果，
但不能让不同来源的原始格式直接进入 PKA selector。

原因是 L1 的输入来自：

- microbench JSON
- Rodinia 本地结果
- mini-transformer full JSON / feature sources
- 后续可能补充的 NCU CSV

这些来源的字段命名、粒度和可信度并不一致。
因此 L1 必须先经过一层工程适配。

这里明确区分两类对象：

1. `KernelValidationRecord`
2. `PkaFeatureRecord`

其中：

- `KernelValidationRecord` 是验证集对象，用于溯源、审计和回归；
- `PkaFeatureRecord` 是 PKA baseline selector 的真正输入。

`KernelValidationRecord` 不是 PKA 方法本身的一部分，
也不代表 PKA 论文内部存在同名处理步骤。
它只是我们为了在混合来源数据上复现 PKA baseline 而引入的工程输入适配层。

#### 3.3.1 `KernelValidationRecord`

`KernelValidationRecord` 至少包含：


- `validation_id`
- `dataset_level`
- `source_type`
- `benchmark_name`
- `kernel_or_case`
- `kernel_invocation_id`
- `kernel_name`
- `exec_time_or_cycle_observed`
- `expected_behavior_axis`
- `pka_feature_vector`
- `feature_status`
- `source_path`

各字段作用如下：

| 字段 | 作用 | 是否允许进入 PKA selector |
|---|---|---|
| `validation_id` | L1 验证对象的稳定 id，用于测试、报告和错误定位 | `No` |
| `dataset_level` | 标明对象属于 `L1` 还是后续 `L2`，防止混用验收标准 | `No` |
| `source_type` | 标明来源类型，例如 microbench、Rodinia、AI workload | `No` |
| `benchmark_name` | 记录 benchmark 名称，用于人工审查和报告 | `No` |
| `kernel_or_case` | 记录该对象对应的 kernel 或 benchmark case 名称 | `No` |
| `kernel_invocation_id` | invocation 级唯一标识，用于 membership 和 writeback | `Identity only` |
| `kernel_name` | 原始 kernel 名称，用于溯源和报告 | `No` |
| `exec_time_or_cycle_observed` | 记录实测时间或周期，用于 weight / audit，不作为 PKA 主特征 | `No` |
| `expected_behavior_axis` | 人工预期行为轴，用于 sanity check，不作为聚类标签 | `No` |
| `pka_feature_vector` | PKA 12 维特征容器，后续转换成 `PkaFeatureRecord.features` | `Only contained 12 features` |
| `feature_status` | 记录 12 维特征是否均已实测；未采齐时标记 acquisition incomplete | `Audit only` |
| `source_path` | 原始数据路径，用于复现和追踪问题 | `No` |

关键约束：

- `expected_behavior_axis` 只能用于 sanity check，不能用于 grouping；
- `kernel_name` 只能用于溯源，不能用于 PKA baseline 主 grouping；
- `source_type` / `benchmark_name` / `dataset_level` 不能进入 selector；
- `pka_feature_vector` 中只有通用 PKA feature spec 定义的 12 个字段可以进入 selector。

#### 3.3.2 `PkaFeatureRecord`

`PkaFeatureRecord` 是 selector 的真正输入对象。

它从 `KernelValidationRecord` 派生得到，
只保留：

- 最小 identity
- PKA 12 维 feature values
- 每个 feature 的状态
- 必要的 audit metadata

建议结构如下：

```json
{
  "record_id": "L1_MB_01",
  "kernel_invocation_id": "l1_bw_32f#1",
  "feature_mode": "pka_l1_measured_only",
  "features": {
    "coalesced_global_loads": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"},
    "coalesced_global_stores": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"},
    "coalesced_local_loads": {"value": 0.0, "status": "measured", "source": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"},
    "thread_global_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_global_ld.sum"},
    "thread_global_stores": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_global_st.sum"},
    "thread_local_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_local_ld.sum"},
    "thread_shared_loads": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_shared_ld.sum"},
    "thread_shared_stores": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed_op_shared_st.sum"},
    "thread_global_atomics": {"value": 0.0, "status": "measured", "source": "smsp__sass_inst_executed_op_global_atom.sum"},
    "num_instructions": {"value": 0.0, "status": "measured", "source": "smsp__inst_executed.sum"},
    "divergence_efficiency": {"value": 0.0, "status": "measured", "source": "smsp__thread_inst_executed_per_inst_executed.ratio"},
    "num_thread_blocks": {"value": 0.0, "status": "measured", "source": "launch_grid_size"}
  },
  "metadata": {
    "kernel_name": "l1_bw_32f",
    "source_path": "experiments/baseline_diagnosis/results/microbench/l1_bw_32f.json",
    "expected_behavior_axis": "L1 bandwidth / coalesced load-heavy"
  }
}
```

Selector 只能读取：

- `record_id`
- `kernel_invocation_id`
- `features`
- `feature_mode`

Selector 不得使用：

- `metadata.kernel_name`
- `metadata.source_path`
- `metadata.expected_behavior_axis`

#### 3.3.3 为什么需要两层对象

两层对象的目的不是增加方法复杂度，
而是防止 PKA baseline 被工程 metadata 污染。

`KernelValidationRecord` 解决：

- 这个验证对象来自哪里；
- 预期行为是什么；
- 原始数据是否可追踪；
- 字段状态是否可信。

`PkaFeatureRecord` 解决：

- 哪些数值真正进入 PKA behavior feature space；
- selector 实际使用了哪些字段；
- 当前结果是 `pka_complete` 还是 `pka_l1_measured_only`。

因此，L1 的数据流应固定为：

`raw local result -> KernelValidationRecord -> PkaFeatureRecord -> pka_baseline selector`

---

## 4. PKA 12 维特征要求

L1 必须以 PKA 12 维信号作为 feature extraction 的目标字段。

通用字段契约见：

- `docs/a-line-pka-feature-general-spec-2026-04-27.md`

目标字段如下：

| 字段 | 含义 | PKA / Nsight metric name | L1 处理要求 |
|---|---|---|
| `coalesced_global_loads` | 合并全局加载 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `coalesced_global_stores` | 合并全局存储 | `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `coalesced_local_loads` | 合并局部加载 | `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_loads` | 线程级全局加载 | `smsp__inst_executed_op_global_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_stores` | 线程级全局存储 | `smsp__inst_executed_op_global_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_local_loads` | 线程级局部加载 | `smsp__inst_executed_op_local_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_shared_loads` | 线程级共享内存加载 | `smsp__inst_executed_op_shared_ld.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_shared_stores` | 线程级共享内存存储 | `smsp__inst_executed_op_shared_st.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `thread_global_atomics` | 全局原子操作 | `smsp__sass_inst_executed_op_global_atom.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `num_instructions` | 总指令数 | `smsp__inst_executed.sum` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `divergence_efficiency` | 分支发散效率 | `smsp__thread_inst_executed_per_inst_executed.ratio` | 必须通过 NCU 实测；缺失则标记 acquisition incomplete |
| `num_thread_blocks` | 线程块数量 | `launch_grid_size` | 必须从 profiler / launch metadata 记录；状态标记为 `measured` |

### 4.1 字段状态

进入 `PkaFeatureRecord.features` 的每个字段必须带状态：

- `measured`

如果任意 PKA 字段没有 measured value，
该 invocation 不得生成可供 selector 消费的 `PkaFeatureRecord`，
只能进入 acquisition gap report。

L1 spec 只定义 `measured` feature status，
避免实现层把替代字段误接入正式 baseline。

### 4.2 Selector 字段策略

L1 阶段正式 selector 只允许使用本 12 维字段中状态为：

- `measured`

的字段。

但每次 selector 运行必须输出：

- 实际使用字段列表；
- 每个使用字段的状态；
- `feature_mode`: `pka_l1_measured_only` 或 `pka_complete`。

未采集到的字段必须保留在 acquisition gap report 中，
用于说明当前 NCU acquisition 还缺什么。
gap report 不得进入正式 `pka_baseline` 主 grouping。

禁止进入 `pka_baseline` 主 grouping 的字段包括：

- `kernel_name`
- `grid_dim` string
- `block_dim` string
- `shape_hint`
- `trace_order`
- family / regime / lane 字段
- squash / batch / delta 机制字段

### 4.3 L1 的最低通过条件

第一版 L1 的最低通过条件与正式 selector 条件一致：

- PKA 12 维字段必须全部以 `measured` 状态稳定生成；
- `num_thread_blocks` 必须来自 profiler / launch metadata 记录；
- 任一字段未采齐时，该 invocation 只能进入 acquisition gap report。

如果 12 维 measured feature table 无法稳定生成，
则 L1 不应继续进入 selector / B 线消费阶段。

### 4.4 Stage-gate 执行约束

L1 必须按 stage-gate 执行，
不能把后续步骤当作“尽力而为”的 smoke test。

执行顺序固定为：

1. 先完成 trace / NCU acquisition；
2. 再生成 12 维 measured `PkaFeatureTable`；
3. 只有 feature table 通过完整性检查后，才能运行 `pka_baseline` selector；
4. 只有 selector 产出 `RepresentativeAnchorTable` 后，才能进入 B 线消费检查。

如果任一 P0 invocation 无法读取或生成完整 12 维 measured feature，
当前 stage 必须停止，
并回到 trace / NCU acquisition 部分继续迭代。

此时允许输出：

- `pka_feature_audit_l1.md`
- `pka_feature_audit_l1.json`
- `pka_acquisition_gap_l1.json`

但禁止输出或消费：

- `representative_anchor_table_l1.json`
- `b_line_consumption_report_l1.md`

原因是 selector 和 B 线消费都依赖已经完成的 PKA baseline 输入。
如果输入阶段没有通过，
后续 anchor / family / regime 结果都没有可信基础。

---

## 5. L1 输出产物

L1 RLCR 完成后至少应产出下面 5 类文件。

### 5.1 `KernelValidationManifest`

用途：

- 固定 L1 输入对象；
- 记录每个对象的来源、状态、预期行为轴和优先级。

建议路径：

- `artifacts/a_line/l1/kernel_validation_manifest_l1.json`

### 5.2 `PkaFeatureTable`

用途：

- 将 L1 对象统一转换成 PKA 12 维 feature table。

建议路径：

- `artifacts/a_line/l1/pka_feature_table_l1.json`

### 5.3 `PkaFeatureAudit`

用途：

- 记录每个 PKA 字段是否已经获得 measured value；
- 明确哪些字段仍需要后续 NCU acquisition 补齐；
- 对未采齐对象输出 acquisition gap report，而不是补齐 feature table。

建议路径：

- `artifacts/a_line/l1/pka_feature_audit_l1.md`
- `artifacts/a_line/l1/pka_feature_audit_l1.json`
- `artifacts/a_line/l1/pka_acquisition_gap_l1.json`

### 5.4 `RepresentativeAnchorTable`

用途：

- 在 L1 feature table 上运行 `pka_baseline` selector；
- 输出 representative anchors、membership 和 weight。

建议路径：

- `artifacts/a_line/l1/representative_anchor_table_l1.json`

### 5.5 `BLineConsumptionReport`

用途：

- 验证 B 线能消费 L1 anchor；
- 输出 family / regime / writeback 最小闭环状态。

建议路径：

- `artifacts/a_line/l1/b_line_consumption_report_l1.md`

---

## 6. L1 RLCR 工作包

### Task 1：Manifest builder

目标：

- 将 `docs/a-line-l1-validation-manifest-2026-04-26.md` 中的对象转成机器可读 manifest。

输入：

- L1 manifest 文档
- 本地已有结果路径

输出：

- `kernel_validation_manifest_l1.json`

验收：

- 每个 P0 对象都有稳定 id、source path、expected behavior axis；
- manifest 能通过 `kernel_validation_manifest_schema.json`。

### Task 2：PKA feature extractor

目标：

- 从 L1 对象中抽取 PKA 12 维 feature table。
- 如果无法抽取完整 12 维 measured feature，
  则迭代 trace / NCU acquisition，
  不进入 selector。

输入：

- PKA NCU CSV / profile report
- profiler / launch metadata
- L1 manifest 中记录的 source path

输出：

- `pka_feature_table_l1.json`
- `pka_feature_audit_l1.json`
- `pka_feature_audit_l1.md`
- `pka_acquisition_gap_l1.json`

验收：

- 只有 12 维均为 measured 的对象才能进入 feature table；
- 每个进入 feature table 的 PKA 字段都有 value、status=`measured` 和 source；
- `num_instructions`、`divergence_efficiency`、`num_thread_blocks` 必须为 measured；
- 未采齐对象必须进入 acquisition gap report。
- 如果任何 P0 对象未采齐 12 维 measured feature，
  本 task 判定为 blocked on acquisition，
  后续 Task 3 / Task 4 不得执行。

### Task 3：PKA baseline selector

目标：

- 在 L1 PKA feature table 上实现或调用 `pka_baseline` selector。

前置条件：

- Task 2 必须通过；
- `pka_acquisition_gap_l1.json` 中不得存在阻塞 P0 对象；
- 输入 feature table 中每个字段 status 都必须为 `measured`。

要求：

- 主 grouping 不依赖 `kernel_name`；
- 主 grouping 不依赖 `grid_dim` / `block_dim`；
- 主 grouping 不使用 compression-side / downstream-side 字段。

输出：

- `representative_anchor_table_l1.json`

验收：

- 每个 anchor 有 explicit membership；
- 每个 anchor 有 coverage count / weight；
- 每个 anchor 有代表对象；
- 输出不包含 family / regime / lane 字段。

### Task 4：B 线消费检查

目标：

- 用 L1 anchor table 验证 B 线是否能稳定消费 A 线输出。

前置条件：

- Task 3 必须通过；
- `representative_anchor_table_l1.json` 必须存在且 schema 完整；
- 不允许直接消费 acquisition gap report 或不完整 feature table。

输出：

- `b_line_consumption_report_l1.md`

验收：

- B 线能读取 anchor table；
- 能生成或更新 family / regime 的最小对象；
- 不要求 family 结论最终正确；
- 只要求接口完整、字段齐、writeback 关系不断裂。

### Task 5：L1 regression tests

目标：

- 把 L1 的核心约束转成测试。

建议测试：

- manifest schema validation；
- PKA feature table required fields；
- selector forbidden field check；
- representative anchor output schema；
- B line consumption smoke test。

---

## 7. 明确不做的事情

L1 RLCR 不做：

- 大规模 benchmark acquisition；
- Rodinia / Altis 全量跑通；
- CUTLASS sweep；
- HeCBench 泛化；
- baseline vs extension 的统计显著性分析；
- compression ratio 最大化；
- family / regime 最终正确性证明。

这些属于 L2 或后续 B/C 线验证。

---

## 8. 验收标准

L1 RLCR 完成时，必须满足下面条件。

### AC-1：L1 manifest 可机器读取

至少所有 P0 对象被写入 JSON manifest，
并能通过 schema 检查。

### AC-2：PKA feature table 可生成

进入 feature table 的所有对象都有 PKA 12 维 measured 字段。
采不齐的 P0 对象必须进入 acquisition gap report，
不得用替代字段补齐。
如果 P0 对象仍存在 acquisition gap，
L1 必须回到 trace / NCU acquisition 继续迭代，
不得进入 selector。

### AC-3：baseline selector 不依赖禁止字段

前置条件：

- AC-2 必须通过；
- 否则 AC-3 不执行。

`pka_baseline` 主 grouping 不使用：

- `kernel_name`
- `grid_dim`
- `block_dim`
- `cross_tb_offset_coverage`
- squash boundary fields
- family / regime / lane 字段

### AC-4：anchor 输出可被 B 线消费

前置条件：

- AC-3 必须通过；
- 否则 AC-4 不执行。

B 线能读取 L1 anchor table，
并完成最小 family / regime / writeback 接口检查。

### AC-5：测试可回归

至少存在一组自动化测试或检查脚本，
能覆盖：

- manifest schema
- feature table completeness
- selector forbidden fields
- anchor output schema
- B line smoke consumption

---

## 9. 与 L2 的接口

L1 完成后，必须向 L2 输出稳定接口：

- `KernelValidationManifest` schema
- `PkaFeatureTable` schema / implicit contract
- `pka_baseline` selector contract
- `RepresentativeAnchorTable` schema / implicit contract
- L1 regression tests

L2 只允许扩大输入规模和增加 compression quality metrics，
不应重新定义 L1 已经固定的核心 schema。

---

## 10. 建议执行顺序

建议下一轮 RLCR 按下面顺序执行：

1. `Manifest builder`
2. `Trace / NCU acquisition`
3. `PKA feature extractor`
4. `PKA feature audit`
5. `Gate: 12 measured PKA features complete`
6. `pka_baseline selector`
7. `RepresentativeAnchorTable` export
8. `B line consumption smoke`
9. `L1 regression tests`

如果第 5 步失败，
必须回到第 2 步继续迭代采集，
不能继续第 6 步。

---

## 11. 简短结论

L1 RLCR 的目标可以压成一句话：

**先用一组小而可解释的 kernel，把 PKA baseline 输入、特征、anchor 输出和 B 线消费接口打稳。**

只有 L1 通过后，L2 才应该开始做大规模 acquisition 和 compression quality evaluation。
