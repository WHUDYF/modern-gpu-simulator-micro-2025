# A 线 PKA Feature General Spec

日期：2026-04-27

## 1. 目标

这份 spec 定义 A 线复现 `PKA baseline` 时必须采用的通用行为特征契约。

它的作用是把下面三件事固定下来：

1. `PKA baseline` 的判断因素到底是哪几个；
2. 每个判断因素对应的 Nsight / profile 指标是什么；
3. 在 L1 / L2 中，哪些字段可以进入 selector，哪些只能进入 audit。

这份 spec 是 L1 和 L2 共同继承的上层契约。

---

## 2. 结论：PKA baseline 的 12 个判断因素

当前确认，A 线要复现的 PKA kernel selection baseline 应以以下 12 个因素作为主行为特征空间。

| 编号 | PKA feature | 中文含义 | PKA / Nsight metric name |
|---|---|---|---|
| `F01` | `coalesced_global_loads` | 合并全局加载 | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` |
| `F02` | `coalesced_global_stores` | 合并全局存储 | `l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum` |
| `F03` | `coalesced_local_loads` | 合并局部加载 | `l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` |
| `F04` | `thread_global_loads` | 线程级全局加载 | `smsp__inst_executed_op_global_ld.sum` |
| `F05` | `thread_global_stores` | 线程级全局存储 | `smsp__inst_executed_op_global_st.sum` |
| `F06` | `thread_local_loads` | 线程级局部加载 | `smsp__inst_executed_op_local_ld.sum` |
| `F07` | `thread_shared_loads` | 线程级共享内存加载 | `smsp__inst_executed_op_shared_ld.sum` |
| `F08` | `thread_shared_stores` | 线程级共享内存存储 | `smsp__inst_executed_op_shared_st.sum` |
| `F09` | `thread_global_atomics` | 线程级全局原子操作 | `smsp__sass_inst_executed_op_global_atom.sum` |
| `F10` | `num_instructions` | 总指令数 | `smsp__inst_executed.sum` |
| `F11` | `divergence_efficiency` | 分支发散效率 | `smsp__thread_inst_executed_per_inst_executed.ratio` |
| `F12` | `num_thread_blocks` | 线程块数量 | `launch_grid_size` |

这 12 个字段共同构成 A 线 `pka_baseline` 的主 grouping feature space。

---

## 3. 字段语义约束

### 3.1 `coalesced_*` 字段

`coalesced_global_loads`、`coalesced_global_stores`、`coalesced_local_loads`
描述的是以 coalesced sector / transaction 视角观察到的访存行为。

这些字段更接近 memory transaction behavior，
不应被简单替换为 kernel source 中的 load/store 指令数量。

### 3.2 `thread_*` 字段

`thread_global_loads`、`thread_global_stores`、`thread_local_loads`、
`thread_shared_loads`、`thread_shared_stores`、`thread_global_atomics`
描述的是线程级指令或操作计数。

这些字段更接近 per-thread executed operation behavior。

### 3.3 `num_instructions`

`num_instructions` 是总执行指令数。

它是 work-size signal，
也是 PKA / Sieve 类压缩方法都高度依赖的规模信号。

### 3.4 `divergence_efficiency`

`divergence_efficiency` 描述发散效率。

它不等同于简单的 branch count，
也不等同于 `1 - divergent_branch_ratio`，除非当前数据无法获得 PKA 原始 metric。

### 3.5 `num_thread_blocks`

`num_thread_blocks` 对应 launch grid size。

它是 kernel 规模信号，
但它不是 `grid_dim` 作为形状标签进入 grouping。
在 PKA baseline 中，允许使用的是数值化后的 thread block count，
而不是把 `grid_dim` 字符串当成 identity / shape bucket。

---

## 4. 数据状态

A 线复现 PKA baseline 的第一准则是：

**正式 baseline selector 只能使用 `measured` 字段。**

这里的 `measured` 指该字段直接来自 Nsight Compute metric、profiler 报告字段、
或 launch metadata 中对应的实测/记录值。
后处理阶段可以做格式归一化、单位转换或数值化，
但不能把近似字段、语义替代字段或人工推断字段伪装成 `measured`。

PKA feature schema 固定，
但具体 NCU source metric 可以按当前 GPU / Nsight Compute 环境解析。
也就是说：

- `pka_feature_name` 必须固定为本 spec 的 12 个字段之一；
- `canonical_metric` 记录本 spec 表中列出的标准 metric；
- `actual_source_metric` 记录当前 profile 中实际采到的 metric；
- `actual_source_metric` 必须来自 Nsight Compute query / profile report、profiler 字段或 launch metadata；
- `actual_source_metric` 必须与 `canonical_metric` 在语义上等价；
- 如果无法找到 measured source metric，则只能进入 acquisition gap report。

该解析机制只允许适配 metric 名称或 profiler 字段来源，
不允许改变 PKA 12 维 feature 的行为语义。

因此，进入 `PkaFeatureTable` 的每个 feature value 只允许一种状态：

| status | 含义 | 是否可进入正式 PKA selector |
|---|---|---|
| `measured` | 直接来自对应 Nsight metric | `Yes` |

如果某个 PKA 字段无法以 `measured` 状态获得，
该 invocation 不得生成可供 selector 消费的 `PkaFeatureRecord`，
只能生成 acquisition gap report。

### 4.1 PKA-complete mode

如果目标是声称：

**已建立正式 PKA baseline**

则进入 selector 的 PKA 字段必须为：

- `measured`

如果某个 PKA 字段无法以 `measured` 状态获得，
该次运行必须标记为 acquisition incomplete，
不能自动补齐后继续声称 `pka_complete`。

`num_thread_blocks` 也应作为实测/记录的 launch metadata 进入 feature table，
状态标记为 `measured`。
如果无法从 profiler / launch record 中取得，则只能进入 acquisition gap report，
不能进入正式 selector。

### 4.2 L1 measured-only mode

L1 阶段仍然执行同一条规则：

1. selector 输入只能来自 `measured` 字段；
2. 采不齐 12 维时，只输出 acquisition gap report；
3. gap report 只能说明缺失的 metric、kernel invocation 和 source path；
4. gap report 不得进入 selector，也不得被包装成 baseline 结果。

---

## 5. Selector 使用策略

### 5.1 允许进入 `pka_baseline` 主 grouping 的字段

在 `pka_complete` 模式下，允许进入主 grouping 的字段仅限：

- `coalesced_global_loads`
- `coalesced_global_stores`
- `coalesced_local_loads`
- `thread_global_loads`
- `thread_global_stores`
- `thread_local_loads`
- `thread_shared_loads`
- `thread_shared_stores`
- `thread_global_atomics`
- `num_instructions`
- `divergence_efficiency`
- `num_thread_blocks`

### 5.2 禁止进入 `pka_baseline` 主 grouping 的字段

下面字段不得进入 PKA baseline 主 grouping：

- `kernel_name`
- `grid_dim` string
- `block_dim` string
- `shape_hint`
- `trace_order`
- `phase_id`
- `route_primitive`
- `execution_template`
- `family_id`
- `regime_id`
- `simulator_lane_id`
- `cross_tb_offset_coverage`
- squash / batch / delta 机制特征

这些字段可以作为：

- metadata
- audit 字段
- extension guardrail
- B 线输入上下文

但不能作为 PKA baseline 主分组轴。

### 5.3 Representative selection

PKA baseline 的 representative selection 应至少记录：

- cluster id
- member ids
- representative id
- representative selection rule
- coverage count
- projected / covered execution time if available

第一版推荐 selection rule：

- `first_chronological`

原因是 PKA 论文中也将 first chronological 作为 practical selection choice。

---

## 6. Feature table schema

建议通用 feature table 的每一行结构如下：

```json
{
  "record_id": "string",
  "kernel_invocation_id": "string",
  "kernel_name": "string",
  "source_path": "string",
  "feature_mode": "pka_complete | pka_l1_measured_only",
  "features": {
    "coalesced_global_loads": {"value": 0.0, "status": "measured", "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "actual_source_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"},
    "coalesced_global_stores": {"value": 0.0, "status": "measured", "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "actual_source_metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"},
    "coalesced_local_loads": {"value": 0.0, "status": "measured", "canonical_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "actual_source_metric": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"},
    "thread_global_loads": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed_op_global_ld.sum", "actual_source_metric": "smsp__inst_executed_op_global_ld.sum"},
    "thread_global_stores": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed_op_global_st.sum", "actual_source_metric": "smsp__inst_executed_op_global_st.sum"},
    "thread_local_loads": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed_op_local_ld.sum", "actual_source_metric": "smsp__inst_executed_op_local_ld.sum"},
    "thread_shared_loads": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed_op_shared_ld.sum", "actual_source_metric": "smsp__inst_executed_op_shared_ld.sum"},
    "thread_shared_stores": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed_op_shared_st.sum", "actual_source_metric": "smsp__inst_executed_op_shared_st.sum"},
    "thread_global_atomics": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__sass_inst_executed_op_global_atom.sum", "actual_source_metric": "smsp__sass_inst_executed_op_global_atom.sum"},
    "num_instructions": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__inst_executed.sum", "actual_source_metric": "smsp__inst_executed.sum"},
    "divergence_efficiency": {"value": 0.0, "status": "measured", "canonical_metric": "smsp__thread_inst_executed_per_inst_executed.ratio", "actual_source_metric": "smsp__thread_inst_executed_per_inst_executed.ratio"},
    "num_thread_blocks": {"value": 0.0, "status": "measured", "canonical_metric": "launch_grid_size", "actual_source_metric": "launch_grid_size"}
  },
  "metadata": {
    "trace_order": 0,
    "grid_dim": "string",
    "block_dim": "string",
    "exec_time_or_cycle_observed": 0.0
  }
}
```

---

## 7. L1 与 L2 的使用差异

### 7.1 L1

L1 可以生成 `pka_l1_measured_only` feature table，
但 A 线正式 selector 仍必须以 `measured` 字段为唯一输入准则。

目的：

- 建立 extractor；
- 建立 audit；
- 跑通 measured-only selector；
- 验证 B 线消费接口。

L1 不负责证明完整 PKA fidelity。
如果 L1 暂时无法采齐 12 维 measured 字段，
该对象只能进入 acquisition gap report，
不能被当作正式 PKA baseline 的替代输入。

### 7.2 L2

L2 应逐步推进到 `pka_complete` feature table。

目的：

- 做大规模 compression quality evaluation；
- 比较 baseline 与 extension；
- 支撑正式结论。

如果 L2 无法为某些对象采齐 12 维 measured 字段，
这些对象必须从正式 selector 输入中排除，
不能声称完成了严格 PKA baseline 复现。

---

## 8. 验收标准

### AC-1：字段集合完全固定

任何 PKA baseline 结果都必须使用本 spec 中列出的 12 个字段作为候选主特征集合。

### AC-2：字段来源可追踪

每个字段必须记录：

- value
- status
- source

### AC-3：selector 字段可审计

每次运行 `pka_baseline` selector 时，必须输出实际进入 selector 的字段列表。

### AC-4：禁止字段不可进入 baseline grouping

`kernel_name`、`grid_dim`、`block_dim`、family / regime / lane 字段，以及 compression-side 机制字段不得进入 `pka_baseline` 主 grouping。

### AC-5：L1 / L2 结论分级

只有当 selector 使用的字段均为 `measured` 时，才允许标记为：

- `pka_complete`

如果任何 PKA 字段没有 measured value，
该 invocation 必须标记为：

- `acquisition_incomplete`

---

## 9. 参考来源

本 spec 的 12 个字段来自 PKA 论文中用于 Principal Kernel Selection 的 microarchitecture-agnostic characteristics 表。

参考：

- `Principal Kernel Analysis: A Tractable Methodology to Simulate Scaled GPU Workloads`, MICRO 2021, Table 2.
