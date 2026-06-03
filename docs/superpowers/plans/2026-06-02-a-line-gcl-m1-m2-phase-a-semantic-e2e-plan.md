# A 线 GCL-M1/M2 Phase A 语义闭环实施计划

## 目标描述

实现 GCL Phase A 的最小语义闭环：

```text
gcl_phase_a_controlled_trace_fixture_v1
  -> canonical graph artifacts
  -> tensorization
  -> minimal RGCN contrastive training
  -> M0-compatible kernel embedding table
  -> M0 selector
  -> cluster / representative anchor / evaluation artifacts
```

本计划只实现 Phase A strict reproduction。首要目标是让代码路径与 `2026-06-01-a-line-gcl-m1-m2-phase-a-semantic-e2e-design.md` 对应，并证明 artifact 能从 trace fixture 一路流到 M0 selector 输出。该阶段不证明 embedding quality、sampling accuracy、simulator speedup，也不引入 Phase B/C 的扩展。

建议实现路径：

```text
experiments/gcl_phase_a/
  fixtures/
  artifacts/
  trace_fixture.py
  graph_builder.py
  tensorizer.py
  rgcn.py
  train.py
  embedding_export.py
  selector.py
  pipeline.py
tests/gcl_phase_a/
```

## RLCR Review Phase 修正要求

重新执行本计划时，必须优先处理上一轮 RLCR Review Phase 暴露出的 3 个 contract 问题。它们属于 Phase A strict reproduction 的一部分，不是 Phase B/C 扩展。

1. `register_version` 必须由 graph builder 稳定生成
   - graph builder 不能假设 trace operand 已经预先写成 `R4.v1.w0` 这类版本化 token。
   - 对同一 warp 内的原始寄存器名，例如 `R4`，必须维护 per-warp register version state。
   - 每次写寄存器生成新版本：`R4.v1.w0`、`R4.v2.w0`。
   - 每次读寄存器连接到当前最新版本，不生成新版本。
   - 测试必须覆盖 raw register reuse：同一个 warp 中 `R4` 被写两次时，graph 中必须产生两个不同 `register_version` node，后续读边连接到最新 producer。

2. `mem_ref` 的 address operand 不能依赖 `Raddr*` 命名
   - graph builder 不能只用 token 名字判断地址寄存器。
   - Phase A 默认规则：对 `LDG*` / `STG*` memory instruction，`source_operands` 中第一个 register operand 作为 address operand。
   - 地址寄存器可以叫 `Raddr...`，也可以是普通寄存器，例如 `R14`、`R22`。
   - 每个 memory instruction 必须生成完整 data-flow chain：

     ```text
     address register_version -> mem_ref pseudo node -> memory instruction
     ```

   - 测试必须覆盖 raw address register：`LDG.E.64.SYS R8, [R14]` 和 `STG.E.64.SYS [R22], R9` 这类输入必须生成 `mem_ref`，不能因为地址名不是 `Raddr*` 被拒绝。

3. disk-backed selector stage 必须持久化修复结果
   - `run_selector_stage_from_disk(out_dir)` 不能只返回 regenerated selector artifact。
   - 它必须把结果写回：

     ```text
     out_dir / selector_artifacts.json
     ```

   - 测试必须覆盖：删除 `selector_artifacts.json` 后调用 `run_selector_stage_from_disk(out_dir)`，函数返回合法 artifact，且磁盘上的 `selector_artifacts.json` 被重新创建。

## 验收标准

- AC-1: Phase A fixture 可以稳定生成 12 个 kernel invocation trace records
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_trace_fixture.py::test_fixture_has_expected_size`
    - 验证 `kernel_invocation_count = 12`、`trace_family_count = 3`、每个 invocation 有 2 个 warp、每个 warp 有 6 条 dynamic instruction entries。
    - 验证所有 trace entry 都包含 `kernel_invocation_id`、`trace_family`、`collection_scope`、`warp_id`、`trace_index`、`pc`、`opcode`、`active_mask`、`destination_operands`、`source_operands`、`observed_dynamic_values`、`source_entry_hash`。
  - 负向测试（预期失败）:
    - 删除任意必需字段后，fixture validator 必须拒绝输入。
    - 将 `collection_scope` 改成非 `selected_warps_fixture` 后，fixture validator 必须拒绝输入。

- AC-2: M1 graph builder 可以生成 12 个 canonical graph artifacts
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_graph_builder.py::test_builds_expected_graph_count`
    - 验证每个 graph 包含 `graph_id`、`kernel_invocation_id`、`collection_scope`、`nodes`、`edges`、`warp_partitions`、`graph_summary`、`graph_hash`。
    - 验证相同 fixture 输入重复构图会得到相同 `graph_hash`。
    - 验证 Phase A graph 至少区分 `instruction`、`register_version`、`input_variable`、`unknown_variable` 和 `pseudo` node。
    - 验证 raw register operands 会在 graph builder 内部被版本化；同一 warp 内同一寄存器多次写入不能 collapse 成同一个 `register_version` node。
  - 负向测试（预期失败）:
    - 打乱同一 warp 内的 `trace_index` 且不重新排序时，graph validator 必须发现 ordering violation。
    - 删除 `warp_partitions` 后，graph artifact validator 必须失败。
    - 人为把 control-flow edge 改成非 consecutive instruction edge，并重新计算 `graph_hash` 后，graph validator 仍必须失败。

- AC-3: `mem_ref` pseudo node 只通过 data-flow 接入，不进入 control-flow 主链
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_graph_builder.py::test_mem_ref_is_data_flow_only`
    - 验证 LDG/STG 对应的 `mem_ref` pseudo node 存在。
    - 验证 `address register_version -> mem_ref -> memory instruction` 使用 data-flow edge。
    - 验证 memory address operand 的识别基于 operand role，而不是 `Raddr*` 命名；普通寄存器地址如 `R14` / `R22` 必须能生成 `mem_ref`。
    - 验证 control-flow edge 只连接 consecutive instruction nodes。
  - 负向测试（预期失败）:
    - 如果出现 `instruction -> mem_ref` 或 `mem_ref -> instruction` 的 `control_flow` edge，测试必须失败。
    - 如果 LDG/STG 缺少 required `mem_ref` pseudo node，测试必须失败。
    - 删除 `address register_version -> mem_ref` data-flow edge，并重新计算 `graph_hash` 后，graph validator 仍必须失败。

- AC-4: Tensorization 生成 strict GCL-Sampler node feature schema
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_tensorizer.py::test_node_features_are_64_wide`
    - 验证 `node_features.shape = [node_count, 64]`。
    - 验证 `node_feature_schema = gcl_m2_phase_a_paper_node_feature_v1`。
    - 验证 `paper_reproduction_mode = strict_gcl_sampler_node_features`。
    - 验证 instruction node 使用 `concat_opcode63_normalized_pc1`，即 `[0:63)` opcode embedding + `[63]` normalized PC scalar。
    - 验证 variable node 使用 `[0:32)` token embedding、`[32:40)` dynamic value statistics、`[40:64)` zero padding。
    - 验证 pseudo node 使用 `[0:16)` token embedding、`[16:64)` zero padding。
  - 负向测试（预期失败）:
    - 在 variable node `[40:64)` 或 pseudo node `[16:64)` 写入非零值时，schema validator 必须失败。
    - instruction node 使用 trace index encoding 或其他 positional encoding 时，schema validator 必须失败。

- AC-5: Tensorization 输出 RGCN 所需结构输入
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_tensorizer.py::test_rgcn_inputs_are_complete`
    - 验证输出包含 `node_features`、`edge_index`、`edge_type`、`warp_partitions`、`graph_batch_metadata`。
    - 验证 `edge_index.shape = [2, edge_count]`，`edge_type.shape = [edge_count]`。
    - 验证 `tensor_hash` 引用 `input_graph_hash`，且同一 graph 重复 tensorization 得到相同 `tensor_hash`。
  - 负向测试（预期失败）:
    - 缺失 `input_graph_hash`、`edge_relation_schema` 或 `node_feature_schema` 时，tensor artifact validator 必须失败。
    - `edge_index` 和 `edge_type` 长度不一致时，validator 必须失败。

- AC-6: Minimal RGCN contrastive training 可以完成 forward / loss / backward / checkpoint
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_rgcn_training.py::test_minimal_training_smoke`
    - 使用 fixture graph batch 生成两个 augmented views。
    - RGCN 输入维度为 64，hidden dimension 为 128，kernel embedding dimension 为 256。
    - Projection head 输出维度为 64。
    - InfoNCE loss 可计算，optimizer step 可执行，checkpoint 可保存。
  - 负向测试（预期失败）:
    - 输入 feature width 不是 64 时，training entrypoint 必须拒绝。
    - 如果 augmentation 导致某个 warp partition 为空，训练必须 reject 或 regenerate，并记录 retry count。

- AC-7: M2 embedding export 使用 canonical non-augmented graph 和 256 维 kernel embedding
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_embedding_export.py::test_exports_m0_compatible_embedding_table`
    - 验证 embedding table 包含 12 rows。
    - 每行包含 `record_id`、`kernel_invocation_id`、`representation_mode`、`embedding_dim`、`embedding`、`source_graph_hash`、`encoder_manifest_hash`、`embedding_hash`、`weight_input`。
    - 验证 `embedding_dim = 256`。
    - 验证 export 使用 canonical graph，而不是 augmented view。
  - 负向测试（预期失败）:
    - 如果尝试导出 projection head 的 64 维 output，export validator 必须失败。
    - 如果 embedding row 缺少 `source_graph_hash` 或 `encoder_manifest_hash`，validator 必须失败。

- AC-8: M0 selector 可以消费 GCL embedding table 并输出 clustering artifacts
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_selector.py::test_selector_outputs_clusters_and_anchors`
    - 验证 selector 执行 z-score normalization、`silhouette_k`、deterministic K-Means、representative anchor selection。
    - 验证输出包含 cluster assignments、representative anchor table、silhouette report、structural evaluation artifacts。
  - 负向测试（预期失败）:
    - embedding table row 数为 0 时，selector 必须失败。
    - embedding_dim 不一致时，selector 必须失败。

- AC-9: End-to-end pipeline 可以一条命令跑完整闭环
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_pipeline.py::test_phase_a_pipeline_e2e`
    - `python -m experiments.gcl_phase_a.pipeline --out artifacts/gcl_phase_a`
    - 验证 `artifacts/gcl_phase_a` 下生成 trace fixture、graph bundle、tensor bundle、training report、checkpoint manifest、embedding table、selector artifacts。
    - 验证 disk-backed M2 export stage 可以从 `graph_bundle.json`、`tensor_bundle.json`、`checkpoint_manifest.json` 和 checkpoint 重建并写回 `embedding_table.json`。
    - 验证 disk-backed selector stage 可以从 `embedding_table.json` 重建并写回 `selector_artifacts.json`。
  - 负向测试（预期失败）:
    - 删除 graph bundle 后直接运行 M2 export，pipeline 必须失败并说明缺失 artifact。
    - 删除 embedding table 后运行 selector，pipeline 必须失败并说明缺失 artifact。
    - 删除 selector artifacts 后运行 selector repair stage，必须重新创建该 artifact；如果只 return 而不写盘，测试必须失败。

- AC-10: Phase A artifact replay deterministic
  - 正向测试（预期通过）:
    - `pytest -q tests/gcl_phase_a/test_replay.py::test_phase_a_artifacts_are_replayable`
    - 固定 random seed 后，trace fixture hash、graph hash、tensor hash、encoder manifest hash、embedding hash、selector manifest hash 可复现。
  - 负向测试（预期失败）:
    - 修改 fixture trace entry 后，graph hash / tensor hash 必须变化。
    - 修改 tensorizer schema 后，tensor hash / encoder manifest hash 必须变化。

## 路径边界

### 上界

允许的最大范围：

- 新增 `experiments/gcl_phase_a` Python package；
- 新增 fixture generator、canonical graph builder、tensorizer、minimal RGCN、contrastive training、embedding export、M0 selector 和 end-to-end pipeline；
- 新增 `tests/gcl_phase_a` 下的单元测试和 end-to-end smoke tests；
- 新增 `artifacts/gcl_phase_a` 作为默认本地输出目录；
- 实现 deterministic artifact hashing 和 manifest validators。

### 下界

最低可接受实现：

- 能生成 `gcl_phase_a_controlled_trace_fixture_v1`；
- 能从 fixture 生成 12 个 canonical graph artifacts；
- 能 tensorize 成 RGCN 输入；
- 能完成一次 minimal RGCN training smoke；
- 能导出 12-row、256 维的 M0-compatible embedding table；
- 能用 M0 selector 生成 cluster / representative anchor / evaluation artifacts。

### 允许与禁止

- 可以使用：
  - Python standard library、`numpy`、`pytest`；
  - `torch` 实现 minimal RGCN 和 projection head；
  - repo-local deterministic K-Means / silhouette implementation，避免额外依赖；
  - JSON artifacts 和 manifest hashes。
- 不可以使用：
  - full-kernel dynamic trace；
  - full-GPU trace；
  - representative SM selection；
  - instruction stream compression；
  - trace index positional encoding；
  - variable node `[40:64)` 非零扩展；
  - pseudo node `[16:64)` 非零扩展；
  - projection head output 作为 selector embedding；
  - augmented graph 覆盖 canonical graph。

## 依赖与顺序

### 里程碑

1. 里程碑 1：Fixture 和 validators
   - 实现 `trace_fixture.py`，生成 `gcl_phase_a_controlled_trace_fixture_v1`。
   - 实现 trace entry validator。
   - 写 AC-1 tests。

2. 里程碑 2：Canonical Graph Builder
   - 实现 instruction / variable / pseudo node construction。
   - 实现 per-warp register version state：write creates new version，read uses latest version。
   - 对 raw register operand 生成稳定 `register_version` node，不依赖 fixture 预先版本化 token。
   - 实现 `control_flow`、`data_source`、`data_destination` edges。
   - 保证 `mem_ref` pseudo node 只通过 data-flow 接入。
   - 对 `LDG*` / `STG*` 使用 operand role 识别 address register，不能只依赖 `Raddr*` 命名。
   - 实现 `warp_partitions`、`graph_summary`、`graph_hash`。
   - 写 AC-2 / AC-3 tests。

3. 里程碑 3：Tensorization 和 Schema Manifest
   - 实现 strict node feature schema。
   - 实现 `node_features`、`edge_index`、`edge_type`、`warp_partitions`、`graph_batch_metadata`。
   - 实现 `tensor_hash` 和 schema validators。
   - 写 AC-4 / AC-5 tests。

4. 里程碑 4：Minimal RGCN Training
   - 实现 relation-aware message passing。
   - 实现 3-layer RGCN：64 -> 128 -> 256。
   - 实现 node -> warp -> kernel readout。
   - 实现 projection head：256 -> 128 -> 64。
   - 实现 two-view augmentation 和 symmetric InfoNCE。
   - 写 AC-6 tests。

5. 里程碑 5：Embedding Export 和 Selector
   - 从 canonical non-augmented graph 导出 256 维 kernel embedding。
   - 实现 M0-compatible embedding table validator。
   - 实现 z-score normalization、silhouette_k、deterministic K-Means、representative anchor selection。
   - 写 AC-7 / AC-8 tests。

6. 里程碑 6：End-to-End Pipeline 和 Replay
   - 实现 `python -m experiments.gcl_phase_a.pipeline --out artifacts/gcl_phase_a`。
   - 保存 graph bundle、tensor bundle、training report、checkpoint manifest、embedding table、selector artifacts。
   - 实现 disk-backed repair stages：
     - `run_embedding_export_stage_from_disk(out_dir)` 必须重建并写回 `embedding_table.json`。
     - `run_selector_stage_from_disk(out_dir)` 必须重建并写回 `selector_artifacts.json`。
   - 写 AC-9 / AC-10 tests。

## 实施说明

- 代码中不要写入 `AC-1` 或 `Milestone 1` 这类 plan 术语。
- Artifact 名称使用稳定的英文标识符。
- 面向人的文档和必要注释可以使用中文，但 JSON keys、module names、manifest keys 和 CLI flags 保持英文。
- fixture generation、augmentation、RGCN initialization、training 和 selector 都必须显式记录 random seed。
- pipeline 遇到缺失或非法 artifact contract 时必须明确失败。
- Phase A 测试只验证结构闭环和 replayability，不验证模型质量。
- 如果当前环境没有 `torch`，实现必须给出清晰的依赖错误，不能静默替换成非 RGCN 路径。
- graph builder 必须接受 raw register operand；fixture 可以使用版本化 token，但实现不能依赖 fixture token 已经版本化。
- memory address 识别必须基于 instruction operand role，不能 hard-code 为 `Raddr*` 命名规则。
- 所有 disk-backed repair stage 必须既返回 artifact，也写回对应 JSON 文件。

## 建议验证命令

```bash
pytest -q tests/gcl_phase_a
python -m experiments.gcl_phase_a.pipeline --out artifacts/gcl_phase_a
python -m pytest -q tests/gcl_phase_a/test_pipeline.py::test_phase_a_pipeline_e2e
python -m pytest -q tests/gcl_phase_a/test_graph_builder.py::test_raw_register_reuse_creates_distinct_register_versions
python -m pytest -q tests/gcl_phase_a/test_graph_builder.py::test_memory_address_role_does_not_require_raddr_name
python -m pytest -q tests/gcl_phase_a/test_pipeline.py::test_selector_stage_from_disk_recreates_selector_artifacts
```
