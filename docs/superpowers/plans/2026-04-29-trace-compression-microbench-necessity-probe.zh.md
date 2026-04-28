# Trace Compression Microbench 必要性探针 中文计划

> **给 agentic workers 的说明：** 仍然需要按 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` 逐任务执行。下面的任务结构与英文执行版一致，只是这里用中文便于阅读和讨论。

**目标：** 先做一个最小证据探针，验证“压缩导出的 trace 结构信号”是否真的比 flat profile 特征更能表达 microbench 匹配所需的行为信息。

**架构：** 在 `experiments/trace_compression_behavior/` 下新增一个小型 Python 实验包。它读取一份精心挑选的 target/candidate catalog，从已有 JSON 记录里提取 flat profile signature 和 compression signature，计算距离矩阵，找出 profile 相似但 compression 不同的冲突对，并输出 JSON/Markdown 报告。第一版只用本地 artifact 和 synthetic fixture，不做在线压缩，也不做 AI 生成。

**技术栈：** Python 3.10+ 标准库，`pytest`，JSON/Markdown 输出。

---

## 范围

这个计划实现的是独立学术线的第一个 gate：

```text
target / candidate traces
  -> flat profile signature
  -> compression signature
  -> distance matrices
  -> conflict pairs
  -> necessity report
```

明确不做的内容：

- CUDA microbench 生成；
- LLM / agent 代码生成；
- 在线 trace 压缩；
- simulator 验证；
- 修改 L1 / PKA selector。

成功的标准是：仓库能产出一份报告，回答下面这个问题：

> 压缩导出的 signature 是否能暴露 flat profile summary 看不到的行为差异？

---

## 文件结构

新增：

- `experiments/trace_compression_behavior/__init__.py`  
  包标记。

- `experiments/trace_compression_behavior/models.py`  
  Catalog、signature、distance、conflict 的 dataclass 和校验帮助函数。

- `experiments/trace_compression_behavior/catalog.py`  
  读取并校验 target/candidate catalog 的 JSON。

- `experiments/trace_compression_behavior/signatures.py`  
  从 catalog 指向的 JSON 记录中提取 flat profile signature 和 compression signature。

- `experiments/trace_compression_behavior/distance.py`  
  归一化数值字段并计算 pairwise distance。

- `experiments/trace_compression_behavior/conflicts.py`  
  检测 profile-similar / compression-different，以及 runtime-similar / compression-different 的 pair。

- `experiments/trace_compression_behavior/report.py`  
  输出机器可读 JSON 和人可读 Markdown 报告。

- `experiments/trace_compression_behavior/run_probe.py`  
  CLI 入口，串起 catalog、提取、距离、冲突检测和报告生成。

- `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`  
  给单元测试用的小型 synthetic catalog。

- `experiments/trace_compression_behavior/fixtures/synthetic_records.json`  
  带有可控 profile / compression 特征的小型记录。

- `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json`  
  第一批真实/本地 catalog，引用仓库里现有 artifact。

- `experiments/trace_compression_behavior/tests/test_models.py`

- `experiments/trace_compression_behavior/tests/test_catalog.py`

- `experiments/trace_compression_behavior/tests/test_signatures.py`

- `experiments/trace_compression_behavior/tests/test_distance.py`

- `experiments/trace_compression_behavior/tests/test_conflicts.py`

- `experiments/trace_compression_behavior/tests/test_report.py`

- `experiments/trace_compression_behavior/tests/test_run_probe.py`

CLI 产生的输出路径：

- `experiments/trace_compression_behavior/results/necessity_probe/signature_table.json`
- `experiments/trace_compression_behavior/results/necessity_probe/profile_distance_matrix.json`
- `experiments/trace_compression_behavior/results/necessity_probe/compression_distance_matrix.json`
- `experiments/trace_compression_behavior/results/necessity_probe/conflict_pairs.json`
- `experiments/trace_compression_behavior/results/necessity_probe/necessity_report.md`
- `experiments/trace_compression_behavior/results/necessity_probe/necessity_report.json`

不要修改：

- `experiments/baseline_diagnosis/pka_baseline_selector.py`
- `experiments/baseline_diagnosis/pka_feature_extractor.py`
- `experiments/baseline_diagnosis/b_line_consumer_l1.py`
- 任何 L1 schema 或 selector tests

---

## 数据契约

### Catalog Entry

每个 catalog 条目都描述一个 target 或 candidate record。

示例结构保持与英文版一致：

```json
{
  "id": "regular_memory_fixture",
  "label": "regular-memory",
  "role": "candidate",
  "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
  "record_pointer": "/records/regular_memory",
  "profile_fields": {
    "runtime": "hardware_metrics.duration_ns",
    "num_instructions": "dynamic_stats.total_dynamic_insts",
    "global_loads": "dynamic_stats.global_loads",
    "global_stores": "dynamic_stats.global_stores",
    "branch_ops": "dynamic_stats.branch_ops",
    "thread_blocks": "dynamic_stats.num_blocks"
  },
  "compression_fields": {
    "instruction_run_coverage": "compression_features.instruction_run_coverage.mean",
    "shared_pc_sequence_coverage": "compression_features.shared_pc_sequence_coverage.mean",
    "warp_pc_override_density": "compression_features.warp_pc_override_density.mean",
    "cross_tb_delta_coverage": "compression_features.cross_tb_delta_coverage.mean",
    "global_address_offset_coverage": "compression_features.global_address_offset_coverage.mean",
    "address_override_density": "compression_features.address_override_density.mean",
    "full_encoding_fallback_rate": "compression_features.full_encoding_fallback_rate.mean"
  }
}
```

允许的 `role`：

- `target`
- `candidate`
- `control`

### Signature Record

提取器对每个 catalog 条目输出一条 signature record：

```json
{
  "id": "regular_memory_fixture",
  "label": "regular-memory",
  "role": "candidate",
  "source_path": "experiments/trace_compression_behavior/fixtures/synthetic_records.json",
  "profile_signature": { ... },
  "compression_signature": { ... },
  "missing_profile_fields": [],
  "missing_compression_fields": [],
  "confidence": "high"
}
```

置信度规则：

- `high`：profile/compression 字段都齐全；
- `medium`：compression 字段缺 1 到 2 个，profile 字段齐全；
- `low`：compression 字段缺超过 2 个，或者 profile 字段有缺失。

### Conflict Pair

冲突对的输出格式：

```json
{
  "left_id": "regular_memory_fixture",
  "right_id": "irregular_gather_fixture",
  "conflict_type": "profile_similar_compression_different",
  "profile_distance": 0.05,
  "compression_distance": 0.72,
  "explanation": "Flat profile counts are close, but memory-structure signature differs: cross_tb_delta_coverage 0.94 vs 0.18 and address_override_density 0.02 vs 0.68."
}
```

### 阈值

初始阈值：

- `profile_similar_threshold = 0.15`
- `runtime_similar_threshold = 0.10`
- `compression_different_threshold = 0.35`
- `compression_similar_threshold = 0.15`

报告里必须明确说明：这些只是第一轮分析参数，不是最终研究常数。

---

## 任务拆分

> 下面的任务顺序、文件路径和验证方式与英文执行版一一对应。这里保留结构，方便你快速看懂整条路线。

### Task 1：创建包骨架和 synthetic fixture

目标：

- 建立 `experiments/trace_compression_behavior/` 包；
- 准备 deterministic 的 synthetic records / catalog；
- 先把 catalog 读取测试跑起来。

关键文件：

- `experiments/trace_compression_behavior/__init__.py`
- `experiments/trace_compression_behavior/fixtures/synthetic_records.json`
- `experiments/trace_compression_behavior/fixtures/synthetic_catalog.json`
- `experiments/trace_compression_behavior/tests/test_catalog.py`

验收：

- catalog 能读出稳定的 entry id；
- duplicate id 能被拒绝；
- 先让测试失败，再做最小实现，然后测试通过。

### Task 2：实现 Catalog 和基础模型

目标：

- 定义 `CatalogEntry`、`Catalog`、`SignatureRecord` 等模型；
- 实现 catalog 加载、路径解析、重复 id 检查、role 校验。

关键文件：

- `experiments/trace_compression_behavior/models.py`
- `experiments/trace_compression_behavior/catalog.py`
- `experiments/trace_compression_behavior/tests/test_catalog.py`

验收：

- catalog 可以稳定加载；
- duplicate / invalid role / 缺失文件路径会报错；
- 单测通过。

### Task 3：提取 flat profile 和 compression signatures

目标：

- 从本地 JSON artifact 中提取 profile signature 和 compression signature；
- 保留缺失字段列表；
- 为每条记录赋予 confidence。

关键文件：

- `experiments/trace_compression_behavior/signatures.py`
- `experiments/trace_compression_behavior/tests/test_signatures.py`

验收：

- regular / irregular 记录的 signature 能抽出来；
- 缺失字段必须显式记录，不能静默补默认值；
- 单测通过。

### Task 4：计算距离矩阵

目标：

- 对 profile signature 和 compression signature 分别计算 pairwise distance；
- 让“数值接近但结构不同”的 pair 能被拉开。

关键文件：

- `experiments/trace_compression_behavior/distance.py`
- `experiments/trace_compression_behavior/tests/test_distance.py`

验收：

- profile 距离和 compression 距离能给出不同排序；
- 单测通过。

### Task 5：检测冲突对

目标：

- 识别 profile-similar / compression-different 的 pair；
- 生成 human-interpretable 的解释文本。

关键文件：

- `experiments/trace_compression_behavior/conflicts.py`
- `experiments/trace_compression_behavior/tests/test_conflicts.py`

验收：

- synthetic fixture 至少能找到一组预期冲突；
- 解释里要点名具体 compression 字段差异。

### Task 6：输出 JSON 和 Markdown 报告

目标：

- 把 signature、distance、conflict 输出成 JSON 和 Markdown；
- Markdown 报告里要能直接读出结论。

关键文件：

- `experiments/trace_compression_behavior/report.py`
- `experiments/trace_compression_behavior/tests/test_report.py`

验收：

- 目录下生成 `necessity_report.json` 和 `necessity_report.md`；
- 报告中要有 conflict 统计和解释。

### Task 7：添加 CLI 入口

目标：

- 让整个探针能通过 `python -m` 运行；
- 输入 catalog，输出报告目录。

关键文件：

- `experiments/trace_compression_behavior/run_probe.py`
- `experiments/trace_compression_behavior/tests/test_run_probe.py`

验收：

- CLI 可运行；
- 结果目录可生成；
- 结束语里打印写入路径。

### Task 8：接入真实本地 catalog

目标：

- 从仓库现有 `mini_transformer` artifact 中挑一批真实记录；
- 形成第一版真实样本 catalog。

关键文件：

- `experiments/trace_compression_behavior/catalogs/initial_probe_catalog.json`
- `experiments/trace_compression_behavior/tests/test_initial_catalog.py`

验收：

- 至少能加载 5 条真实记录；
- 至少能输出一份真实样本的 report；
- 如果暂时没有冲突对，也要如实记录“样本多样性不足”。

### Task 9：运行探针并保存 baseline 结果

目标：

- 跑 synthetic probe；
- 跑 real-artifact probe；
- 保留一份 baseline 结果。

关键输出：

- `experiments/trace_compression_behavior/results/synthetic_probe/*`
- `experiments/trace_compression_behavior/results/necessity_probe/*`

验收：

- synthetic report 至少展示出预期的冲突模式；
- real report 至少说明当前 catalog 是否足够多样；
- 小于 1MB 的结果可以提交，大于 1MB 的结果不强求入库。

### Task 10：写研究解释说明

目标：

- 把本轮 probe 的实际结果写成一份中文/英文都能对照的解释 note；
- 明确“支持 / 暂缓 / 需要扩展样本”的判断。

关键文件：

- `docs/trace-compression-behavior-necessity-probe-2026-04-29.md`

验收：

- 文件中没有模板残留；
- 结论明确，不写成任务提醒；
- 能读出下一步是否进入 microbench matching。

---

## 最终验证

建议最终跑：

```bash
pytest experiments/trace_compression_behavior/tests -q
python3 -m experiments.trace_compression_behavior.run_probe \
  --catalog experiments/trace_compression_behavior/fixtures/synthetic_catalog.json \
  --output-dir /tmp/trace_compression_behavior_synthetic_probe
git diff --check
```

期望结果：

- tests 全通过；
- CLI 正常写出报告；
- `git diff --check` 通过。

---

## 交付标准

这份计划完成时，应满足：

1. 探针包可以在 deterministic fixture 上运行；
2. synthetic fixture 至少能展示一组 profile-similar / compression-different 的模式；
3. 第一版真实 catalog 可以加载并分析；
4. 研究解释说明能写清楚：当前结果是支持、削弱还是暂时不能判断这条学术线；
5. 整个过程没有修改 L1 selector、schema 或 B-line 代码。

英文执行版仍然保留在：

- `docs/superpowers/plans/2026-04-29-trace-compression-microbench-necessity-probe.md`
