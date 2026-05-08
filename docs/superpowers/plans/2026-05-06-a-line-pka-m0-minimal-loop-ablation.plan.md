# A 线 PKA-M0 Minimal Loop 与消融实验 Implementation Plan

日期：2026-05-06

## 1. Goal Description

实现 A 线 `PKA-M0` 算法最小闭环，用于在不依赖真实 NCU measured acquisition 的情况下，先跑通 PKA-like selector 的核心算法链路：

```text
12D fixture feature table
  -> schema / numeric validation
  -> log1p + z-score preprocessing
  -> numpy SVD PCA
  -> deterministic farthest-first k-means
  -> nearest-centroid representative anchor selection
  -> compression evaluation
  -> deterministic replay / audit artifacts
```

`PKA-M0` 是 algorithmic fixture mode，不是 formal measured PKA reproduction。它的结果只能用于 selector、clustering、anchor selection 和消融实验的算法闭环验证，不得写入 formal PKA-M1 measured artifact，也不得作为论文中的 measured reproduction 结论。

## 2. Normative References

实现、review 和 stop-gate 判断必须遵循以下 spec。若本 plan 与 spec 冲突，以 spec 为准。

- `docs/superpowers/specs/2026-05-06-a-line-pka-minimal-loop-ablation-design.md`
- `docs/superpowers/specs/2026-05-06-a-line-pka-m0-pca-kmeans-classification-design.md`
- `docs/superpowers/specs/2026-05-06-a-line-pka-m0-pca-implementation-explainer-design.md`
- `docs/superpowers/specs/2026-05-06-a-line-pka-m0-kmeans-implementation-explainer-design.md`

## 3. Non-Negotiable Constraints

- M0 artifact 必须使用 `pka_m0_*` 独立路径。
- M0 不得写入或覆盖 `artifacts/a_line/l1/pka_feature_table_l1.json`。
- M0 不得写入或覆盖 `artifacts/a_line/l1/representative_anchor_table_l1.json`。
- M0 不得触发 formal B-line consumption。
- M0 fixture 中的 `status: measured` 只表示算法输入完整，不表示真实 profiler measured。
- Selector / pipeline 不得读取 `kernel_name`、`source_path`、`expected_behavior_axis`、`family`、`regime`、`shape_hint`、`trace_order` 或 B-line semantic metadata 来决定 clustering。
- PCA 主路径必须使用 numpy SVD 自实现，不得使用 `sklearn.decomposition.PCA`。
- K-means 主路径必须使用 deterministic farthest-first 自实现，不得使用 `sklearn.cluster.KMeans`。
- Bucketed、threshold、exact-vector grouping 只能作为后续消融策略，不能替代 M0 主路径。
- Cluster label 只能用于 debug / report，不得影响 clustering、membership 或 representative selection。

## 4. Acceptance Criteria

### AC-M0-1：Fixture 输入存在且 schema-valid

Positive Tests：

- `experiments/baseline_diagnosis/fixtures/pka_m0_feature_table_l1.json` 存在。
- fixture 包含 8 条第一版 records：global load、global store、shared memory、atomic、compute、divergence、local memory、mixed memory-compute。
- 每条 record 包含 `record_id`、`kernel_invocation_id`、`feature_mode: pka_m0_algorithmic_fixture`、`source_type: algorithmic_fixture` 和完整 12 维 `features`。
- 每个 feature 都是 numeric 且 `status: measured`。

Negative Tests：

- 缺少任一 12D feature 时 validation fail。
- 任一 feature 非 numeric 时 validation fail。
- 任一 count-like feature 为负数时 validation fail。
- 任一 record 的 `feature_mode` 不是 `pka_m0_algorithmic_fixture` 时 validation fail。

### AC-M0-2：M0 与 formal M1 artifact 隔离

Positive Tests：

- M0 输出只写入 `artifacts/a_line/l1/pka_m0_*` 文件。
- 运行 M0 pipeline 前后，formal `pka_feature_table_l1.json` 不被覆盖。
- 运行 M0 pipeline 前后，formal `representative_anchor_table_l1.json` 不被覆盖。

Negative Tests：

- M0 pipeline 写入 formal measured feature table 时测试失败。
- M0 pipeline 写入 formal representative anchor table 时测试失败。

### AC-M0-3：Forbidden-field guard 生效

Positive Tests：

- M0 selector / pipeline 只读取 `record_id`、`kernel_invocation_id`、`features`、`feature_mode` 和必要的 M0 control fields。
- `forbidden_field_audit.status == passed`。
- artifact 记录实际读取字段。

Negative Tests：

- fixture 中出现 `kernel_name` 且 selector 尝试读取时 pipeline fail。
- fixture 中出现 `expected_behavior_axis` 且 selector 尝试读取时 pipeline fail。
- selector 使用 B-line semantic metadata 决定 cluster 时测试失败。

### AC-M0-4：Preprocessing 固定为 log1p / clip / z-score

Positive Tests：

- count-like features 使用 `log1p(value)` 后再 z-score。
- `divergence_efficiency` 使用 clip to `[0, 1]` 后再 z-score。
- zero-std feature normalized value 全部为 0，并记录到 `zero_std_features`。
- normalization config 记录 feature order、mean、std、log1p fields、ratio fields、clipping count。

Negative Tests：

- count-like feature 未做 log1p 时测试失败。
- ratio feature 未记录 clipping 信息时测试失败。
- feature order 依赖 JSON natural order 时测试失败。

### AC-M0-5：PCA 使用 numpy SVD 自实现并输出 projection artifact

Positive Tests：

- PCA 通过 `np.linalg.svd(X_centered, full_matrices=False)` 实现。
- 不导入或调用 `sklearn.decomposition.PCA`。
- 输出 `artifacts/a_line/l1/pka_m0_pca_projection_l1.json`。
- PCA artifact 包含 `method: numpy_svd`、`pca_input_mean`、`components`、`explained_variance`、`explained_variance_ratio`、`transformed_coordinates`、`record_ids` 和 `input_fixture_hash`。
- `transformed_coordinates` shape 是 `n_records x pca_components`。

Negative Tests：

- `n_records < 3` 时 pipeline fail。
- `sum(explained_variance) == 0` 时 fail，并报告 `pca_degenerate_input`。
- PCA artifact 缺少 `input_fixture_hash` 时测试失败。
- 使用 sklearn PCA 时测试失败。

### AC-M0-6：K-means 使用 deterministic farthest-first 自实现

Positive Tests：

- k-means 输入来自 PCA artifact 的 `transformed_coordinates`。
- `k = ceil(sqrt(n_records))`，并 clamp 到 `[2, n_records]`。
- 第一个 centroid 是 `record_id` 字典序最小的 record。
- 后续 centroid 使用 farthest-first，tie-breaker 为 `record_id` 字典序最小。
- assignment 使用 squared Euclidean distance，tie-breaker 为最小 `cluster_id`。
- 输出 `artifacts/a_line/l1/pka_m0_kmeans_clusters_l1.json`。
- artifact 包含 `initial_centroid_record_ids`、`initialization_trace`、`centroids`、`assignments`、`members_by_cluster`、`distance_to_centroid`、`inertia`、`iterations_run`、`converged`。

Negative Tests：

- 使用 `sklearn.cluster.KMeans` 时测试失败。
- random initialization 时测试失败。
- assignment 后 record 丢失或重复时测试失败。
- PCA artifact hash 缺失或 shape 不合法时测试失败。

### AC-M0-7：Cluster label 只用于 debug / report

Positive Tests：

- cluster label 基于 cluster members 的原始 12D feature mean 生成。
- label artifact 记录 `label_rule_version`、cluster mean feature vector、chosen label 和 label evidence。
- membership 与 representative selection 不依赖 label。

Negative Tests：

- label 反向影响 membership 时测试失败。
- 使用 `expected_behavior_axis` 或 `kernel_name` 生成 cluster membership 时测试失败。

### AC-M0-8：Representative anchor 来自 nearest centroid

Positive Tests：

- 每个 cluster 输出一个 anchor。
- representative 是 cluster 内到 centroid squared distance 最小的真实 record。
- 如果距离并列，选择 `record_id` 字典序最小的 record。
- 输出 `artifacts/a_line/l1/pka_m0_representative_anchor_table_l1.json`。
- anchor table 包含 `anchor_id`、`cluster_id`、`representative_record_id`、`members`、`weight`、`representative_distance_to_centroid`、`cluster_label`、`forbidden_field_audit`。

Negative Tests：

- representative 是 centroid 平均点而不是真实 record 时测试失败。
- anchor membership 不是来自 k-means assignments 时测试失败。
- 使用 `first_chronological` 作为 M0 主路径时测试失败。

### AC-M0-9：Compression evaluation 输出结构性指标

Positive Tests：

- 输出 `artifacts/a_line/l1/pka_m0_compression_evaluation_l1.json`。
- evaluation 包含 compression ratio、coverage count、weighted coverage、cluster feature variance、top-k coverage、deterministic replay hash。
- evaluation 记录 input fixture hash、PCA artifact hash、k-means artifact hash、anchor table hash。

Negative Tests：

- evaluation 声称 simulator accuracy 或 measured speedup 时测试失败。
- evaluation 无法追溯 input fixture 或 anchor table 时测试失败。

### AC-M0-10：Pipeline CLI 一键生成全部 M0 artifacts

Positive Tests：

- 提供一个可运行 CLI，例如 `python experiments/baseline_diagnosis/pka_m0_pipeline.py`。
- 默认输入是 `experiments/baseline_diagnosis/fixtures/pka_m0_feature_table_l1.json`。
- 默认输出四个主 artifacts：PCA projection、k-means clusters、representative anchor table、compression evaluation。
- CLI 返回 0 表示所有 M0 gates passed。

Negative Tests：

- 只生成部分 artifact 时返回非零。
- 任一 forbidden-field audit failed 时返回非零。
- 缺少 fixture 时返回非零。

### AC-M0-11：Deterministic replay 稳定

Positive Tests：

- 连续两次运行 M0 pipeline，主要 artifact hash 保持一致。
- PCA coordinates、k-means assignments、anchor table 在固定 fixture 下保持一致。

Negative Tests：

- 同一输入连续运行产生不同 cluster assignments 时测试失败。
- artifact 中缺少 deterministic replay hash 时测试失败。

### AC-M0-12：Regression tests 覆盖主路径和负路径

Positive Tests：

- 新增测试文件，例如 `experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py`。
- 测试覆盖 fixture validation、preprocessing、PCA、k-means、anchor、evaluation、artifact isolation、forbidden-field guard。
- `pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py` 通过。
- `pytest -q experiments/baseline_diagnosis/test_l1_regression.py` 不因 M0 变更回归。

Negative Tests：

- 缺少负路径测试时 review 不能通过。
- 只测试 happy path 时 review 不能通过。

## 5. Path Boundaries

### Upper Bound

- 实现完整 PKA-M0 fixture、preprocessing、PCA、k-means、cluster label、anchor、evaluation 和 deterministic replay。
- 输出全部 M0 artifacts。
- 覆盖所有 AC-M0-1 到 AC-M0-12。
- 保留可扩展的消融配置接口，但不要求实现所有消融组合。
- 不触碰 formal PKA-M1 measured acquisition。

### Lower Bound

- 至少实现一条完整 M0 主路径：8 条 fixture -> preprocessing -> numpy SVD PCA -> deterministic k-means -> nearest-centroid anchors -> evaluation。
- 必须输出四个主 artifacts。
- 必须有测试证明 M0 artifact 与 formal M1 artifact 隔离。
- 必须有测试证明 PCA 和 k-means 不依赖 sklearn。
- 必须有 forbidden-field negative test。

### Allowed Choices

Can use:

- Python standard library。
- `numpy`。
- JSON artifacts。
- 独立 M0 module / CLI。
- 独立 M0 fixtures。
- 独立 M0 tests。

Cannot use:

- `sklearn.decomposition.PCA`。
- `sklearn.cluster.KMeans`。
- random k-means initialization。
- formal measured artifact path。
- B-line semantic metadata。
- selector grouping logic 中的 forbidden fields。
- proxy / derived / section-label fallback。

## 6. Dependencies and Sequence

### Milestone 1：M0 Fixture 与 Validation

目标：

- 创建 M0 fixture 文件。
- 实现 fixture loader 和 validator。
- 固定 12D feature order。
- 实现 forbidden-field audit 的基础记录。

建议文件：

- `experiments/baseline_diagnosis/fixtures/pka_m0_feature_table_l1.json`
- `experiments/baseline_diagnosis/pka_m0_pipeline.py`
- `experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py`

验收：

- AC-M0-1。
- AC-M0-2 的基础隔离检查。
- AC-M0-3 的基础 forbidden-field 测试。

### Milestone 2：Preprocessing 与 PCA

目标：

- 实现 log1p / clip / z-score。
- 实现 numpy SVD PCA。
- 输出 `pka_m0_pca_projection_l1.json`。

验收：

- AC-M0-4。
- AC-M0-5。
- PCA-only spec 中的 fail 条件。

### Milestone 3：Deterministic K-Means

目标：

- 实现 deterministic farthest-first initialization。
- 实现 assignment / update / empty cluster handling / stop condition / inertia。
- 输出 `pka_m0_kmeans_clusters_l1.json`。

验收：

- AC-M0-6。
- K-means-only spec 中的 fail 条件。

### Milestone 4：Cluster Label 与 Anchor Table

目标：

- 基于 cluster 原始 12D mean 生成 debug label。
- 实现 nearest-centroid representative selection。
- 输出 `pka_m0_representative_anchor_table_l1.json`。

验收：

- AC-M0-7。
- AC-M0-8。

### Milestone 5：Evaluation 与 Replay Hash

目标：

- 计算 compression ratio、coverage、weighted coverage、cluster feature variance、top-k coverage。
- 生成 deterministic replay hash。
- 输出 `pka_m0_compression_evaluation_l1.json`。

验收：

- AC-M0-9。
- AC-M0-11。

### Milestone 6：End-to-End CLI 与 Regression

目标：

- 提供一键运行 CLI。
- 确认所有 M0 artifacts 一次生成。
- 跑 M0 tests 和 L1 regression tests。

验收：

- AC-M0-10。
- AC-M0-12。

## 7. Task Breakdown

### T1：创建 M0 fixture

- 添加 8 条 `algorithmic_fixture` records。
- 每条 record 覆盖一个可解释行为轴。
- 保证 12D 完整、numeric、非负 count-like、ratio 合法。

### T2：实现 fixture loader / validator

- 读取 fixture。
- 检查 required fields。
- 检查 12D feature set。
- 检查 numeric status。
- 检查 M0 mode 标记。

### T3：实现 forbidden-field audit

- 定义 allowed field set。
- 记录 pipeline 实际读取字段。
- 如果读取 forbidden metadata，fail。

### T4：实现 preprocessing

- 固定 feature order。
- count-like 执行 log1p。
- ratio 执行 clip。
- 执行 z-score。
- 记录 normalization config。

### T5：实现 PCA

- 使用 numpy SVD。
- 输出 PCA projection artifact。
- 实现 degenerate input fail。

### T6：实现 deterministic k-means

- 实现 `ceil_sqrt_n` k strategy。
- 实现 farthest-first initialization。
- 实现 assignment/update/empty-cluster/stop/inertia。
- 输出 k-means artifact。

### T7：实现 cluster label

- 基于 cluster 原始 feature mean 生成 label。
- 保证 label 不参与 membership。

### T8：实现 anchor table

- 从 k-means assignments 生成 anchors。
- 使用 nearest-centroid representative。
- 输出 M0 anchor table。

### T9：实现 evaluation

- 计算结构性压缩指标。
- 写 evaluation artifact。
- 记录 replay hashes。

### T10：实现 CLI

- 默认输入 fixture。
- 默认输出 artifact directory。
- 返回码反映 M0 gate 状态。

### T11：实现 regression tests

- 覆盖 AC-M0-1 到 AC-M0-12。
- 包含正路径和负路径。

## 8. Implementation Notes

- 代码中不要使用 plan 术语作为业务逻辑，例如不要用 `T1`、`AC-M0-1` 作为函数名。
- 推荐把纯算法函数保持小而独立，便于测试。
- 推荐 CLI 只负责 orchestration，算法逻辑放在可导入函数中。
- JSON 输出必须 `indent=2`，并保持 deterministic key / list order。
- Hash 建议使用 SHA-256，hash 输入应为 canonical JSON bytes。
- 如果当前已有 `pka_baseline_selector.py` 与新 M0 逻辑冲突，优先新增独立 M0 module，避免破坏 formal selector。

## 9. Required Verification

最小验证命令：

```bash
python experiments/baseline_diagnosis/pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/tests/test_pka_m0_pipeline.py
pytest -q experiments/baseline_diagnosis/test_l1_regression.py
```

如果实现新增了专用 lint / schema check，可以额外运行，但不能替代上述命令。

## 10. Stop-Gate Completion Rule

本 plan 的 RLCR completion 必须满足：

- M0 fixture 存在且 validation passed。
- PCA artifact 存在且 method 是 `numpy_svd`。
- K-means artifact 存在且 initialization 是 `deterministic_farthest_first`。
- M0 anchor table 存在且 anchors 来自 k-means assignments。
- M0 evaluation artifact 存在。
- Forbidden-field audit passed。
- Formal PKA-M1 artifacts 未被覆盖。
- Required verification 全部通过。

如果只实现 fixture validation，没有 PCA + k-means，不得 COMPLETE。

如果只实现 PCA，没有 k-means 和 anchor，不得 COMPLETE。

如果输出写入 formal PKA-M1 artifact 路径，不得 COMPLETE。

## 11. Risks

- 当前已有 `pka_baseline_selector.py` 中可能存在旧 PCA/k-means 实现。该实现不能直接视为满足本 plan，因为本 plan 要求 M0 artifact 隔离、numpy SVD PCA、deterministic farthest-first k-means 和 forbidden-field audit。
- M0 fixture 是人工构造输入，不能被误解为 measured PKA reproduction。
- M0 的 compression evaluation 只计算结构性指标，不代表 simulator accuracy。

## 12. 简短结论

本 plan 的目标是让 A 线先获得一个严格隔离、可复现、可审计的 PKA-M0 算法闭环。

完成后，我们可以在不等待 NCU measured acquisition 的情况下运行 PCA/k-means selector 和消融实验；同时仍保持 M1 measured baseline 的科学边界。
