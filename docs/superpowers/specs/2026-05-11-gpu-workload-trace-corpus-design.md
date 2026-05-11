# GPU Workload Trace Corpus 设计

日期：2026-05-11

## 1. 研究问题

本设计定义一个可复用的 GPU workload trace corpus，用于支撑 Photon 压缩、PKA selector、后续训练集构建和论文实验。

它要回答的问题是：

> 当 workload 从单 kernel benchmark 扩展到完整真实应用、完整神经网络和大规模 kernel sequence 时，Photon 是否仍能在大量 kernel instances、大 kernel 和不规则 workload 上保持可解释、可复现、可量化的压缩收益？

当前已经下载的 `2.8G` 左右 workload 源码只是 `L0 source/harness` 层。它不能代表最终训练集或 claim-bearing trace corpus 的规模。真正的证据规模应来自：

- 完整网络 / 完整应用运行产生的 kernel launch sequence。
- 每个 kernel invocation 的 measured metadata 和 NCU feature。
- dynamic trace、trace stats、compressed trace 和 replay / validation artifact。
- 不同 input shape、batch、sequence length、precision、graph size、problem size 下的系统化采样。

## 2. 目标

本 corpus 的目标是形成一个长期可扩展的 GPU workload 数据资产，而不是只为当前 M1.5 实验临时下载 benchmark。

目标包括：

- 覆盖论文相关 benchmark suite：Rodinia、Parboil、SHOC、Altis、DeepBench、CUTLASS、MLPerf-style AI workload、Gunrock / Pannotia graph workload、LAMMPS / GROMACS 等 HPC full applications。
- 纳入完整网络级 workload，而不仅是 kernel primitive。
- 支持 Photon 对完整 workload trace 的压缩效率验证。
- 支持 PKA / M1 / M1.5 的 measured feature table 扩展。
- 支持后续模型训练：selector、kernel family classifier、trace compressibility predictor、cost model、failure predictor。
- 每个样本都有 provenance、输入配置、运行命令、环境、采集状态和 artifact hash。

## 3. 非目标

本设计不直接完成以下事项：

- 不立即下载所有模型权重、完整数据集或大型 simulation input decks。
- 不声称当前源码下载已经形成训练集。
- 不把 synthetic input 等同于真实 dataset run。
- 不把 `placeholder`、`modeled`、`pending_measurement` 行作为最终 claim-bearing 证据。
- 不在本阶段做 simulator accuracy 或 speedup 结论。
- 不把 license 受限的 SPEC ACCEL package 纳入自动下载。

## 4. Corpus 分层

### L0：Source / Harness Layer

内容：

- benchmark / application 源码。
- build system。
- benchmark launcher。
- sparse / partial clone 状态。
- source commit hash。

状态标签：

- `source_available`
- `source_sparse_available`
- `source_unavailable`
- `license_blocked`

当前 L0 已有来源：

- `gpu-rodinia`
- `gpu-parboil`
- `shoc`
- `altis`
- `deepbench`
- `cutlass`
- `mlperf-inference`
- `gunrock`
- `pannotia`
- `hecbench`
- `lammps`
- `gromacs`

### L1：Workload Registry Layer

内容：

- workload id。
- suite / source。
- workload category。
- build command。
- run command template。
- input family。
- shape grid。
- expected kernel count class。
- expected large-kernel class。
- expected irregularity class。
- license / dataset requirement。

推荐状态标签：

- `registry_candidate`
- `build_ready`
- `run_ready`
- `input_required`
- `license_review_required`
- `deferred`

### L2：Launch / Kernel Metadata Layer

内容：

- kernel invocation id。
- kernel name。
- launch order。
- grid / block shape。
- runtime / timing unit。
- operator mapping（如果可得）。
- workload phase。
- parent workload run id。

这个层用于回答：

- 一个完整 workload 有多少 kernel instances。
- kernel sequence 是否足够复杂。
- 哪些 kernel 是大 kernel。
- 哪些 workload 适合作为 Photon stress workload。

### L3：Measured Feature Layer

内容：

- NCU / profiler measured metrics。
- M1 / M1.5 12D PKA feature。
- timing / memory / occupancy / instruction / divergence / atomic / shared-memory 等扩展特征。
- feature provenance。
- query metrics artifact。
- capture stderr / exit code / environment manifest。

禁止：

- proxy feature。
- default-zero fill。
- semantic fallback。
- section label fallback 伪装 measured。

状态标签：

- `measured`
- `capture_failed`
- `permission_blocked`
- `environment_blocked`
- `metric_unavailable`
- `parser_failed`

### L4：Trace Artifact Layer

内容：

- raw dynamic trace。
- trace stats。
- compressed trace。
- compression metadata。
- replay / parser validation report。
- per-kernel and per-workload compression ratio。
- trace acquisition time。
- compression time。
- decompression / replay time。

状态标签：

- `trace_measured`
- `trace_too_large`
- `trace_partial`
- `trace_export_failed`
- `trace_replay_failed`

### L5：Training Dataset Layer

内容：

- train / validation / test split。
- sample metadata。
- label definitions。
- feature tensor / table。
- compression labels。
- kernel family labels。
- corpus version。

这个层只允许消费 L1-L4 中 provenance 完整的样本。不能直接从源码目录生成 claim-bearing training row。

## 5. Workload 类型

### 5.1 Benchmark Kernel Suite

用途：

- 验证工具链。
- 覆盖典型 kernel pattern。
- 提供可控输入规模。

候选：

- Rodinia：`bfs`、`cfd`、`hotspot`、`lavaMD`、`backprop`、`nn`、`pathfinder`、`srad`、`streamcluster`。
- Parboil：`bfs`、`cutcp`、`histo`、`lbm`、`mri-q`、`sgemm`、`spmv`、`stencil`。
- SHOC：`GEMM`、`FFT`、`MD`、`SpMV`、`Stencil`、`Reduction`、`Scan`。
- CUTLASS / DeepBench：GEMM、batched GEMM、conv-like primitive、DNN primitive。

这些 workload 主要作为 control / calibration，不应单独支撑 Photon 完整 workload 主张。

### 5.2 Full Network Workload

用途：

- 验证完整神经网络 pipeline 的 kernel sequence 压缩。
- 构建后续训练集核心样本。
- 暴露 high-kernel-count 和 operator diversity。

第一批候选：

- `ResNet50` full inference。
- `BERT-base` full inference。
- `Llama/GPT-style small decoder`，包含 prefill 和 decode。
- `Stable Diffusion UNet` 或 reduced text-to-image pipeline。
- `DLRM` recommendation inference。
- `R-GAT / GNN` inference。
- `Whisper` speech model inference。
- `3D-UNet` medical segmentation inference。

每个完整网络必须至少记录：

- framework / implementation。
- model source。
- weight source。
- dataset or synthetic input mode。
- batch size。
- sequence length / image size / graph size。
- precision。
- kernel launch count。
- trace size。
- measured feature coverage。

### 5.3 Irregular / Graph / Sparse Workload

用途：

- 验证 Photon 对非规则 kernel sequence 和非规则访存的优势。

候选：

- Gunrock：BFS、SSSP、PageRank、Connected Components、Triangle Counting。
- Pannotia：graph traversal / graph analytics workload。
- sparse：SpMV、SpMM、embedding lookup。
- histogram、hash table、join、scatter/gather。

这些 workload 的关键不是源码大小，而是数据集规模和图结构复杂度。

### 5.4 HPC Full Application

用途：

- 验证真实科学应用的完整 step / timestep trace。
- 覆盖 molecular dynamics、CFD、stencil、particle simulation 等大 kernel 和复杂 phase。

候选：

- LAMMPS。
- GROMACS。
- AMBER / NAMD / OpenMM，后续可加入。
- LULESH / miniAMR / XSBench / Nekbone 类 mini-app，后续可加入。

HPC full app 的输入 deck 必须单独登记，不允许用未记录来源的临时输入产生 claim-bearing trace。

## 6. Shape Grid

完整网络和 full application 不应只跑一个输入。每个 workload 应定义 shape grid，形成可训练、可泛化的数据分布。

### AI 网络

推荐维度：

- batch size：`1`、`4`、`8`、`16`。
- sequence length：`128`、`512`、`1024`、`2048`。
- image size：`224`、`512`、`1024`。
- precision：`fp32`、`fp16`、`bf16`，按硬件能力选择。
- mode：`inference` 优先，`training` 后续加入。

### Graph / Sparse

推荐维度：

- graph size：small / medium / large。
- edge density：sparse / medium / dense。
- degree distribution：regular / power-law / skewed。
- input source：real graph / synthetic graph。

### HPC

推荐维度：

- system size。
- timestep count。
- domain decomposition。
- precision。
- physics / pair style / solver mode。

## 7. 目录结构

外部 corpus 根目录：

```text
/home/dyf/workloads/trace-compressions-industrial-codex-workload/
```

建议结构：

```text
sources/
  <source_repo>/

inputs/
  <workload_id>/
    README.md
    checksums.tsv
    license.md

runs/
  <corpus_version>/
    <workload_id>/
      <run_id>/
        run_config.json
        environment.json
        build.log
        run.log
        launch_metadata.json
        ncu/
        traces/
        compression/
        validation/

registry/
  workload_registry.json
  corpus_schema.json
  source_registry.json

reports/
  acquisition_report.md
  coverage_report.md
  corpus_summary.md
```

仓库内只保存 schema、manifest、脚本和报告，不保存大 trace、权重或数据集。

## 8. Schema 草案

### 8.1 Source Record

```json
{
  "source_id": "mlperf-inference",
  "source_type": "benchmark_suite",
  "url": "https://github.com/mlcommons/inference.git",
  "local_path": "/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources/mlperf-inference",
  "commit": "7b11eeb",
  "clone_mode": "sparse_partial",
  "license_status": "needs_review"
}
```

### 8.2 Workload Record

```json
{
  "workload_id": "bert_base_inference_bs1_seq128_fp16",
  "source_id": "mlperf_or_hf_transformers",
  "workload_family": "full_network",
  "claim_role": "claim_bearing",
  "expected_kernel_count_class": "high",
  "expected_large_kernel_class": "medium",
  "expected_irregularity_class": "medium",
  "build_status": "pending",
  "input_status": "pending",
  "license_status": "needs_review"
}
```

### 8.3 Run Record

```json
{
  "run_id": "bert_base_inference_bs1_seq128_fp16_2026-05-11_0001",
  "workload_id": "bert_base_inference_bs1_seq128_fp16",
  "mode": "inference",
  "input_mode": "synthetic",
  "precision": "fp16",
  "shape": {
    "batch_size": 1,
    "sequence_length": 128
  },
  "launch_count": null,
  "trace_status": "pending",
  "measured_feature_status": "pending"
}
```

## 9. Milestone Gates

### Gate C0：Source Mirror Ready

输入：

- source repository list。
- external storage path。

输出：

- `source_registry.json`。
- `clone_status.tsv`。
- `workload_download_status.md`。

通过条件：

- 每个 source 有 URL、local path、commit hash、clone mode。
- sparse / partial source 明确标注。
- 下载失败有 reason。

失败状态：

- `SOURCE_MIRROR_PARTIAL`。

当前状态：

- 已基本满足，但还缺 machine-readable `source_registry.json`。

### Gate C1：Workload Registry Ready

输入：

- L0 sources。
- manual suite scan。

输出：

- `workload_registry.json`。
- `workload_registry.md`。

通过条件：

- 每个 candidate workload 有 source、category、claim role、expected pressure class。
- full network workload 与 benchmark kernel workload 分开标记。
- license / input requirement 明确。

失败状态：

- `REGISTRY_INCOMPLETE`。

### Gate C2：Input / Weight Plan Ready

输入：

- workload registry。
- full network candidate list。

输出：

- `input_asset_plan.md`。
- `input_asset_registry.json`。

通过条件：

- 每个 full network 有 model weight source、dataset / synthetic input policy、expected size、license status。
- 每个 graph / HPC workload 有 input deck / graph dataset source。
- 不允许未登记数据进入 claim-bearing run。

失败状态：

- `INPUT_ASSET_PLAN_INCOMPLETE`。

### Gate C3：Trace Acquisition Smoke Ready

输入：

- build-ready workloads。
- small input configs。

输出：

- smoke run artifacts。
- launch metadata。
- at least one small trace per workload category。

通过条件：

- 至少一个 benchmark kernel、一个 full network、一个 irregular workload、一个 HPC app 完成 smoke trace。
- trace directory layout 被后续 parser 接受。
- failures recorded as structured gaps。

失败状态：

- `TRACE_ACQUISITION_BLOCKED`。

### Gate C4：Claim-Bearing Corpus Ready

输入：

- registry。
- input assets。
- acquisition harness。

输出：

- measured run artifacts。
- trace artifacts。
- compression artifacts。
- corpus coverage report。

硬性通过条件：

- 至少 `5` 个 full-network workload 有 measured launch metadata 和 trace artifact。
- 至少 `20` 个 benchmark / application workload 有 measured launch metadata。
- 至少 `100` 条 kernel invocation rows 有 measured metadata。
- 至少 `50` 条 kernel invocation rows 有完整 measured feature。
- 至少 `1` 个 large-kernel workload、`1` 个 high-kernel-count workload、`1` 个 irregular workload 有 claim-bearing trace artifact。

失败状态：

- `PARTIAL: infrastructure complete, evidence incomplete`。

### Gate C5：Training Dataset Ready

输入：

- claim-bearing corpus。
- split policy。
- label definitions。

输出：

- `dataset_manifest.json`。
- train / validation / test split。
- dataset card。

通过条件：

- split 不按同一 workload 的相邻 shape 泄漏。
- 每个 sample 有 source hash、run hash、trace hash。
- label definition 明确。
- no placeholder label。

失败状态：

- `DATASET_PARTIAL`。

## 10. 完成状态定义

### COMPLETE

同时满足：

- Gate C0-C5 全部通过。
- 至少一个 claim-bearing corpus version 可用于 Photon 和后续训练。
- dataset card 明确列出覆盖范围和限制。

### PARTIAL

满足：

- source / registry / harness 已完成。
- measured trace 或 dataset split 不足以支撑 claim。

### BLOCKED

出现以下任一情况：

- 必需数据集或模型权重无法合法获取。
- trace acquisition prerequisite 不满足。
- GPU / NCU / NVBit 环境不可用。

### NEGATIVE

measured evidence 显示：

- Photon 对目标 stress workload 没有压缩收益，或收益低于预注册阈值。

## 11. 下一步

1. 生成 machine-readable `source_registry.json`。
2. 扫描当前 12 个 source，生成 `workload_registry.json` 草案。
3. 对完整网络 workload 单独写 `input_asset_plan.md`。
4. 优先展开 MLPerf / full-network 相关目录，但不下载权重和数据集，直到 input asset plan 明确。
5. 先做 small full-network smoke run，再扩展到 shape grid。

