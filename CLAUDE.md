# CLAUDE.md

## 文档语言要求

所有诊断报告、实验总结、分析文档（`.md` 格式）均使用**中文**撰写。
代码注释、JSON 字段名、变量名、命令仍使用英文。

## Plan 生成准则

后续生成任何研究、实验、RLCR 或证据流水线 plan 时，必须遵守本节规则。核心原则是：

```text
基础设施完成，不等于证据完成。
```

### 1. Plan 必须自包含

plan 开头必须用中文写清楚：

- 研究要回答的具体问题是什么；
- 要测量的对象是什么；
- 最终判断依据是什么；
- 哪些内容明确不在范围内；
- 当前类比或外部方法是什么意思。

如果使用 DiffTest、SMARTS、SimPoint、Accel-Sim 等外部概念，必须在 plan 中直接解释其在本项目里的含义，并给出参考链接。不能假设执行模型已经知道这些概念。

### 2. 使用 milestone gates，而不是只写 task list

不能只列任务。必须定义 gate，并说明每个 gate 的通过条件。

推荐结构：

```text
Gate A: Infrastructure Ready
Gate B: Trace Acquisition Ready
Gate C: Claim-Bearing Measurement Complete
Gate D: Go/No-Go Decision Complete
```

每个 gate 必须列出：

- 必需输入；
- 必需输出 artifact；
- 允许的数据标签；
- 不允许的数据标签；
- 失败时的状态名称。

### 3. 把 measured-data requirement 写成硬性 AC

如果研究目标需要实测数据，必须把 measured-data requirement 写成硬性 acceptance criterion。

硬性 AC 的含义：

```text
没有 measured data，就不允许判定 plan 完成。
```

例如 GPU trace frontend 这类研究，至少一个 claim-bearing workload 必须有 measured：

- trace export time；
- frontend timing breakdown；
- redundancy profile；
- simulator backend time；
- result analysis time；
- complete-flow burden ratio。

`modeled`、`placeholder`、`pending_measurement` 只能用于开发阶段，不能满足最终 measured-data AC。

### 4. 先确认 trace acquisition，再写 burden-ratio 证据计划

在生成 burden-ratio、complete-flow、evidence-table 类 plan 前，必须先检查 trace acquisition prerequisite。

最低检查项：

```text
NVBit-instrumented training harness exists
trace export smoke test passes
trace directory layout is accepted by the simulator
small batch trace can be replayed
```

如果这些条件不满足，plan 应该改写为 trace acquisition plan，而不是继续假设 measured trace 已经存在。

### 5. placeholder 可以用于开发，但 final gate 必须拒绝

artifact generator 可以在开发阶段写 placeholder 行来跑通格式、schema 和表格生成。

但 final gate 必须拒绝 claim-bearing 行中的以下值：

- `placeholder`
- `pending_measurement`
- `modeled`，当该字段要求 measured 时
- `null` telemetry values

如果这些值仍存在，最终状态必须是：

```text
PARTIAL: infrastructure complete, evidence incomplete
```

不能写成：

```text
COMPLETE
```

### 6. 区分 control validation 和 claim-bearing proof

control workload 只能证明工具链、插桩、schema、artifact emission 能跑通。它不能替代研究主张的证据。

plan 必须分别记录：

```text
control_validation_passed: true/false
claim_bearing_measurement_passed: true/false
```

例如：

- `Rodinia nn` 可以作为 control workload；
- `BERT-base`、`Llama 3.1 8B` 这类目标 workload 才能作为 claim-bearing workload。

如果只有 control workload 跑通，结论只能是：

```text
instrumentation validated
claim not proven
```

### 7. optional tail attempt 必须是真正可选

像 `Llama 3.1 8B full step` 这类高风险、大成本目标，可以作为 nice-to-have tail attempt。

但 plan 必须明确：

- 它不能阻塞主线 artifact；
- 它不能覆盖已有完整结果；
- 它只能在必需证据线完成后尝试；
- 多次失败时必须记录 attempt count、failure reason、partial artifacts 和 abandoned/measured 状态。

### 8. RLCR 结束状态必须诚实

如果 claim-critical task 仍未完成，RLCR 最终状态不能写 `complete`。

推荐状态名：

```text
COMPLETE: evidence complete and go/no-go verdict produced
PARTIAL: infrastructure complete, evidence incomplete
BLOCKED: required prerequisite unavailable
NEGATIVE: measured evidence rejects the optimization line
```

如果 central evidence table、claim-bearing timing、batch scaling、trace acquisition 等任务仍然缺失，必须写 `PARTIAL` 或 `BLOCKED`。

### 9. Artifact 必须区分 measured、modeled、placeholder、control

所有 JSON/Markdown artifact 都必须显式标注数据来源。

推荐字段：

```json
{
  "data_label": "measured | modeled | placeholder | control",
  "claim_bearing": true,
  "measurement_unit": "slice | step | control",
  "source_artifact": "...",
  "provenance": "..."
}
```

claim-bearing 结论只能使用 `data_label = measured` 的行。control 和 modeled 行可以进入附录或 planning table，但不能满足最终 go/no-go。

### 10. 下一步计划优先补证据，不优先继续造工具

当已有 schema、calculator、report generator、instrumentation 后，下一版 plan 应优先解决证据缺口。

例如 GPU trace frontend 工业线的下一版 plan 应优先写成：

```text
BERT/Llama Trace Acquisition And Claim-Bearing Measurement Plan
```

而不是继续扩展 calculator 或 summary table。

## Zellij Manipulation

This project uses Zellij for terminal multiplexing. Use the following Python scripts to interact with other Zellij panes:

```bash
# Discover sessions and panes
python3 /home/dyf/zellij-manipulation-skill/scripts/find-sessions.py
python3 /home/dyf/zellij-manipulation-skill/scripts/find-panes.py

# Read pane content (e.g. check simulation output)
python3 /home/dyf/zellij-manipulation-skill/scripts/dump-pane.py --tab sim --lines 80

# Run a command in another pane
python3 /home/dyf/zellij-manipulation-skill/scripts/run-in-pane.py --tab sim -- <command>

# Send keystrokes to another pane
python3 /home/dyf/zellij-manipulation-skill/scripts/send-keys.py --tab sim --text "echo hello"
python3 /home/dyf/zellij-manipulation-skill/scripts/send-keys.py --tab sim --control enter
python3 /home/dyf/zellij-manipulation-skill/scripts/send-keys.py --tab sim --control ctrl-c

# Create / rename tabs
python3 /home/dyf/zellij-manipulation-skill/scripts/new-tab.py --name <tab-name>
python3 /home/dyf/zellij-manipulation-skill/scripts/rename-tab.py --tab <old> --name <new>
```

Workflow: use `find-panes.py` first to discover available tabs, then use `--tab <name>` to target a specific pane. If a tab has multiple terminal panes, add `--pane-id terminal_XX`.

## GPU Simulator

- Binary: `gpu-simulator/bin/release/accel-sim.out`
- Configs: `gpu-simulator/gpgpu-sim/configs/tested-cfgs/` (gpgpusim.config) and `gpu-simulator/configs/tested-cfgs/` (trace.config)
- Example traces: `exampleTraces/rodinia2/12.8/`
- Environment setup: `source gpu-simulator/setup_environment_no_git.sh` (requires `CUDA_INSTALL_PATH=/usr/local/cuda-12.8`)

Run example (from project root `simulator-remodeled/`):

```bash
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
source gpu-simulator/setup_environment_no_git.sh

OMP_NUM_THREADS=4 gpu-simulator/bin/release/accel-sim.out \
    -config gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_RTXA6000/gpgpusim.config \
    -config gpu-simulator/configs/tested-cfgs/SM86_RTXA6000/trace.config \
    -trace exampleTraces/rodinia2/12.8/nn-rodinia-2.0-ft/__data_filelist_4_3_30_90___data_filelist_4_3_30_90_result_txt/traces/dynamic_trace.pb
```
