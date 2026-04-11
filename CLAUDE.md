# CLAUDE.md

## 文档语言要求

所有诊断报告、实验总结、分析文档（`.md` 格式）均使用**中文**撰写。
代码注释、JSON 字段名、变量名、命令仍使用英文。

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
