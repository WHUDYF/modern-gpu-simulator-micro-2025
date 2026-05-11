# 进度日志

## 会话：2026-05-08

### 阶段 1：需求与边界
- **状态：** complete
- 执行的操作：
  - 确认当前 worktree 是 `trace-compressions-industrial-codex-workload`。
  - 检查磁盘空间，`/home` 可用空间约 6.1T。
  - 确定源码放在 worktree 外部 `/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources`。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 阶段 2：源码获取
- **状态：** complete
- 执行的操作：
  - 创建 clone manifest 和可重复运行的下载脚本。
  - 下载 12 个 workload source repositories。
  - 对 `mlperf-inference` 和 `hecbench` 使用 sparse/partial clone 绕过普通 shallow clone 停滞。
  - 生成外部状态文件 `/home/dyf/workloads/trace-compressions-industrial-codex-workload/clone_status.tsv`。
- 创建/修改的文件：
  - `docs/workload_source_manifest.md`
  - `scripts/clone_workload_sources.sh`
  - `docs/workload_download_status.md`

### 阶段 3：盘点与筛选
- **状态：** complete
- 执行的操作：
  - 将 workload suite 目标升级为 GPU workload trace corpus。
  - 定义 L0 source/harness、L1 workload registry、L2 launch metadata、L3 measured feature、L4 trace artifact、L5 training dataset 分层。
  - 创建 GPU Workload Trace Corpus 设计文档。
- 创建/修改的文件：
  - `docs/superpowers/specs/2026-05-11-gpu-workload-trace-corpus-design.md`

### 阶段 4：交付
- **状态：** complete
- 执行的操作：
  - 已检查设计文档是否存在待填项。
  - 已检查 git 状态。
  - 根据用户批准，创建 Gate C0/C1 registry 实施计划。
- 创建/修改的文件：
  - `docs/superpowers/plans/2026-05-11-gpu-workload-trace-corpus-registry.md`

### 阶段 5：Gate C0/C1 Registry
- **状态：** complete
- 执行的操作：
  - 生成 source registry。
  - 生成 workload registry draft。
  - 运行 registry tests 和 baseline tests。
- 创建/修改的文件：
  - `scripts/generate_source_registry.py`
  - `scripts/generate_workload_registry.py`
  - `tests/test_workload_registry_tools.py`
  - `registry/source_registry.json`
  - `registry/source_registry.md`
  - `registry/workload_registry.json`
  - `registry/workload_registry.md`

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 磁盘空间检查 | `/home` | 足够下载源码候选池 | 约 6.1T 可用 | pass |
| clone 脚本 | 12 个第三方源码来源 | 生成可读 commit 状态 | 全部有本地 commit | pass |
| 进程检查 | clone/sparse 相关进程 | 无残留下载进程 | 无残留 | pass |
| 设计文档 | 完整网络和训练集需求 | 有 corpus 分层、gate、完成状态 | 已创建 | pass |
| 文档自检 | `rg TBD/TODO/待定/FIXME` | 无未完成占位 | 仅命中禁止 placeholder 的规则说明 | pass |
| 实施计划 | Gate C0/C1 | 覆盖 source registry 和 workload registry draft | 已创建 | pass |
| registry tests | `pytest -q tests/test_workload_registry_tools.py` | 12 passed | 12 passed | pass |
| source registry generation | `python scripts/generate_source_registry.py --generated-at 2026-05-11T00:00:00+00:00` | exit 0 | exit 0 | pass |
| workload registry generation | `python scripts/generate_workload_registry.py` | exit 0 | exit 0 | pass |
| baseline tests | `pytest -q tests/test_build_kernel_cards.py tests/test_build_middle_layer.py tests/test_check_analysis_cards.py` | 22 passed | 22 passed | pass |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-05-08 | `mlperf-inference` 普通 shallow clone 长时间停在约 257M | 1 | 终止该 clone，改用 sparse/partial clone |
| 2026-05-08 | `hecbench` 普通 shallow clone 长时间停在约 257M | 1 | 终止该 clone，改用 sparse/partial clone 并展开源码目录 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 阶段 5：Gate C0/C1 Registry 已完成 |
| 我要去哪里？ | 进入 Gate C2/C3 build/run/input/license 验证 |
| 目标是什么？ | 建立可复用 GPU workload trace corpus，为 Photon 和后续训练集服务 |
| 我学到了什么？ | 见 findings.md |
| 我做了什么？ | 见上方记录 |
