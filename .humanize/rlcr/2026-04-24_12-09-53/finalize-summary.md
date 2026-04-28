# RLCR Final Summary — Backend Execution Bridge

日期：2026-04-24 至 2026-04-27
总轮次：16 (Round 0 至 Round 15)
Codex 模型：gpt-5.4 (high effort)

## 实施目标

补上从 `backend_run_manifest_v1.json` 到真实 simulator 执行的 "backend execution bridge"，完成链路：`manifest -> command plan -> smoke execution -> minimal result summary`。

## 验收状态

| AC | 描述 | 状态 | 验证轮次 |
|----|------|------|----------|
| AC-1 | manifest 生成稳定 command plan | 通过 | Round 4 |
| AC-2 | 每个 run_id 建立稳定输出目录和元数据 | 通过 | Round 4 |
| AC-3 | 执行 smoke runs 并保留原始输出 | 通过 | Round 4 |
| AC-4 | 提取最小指标写入 result_summary | 通过 | Round 4 |
| AC-5 | 保持 A/B/C 三线职责边界 | 通过 | Round 0 |

全部 5 条验收准则已于 Round 4 满足。Round 5-12 持续加固；Round 13-15 转向 A-Line L1 RLCR 计划生成及两个 bug 修复。

## 关键文件

- `experiments/backend_pipeline/execution_bridge.py` — 执行桥主干
- `experiments/backend_pipeline/workload_profiles.py` — workload profile 解析与 smoke trace builder
- `experiments/backend_pipeline/run_backend_execution.py` — CLI 入口
- `experiments/backend_pipeline/tests/test_execution_bridge.py` — 执行桥测试
- `experiments/backend_pipeline/tests/test_backend_builder.py` — builder 测试
- `docs/superpowers/plans/2026-04-24-backend-execution-bridge-implementation.plan.md` — 实施计划
- `docs/superpowers/plans/2026-04-27-a-line-l1-rlcr.zh.plan.md` — A-Line L1 RLCR 中文计划（Round 13 gen-plan 产出）

## 测试结果

`pytest -q experiments/backend_pipeline/tests/test_execution_bridge.py experiments/backend_pipeline/tests/test_backend_builder.py experiments/backend_pipeline/tests/test_plan_backend_validation.py experiments/backend_pipeline/tests/test_apply_backend_writeback.py` → **30 passed**

## Round 14 Bug 修复

1. **Custom profile smoke trace builder 继承问题** (`workload_profiles.py`)：自定义 profile 在 smoke 模式下不再静默继承内置 `smoke_trace_builder`，缺失时抛出 `ValueError`。
2. **Profile working directory 未被 command script 遵循** (`execution_bridge.py`)：`render_command_script()` 现在优先使用 `simulator_working_directory`。

## 遗留事项

- 当前 smoke 执行使用 `-gpgpu_max_cycle 10`（精简 trace），为刻意设计选择，非缺陷
- 全自动 writeback 不在第一版范围内
- 仅支持 `mini_transformer_v4`；多 workload 扩展保留在架构边界内

## 分支信息

- 分支：`dyf/docs/frontend-anchor-model`
- Base：`main` @ `742b23d8c93780450cbfd86c24c23b243ae3db08`
- 共计 206 文件变更，80760 行新增，1401 行删除
