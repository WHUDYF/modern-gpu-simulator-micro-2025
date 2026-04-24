# 后端执行桥第一版实施计划

## Goal Description

在当前仓库已经具备：

- `frontend anchor`
- `middle structure`
- `backend validation planning`
- `writeback` 协议接口

的基础上，补上缺失的 `backend execution bridge`，使当前方法链第一次能够从规划对象进入真实 simulator 执行侧。

本计划的长期设计目标对应完整链路：

`manifest -> execute -> collect -> parse -> result_summary -> writeback`

但本计划第一版实施范围明确收紧为：

`manifest -> command plan -> smoke execution -> minimal result summary`

也就是说，第一版实现的核心不是把整条后端系统一次性做满，而是先证明下面三件事成立：

1. 现有 `backend_run_manifest_v1.json` 能稳定映射成真实 simulator 命令
2. 至少少量 smoke runs 能真实执行、稳定落盘并保留原始输出
3. 这些输出能进入统一的 `backend_result_summary_v1.json` schema

第一版实现只服务当前 `mini_transformer_v4`，但架构与接口必须为未来多 workload 扩展保留清晰边界。

---

## Acceptance Criteria

- AC-1: 系统能够从现有 `backend_run_manifest_v1.json` 生成稳定的 command plan。
  - Positive Tests (expected to PASS):
    - 给定合法 manifest 和 `mini_transformer_v4` workload profile，能够生成一组 run specs，每行都包含 `run_id`、`command`、`output_dir`、`stdout_path`、`stderr_path`、`metadata_path`
    - 相同输入重复生成 command plan 时，命令内容与目录规划保持稳定
    - command plan 中能够解析出 trace 路径、config 路径与 scenario 对应关系
  - Negative Tests (expected to FAIL):
    - manifest 缺少 `run_id`、`regime_id` 或 `parameter_scenario_id` 时，不得继续生成 command plan
    - workload profile 缺 trace/config 映射时，不得生成不完整命令
    - scenario 无法映射到参数方向时，不得静默降级为默认命令

- AC-2: 系统能够为每个 `run_id` 建立稳定的输出目录和运行元数据约定。
  - Positive Tests (expected to PASS):
    - 每个 `run_id` 对应唯一目录，例如 `experiments/backend_pipeline/runs/<workload_id>/<run_id>/`
    - 每个 run 目录都能生成 `run_metadata.json`、`command.sh`、`stdout.log`、`stderr.log`
    - metadata 中至少记录 `run_id`、`workload_id`、`priority_source`、`regime_id`、`parameter_scenario_id`、命令摘要
  - Negative Tests (expected to FAIL):
    - 两个不同 `run_id` 落到同一输出目录时，应判定目录规划错误
    - 缺少 metadata 或日志路径时，应判定为不满足执行桥要求
    - 输出目录依赖随机命名且无法由 `run_id` 反查时，应判定失败

- AC-3: 系统能够执行第一轮 smoke runs，并保留真实原始输出。
  - Positive Tests (expected to PASS):
    - 给定少量选定 manifest 行，系统能真实调用 simulator 命令
    - 至少有一小组 smoke runs 能完成执行并写出 stdout/stderr 与执行状态
    - 对于成功 run，系统能记录 `start_time`、`end_time`、`exit_code`、`execution_status`
    - 对于失败或超时 run，系统也能保留原始日志并明确标注失败原因
  - Negative Tests (expected to FAIL):
    - 路径不存在、环境变量缺失、命令不可执行时，不得把 run 标为 success
    - 执行失败但没有 stdout/stderr 保留时，应判定不满足可追溯性要求
    - 超时 run 若被误记为普通成功 run，应判定失败

- AC-4: 系统能够从 smoke run 输出中提取最小指标并写入统一 `backend_result_summary_v1.json`。
  - Positive Tests (expected to PASS):
    - parser 至少能生成包含 `run_id`、`workload_id`、`family_id`、`regime_id`、`priority_source`、`parameter_scenario_id`、`execution_status`、`result_status`、`exit_code`、`sim_cycles`、`elapsed_wall_time`、`parse_note` 的 summary rows
    - 当 `sim_cycles` 可解析时，字段写入数值；不可解析时显式写 `null` 并附带 `parse_note`
    - `backend_result_summary_v1.json` 的每一行都能由 `run_id` 追溯到 run 目录
  - Negative Tests (expected to FAIL):
    - parser 找不到目标字段但仍将 row 记为 `result_status=success`，应判定失败
    - summary 中缺 `run_id` 或 `execution_status` 时，应判定 schema 不合格
    - 同一 `run_id` 被重复写入且没有显式处理时，应判定失败

- AC-5: 第一版实现必须保持现有 A/B/C 三线职责边界，不重写结构层与 writeback 规则。
  - Positive Tests (expected to PASS):
    - execution bridge 只消费现有 manifest / profile / scenario 信息，不重新生成 `family / regime / lane`
    - 第一版实现只服务 `mini_transformer_v4`，但 workload 特定知识收敛在 profile 层，而不是散落在执行主干中
    - `backend_result_summary_v1.json` 能被后续 writeback 使用，但第一版不强制做满自动 writeback
  - Negative Tests (expected to FAIL):
    - 在执行桥代码中重新定义 family assignment 或 regime priority，应判定越界
    - 为了跑通第一版而把 `mini_transformer_v4` 的 trace/config 路径全部硬编码进主执行逻辑，应判定为破坏扩展边界
    - 为追求“一次性全自动”而把 writeback 重写并耦合进 execution bridge，应判定 scope 失控

---

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

在不引入过度工程化的前提下，第一版可以完成：

- `workload profile resolver`
- `command builder`
- `run executor`
- `minimal metric parser`
- `backend_result_summary_v1.json` writer
- 少量真实 smoke execution
- 对应单元测试与 smoke test 脚本

并确保：

- `run_manifest` 到 `result_summary` 全链路可走通
- run 目录与日志可追溯
- parser 失败与执行失败可区分

### Lower Bound (Minimum Acceptable Scope)

至少完成：

- 从现有 manifest 生成稳定 command plan
- 统一 run 目录与日志规范
- 跑通少量真实 smoke runs
- 写出最小 `backend_result_summary_v1.json`

并保证：

- 每条成功或失败 run 都可按 `run_id` 回溯
- summary schema 稳定，不依赖人工临时解释

### Allowed Choices

- Can use:
  - Python CLI
  - JSON 作为 run metadata / summary 主格式
  - shell command wrapper 生成 `command.sh`
  - 现有 `experiments/backend_pipeline/` 目录结构
  - 固定 `mini_transformer_v4` workload profile 作为第一版唯一路径
- Cannot use:
  - 重写 `frontend anchor`
  - 重写 `family / regime / lane`
  - 扩展第二个 workload 作为第一版前置条件
  - 将完整 writeback 自动化作为第一版硬要求
  - 引入集群调度、队列系统或大规模并行执行框架

---

## Feasibility Hints and Suggestions

### Conceptual Approach

推荐按五段式实现，但只把前三段做硬、后两段保持最小化：

1. `profile resolver`
   - 根据 `workload_id` 提供 trace/config/env/parser 约定
   - 第一版仅支持 `mini_transformer_v4`

2. `command builder`
   - 读取 manifest 行和 profile
   - 生成稳定 run spec 与目录规划

3. `run executor`
   - 接收 run spec
   - 落盘命令、metadata、stdout、stderr
   - 运行少量 smoke cases

4. `minimal parser`
   - 从 run 输出中抽最小指标
   - 抽不到字段时显式失败，不伪造成功结果

5. `summary writer`
   - 将 parser rows 写成统一 `backend_result_summary_v1.json`

### Relevant References

- `/home/dyf/modern-gpu-simulator-micro-2025/docs/superpowers/specs/2026-04-24-backend-execution-bridge-design.md`
- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/plan_backend_validation.py`
- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/apply_backend_writeback.py`
- `/home/dyf/modern-gpu-simulator-micro-2025/experiments/backend_pipeline/results/mini_transformer_v4/backend_run_manifest_v1.json`
- `/home/dyf/modern-gpu-simulator-micro-2025/docs/integrated-method-baseline-2026-04-23.md`

### Scope Compression Rule

如果中途出现 scope 膨胀，优先保留：

1. `command generation`
2. `output directory convention`
3. `minimal parser`
4. `minimal result summary`

不要优先去做：

- 完整 writeback 自动化
- 完整 baseline 比较统计
- 多 workload 并行接入

---

## Dependencies and Sequence

### Milestones

1. Milestone 1: 固定 execution bridge 输入输出接口
   - Phase A: 定义 `mini_transformer_v4` workload profile 输入字段
   - Phase B: 定义 run spec、metadata、summary row 的最小 schema

2. Milestone 2: 落地 command plan 与目录规范
   - Phase A: 从 manifest 生成稳定命令
   - Phase B: 生成统一输出目录、日志与 metadata 约定

3. Milestone 3: 跑通第一轮 smoke execution
   - Phase A: 选少量 smoke runs
   - Phase B: 真实执行并保留原始输出

4. Milestone 4: 落地最小 parser 与 summary writer
   - Phase A: 提取 `sim_cycles` / `elapsed_wall_time` / 执行状态
   - Phase B: 写入统一 `backend_result_summary_v1.json`

5. Milestone 5: 完成回归测试与人工核对
   - Phase A: 增加单元测试和样例测试
   - Phase B: 核对 smoke outputs 与 summary 一致性

### Task Breakdown

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 设计 `mini_transformer_v4` workload profile schema 与解析入口 | AC-1, AC-5 | coding | - |
| task2 | 实现 manifest 到 run spec 的 command builder | AC-1, AC-2 | coding | task1 |
| task3 | 实现统一 run 目录、metadata、日志文件约定 | AC-2 | coding | task2 |
| task4 | 实现 smoke execution runner | AC-3 | coding | task3 |
| task5 | 实现最小 parser 与 `backend_result_summary_v1.json` writer | AC-4 | coding | task4 |
| task6 | 增加单元测试、样例测试与 smoke test 验证 | AC-1, AC-2, AC-3, AC-4 | coding | task5 |

---

## Implementation Notes

### Code Style Requirements

- 代码与注释中不要写 `AC-1`、`Milestone`、`Phase` 等计划术语
- 这些术语只属于计划文档，不属于最终实现
- 代码命名应使用领域语义，例如：
  - `workload_profile`
  - `run_spec`
  - `execution_status`
  - `result_summary_row`
  - `parser_report`

### Boundary Rules

- `execution bridge` 是执行层，不是对象定义层
- 不允许在实现中重新生成 `family / regime / lane`
- `backend_result_summary_v1.json` 是执行层输出接口，不是方法结论文件
- 第一版允许保留手工选择 smoke runs，但不允许手工拼 summary

### First-Version Success Definition

第一版的“成功”不是：

- 跑很多 run
- 做完整平台
- 证明所有 baseline 优势

第一版的“成功”是：

- 证明当前 `backend_run_manifest_v1.json` 能驱动真实 simulator 命令
- 证明少量真实 runs 能稳定落盘并收集
- 证明这些输出能进入统一结果 schema，为后续 writeback 与优化调参提供基础
