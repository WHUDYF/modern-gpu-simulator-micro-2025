# 后端执行桥设计

日期：2026-04-24

## 1. 文档目的

这份设计文档用于定义当前方法链中缺失的 `backend execution bridge`。

当前仓库已经具备：

- frontend anchor 输入层
- middle structure 对象层
- backend validation planning / writeback 协议层

但仍然缺少一层把 `backend_run_manifest_v1.json` 真正接到 simulator 执行侧的桥接层。

本设计文档要解决的问题是：

1. 如何把 `run manifest` 稳定映射成真实 simulator 命令
2. 如何让少量真实 runs 落盘并保留统一目录约定
3. 如何把 simulator 输出收集并写入统一的 `backend_result_summary_v1.json`
4. 如何在不破坏现有 A/B/C 三线对象边界的前提下，为后续完整 writeback 闭环预留接口

---

## 2. 背景与当前缺口

当前统一方法基线已经可以写成：

`frontend anchor -> family -> regime -> importance -> lane -> backend validation planning -> writeback interface`

其中：

- `plan_backend_validation.py` 已经能够生成：
  - `backend_run_manifest_v1.json`
  - `backend_scenario_matrix_v1.json`
  - `backend_baseline_plan_v1.json`
  - `backend_result_summary_v1.json` 空模板
- `apply_backend_writeback.py` 已经能够消费 `result summary`

当前缺口不在规划层，也不在 writeback 协议层，而在于：

**`backend_run_manifest_v1.json` 还不能直接驱动真实 simulator 执行。**

这导致当前方法链虽然已经形成：

- 对象闭环
- 规划闭环
- 接口闭环

但还没有形成：

**真实执行闭环**

---

## 3. 目标与非目标

### 3.1 长期目标

长期目标是让当前方法链形成完整的执行回流通路：

`manifest -> execute -> collect -> parse -> result_summary -> writeback`

也就是说，后端执行桥最终不只负责“发命令”，还负责：

- 统一执行记录
- 统一结果收集
- 统一结果摘要
- 为 writeback 提供稳定输入

### 3.2 第一版实现目标

第一版实现不追求把完整链路一次性做满。

第一版只要求形成一个最小可证闭环：

`manifest -> command plan -> smoke execution -> minimal result summary`

第一版成功的核心标准只有三条：

1. `run_manifest` 能稳定映射成真实 simulator 命令
2. 至少少量 smoke runs 能成功落盘并收集结果
3. 结果能进入统一的 `backend_result_summary_v1.json` schema

### 3.3 非目标

第一版明确不覆盖：

- 完整自动化大规模批跑系统
- 多 workload 同时上线
- 重写现有 B 线 family / regime 规则
- 一次性完成完整 writeback 自动闭环
- 完整 baseline 统计对比与优先级有效性证明
- 集群调度、队列管理、资源分配平台化

---

## 4. 设计路线选择

本设计考虑过三种路线：

### 4.1 路线 A：先做 `mini_transformer_v4` 专用执行器，后续再抽象

优点：

- 最快落地
- 最容易先看到跑通结果

缺点：

- 会把 workload 特定逻辑硬编码进执行主干
- 后续扩 workload 时很可能重写 command builder、parser 和目录规则
- 与“方法本身具有 workload 可扩展性”的目标不一致

### 4.2 路线 B：做轻量通用执行桥，第一版仅接 `mini_transformer_v4`

优点：

- 主干接口从第一天起就是 workload 可扩展的
- 第一版 scope 仍然可控
- 可以在不扩第二个 workload 的前提下，为后续 workload 扩展保留清晰扩展点

缺点：

- 比纯 hardcode 多一层 profile 抽象
- 需要在 spec 中更清楚地定义接口边界

### 4.3 路线 C：直接做完整通用执行框架

优点：

- 远期扩展空间最大

缺点：

- 当前阶段明显过重
- 会把验证方法有效性的优先级让位给框架建设

### 4.4 最终选择

本设计采用路线 B：

**做一个轻量通用 execution bridge，第一版实现只支持 `mini_transformer_v4`，但接口和数据模型按未来多 workload 扩展来设计。**

---

## 5. 系统边界

本次设计新增的系统层为：

`backend execution bridge`

它位于：

- 上游：`backend_run_manifest_v1.json`
- 下游：`backend_result_summary_v1.json`

它不负责：

- 重写 `frontend anchor`
- 重写 `family / regime / lane`
- 重新计算 priority
- 重新定义 writeback 规则

它负责：

- 把 manifest 行转成真实执行请求
- 组织目录和运行元数据
- 调起 simulator 命令
- 收集最小结果
- 把结果写入统一 summary schema

因此，它的职责是：

**把规划对象变成真实执行对象，并把真实执行结果变成可回流对象。**

---

## 6. 核心架构

### 6.1 总体数据流

完整目标链路定义为：

`backend_run_manifest -> workload profile -> command builder -> run executor -> output collector -> metric parser -> backend_result_summary -> writeback`

其中第一版实际实现只保证：

`backend_run_manifest -> workload profile -> command builder -> run executor -> metric parser -> backend_result_summary`

说明：

- `output collector` 是完整链路中的独立语义步骤
- 第一版实现中暂不单独拆文件级组件，而是把它并入 `run executor` 和 `metric parser` 的交界处
- 后续若输出形态复杂化，再把它提升为独立组件

### 6.2 组件划分

#### 组件 1：Workload Profile Resolver

职责：

- 根据 `workload_id` 或运行上下文加载 workload 执行描述
- 提供 trace/config 根路径、默认环境变量、支持的 parser、默认指标抽取规则

设计原则：

- workload 特定知识只留在 profile 中
- execution bridge 主干不直接硬编码 workload 业务语义

第一版要求：

- 只实现 `mini_transformer_v4` profile
- 但 profile 接口要支持未来新增 workload

#### 组件 2：Command Builder

职责：

- 将 manifest 行、workload profile、scenario 定义拼接成稳定的 run spec

run spec 至少包含：

- `run_id`
- `workload_id`
- `priority_source`
- `regime_id`
- `parameter_scenario_id`
- `command`
- `env`
- `output_dir`
- `stdout_path`
- `stderr_path`
- `metadata_path`

设计原则：

- 只做命令生成，不负责执行
- 一条 manifest 行生成一条稳定 run spec

#### 组件 3：Run Executor

职责：

- 接收 run spec
- 真实调用 simulator
- 记录运行状态

输出至少包括：

- `run_id`
- `start_time`
- `end_time`
- `exit_code`
- `execution_status`
- `output_dir`

第一版要求：

- 只支持少量 smoke execution
- 不做并发调度
- 不做集群适配

#### 组件 4：Metric Parser

职责：

- 从 run 输出目录中提取第一版最小指标

第一版最小字段包括：

- `run_id`
- `result_status`
- `exit_code`
- `execution_status`
- `sim_cycles`
- `elapsed_wall_time`
- `parse_note`

设计原则：

- 解析不到字段时显式写 `null` 或失败原因
- 不允许把 parse failure 伪装成 success

#### 组件 5：Result Summary Writer

职责：

- 将 parsed rows 写成稳定的 `backend_result_summary_v1.json`

设计原则：

- 只负责统一 schema 落盘
- 不重新做策略判断
- 不直接负责 writeback

---

## 7. 数据模型

### 7.1 输入对象：manifest row

执行桥消费现有 `backend_run_manifest_v1.json`，不修改其主语义。

第一版假定单行至少包含：

- `run_id`
- `family_id`
- `regime_id`
- `priority_source`
- `priority_rank`
- `simulator_lane_id`
- `parameter_scenario_id`
- `recommended_tuning_target`
- `validation_role`
- `expected_signal`

现有 manifest 行未必显式携带 `workload_id`。

因此第一版约定：

- 执行桥 CLI 或 profile resolver 必须提供一个明确 `workload_id`
- 该 `workload_id` 在第一版固定为 `mini_transformer_v4`
- 后续若 manifest 升级为显式带 `workload_id`，执行桥优先使用 manifest 字段

### 7.2 中间对象：run spec

新增中间执行对象 `run spec`，用于把“规划信息”和“执行信息”分开。

它至少包含：

- `run_id`
- `workload_id`
- `command`
- `env`
- `cwd`
- `output_dir`
- `stdout_path`
- `stderr_path`
- `metadata_path`
- `trace_path`
- `config_paths`

### 7.3 输出对象：minimal result summary row

第一版 `backend_result_summary_v1.json` 的最小行结构定义为：

- `run_id`
- `workload_id`
- `family_id`
- `regime_id`
- `priority_source`
- `parameter_scenario_id`
- `execution_status`
- `result_status`
- `exit_code`
- `sim_cycles`
- `elapsed_wall_time`
- `parse_note`
- `summary_version`

说明：

- `execution_status` 表示命令是否成功执行
- `result_status` 表示执行结果是否可用于后续 interpretation
- 两者不能混用

---

## 8. 目录与文件约定

第一版必须固定统一输出目录约定。

建议目录形态如下：

`experiments/backend_pipeline/runs/<workload_id>/<run_id>/`

每个 `run_id` 目录下至少包含：

- `run_metadata.json`
- `stdout.log`
- `stderr.log`
- `command.sh`
- `parser_report.json`

如果 simulator 产生额外 stats 文件，则保留原始文件，不重命名其内部格式，只在 metadata 中记录路径。

目录约定的目的不是好看，而是保证：

- 每个 run 都可追溯
- parser 可以稳定找到输入
- 失败 run 也能保留诊断信息

---

## 9. 错误处理

第一版将错误分为四类。

### 9.1 Planning Error

发生在：

`manifest -> run spec`

典型原因：

- manifest 缺字段
- workload profile 缺失
- scenario 映射不存在
- trace/config 路径无法解析

处理原则：

- fail-fast
- 不进入执行阶段
- 状态标记为 `plan-failed`

### 9.2 Execution Error

发生在：

真实 simulator 命令执行阶段

典型原因：

- 命令启动失败
- 环境变量不完整
- trace/config 路径不存在
- 超时
- 非零退出码

处理原则：

- 保留 stdout/stderr
- 保留 exit code
- 状态标记为 `run-failed` 或 `timeout`

### 9.3 Parse Error

发生在：

输出已落盘但 parser 无法抽字段

典型原因：

- 目标 stats 文件不存在
- 输出格式不匹配
- `sim_cycles` 等关键字段缺失

处理原则：

- 允许“执行成功但解析失败”
- 明确标记为 `parse-failed`
- 不假装生成有效结果

### 9.4 Summary Error

发生在：

结果写入 `backend_result_summary_v1.json` 时

典型原因：

- `run_id` 对不齐
- 重复 row
- 缺关键字段

处理原则：

- 阻止 summary 落盘
- 优先保证 summary 的结构可信

---

## 10. 第一版实现收紧策略

### 10.1 完整 spec 与第一版实现的关系

本 spec 覆盖完整链路：

`manifest -> execute -> collect -> parse -> result_summary -> writeback`

但第一版 implementation 只承诺做到：

`manifest -> command plan -> smoke execution -> minimal result summary`

### 10.2 第一版优先级

若 scope 出现膨胀，第一版优先保留：

1. `command generation`
2. `output directory convention`
3. `minimal parser`
4. `minimal result summary`

第一版可以暂时不做满：

- 完整 writeback 自动化
- 全量 manifest 批跑
- 多 workload profile 同时接入
- 完整指标矩阵

### 10.3 第一版 workload 约束

第一版实现只服务：

- `mini_transformer_v4`

但架构约束必须满足：

- 执行桥主干不写死 workload 逻辑
- workload 特定知识通过 profile 注入

这样做的目的是：

- 当前先验证 B 线对象能否驱动真实后端执行
- 未来扩其他 workload 时不需要推倒 execution bridge 主干

---

## 11. 测试策略

第一版测试分四层。

### 11.1 单元测试

覆盖：

- profile 解析
- command builder
- 目录规划
- minimal parser
- result summary writer

### 11.2 样例测试

给定少量 manifest 行，验证：

- 生成的命令稳定
- 输出路径稳定
- metadata 完整

### 11.3 Smoke Test

真实跑 1 到 3 个最小 run，验证：

- 命令可执行
- 至少少量 run 成功落盘
- parser 可从真实输出中抽最小字段

### 11.4 回归测试

固定 `backend_result_summary_v1.json` 的最小 schema，避免后续字段漂移。

---

## 12. 第一版验收标准

### 12.1 硬门槛

第一版必须满足：

1. 给定 `backend_run_manifest_v1.json`，系统能生成稳定 `command plan`
2. 每个 `run_id` 都有稳定输出目录、日志路径和 metadata 路径
3. 至少少量 smoke runs 能真实执行并成功落盘
4. parser 能从这些 run 中抽出最小字段
5. 系统能写出统一的 `backend_result_summary_v1.json`

### 12.2 最小字段要求

第一版 summary 至少包含：

- `run_id`
- `workload_id`
- `family_id`
- `regime_id`
- `priority_source`
- `parameter_scenario_id`
- `execution_status`
- `result_status`
- `exit_code`
- `sim_cycles`
- `elapsed_wall_time`
- `parse_note`

### 12.3 有效性判断

第一版“证明有效”不是指：

- 已经大规模跑了很多 cases
- 已经完整证明 `importance-guided` 优于所有 baseline

第一版“证明有效”是指：

1. 整个 execution bridge 通路被打通
2. 当前 B 线对象能真实驱动后端执行
3. 这个通路为后续优化调参和 baseline 对照提供可执行基础

---

## 13. 后续扩展方向

在第一版闭环成立之后，再推进下面这些方向：

- 扩展第二个 workload profile
- 增加更多 scenario 类型
- 增加更多 parser 指标
- 让 `backend_result_summary` 更好地衔接 `apply_backend_writeback.py`
- 做 `importance-guided`、`time-only`、`no-priority` 的真实对照
- 支持更系统的 run budgeting 和实验批量化

这些方向属于后续演进，不属于第一版硬要求。

---

## 14. 简短结论

当前最缺的不是新的方法对象，而是把已有方法对象真正接到 simulator 的执行桥。

因此，本设计的核心判断是：

1. 新增一个轻量通用 execution bridge
2. 接口从第一天起按多 workload 扩展设计
3. 第一版实现只先服务 `mini_transformer_v4`
4. 第一版先打通：
   `manifest -> command plan -> smoke execution -> minimal result summary`
5. 在此基础上，再逐步推进完整 writeback 与 baseline 优化验证

这个设计既保证：

- 当前能尽快证明 idea 通路可行

又保证：

- 后续扩 workload 时不用推倒重来
