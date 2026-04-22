# 前端压缩实现计划

## 目标说明

实现仓库方法链中第一版 A 线前端压缩流水线。该实现必须从双数据源输入构建稳定的
`KernelInvocationRecord` 表，在三种前端分组模式
（`name-only`、`PKA-like coarse`、`hybrid`）下生成 representative anchors，
导出可被后续 B 线消费的 `Representative Anchor Table`，并同时产出最小证据产物
（`Comparison Table` 与 `Case Note`），用于说明为什么在前端加入轻量 clustering
是有价值的。

本计划的范围是明确收紧的：

- 它必须保持为 `Constrained PKA Extension`，而不是新的 sampled-simulation 方法。
- 它不能输出 family、regime 或 simulator lane 结论。
- v1 允许存在 provisional 字段，但每个 provisional 字段都必须显式标注来源与状态。

## 验收标准

- AC-1: 存在一个双数据源 invocation builder，能够为 `mini_transformer_v4`
  生成稳定的 `KernelInvocationRecord` 数据集。
  - 正向测试（预期 PASS）：
    - 一个 CLI 入口能接收两个 source 输入，并将标准化后的 invocation 表写入结果路径。
    - 生成后的表包含必需的 identity/context 字段：
      `kernel_invocation_id`、`kernel_name`、`trace_order`、`grid_dim`、
      `block_dim`，以及承载特征的字段。
    - 该表的生成流程可以直接在仓库内的 `mini_transformer_v4` 输入上运行，
      不依赖人工编辑数据。
  - 反向测试（预期 FAIL）：
    - 缺少 `enhanced_execution_info.json` 时，返回清晰的校验错误。
    - 缺少 merged feature JSON 时，返回清晰的校验错误。
    - 若输入无法对齐成 invocation records，流程必须显式报错，不能静默输出部分行。

- AC-2: 前端 selector 必须且只能支持三种 v1 分组模式：
  `name-only`、`PKA-like coarse`、`hybrid`。
  - 正向测试（预期 PASS）：
    - 用户可以基于同一份 invocation 表分别请求三种模式。
    - `name-only` 仅按 `kernel_name` 分组。
    - `PKA-like coarse` 仅按显式 coarse metadata/signature 分桶，不进行 clustering。
    - `hybrid` 先执行相同的 coarse bucketing，再在桶内进行 lightweight clustering。
  - 反向测试（预期 FAIL）：
    - 未知 selector mode 会被明确拒绝，并返回 CLI/config 错误。
    - `PKA-like coarse` 不得调用 clustering 内部逻辑。
    - `hybrid` 不得跳过 coarse bucketing 而直接全局聚类。

- AC-3: 存在一个 `Representative Anchor Table` exporter，能够输出满足
  下游集成要求的 A 线主线对象。
  - 正向测试（预期 PASS）：
    - 导出的 anchor 表包含：
      `rep_kernel_id`、`kernel_name`、`cluster_id`、`member_invocations`、
      `coverage_count`、`coverage_weight`、`time_weight`。
    - 若数据可用，则同时导出强烈建议字段：
      `trace_order_summary`、`shape_hint_summary`、`grid_dim`、`block_dim`。
    - 对非最终值导出 source/provisional 字段，例如：
      `coverage_weight_source`、`time_weight_source`、
      `member_invocations_status`、`heterogeneity_flag`。
  - 反向测试（预期 FAIL）：
    - 缺少必填字段时仍然声称导出成功。
    - exporter 将 family、regime、route primitive、execution template 或
      simulator lane 标签写入 anchor 表。
    - provisional 字段没有来源/状态说明。

- AC-4: 存在最小证据输出，并且与主线输出严格分离。
  - 正向测试（预期 PASS）：
    - 流水线写出一张 `Comparison Table`，至少包含：
      `method`、`num_anchors`、`time_weight_covered`、`avg_cluster_size`、
      `intra_cluster_exec_time_var`、`intra_cluster_inst_var`、
      `split_cases_count`、`notes`。
    - 流水线写出一份短的 `Case Note`，至少解释一个 `hybrid`
      能区分而简单 baseline 不能区分的代表性 split case。
    - 实现明确标注这些产物是 evidence outputs，而不是下游主线输入。
  - 反向测试（预期 FAIL）：
    - 在声称 clustering 有价值的同时缺少证据产物。
    - Comparison 输出被直接当成 B 线的标准输入表。
    - Case Note 给出最终的 family/regime 结论，而不是前端层观察。

- AC-5: squash 相关信息作为 context/guardrail 接入，而不是变成前端主分组轴。
  - 正向测试（预期 PASS）：
    - 实现或文档将 squash-derived summary 只放在辅助上下文或 guardrail 位置。
    - v1 的 squash 特征集遵循约定的 resource-balanced 策略，并明确区分
      必须真实的核心特征与可选增强特征。
    - squash 集成可以触发 heterogeneity 或 boundary 警告，但不直接给出
      family/regime 结论。
  - 反向测试（预期 FAIL）：
    - squash 替代前端 selector 的主特征路径。
    - squash segments 被直接当成最终前端 anchors、families 或 regimes。
    - 将后段结构层的重型特征引入 v1 squash 特征集。

- AC-6: v1 交付必须包含一份 `Frontend Compression Note`，使初版结果可审计、可集成。
  - 正向测试（预期 PASS）：
    - note 解释 anchors 是如何得到的。
    - note 解释 `member_invocations` 的粒度。
    - note 说明 `coverage_weight` 与 `time_weight` 是 measured、derived、
      provisional 还是 placeholder。
    - note 明确当前前端输出最可能的偏差来源。
  - 反向测试（预期 FAIL）：
    - note 遗漏 placeholder/provisional 披露。
    - note 在字段仍部分近似的情况下，错误宣称 anchors 来自“真实压缩”。

## 路径边界

### 上界（最大可接受范围）

可接受的实现可以新增：

- 一个可复用的 invocation-table builder 模块
- 一个可复用的 frontend selector 模块
- 一个可复用的 anchor exporter 模块
- 一个可复用的 comparison/case-note report builder
- 覆盖三种 selector mode 与主线输出 schema 的最小测试
- 与当前 spec 一致的轻量 squash summary/guardrail 集成

还可以新增：

- 面向 `mini_transformer_v4` 的专用结果目录
- 一个运行完整 A 线前端流程的主 CLI 入口

### 下界（最小可接受范围）

最小可接受实现为：

- 一个可运行的 CLI 入口
- 一条内部 table-builder 路径
- 三种可选分组模式
- 一份导出的 `Representative Anchor Table`
- 一份导出的 `Comparison Table`
- 一份简短的 `Frontend Compression Note`

只有在输出 schema 稳定、且主线输出与证据输出明确分离时，这个最小范围才成立。

### 允许的选择

- 可以使用：
  - 现有 `experiments/baseline_diagnosis` Python 工具链
  - 现有 merged feature JSON 与 enhanced execution JSON 输入
  - 仓库内 `mini_transformer_v4` 的本地结果文件
  - 环境中已有的轻量 clustering 依赖，或者更简单的仓库内实现
  - 放在 experiment result 目录下的额外 JSON/Markdown 输出
- 不可以使用：
  - 在 A 线输出中加入新的 family/regime/simulator-lane 标签
  - 重写仓库现有方法链
  - 宣称“完整复现 PKA”
  - 任何让 squash 变成前端主分组逻辑的方案
  - 将计划术语直接写进面向实现的字段名

## 依赖与执行顺序

### 里程碑 1：标准化输入

- 阶段 A：确认 `mini_transformer_v4` 所用的双数据源文件及其对齐策略
- 阶段 B：定义并生成 `KernelInvocationRecord`
- 阶段 C：为必需字段与对齐失败场景添加表级验证/测试

### 里程碑 2：构建前端 selectors

- 阶段 A：实现 `name-only` 分组
- 阶段 B：实现 `PKA-like coarse` 分组
- 阶段 C：在 coarse bucket 之上实现 `hybrid`
- 阶段 D：补充 selector 测试，证明三种模式彼此区分且非法 mode 会被拒绝

### 里程碑 3：导出主线产物

- 阶段 A：导出 `Representative Anchor Table`
- 阶段 B：补充 source/provisional 标注字段
- 阶段 C：撰写 `Frontend Compression Note`
- 阶段 D：验证 A 线必需字段齐全且保持 downstream-safe

### 里程碑 4：生成证据产物

- 阶段 A：生成 `Comparison Table`
- 阶段 B：生成简短的 `Case Note`
- 阶段 C：确认 evidence outputs 不被误当成主线输入

### 里程碑 5：接入 squash guardrails

- 阶段 A：对齐 squash 所需字段与 merged/NCU 现有字段命名
- 阶段 B：决定哪些 squash-derived summary 字段进入 invocation/context 数据
- 阶段 C：加入 guardrail 导向的 squash 字段，但不让 squash 成为主分组轴

## 实施说明

- 代码中不要出现 `AC-1`、`Milestone 2`、`Upper Bound` 之类计划术语。
- 主线输出与证据输出必须存放在不同文件中，并采用清晰可辨的命名。
- A 线输出 schema 必须保持保守、可集成。
- 当某字段是 provisional 时，必须在输出数据或 note 中显式标注，不能隐藏不确定性。
- 优先采用稳定 ID 与可重复的输出顺序，保证 A/B/C 三条线后续容易拼接。
- 如果某个实现选择实质性偏离当前 `Constrained PKA Extension` 定位，必须先记录偏离点，再决定是否继续扩展。
