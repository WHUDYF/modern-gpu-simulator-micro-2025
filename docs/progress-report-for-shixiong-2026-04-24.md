# 当前工作进展汇报

日期：2026-04-24

## 1. 当前我们在做什么

我们当前工作的核心，不是单纯做 trace compression，也不是单纯做 simulator 调参，
而是在这两者之间补上一层原本缺失的：

**从 workload 到 simulator validation 的结构化决策层。**

如果把当前方法链压成一句话，可以写成：

`frontend anchor -> family -> regime -> importance -> lane -> backend validation`

也就是说，我们希望把原本比较依赖人工经验的调参流程，变成一条：

- 有对象边界
- 有优先级
- 有验证入口
- 有结果回写

的方法链。

---

## 2. 我们目前已经完成了什么

在展开 A/B/C 三条线之前，先说明两个我们反复使用的词：

### 2.0.1 什么叫“输入链条”

这里的“输入链条”，不是指一个抽象概念，
而是指：

**上游原始数据如何被逐步读取、对齐、标准化，最后变成下游可以稳定消费的对象。**

例如：

- 哪些 JSON 是输入
- 哪些字段来自 identity source
- 哪些字段来自 feature source
- 哪一步把它们拼成统一对象
- 最后输出成哪张对象表

所以当我们说“完成了输入链条”，意思不是“我们有输入文件了”，
而是：

**我们已经把原始 source files 变成了稳定、可校验、可重复生成的标准输入对象。**

### 2.0.2 什么叫“builder”

这里的 `builder` 不是某个外部框架，而是仓库里一类非常具体的程序角色：

- 读取一组固定输入
- 做标准化、映射、组织、校验
- 生成一组稳定的结构化产物

也就是说，builder 不只是“脚本”，而是：

**把上游数据构造成下游对象表的构建器。**

比如：

- A 线 builder：把原始 invocation source 构造成 anchor 输入表
- B 线 builder：把 anchor 和规则构造成 family / regime / lane
- C 线 builder：把这些对象构造成 run manifest / scenario / writeback
- execution bridge：把 manifest 进一步构造成真实 simulator run 和 result summary

### 2.1 A 线：frontend anchor 输入层

这一层的目标是：

把 workload 中原始 kernel invocation 压缩成后段可以消费的 representative anchors。

目前已经完成：

- dual-source 输入链
- invocation table builder
- representative anchor table
- comparison table
- case note / frontend note

当前 `mini_transformer_v4` 上的规模是：

- `14` 条 invocation records
- `8` 条 representative anchors

更具体地说，A 线内部已经完成了三步：

#### 第一步：把原始 source files 变成统一输入对象

A 线当前采用的是 `dual-source` 输入：

- 一组 source 负责 identity / context
  - 例如 `kernel_name`、`trace_order`、`grid_dim`、`block_dim`
- 一组 source 负责 feature / weight
  - 例如 `dynamic_inst_count`、`exec_time`、compression / hardware features
- 可选再接一个 `squash` source 作为 guardrail / segment 辅助信息

这一步不是简单把 JSON 拼起来，
而是会先检查：

- 两边是否能按 invocation 对齐
- 是否存在字段缺失
- 是否有顺序或单位不一致

#### 第二步：生成标准输入表

这一步由 invocation table builder 完成。

它的作用是：

把原始 dual-source 数据统一成一张稳定的 `KernelInvocationRecord` 表。

这张表里已经包含：

- `kernel_invocation_id`
- `kernel_name`
- `trace_order`
- `grid_dim`
- `block_dim`
- `dynamic_inst_count`
- `exec_time`
- `feature_vector`

也就是说，A 线并不是直接让后面去读原始 JSON，
而是先产出一张标准输入表。

#### 第三步：基于输入表选择 representative anchors

这一步由 frontend selector 和 exporter 完成。

当前 A 线不是只给一个分组结果，而是比较了三种前端模式：

- `name-only`
- `pka-like-coarse`
- `hybrid`

最后导出：

- representative anchor table
- comparison table
- case note
- frontend note

所以 A 线最终交给后面的，不是“更多 feature”，而是：

**一组已经过选择、已经带有 membership 和 weight 的 representative anchors。**

这一层的意义是：

我们已经不再直接拿原始 invocation 做后续结构分析，
而是有了一层相对稳定、可解释的输入对象。

### 2.2 B 线：middle structure 对象层

这一层是当前方法的核心主干。

它的目标是：

把 frontend anchor 进一步组织成 simulator 侧真正可以消费的结构对象。

目前已经完成：

- family-centered YAML rule config
- middle-layer builder
- anchors / families / regimes / lanes artifacts
- importance scoring sheet
- writeback lane-to-regime mapping

当前 `mini_transformer_v4` 上的规模是：

- `9` 个 anchors
- `4` 个 families
- `9` 个 regimes
- `9` 条 lanes

更具体地说，B 线内部做的是下面几步：

#### 第一步：读取上游对象和规则真源

B 线不是纯粹靠统计聚类自动长出来的。

它一方面读取：

- A 线或 full-feature 相关输入
- squash / batch / APE 等证据

另一方面读取：

- family-centered YAML rule config

这个 YAML 不是普通配置文件，
而是当前 B 线对象边界的 source of truth。

#### 第二步：构造 anchors / families / regimes / lanes

middle-layer builder 负责把上游输入一步步构造成：

- anchors
- families
- regimes
- lanes

这几层对象分别对应：

- `anchor`
  - 前端压缩后可继续组织的基本对象
- `family`
  - 一组共享主要硬件执行模板、共享大方向调参语义的对象
- `regime`
  - family 内真正进入 backend 的细粒度对象
- `lane`
  - 后端参数空间里的验证 / 干预入口

所以 B 线不是“再做几张分析表”，
而是在构建一套 simulator-side decision objects。

#### 第三步：生成 importance 和 writeback 相关中间对象

除了 `anchors / families / regimes / lanes`，
B 线还会继续生成：

- importance scoring sheet
- writeback lane-to-regime mapping
- bundle

也就是说，B 线不只是定义对象，
还在定义：

- 对象的重要性
- 对象进入后端时该走哪条 lane
- 后续 writeback 该怎么对接

这一层的意义是：

我们不再只是“分析出一些 representative kernels”，
而是把它们提升成了一组可以直接用于后端决策的对象：

- family：共享机制层
- regime：backend 直接入口对象
- lane：参数方向映射对象

### 2.3 C 线：backend planning 层

这一层的目标是：

把 middle-layer 对象转成后端的验证规划。

目前已经完成：

- backend builder
- validation worksheet
- scenario matrix
- run manifest
- writeback map

当前已经形成的后端规划产物包括：

- `36` 条 run manifest
- `24` 条 scenario matrix
- `9` 条 writeback map

更具体地说，C 线内部的工作是：

#### 第一步：把 middle-layer objects 翻译成 backend planning objects

C 线读取的上游，不再是原始 trace，
而是 B 线已经构造好的结构对象：

- family
- regime
- lane

然后把它们翻译成后端真正关心的规划对象：

- 哪些 regime 应该先验证
- 每条 lane 对应什么 parameter direction
- baseline 如何定义
- scenario 应该怎样组织

#### 第二步：生成可执行前的验证规划

这一层 builder / planner 负责生成：

- validation worksheet
- scenario matrix
- priority lane table
- run manifest
- writeback map

所以 C 线当前的真实作用不是执行，而是：

**把结构对象层推进成后端验证计划层。**

#### 第三步：为 execution 和 writeback 预留接口

也就是说，C 线不仅规划“该跑什么”，
还定义了：

- 结果回来之后如何汇总
- 后续如何把结果回写到 regime / family / anchor 状态

因此它已经形成了：

- 规划闭环
- 接口闭环

只是还没有完全形成“真实执行闭环”

这一层的意义是：

我们已经不仅仅停留在“对象定义”，
而是进一步把对象映射到了：

- 哪些 regime 值得先看
- 每个 regime 应该走哪条 validation lane
- 用哪些 scenario 去做第一轮验证

---

## 3. 这段时间最重要的新进展

最近最关键的新进展，是我们补上了：

**backend execution bridge v1**

这部分工作的目标是：

把原来只有 `backend_run_manifest` 的规划对象，
真正接到 simulator 执行侧。

也就是说，把链路从：

`manifest`

推进成：

`manifest -> command plan -> execution -> result summary`

目前已经完成：

- command plan 生成
- run 目录规范
- command / metadata / stdout / stderr 落盘
- parser report
- backend result summary
- 第一轮真实 smoke execution

这里也可以进一步拆成三层理解：

#### 第一层：workload profile

这一层回答的是：

**“这个 workload 在 execution 层应该怎么跑？”**

它会定义：

- simulator binary 在哪
- full trace 在哪
- smoke trace 怎么构造
- config 路径是什么
- 环境变量是什么
- parser 该从哪里提字段

#### 第二层：execution bridge

这一层回答的是：

**“一条 run manifest 记录，怎么变成一次真实 simulator run？”**

它负责：

- 从 manifest 生成 command plan
- 生成每个 `run_id` 的独立输出目录
- 落盘 `command.sh / run_metadata.json / stdout.log / stderr.log`
- 执行命令
- 收集 parser report
- 生成 result summary

#### 第三层：execution CLI

这一层就是实际运行入口。

它负责接收：

- workload id
- manifest 路径
- smoke mode
- timeout
- 运行数量上限

然后调用 profile 和 execution bridge 完成真正执行

因此，execution bridge 的真实意义是：

**把 C 线的 backend planning，第一次推进成了真实执行。**

这意味着：

我们现在已经不是只有“规划闭环”，
而是第一次形成了：

**初步执行闭环**

---

## 4. 当前已经有哪些可复用产物

当前仓库里已经有几类比较稳定的产物：

### frontend 产物

- invocation table
- representative anchor table
- comparison table
- case note

### middle-layer 产物

- anchors.json
- families.json
- regimes.json
- lanes.json
- importance_scoring_sheet.json
- bundle.json

### backend 产物

- backend_run_manifest_v1.json
- backend_scenario_matrix_v1.json
- backend_command_plan_v1.json
- backend_execution_records_v1.json
- backend_result_summary_v1.json
- backend_writeback_map_v1.json

这些产物说明：

当前工作已经不再只是文档讨论，
而是有了一套可运行、可测试、可复现的实验基础设施。

---

## 5. 当前工作的主要价值

我认为当前工作的价值主要体现在三点：

### 5.1 方法价值

我们补上了一层以前常常缺失的中间层：

从 representative objects 到 simulator decision objects 的组织层。

这使得后续调参不再只是：

- 按 kernel 名字看
- 按时间占比看
- 靠人工经验挑对象

而是可以按：

- family
- regime
- lane

这样的结构化对象来推进。

### 5.2 工程价值

现在这套工作已经有：

- rule config
- builder
- artifacts
- execution bridge
- tests

所以它不只是概念，而是已经变成了一套可运行的研究工程基线。

### 5.3 研究组织价值

过去 workload 分析和 simulator 调参之间经常靠人工拼接，
导致对象边界不稳、优先级漂移、结论难复现。

我们现在做的事情，本质上是在固定这段最容易失真的中间过程。

---

## 6. 当前还没有完全完成的部分

虽然第一版主干已经基本成型，但目前还有一个关键问题在收尾：

**要把 smoke execution 和 formal validation 完全分开。**

当前 review 指出的核心问题是：

1. 默认 profile 不能直接走 smoke trace
2. `gpgpu_max_cycle=10` 这种 smoke 限制不能当作默认 validation 配置
3. parser 解析出 `sim_cycles`，不能直接等价成正式 validation success

这意味着：

当前 execution bridge 已经能够跑，
但最后还要把“能跑通”和“能作为正式方法证据”这两个层次分开。

这是当前最关键的收尾工作。

---

## 7. 当前我们最准确的阶段判断

我认为当前阶段最准确的判断是：

**第一版主干已经完成，当前正在做执行语义收尾。**

更具体地说：

- A 线输入层：已经完成第一版
- B 线结构层：已经完成第一版主干
- C 线规划层：已经完成第一版主干
- execution bridge：已经完成主体实现
- 当前剩余工作：把 smoke 通路和正式 validation 通路彻底拆开

因此，当前不能再说我们只是“在想方法”，
也不能过度说我们“已经完成全部验证”。

更准确的说法是：

**我们已经把方法链从概念、结构、规划推进到了初步执行。**

---

## 8. 接下来最合理的推进方向

接下来最合理的推进顺序是：

### 第一步：收尾 execution 语义分离

先明确：

- 默认路径走 full trace
- 只有显式 smoke mode 才走 trimmed trace
- smoke 成功不自动升格为 formal validation success

### 第二步：固定 execution bridge v1 的结果 schema

包括固定：

- execution_mode
- result_status
- parse_status
- parser_report
- backend_result_summary

### 第三步：再用 execution evidence 回看 B 线规则

也就是等执行层语义干净后，再回头判断：

- 哪些 family 合理
- 哪些 regime 拆得过细
- 哪些 lane 没信息增益

这样后续优化才会更扎实。

---

## 9. 一句话总结

如果用一句话总结当前工作进展，我会这样说：

**我们已经把 workload-driven 的分析结果，从 frontend 压缩对象推进成了 simulator 可消费的 decision objects，并且开始把这些对象真正接入后端执行验证。**

当前最关键的剩余工作，不是再补更多对象，而是把“执行桥”的证据语义彻底收干净，让第一版从“能跑”走到“能作为正式验证依据”。 
