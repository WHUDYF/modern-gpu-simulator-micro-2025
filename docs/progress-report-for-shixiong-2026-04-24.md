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
