# AI GPU 方向组会进展总结

日期：2026-04-05

## 1. 当前研究主线的重新收敛

这一阶段的工作重点，已经从“单纯做 GPU trace 压缩”收敛为一条更适合体系结构投稿的主线：

`difftest 风格的压缩与验证思想 -> 可保留 warp-level 行为语义的 GPU trace 表示 -> AI workload 中 warp scheduling 失效模式分析 -> 寻找调度机制设计机会`

这里的关键变化是：

- `trace compression` 不再只是工程优化，而要上升为 `behavior abstraction`
- `difftest` 的价值不在于直接复刻 CPU DiffTest，而在于借用其 `squash / delta / batch / event-level checking` 思路，提升 GPU trace 的可验证性和可分析性
- 论文主问题不再是“trace 变小了多少”，而是“这种分层表示是否保留了足够的 warp-level 语义，支持 AI workload 的调度行为分析和架构优化”

## 2. 为什么将主问题转到 warp scheduling

目前判断，`warp scheduling` 比单纯做 memory trace 压缩更像体系结构方向的核心问题。

原因如下：

- 传统 dense matmul / GEMM 虽然重要，但通常执行更规则，warp 调度效应容易被算力吞吐掩盖
- LLM inference，尤其是 `decode` 阶段的 `attention + KV-cache`，更容易出现 warp 间推进不均衡、memory stall 长尾、以及 ready warp 与高收益 warp 不一致的问题
- 现有 trace 压缩结构中已经包含 `shared PC sequence`、`warp diff`、地址 delta / override、run-length 等信息，这些天然接近 warp scheduling 相关语义

因此，当前的核心假设是：

> 在 LLM decode 阶段，attention 与 KV-cache 共同破坏了 warp-level execution regularity；如果能够用保留行为语义的 trace 表示把这种失效模式提取出来，就有机会解释现有 warp scheduler 的盲点，并进一步寻找新的调度机制。

## 3. 当前已经完成的工作

### 3.1 `modern-gpu-simulator-micro-2025` 主线工作

- 已明确 trace compression 不应只停留在离线格式层，而要进入 simulator 主流程
- 当前仓库中已经形成一条 `L1 -> L4` 压缩路线，重点包括：
  - 指令级 delta / flags
  - run-length squash
  - 跨 warp PC 去重
  - 跨 threadblock delta
- parser / simulator 接入已经开始推进，当前本地未提交改动已经覆盖：
  - `v5` 压缩 trace 解码
  - `v6` run-length 展开解码
  - `trace_driven` 中的版本分发
- 已有文档已经把主线梳理出来，包括：
  - `docs/trace-compression-for-microbench-agent.md`
  - `docs/ai-workload-driven-workflow.md`
  - `docs/ai-research-roadmap-for-group-meeting.md`

### 3.1.1 当前压缩测试结果

目前在 `util/trace-compress/test_roundtrip` 中，功能性测试已经全部通过。

当前实际运行结果为：

- `test_v5_roundtrip`：通过
- `test_v5_divergent`：通过
- `test_v5_compression_ratio`：通过
- `test_v6_roundtrip`：通过
- `test_v6_with_long_run`：通过
- `test_v7_roundtrip`：通过
- `test_v7_divergent_warps`：通过
- `test_v8_roundtrip`：通过
- `test_v8_fallback`：通过

合计：`9 passed, 0 failed`

其中，当前已经能明确给出的压缩比结果，是 `v4 -> v5` 在一个合成 threadblock 上的单元测试结果：

- 原始大小：`258 B`
- 压缩后大小：`132 B`
- 压缩后占原始大小：`51.16%`
- 等价压缩倍数：约 `1.95x`

这里需要特别说明：

- 这个 `51.16%` 是当前单元测试里**唯一已经直接打印出来的压缩比**
- 它对应的是一个合成 threadblock，不是 rodinia、microbenchmark 或 AI 模型 trace 的总体压缩比
- `v6 / v7 / v8` 当前已经完成 roundtrip 正确性验证，但还没有在组会材料中形成独立的压缩比数字
- 面向真实 trace 的批量压缩比报告，仍然是下一阶段需要补齐的实验项

### 3.2 `difftest` 相关思路梳理

- 已明确 `difftest` 对当前工作的真正价值，不是单独形成另一条论文主线，而是提供一套方法学：
  - `Squash`：跨周期事件压缩
  - `Delta`：只记录变化部分
  - `Batch`：批量传输
  - `event-level checker`：逐事件对齐验证
  - `snapshot/replay`：长运行调试加速
- 已形成一份面向 GPU simulator 的迁移思考文档：
  - `difftest/docs/gpu-simulator-integration.md`

当前结论是：`difftest` 更适合作为 GPU trace 表示与验证层的“支撑方法”，而不是单独作为主论文故事。

### 3.3 AI workload 选择上的收敛

已经做出的关键决策如下：

- 不先做训练，先做 `inference`
- 不先做 full forward，先做 `decode`
- 不先做大模型，先用 `GPT-2 small` 打通链路
- 主研究对象先放在 `attention + KV-cache`
- `matmul / GEMM` 保留为后续对照组

这样做的目的，是用最小成本先确认：

- decode trace 是否可稳定生成
- attention 相关 kernel 是否可被稳定识别
- context length 增大时，trace 结构是否出现可解释的 warp-level 变化

## 4. 新增的实验脚手架

为了尽快验证这条设计是否成立，当前已经新增一套最小实验脚手架：

- `experiments/gpt2_decode/run_decode.py`
- `experiments/gpt2_decode/run_trace.sh`
- `experiments/gpt2_decode/summarize_runs.py`
- `experiments/gpt2_decode/tests/test_summarize_runs.py`

其功能分别是：

- `run_decode.py`
  - 使用 `PyTorch + HuggingFace`
  - 加载 `GPT-2 small`
  - 将 `prefill` 放在 ROI 之外
  - 用 `cudaProfilerStart/Stop` 只圈住 `decode`
- `run_trace.sh`
  - 批量运行 `context length = 128 / 512 / 1024`
  - 每个点默认运行 3 次
  - 自动写入独立 trace 目录
- `summarize_runs.py`
  - 汇总 `run.log`
  - 统计 `dynamic_trace.pb`、`threadblocks/`、`stats.csv`
  - 输出一张 `summary.csv`
- `test_summarize_runs.py`
  - 对汇总逻辑做最小单元测试

当前已经完成的最小验证：

- `summarize_runs.py` 单测通过
- Python 语法检查通过
- shell 脚本语法检查通过

尚未完成的部分：

- 当前机器默认 Python 环境中还没有 `torch / transformers`
- 尚未实际运行 `GPT-2 decode` trace

## 5. 近期一周的实验目标

接下来一周的目标不是出最终论文结果，而是回答“这条主线是否值得继续重投入”。

第一轮实验设置为：

- 模型：`GPT-2 small`
- 路径：`PyTorch + HuggingFace`
- 模式：`decode only`
- `batch = 1`
- `gen_tokens = 1`
- `context length = 128 / 512 / 1024`
- 每个点运行 3 次

这一周重点看 4 件事：

1. `decode` 的 ROI trace 能否稳定生成
2. `stats.csv` 中能否识别出 attention 相关 kernel
3. trace 大小、kernel 数、threadblock 数据量是否随 context length 稳定变化
4. 是否已经能从压缩结构中看到后续 warp scheduling 分析需要的趋势

如果第一周结果成立，第二周将进入更像论文问题的分析：

- 对 attention / KV-cache 与 matmul 做对照
- 定义 warp-level 行为特征
- 开始寻找 scheduler 失效模式

## 6. 当前准备关注的 warp scheduling 解释型问题

目前最值得优先观察的不是“调度器如何设计”，而是“现有调度器在 AI decode 中到底哪里失效”。

计划优先关注的现象包括：

- warp progress skew
- divergence persistence
- memory-stall clustering
- ready-but-low-payoff issue
- long-tail warp ratio

如果这些现象在 `attention + KV-cache` 中明显，而在 `matmul` 中不明显，那么就形成了很强的解释型结果。

## 7. 4 个月投稿窗口下的阶段计划

### 阶段 1：打通链路并验证可行性

目标：

- 完成 `GPT-2 decode` trace 生成
- 跑通汇总分析
- 确认 trace 中存在可稳定观测的 warp-level 结构变化

### 阶段 2：形成解释型结果

目标：

- 从 attention / KV-cache trace 中提取 warp scheduling 相关特征
- 与 matmul 做对照
- 找出一个清晰、稳定、可解释的 scheduler 失效模式

### 阶段 3：寻找设计机会

目标：

- 不强行做大机制
- 只针对前一阶段发现的一个核心问题，探索是否能提出一个小而硬的 scheduler 改进

### 阶段 4：论文化收敛

目标：

- 将 `trace abstraction + event-level validation + scheduling insight` 收敛为单一论文主线
- 如果设计机会成立，则补充为“解释型 + 设计型”组合贡献
- 如果设计机会不足，则至少形成一篇以体系结构 insight 为核心的解释型论文

## 8. 当前风险

当前主要风险有以下几类：

- `decode` trace 中 attention kernel 不够干净，ROI 仍可能混入较多无关 kernel
- `GPT-2 small` 过于简单，可能不够暴露真实的 scheduler 失效模式
- trace 特征与真实 warp scheduling 行为之间的映射，可能需要进一步补 simulator 级 checker 才能增强可信度
- 如果只有解释型结果，没有新的调度设计点，则体系结构投稿力度可能不足
- 当前压缩结果里，只有 `v5` 合成测试给出了明确压缩比；真实 workload 上 `v5-v8` 的分层压缩比还没有完整实验数据

## 9. 当前决策

基于当前进展，现阶段做出的决策是：

- 继续以体系结构投稿为主线
- `difftest` 作为方法学支撑，而不是独立主故事
- 先做解释型，再看是否自然长出设计型机会
- 先用 `GPT-2 small decode` 快速确认设计可行性
- attention / KV-cache 为主线，matmul 为对照

## 10. 一句话总结

当前工作已经从“做一个新的 GPU trace 压缩工具”收敛为：

> 构建一套借鉴 `difftest` 思想、可保留 warp-level 行为语义的 GPU trace 表示与验证方法，并用它分析 LLM decode 中 attention / KV-cache 引发的 warp scheduling 失效模式，进一步寻找体系结构上的调度优化机会。
