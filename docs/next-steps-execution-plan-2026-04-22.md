# 下一阶段执行计划（1~2 周）

日期：2026-04-22

## 1. 文档目的

这份文档用于把接下来 1~2 周最值得推进的工作，整理成一份可执行计划表。

当前阶段最重要的判断是：

**不应该等前端 compression 完全建立好之后再推进后段方法，而应该让前端锚点、中间结构层和后端验证计划并行生长。**

因此，这份计划的目标不是给出一个串行路线，而是给出：

- 哪些任务可以并行
- 哪些任务是最小闭环的关键
- 哪些任务必须优先

---

## 2. 当前总目标

接下来 1~2 周的总目标不是“把整篇论文写完”，而是完成一个：

**最小可运行方法闭环**

也就是至少形成下面这条链：

`representative anchors -> family -> regime -> importance ratio -> tuning priority`

并且让这条链具备：

- 明确的输入接口
- 明确的结构对象
- 明确的优先级定义
- 明确的验证计划

---

## 3. 三条并行工作线

当前建议同时推进三条线：

### A 线：前端锚点线

目标：

- 给后段方法提供可信输入来源

关键词：

- `PKA`
- representative anchors
- membership / weight

### B 线：中间结构线

目标：

- 让 family / regime / importance ratio 真正落成对象表

关键词：

- `squash`
- `batch`
- `family`
- `regime`
- `importance`

### C 线：后端验证线

目标：

- 提前把 simulator tuning / validation 的验证路径定义清楚

关键词：

- priority-guided tuning
- simulator lane
- validation metrics

---

## 4. 优先级排序

当前最合理的任务优先级是：

### Priority 1：把中间结构层做硬

因为这是你们真正的主贡献所在。

### Priority 2：建立最小可信前端锚点

因为 reviewer 会关心输入是否可信。

### Priority 3：定义后端验证方式

因为没有验证计划，importance ratio 只是概念。

---

## 5. 详细执行计划表

| 编号 | 工作线 | 任务 | 目标产出 | 优先级 | 预计结果 |
|---|---|---|---|---|---|
| `T1` | B | 完成 mini-transformer 第一版 Family Table 真实字段填充 | 更新后的 family table | `High` | 至少把 `coverage / time / decision` 的来源说明清楚 |
| `T2` | B | 完成 mini-transformer 第一版 Regime Table 真实字段填充 | 更新后的 regime table | `High` | 至少形成稳定的 regime 优先级顺序 |
| `T3` | B | 把 importance ratio 从文档定义推进到可填模板 | importance template / scoring sheet | `High` | 能对每个 family 给出可追踪分数来源 |
| `T4` | A | 定义最小版 representative anchor table | anchor table schema + 示例 | `High` | 后续 family 层输入接口固定 |
| `T5` | A | 用 PKA 作为锚点，整理前端最小输入链 | PKA-style frontend note / pseudo pipeline | `Medium-High` | reviewer 可接受的 frontend story 成型 |
| `T6` | A | 从 STEM+ROOT 吸收 heterogeneity refinement 原则 | heterogeneity refinement note | `Medium` | 解释为何前端不能只停在 PKA cluster |
| `T7` | C | 定义 simulator lane 的第一版对象映射 | lane mapping note | `Medium-High` | 明确每个 regime 进入哪个 validation lane |
| `T8` | C | 定义第一组 tuning parameter scenarios | parameter scenario list | `Medium-High` | 不同 family 对应不同参数方向 |
| `T9` | C | 定义 importance-ratio 的 baseline 对照方式 | baseline comparison note | `Medium` | 明确 random / time-only / importance-guided 的对照 |
| `T10` | C | 写出第一版 workflow 验证脚本思路 | validation workflow draft | `Medium` | importance ratio 不再只是排序概念 |
| `T11` | 文档 | 把当前方法链压成 2~3 页可汇报版本 | concise method summary | `Medium` | 后续导师汇报与内部统一口径 |
| `T12` | 文档 | 把 related work 再压成 PPT 可用版本 | related-work condensed slides note | `Low-Medium` | 汇报材料更稳 |

---

## 6. 第一周建议

第一周的目标应该是：

**把中间结构层彻底做硬。**

建议任务顺序：

### Day 1~2

- `T1` 完成 family table 的真实字段填充
- `T2` 完成 regime table 的真实字段填充

### Day 3

- `T3` 把 importance ratio 变成可填模板

### Day 4

- `T4` 建立 anchor table 最小接口
- `T5` 写清楚 PKA-style frontend story

### Day 5

- `T7` 定义 simulator lane 映射
- `T8` 定义第一组 tuning parameter scenarios

第一周结束的理想状态是：

- 已经有：
  - anchor table
  - family table
  - regime table
  - importance template
  - lane mapping

这时方法闭环已经初步长出来了。

---

## 7. 第二周建议

第二周的目标应该是：

**把验证逻辑做清楚。**

建议任务顺序：

### Day 6~7

- `T6` 吸收 STEM+ROOT 异质性 refinement 原则
- `T9` 定义 baseline comparison

### Day 8~9

- `T10` 写 importance-guided validation workflow

### Day 10

- `T11` 方法链压缩成汇报文稿
- `T12` related work 压成 PPT 版本

第二周结束的理想状态是：

- 不只是方法对象存在
- 而且已经清楚知道：
  - 怎么验证 importance ratio
  - 怎么和 baseline 比
  - 怎么进入 simulator tuning workflow

---

## 8. 当前最小闭环定义

如果必须优先保一个最小闭环，那建议保下面这 6 个任务：

1. `T1` Family Table 真实字段填充
2. `T2` Regime Table 真实字段填充
3. `T3` Importance 模板
4. `T4` Anchor Table 最小接口
5. `T7` Simulator lane 映射
6. `T9` Importance baseline 对照方式

只要这 6 个任务完成，你们就已经拥有：

- 一个可信输入接口
- 一套结构对象
- 一套优先级定义
- 一套后续验证比较框架

这时即使前端 compression 还没完全复刻好，方法也已经具备了继续推进的条件。

---

## 9. 当前最不建议的做法

### 9.1 不要先完整复刻所有前端论文

这会拖慢节奏，并且吞掉你们的主创新。

### 9.2 不要继续只停留在文档讨论

现在最需要的不是再多讲一轮概念，而是让：

- 表
- 字段
- 优先级
- lane

都实体化。

### 9.3 不要把 validation 放到最后再想

importance ratio 如果没有验证设计，就会一直停留在“看起来合理”。

---

## 10. 当前阶段的简短结论

如果把这份计划压成最短形式，可以写成：

1. 接下来 1~2 周不应串行等待前端完成，而应并行推进前端锚点、中间结构层和后端验证计划。
2. 当前主贡献最值得优先推进的是中间结构层，即 `family / regime / importance ratio`。
3. 前端只需要先建立最小可信锚点，不需要完整复刻所有相关工作。
4. 后端验证计划必须提前生长，否则 importance ratio 无法真正形成方法价值。
5. 当前最小闭环是：`anchor table -> family table -> regime table -> importance template -> simulator lane -> baseline validation`。
