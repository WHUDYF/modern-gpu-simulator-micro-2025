# Squash + Batch Family 判据框架计划

## Goal Description

基于现有 `mini_transformer_v4` 诊断产物，构建第一版 `squash + batch` family 判据原型。该原型的首要交付物不是单张卡片本身，而是一份以方法说明为主的 `family synthesis`：它要说明 family 判据框架如何工作、为什么边界 case 是第一轮的核心、以及这种结构为何能把后续 simulator 验证从“逐 kernel 猜”压缩为“少数验证主线 + 少量例外”。analysis cards 与 family cards 作为支撑这一 synthesis 的证据层与解释层存在。

## Acceptance Criteria

以下 AC 采用“正向测试 + 负向测试”的方式描述，重点验证的是方法结构是否成立，而不是数值最优。

- AC-1: 建立独立的 family-criteria 工作区，并明确 kernel 分析卡、family 卡片与边界 case 文档的 schema
  - Positive Tests (expected to PASS):
    - `docs/family_criteria/README.md` 存在，并写明 scope / non-goals
    - `docs/family_criteria/mini_transformer_v4/kernel_card_schema.md` 存在，并包含分析卡的固定区块
    - `docs/family_criteria/mini_transformer_v4/analysis_cards/`、`family_cards/`、`boundary_cases/` 与 `outlier_analysis/` 目录存在
    - `boundary_cases/` 与 `outlier_analysis/` 均包含说明文档，明确其职责
  - Negative Tests (expected to FAIL):
    - schema 文档缺少 mixed / outlier / ambiguous 的位置定义
    - workspace 只创建目录但没有说明文档，导致后续无法判断输出格式是否合规
    - 边界 case 与 outlier 信息没有独立存放位置，只能散落在其他卡片中

- AC-2: 提供一个可重复使用的最小证据提取入口，统一指向 `mini_transformer_v4` 现有诊断来源
  - Positive Tests (expected to PASS):
    - Python 脚本能返回 E0/E1/E2/E4/E5 报告路径及 baseline APE JSON 路径
    - Python 脚本能返回与 spec 完全一致的六个代表 kernel 默认名单：`gemm_tiled`、`attention_score`、`residual_add`、`softmax_kernel`、`context_mul`、`layernorm_kernel`
    - 对应 pytest 测试通过
  - Negative Tests (expected to FAIL):
    - 缺少任一核心报告路径时仍静默通过
    - 代表 kernel 名单与 spec 约定不一致，或顺序漂移但未触发测试失败

- AC-3: 第一轮优先完成两组关键边界 case 文档，并以区分点分析为主给出分级结论
  - Positive Tests (expected to PASS):
    - 至少存在两份边界 case 文档，分别覆盖 `gemm_tiled vs attention_score` 与 `softmax_kernel vs context_mul`
    - 每份边界 case 文档同时记录共享点与区分点，并明确以区分点分析为主
    - 每份边界 case 文档都给出分级结论，例如：`强共享 / 弱共享 / 边界未定`
    - 每份边界 case 文档都给出当前阶段的执行建议，且以 family 划分建议为主、验证组织建议为辅
    - 每份边界 case 文档都至少引用两处明确证据来源（E0-E5 报告章节或 baseline APE JSON）
  - Negative Tests (expected to FAIL):
    - 边界 case 只写共享点，不分析为什么不能轻易并类
    - 边界 case 直接给出绝对二选一结论，却不保留分级判断或边界未定状态
    - 边界 case 文档只说结论，不给证据出处

- AC-4: 在边界 case 结论基础上回填 analysis cards，使其成为偏方法论总结的证据层对象
  - Positive Tests (expected to PASS):
    - 六张分析卡全部存在
    - 每张分析卡都严格遵循同一模板，包含执行模式粗分、关键指标、主导资源候选、family 归属判断
    - 每张分析卡至少引用两处明确证据来源（E0-E5 报告章节或 baseline APE JSON）
    - 每张分析卡都包含至少一条边界说明或不确定性说明
    - 至少一张卡片显式写出 `mixed / ambiguous` 或“边界不稳”的情况
    - 存在一个固定检查方式，能够统一检查所有分析卡是否包含必需小节、证据来源和边界/不确定性说明
    - analysis card 的表述偏向方法论总结，而不只是工程记录
  - Negative Tests (expected to FAIL):
    - 某张分析卡只列指标，不给出主导资源候选
    - 某张分析卡给出归类判断，但没有写明边界说明或不确定性
    - 某张分析卡写了结论，却没有证据出处
    - 全部分析卡都表现得“过于确定”，没有保留第一版应有的不确定性

- AC-5: 从 analysis cards 中归纳出第一版 family 解释卡片，采用“粗类现象型 + 子类机制型”的分层命名，并以边界清晰性优先
  - Positive Tests (expected to PASS):
    - family 卡片至少覆盖：`compute-heavy -> ...`、`memory-heavy -> ...`、`mixed -> ...` 与 `outliers`
    - 每张 family 卡片名称都显式包含一个粗类标签和一个机制型子类标签，`outliers` 卡片除外
    - 每张 family 卡片都包含：核心解释、代表 kernel、边界条件、不确定性、后续意义
    - 每张 family 卡片明确说明“它是什么”以及“它不是什么”
    - 每个代表 kernel 至少在一个 family 卡片或 outlier 卡片中被引用
  - Negative Tests (expected to FAIL):
    - family 只按算子名重命名，没有共享架构解释
    - family 名称只有单层标签，无法区分粗类与机制子类
    - family 卡片只描述内部共性，不写边界条件或排除条件
    - outlier 卡片不存在，导致无法容纳无法稳定归类的 kernel

- AC-6: 输出一份以方法说明为主的 family synthesis，明确这套框架如何工作并如何压缩后续验证组织问题
  - Positive Tests (expected to PASS):
    - synthesis 明确写出：family 先于处方，family 的意义是把“逐 kernel 猜”压缩成“少数验证主线 + 少量例外”
    - synthesis 同时说明：这套 family 判据框架如何工作、为什么边界 case 是第一轮核心、为什么 analysis cards 与 family cards 是方法支撑层
    - synthesis 明确列出第一版 family、outlier 以及 version-1 limits：不展开 delta、不定量衡量节省比例、不输出具体处方
  - Negative Tests (expected to FAIL):
    - synthesis 只总结 family 名称，不解释框架为何成立
    - synthesis 直接跳到 simulator 参数建议，越过当前 scope
    - synthesis 把 analysis cards 或 family cards 当成最终目标，而不是方法论支撑对象

- AC-7: 将 family-criteria 原型与现有 draft/spec 对齐，使其成为后续 plan / implementation 的稳定前置
  - Positive Tests (expected to PASS):
    - `draft_squash_batch.md` 增加 family workspace 的引用
    - `2026-04-15-squash-batch-family-criteria-design.md` 增加 prototype status 段落
    - 这些状态说明与实际生成的边界 case 文档、analysis cards、family cards、synthesis 一致
  - Negative Tests (expected to FAIL):
    - spec 仍停留在纯设计层，没有反映原型状态
    - `draft_squash_batch.md` 写成“family 原型已经产出完整卡片集”，但 spec 仍写成“仅停留在设计阶段”
    - draft 引用了不存在的 workspace / 文件路径，而 spec 引用了另一组路径
    - draft 声称第一版已支持验证主线复用，spec 却仍声明第一版尚未产出 family 综合说明

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

实现包含：

- 独立的 family-criteria 文档工作区
- 可重复使用的最小证据提取脚本和测试
- 两组关键边界 case 文档
- 六张完整 kernel 分析卡
- 3 张 family 卡片 + 1 张 outlier 卡片
- 一份以方法说明为主的 family synthesis
- 对 draft/spec 的状态回挂

该版本可以明确展示：

- 边界 case 如何逼出 family 判据
- “执行模式粗分 + 资源主导边界”如何形成 family
- mixed / ambiguous / outlier 如何被显式保留
- family 如何自然压缩后续验证思路

### Lower Bound (Minimum Acceptable Scope)

最小可接受实现包含：

- family-criteria workspace 与 schema 文档
- 至少两份边界 case 文档
- 六张手工 kernel 分析卡
- 至少 2 张 family 卡片 + 1 张 outlier 卡片
- 一份简短综合说明，明确 family 判据框架如何工作以及它对验证组织的定性意义

即使没有更进一步的脚本自动化，只要 family 结构和边界表达已经成立，也满足第一版目标。

### Allowed Choices

- Can use:
  - 现有 `mini_transformer_v4` E0-E5 报告与 APE JSON
  - Markdown 作为边界 case、分析卡和 family 卡片载体
  - Python 脚本只做最小证据入口与一致性检查
  - pytest 做基础回归检查
- Cannot use:
  - 将已有 simulator 经验直接写进 family 判据
  - 直接展开 delta 或 simulator 处方层
  - 把 family 划分写成自动阈值系统并声称已稳定
  - 在第一版里定量声称节省了多少验证成本

## Feasibility Hints and Suggestions

> **Note**: 本节是实现思路参考，不是强制规定。

### Conceptual Approach

第一版建议走“文档先行、脚本最小化、边界先行”的路线：

1. 建立 `docs/family_criteria/mini_transformer_v4/` 工作区
2. 用一个最小 Python 脚本统一指向现有 E0/E1/E2/E4/E5 证据和代表 kernel 名单
3. 优先围绕 `gemm_tiled vs attention_score` 与 `softmax_kernel vs context_mul` 撰写边界 case 文档
4. 将边界 case 结论回填到六个代表 kernel 的 analysis cards
5. 从 analysis cards 中提炼 family 卡片与 outlier 卡片
6. 写一份以方法说明为主的 synthesis，明确 family 对后续验证组织的意义
7. 把原型状态回挂到 draft 和 spec

核心不是让脚本自动决定 family，而是先保证“边界 case -> analysis cards -> family cards -> synthesis”这条人工可解释链路稳定成立。

### Relevant References

- `draft_squash_batch.md` - 当前 squash + batch idea 的高层草稿
- `docs/superpowers/specs/2026-04-15-squash-batch-family-criteria-design.md` - 已批准的设计 spec
- `experiments/baseline_diagnosis/results/mini_transformer_v4/E0_baseline.md` - baseline 观测基线
- `experiments/baseline_diagnosis/results/mini_transformer_v4/E1_squash.md` - squash 结构来源
- `experiments/baseline_diagnosis/results/mini_transformer_v4/E2_batch.md` - batch 结构来源
- `experiments/baseline_diagnosis/results/mini_transformer_v4/E4_full.md` - 现有多机制综合解释
- `experiments/baseline_diagnosis/results/mini_transformer_v4/E5_stageC_validation.md` - 当前验证闭环与可验证性边界
- `experiments/baseline_diagnosis/results/mini_transformer_v4/baseline_ape.json` - 结构化指标支持

## Dependencies and Sequence

### Milestones

1. **结构搭建**
   - Phase A: 建立 family workspace
   - Phase B: 明确 kernel card schema

2. **证据入口**
   - Phase A: 增加最小 Python 入口脚本
   - Phase B: 用 pytest 固化路径和代表 kernel 名单

3. **边界先行**
   - Phase A: 优先完成两组关键边界 case 文档
   - Phase B: 用边界 case 反推 analysis card 应保留的字段和判断方式

4. **案例结构化**
   - Phase A: 回填 analysis cards
   - Phase B: 由 analysis cards 归纳 family / outlier 卡片

5. **定性闭环**
   - Phase A: 写以方法说明为主的 family synthesis
   - Phase B: 将实现状态回挂到 draft/spec

依赖关系上：

- analysis cards 依赖边界 case 结论
- family 卡片依赖 analysis cards
- synthesis 依赖 family 卡片
- draft/spec 对齐依赖前面所有 family 原型产物已经存在

## Task Breakdown

每个任务必须包含且仅包含一个 routing tag：

- `coding`: 由 Claude 实现
- `analyze`: 通过 Codex 执行（`/humanize:ask-codex`）

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 建立 `docs/family_criteria/mini_transformer_v4/` workspace，补充 `boundary_cases/`、`outlier_analysis/` 目录与说明文档，并写 README 与 kernel card schema | AC-1 | coding | - |
| task2 | 增加最小 Python 证据提取入口与 pytest，固定核心报告路径，并将代表 kernel 名单严格绑定到 spec 约定 | AC-2 | coding | task1 |
| task3 | 优先围绕 `gemm_tiled vs attention_score` 与 `softmax_kernel vs context_mul` 生成边界 case 文档，采用分级结论并以前者为主给出 family 划分建议 | AC-3 | coding | task2 |
| task4 | 将边界 case 结论回填到六个代表 kernel 的 analysis cards，要求每张卡明确证据来源、边界说明与 tentative / ambiguous / outlier 信息，并补充一个固定检查方式统一验证分析卡完整性 | AC-4 | coding | task3 |
| task5 | 基于 analysis cards 归纳第一版 family 卡片与 outlier 卡片，严格采用“粗类现象型 + 子类机制型”的命名，并以边界清晰性优先 | AC-5 | coding | task4 |
| task6 | 写 `mini_transformer_v4` family synthesis，以方法说明为主解释 family 判据框架如何工作及其如何组织后续验证主线 | AC-6 | coding | task5 |
| task7 | 把原型状态回挂到 `draft_squash_batch.md` 与 spec 中，完成结构对齐 | AC-7 | coding | task6 |

## Claude-Codex Deliberation

### Agreements

- 第一版应以 `mini-transformer` 单点原型为主，而不是直接做通用系统
- family 判据框架的核心是“执行模式粗分 + 资源主导边界”
- 输入采用“指标为主、语义约束、经验隔离”
- mixed / ambiguous / outlier 必须显式保留
- family 命名采用“粗类现象型 + 子类机制型”的分层形式
- family 卡片同时服务论文表达与后续验证组织
- 第一轮应优先通过边界 case 逼出判据，而不是先把所有 cards 铺满

### Resolved Disagreements

- **Family 的主定义**：最终采用“解释层定义”为主，而不是直接把验证调度写成主定义；验证分流作为落地后果保留
- **判据优先级**：执行模式不作为最终边界，而是作为粗分层；真正决定 family 边界的是资源主导特征
- **经验信息使用**：在未引入 delta 闭环前，不允许把 simulator 经验直接写进判据
- **迭代顺序**：从“先 analysis cards 再 family cards”调整为“先边界 case，再 analysis cards，再 family cards”

### Convergence Status

- Final Status: `converged`

## Pending User Decisions

- DEC-1: 第一版 family 数量
  - Claude Position: 中等数量最稳，兼顾清晰度与可解释性
  - Codex Position: 认同中等粒度，避免过粗或过碎
  - Tradeoff Summary: 过粗会丢掉结构，过细会让第一版显得脆弱
  - Decision Status: `中等粒度`

- DEC-2: 第一版稳定性目标
  - Claude Position: 不明显偏向“只解释 mini-transformer”或“强行追求跨 workload”
  - Codex Position: 认同折中版本
  - Tradeoff Summary: 当前阶段更适合两边都不过度牺牲
  - Decision Status: `折中`

- DEC-3: 双强信号 kernel 的处理
  - Claude Position: 保留为 mixed / ambiguous，并视情况作为独立 outlier
  - Codex Position: 认同不能强行并入已有 family
  - Tradeoff Summary: 强行归类会破坏 family 边界可信度
  - Decision Status: `mixed/ambiguous，必要时升级为 outlier`

- DEC-4: outlier 第一版定义
  - Claude Position: 先宽松定义，再后续收紧
  - Codex Position: 认同
  - Tradeoff Summary: 第一版需要结构保留区，而不是过早过滤
  - Decision Status: `先宽松定义`

- DEC-5: family 命名方式
  - Claude Position: 粗类现象型 + 子类机制型
  - Codex Position: 认同分层命名
  - Tradeoff Summary: 单层命名不足以表达结构和边界
  - Decision Status: `分层命名`

- DEC-6: family 卡片用途
  - Claude Position: 同时服务论文与验证组织
  - Codex Position: 认同
  - Tradeoff Summary: 维持单一对象，避免做两套体系
  - Decision Status: `两者并重`

- DEC-7: 不确定性表达
  - Claude Position: 显式保留
  - Codex Position: 认同
  - Tradeoff Summary: 第一版只给内部看，没必要隐藏结构性不确定性
  - Decision Status: `显式保留不确定性`

- DEC-8: spec 聚焦点
  - Claude Position: 方法框架为主，`mini-transformer` 为首个应用场景
  - Codex Position: 认同
  - Tradeoff Summary: 方法与落地需要同时存在
  - Decision Status: `框架为主，案例为辅`

- DEC-9: 第一版交付优先级
  - Claude Position: synthesis 优先，其次 family cards，最后 analysis cards
  - Codex Position: 认同，但 analysis cards 仍需保持较强可读性
  - Tradeoff Summary: synthesis 承担方法论表达，cards 承担证据与解释支撑
  - Decision Status: `synthesis > family cards > analysis cards`

## Implementation Notes

### Code Style Requirements

- 实现代码与注释中不得出现 `AC-`、`Milestone`、`Phase`、`Step` 等计划术语
- 这些术语只属于计划文档，不应进入最终代码或正式分析产物
- Markdown 产物应保持中文叙述，代码、变量、命令使用英文
