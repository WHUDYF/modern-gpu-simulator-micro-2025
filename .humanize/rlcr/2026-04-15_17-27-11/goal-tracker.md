# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal

基于现有 `mini_transformer_v4` 诊断产物，构建第一版 `squash + batch` family 判据原型。该原型的首要交付物不是单张卡片本身，而是一份以方法说明为主的 `family synthesis`：它要说明 family 判据框架如何工作、为什么边界 case 是第一轮的核心、以及这种结构为何能把后续 simulator 验证从“逐 kernel 猜”压缩为“少数验证主线 + 少量例外”。analysis cards 与 family cards 作为支撑这一 synthesis 的证据层与解释层存在。

## Acceptance Criteria

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->

- AC-1: 建立 `docs/family_criteria/mini_transformer_v4/` 工作区，并同时提供 analysis / family / boundary / outlier 四类产物的目录与 schema 说明。
- AC-2: 提供一个最小、可重复的证据提取入口，能够稳定返回 `mini_transformer_v4` 的核心报告路径与六个代表 kernel，且与 spec 保持一致。
- AC-3: 第一轮优先完成两组关键边界 case 文档：`gemm_tiled vs attention_score` 与 `softmax_kernel vs context_mul`；文档必须同时记录共享点与区分点，以区分点分析为主，给出分级结论与当前阶段的执行建议。
- AC-4: 在边界 case 结论基础上回填六张 analysis cards；每张卡都必须使用统一模板、引用明确证据、写明边界说明或不确定性，并可通过固定检查方式做一致性校验。
- AC-5: 从 analysis cards 中归纳出第一版 family cards 与 outlier card；family 命名必须采用“粗类现象型 + 子类机制型”，并优先保证边界清晰性，而不是仅按算子重命名。
- AC-6: 输出一份以方法说明为主的 family synthesis，明确解释 family 判据框架如何工作、为什么边界 case 是第一轮核心，以及为什么这套结构能压缩后续 simulator 验证组织问题。
- AC-7: 将原型状态回挂到 `draft_squash_batch.md` 与 spec 中，确保 draft/spec 对“当前已完成什么、尚未完成什么”的描述一致。

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 1 (Updated: Round 0)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initialized RLCR goal tracker from the refined family-criteria plan | Required before any implementation work in Round 0 | Extracted AC-1 ~ AC-7 and fixed task routing |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task1 | AC-1 | pending | coding | claude | Create workspace plus boundary/outlier directories and schema docs |
| task2 | AC-2 | pending | coding | claude | Add minimal evidence extractor and pytest guardrails |
| task3 | AC-3 | pending | coding | claude | Write first two boundary case documents with graded judgments |
| task4 | AC-4 | pending | coding | claude | Backfill six analysis cards from boundary-case conclusions |
| task5 | AC-5 | pending | coding | claude | Derive family cards and outlier card with boundary-first framing |
| task6 | AC-6 | pending | coding | claude | Write family synthesis with method-first emphasis |
| task7 | AC-7 | pending | coding | claude | Reflect prototype status back into draft/spec |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
