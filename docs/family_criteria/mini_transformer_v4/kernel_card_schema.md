# mini_transformer_v4 Kernel Analysis Card Schema

每张 analysis card 必须包含以下固定区块，后续所有 kernel 一律按同一模板填写：

## 1. Basic Info

- kernel name
- operator semantics
- workload role
- representative note

## 2. Execution Mode

- `compute-heavy` / `memory-heavy` / `mixed` / `uncertain`
- 若为暂定判断，应显式标明

## 3. Key Observed Metrics

- achieved occupancy
- compute throughput
- dram throughput
- l1/l2 hit behavior
- warp cycles
- shmem usage
- waves / launch shape / block limit（若相关）

## 4. Dominant Resource Candidates

- 一个 primary candidate
- 可选一个 secondary candidate
- 若边界不稳，应显式写明

## 5. Family Decision

- tentative family
- boundary note
- ambiguity / outlier note（若相关）

## Evidence Rule

每张 analysis card 至少引用两处明确证据来源：

- E0-E5 诊断报告的具体章节、表格或观测点
- `baseline_ape.json` 的具体字段或 kernel 对应项

## Validation Rule

每张 analysis card 必须通过统一检查：

- 是否包含五个固定区块
- 是否包含至少两处证据引用
- 是否包含边界说明或不确定性说明
