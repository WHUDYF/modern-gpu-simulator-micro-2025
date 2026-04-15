# Family Criteria Workspace

本目录用于存放第一版 `squash + batch` family 判据原型的文档与中间产物。

## Scope

- 输入 workload：`mini_transformer_v4`
- 输入证据：现有 E0-E5 诊断报告与 APE JSON
- 输出对象：
  - `boundary_cases/`：边界 case 文档
  - `analysis_cards/`：kernel 分析卡
  - `family_cards/`：family 解释卡片
  - `outlier_analysis/`：离群 kernel 的单独分析与后续判断记录
  - `mini_transformer_v4_family_synthesis.md`：以方法说明为主的综合文档

## Non-Goals

- 不展开 `delta`
- 不输出 simulator 具体参数处方
- 不做跨 workload 泛化验证
- 不声称第一版 family 规则已稳定

## Version-1 Principle

第一版优先处理关键边界 case，再由边界 case 反推 analysis cards、family cards 与 synthesis。
