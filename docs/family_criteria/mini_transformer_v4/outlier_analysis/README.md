# Outlier Analysis

本目录用于保存无法稳定并入现有 family 的 kernel 的单独分析记录。

## Role

第一版中，outlier 不是噪声桶，而是结构保留区。  
当某个 kernel：

- 同时呈现多个强信号
- 无法稳定并入已有 family
- 或其边界仍明显不稳

应先在本目录中保留单独分析，而不是强行并类。

## Version-1 Principle

第一版对 outlier 采用宽松定义：

**不能稳定并入已有 family 的 kernel，可以先视为 outlier。**

后续再根据其是否真正影响验证组织，决定是否收紧定义。

## Relationship to Family Cards

- `family_cards/` 负责记录稳定 family
- `outlier_analysis/` 负责记录尚未被 family 体系稳定吸收的对象

两者共同构成第一版结构化输出。
