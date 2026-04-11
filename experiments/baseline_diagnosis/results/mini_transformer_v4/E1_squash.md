# 诊断报告：mini-transformer v4 [E1_squash]

**日期：** 2026-04-11
**硬件：** RTX 3080 Ti (SM_86)
**启用机制：** 仅 Squash

---

## Squash 输出：6 段分解

| 段 | 内核范围 | 凝聚度 | 代表 kernel |
|----|---------|:------:|------------|
| 0 | gemm×4, attention_score, softmax, context_mul | 0.933 | gemm_tiled_1 |
| 1 | gemm_tiled（输出投影） | 1.000 | gemm_tiled_8 |
| 2 | residual_add（注意力后） | 1.000 | residual_add_9 |
| 3 | layernorm, ffn_gemm×2 | 0.923 | layernorm_10 |
| 4 | residual_add（FFN 后） | 1.000 | residual_add_13 |
| 5 | layernorm（FFN 后） | 1.000 | layernorm_14 |

TB 级：全部 kernel 0 个 HOT 字段（内部均匀）。

---

## 相对 v1 的关键变化

**v1（broken）的 Squash：** 8 段，段 1 凝聚度=0.850，attention_score 和 softmax 异质性高。

**v4（clean）的 Squash：** 6 段，最低凝聚度=0.923。attention_score 和 softmax
被归入段 0（与 GEMM 同段），cohesion=0.933。

原因：v4 中 attention_score（compute=95.2%）和 softmax（compute=85.5%）的
行为特征与 gemm_tiled（compute=90.9%）已高度相似——均为计算密集型。
软件修复消除了 v1 中的"异常段"信号，Squash 不再将它们作为独立行为区域隔离。

---

## 发现与判定

**发现：** v4 的段结构更紧凑（8→6 段），段内凝聚度更高（最低 0.923 vs 0.850）。
这确认了软件层清洗对 workload 特征的影响是实质性的，而非微小调整。

**段 3**（layernorm + ffn_gemm×2，cohesion=0.923）：
layernorm 夹在两个 GEMM 之间，凝聚度略低，反映 layernorm 的混合计算/归约特性
与纯计算的 GEMM 有细微差异——但差异来自算法逻辑，非软件缺陷。

**判定：确认性（Confirming）**

Squash 的作用与 v1 相同：正确分解了 Transformer 层结构，但这一结构从
kernel 名称也可推断。Squash 在 v4 上提供的独立价值仍然有限。

**模拟复用建议（新增）：** 段 0 内 7 个 kernel（gemm×4 + attention_score + softmax + context_mul）
行为相似（凝聚度=0.933），可以用段内任一 kernel 的 trace 代表整段进行模拟，
降低 trace 采集和模拟成本。
