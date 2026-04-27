# A 线 L2 附加验证集 Acquisition List（第一版）

日期：2026-04-26

## 1. 目的

这份清单用于定义：

**A 线 `L2` 附加验证集应如何获取，以及当前最推荐的获取顺序。**

L2 的主要角色是：

- 验证 A 线 compression 的实现程度
- 检查 baseline 与 extension 的压缩效果差异
- 提供足够多的 kernel / invocation 对象用于 clustering experiment

因此，L2 的核心不是“单个对象是否极易人工解释”，
而是：

- 数据量足够
- 行为类型覆盖足够
- acquisition 成本可控

---

## 2. L2 推荐来源

### 第一阶段主来源

1. `Rodinia`
2. `Altis`
3. `Parboil`
4. `PolyBench/GPU`

### 第二阶段扩样本来源

1. `CUTLASS Profiler`
2. `HeCBench`

---

## 3. L2 acquisition list

| ID | 来源 | 获取目标 | 优先级 | 目标用途 | 预计产出 | 当前状态 | 备注 |
|---|---|---|---|---|---|---|---|
| `L2_RD_CORE` | Rodinia | 获取 `nn`, `backprop`, `bfs`, `lud`, `nw` 等核心 benchmark kernel | `P0` | 第一阶段 compression benchmark 主干 | `10 ~ 20` 个稳定 kernel / 多个 invocation | `needs_acquisition` | 最先 bring-up |
| `L2_ALTIS_CORE` | Altis | 选择 `3 ~ 5` 个能稳定编译和 profile 的 GPU benchmarks | `P0` | 现代 benchmark 补充 | `8 ~ 15` 个 kernel | `needs_acquisition` | 作为 Rodinia 后第一扩展集 |
| `L2_PARBOIL_CORE` | Parboil | 选择 `3 ~ 4` 个行为差异明显的 benchmark | `P1` | 补充 benchmark 多样性 | `6 ~ 10` 个 kernel | `needs_acquisition` | bring-up 成本中等 |
| `L2_POLYBENCH_CORE` | PolyBench/GPU | 选择 `3 ~ 4` 个 dense / regular kernels | `P1` | 规则数值 kernel 补点 | `6 ~ 10` 个 kernel | `needs_acquisition` | dense 结构补充 |
| `L2_CUTLASS_SWEEP` | CUTLASS Profiler | 参数 sweep 生成大量 dense compute kernels | `P1` | 扩样本、做 compression robustness test | `30 ~ 200` 个 kernels / cases | `needs_acquisition` | 样本增长最快 |
| `L2_HECBENCH_EXT` | HeCBench | 第二阶段引入更大规模 benchmark 集 | `P2` | 泛化验证 | `20+` benchmarks / 大量 kernels | `needs_acquisition` | 第一阶段不建议全量上 |

---

## 4. 第一阶段推荐获取顺序

### Step 1：Rodinia

原因：

- 社区熟悉
- 行为类型多
- 与既有 GPU simulation 相关工作更容易对话

当前建议：

- 先从 `nn`, `backprop`, `bfs`, `lud`, `nw` 开始

### Step 2：Altis

原因：

- 比经典套件更现代
- 更能补今天 GPU 系统环境下的 benchmark 行为

当前建议：

- 先选 `3 ~ 5` 个稳定 benchmark 做第一批 bring-up

### Step 3：Parboil + PolyBench/GPU

原因：

- 用于补更多真实 kernel 类型
- 增强 compression space 覆盖性

### Step 4：CUTLASS Profiler

原因：

- 当第一阶段 benchmark kernel 数量仍不够时
- 它是最快的扩样本手段

### Step 5：HeCBench

原因：

- 适合作为第二阶段大规模泛化验证
- 不适合作为当前第一步 bring-up

---

## 5. 每一类来源的实际角色

### Rodinia

角色：

- 第一阶段 benchmark kernel 主干集合

### Altis

角色：

- 第一阶段现代 benchmark 补充

### Parboil / PolyBench

角色：

- 第一阶段多样性补充集

### CUTLASS Profiler

角色：

- 扩样本层
- dense parameter sweep layer

### HeCBench

角色：

- 第二阶段泛化层

---

## 6. acquisition 不是下载源码这么简单

L2 的真正目标不是“把 benchmark 仓库 clone 下来”，
而是形成一条稳定的数据生成链：

`benchmark source -> executable -> stable run -> NCU capture -> PKA feature table -> manifest entry`

因此每个 acquisition 项都应最终回答：

1. 这个 benchmark 能否稳定编译？
2. 能否稳定运行？
3. 能否稳定采集 NCU？
4. 能否映射到 A 线需要的 PKA feature table？

如果其中任一步不稳定，
就不能把它正式记入 L2 主集合。

---

## 7. L2 的目标规模

第一阶段建议目标：

- `30 ~ 50` 个可验证 kernel 对象

如果加入参数变体、invocation variants 或 CUTLASS sweep，
第二阶段很容易扩展到：

- `100 ~ 300` 个 compression objects

这个规模已经足以做第一轮 A 线 compression evaluation。

---

## 8. 需要为 L2 单独记录的字段

建议每个 L2 acquisition entry 至少记录：

- `benchmark_name`
- `source_repo_or_origin`
- `build_status`
- `run_status`
- `ncu_status`
- `feature_table_status`
- `expected_behavior_axis`
- `target_priority`
- `notes`

如果没有这些状态字段，
L2 很容易退化成“下载了很多东西，但不知道哪些真的可用”。

---

## 9. 当前最建议的简短结论

如果压成最短形式：

1. L2 不应从 HeCBench 全量开始，而应先从 `Rodinia + Altis` 起步。
2. `Parboil + PolyBench/GPU` 适合作为第一阶段补充集。
3. `CUTLASS Profiler` 最适合在 baseline 稳定后快速扩样本。
4. `HeCBench` 更适合第二阶段做泛化验证，而不是当前第一步 bring-up。
5. L2 的关键不是下载源码，而是打通 `run -> profile -> feature table -> manifest` 的完整 acquisition pipeline。
