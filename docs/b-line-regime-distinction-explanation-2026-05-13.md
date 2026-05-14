# B 线 Regime 如何区分

日期：2026-05-13

## 1. 一句话定义

`regime` 是 squash 提取出的同一个 `phase` 内部，一段可复用执行工作区间。

它不是按 kernel 名字拆，也不是按算子名字拆，而是按下面这组结构共同决定：

```text
squash
  -> phase context
  -> family
  -> Route Primitive
  -> Hardware Execution Template
  -> shape / size regime
  -> resource signature
  -> weights / validation role
```

更直白地说：

```text
squash / phase 说明“这个对象处在 workload 时间轴的哪个稳定上下文”
family 说明“在这个 phase 内，哪些对象硬件执行机制大类相近”
regime 说明“在这个 phase 和 family 内，哪些对象真的可以作为同一个 backend 验证对象”
lane 说明“这个 regime 要沿哪个 simulator 参数方向验证”
```

## 2. 为什么先 phase，再 family / regime

主流程应该先由 `squash` 提取不同 phase 和对应上下文参数，再在每个 phase 内展开 family、regime、lane。

原因是 phase 表示 workload 时间结构。对象是否能共享 simulator reasoning，不只取决于硬件执行机制，还取决于它处在 workload 时间轴的哪个稳定上下文中。

因此正确顺序是：

```text
workload
  -> squash / phase
  -> per-phase family
  -> per-phase regime
  -> lane
```

不是：

```text
workload
  -> global family
  -> phase
  -> regime
```

## 3. 为什么 phase 内 family 之后还需要 regime

即使在同一个 phase 里，`family` 仍然太粗。

比如多个对象都属于 dense compute family，说明它们在 GPU 上大体都像 dense tiled compute。但它们仍然可能在以下方面不同：

- 出现的 phase 不同；
- 算法路径角色不同；
- shape / size 区间不同；
- resource signature 不同；
- 时间权重或决策角色不同。

如果 phase 内的 family 后面直接接 C 线，C 线会不知道它到底在验证哪个对象。

因此需要 `regime`：

```text
family = 共享机制组织层
regime = 进入 C 线的代表执行工作区间
```

## 4. 第 1 层：先看 phase context

Regime 首先只在同一个 phase context 内部讨论。

如果两个 anchors 处在不同稳定 phase，默认不应进入同一个 regime，即使它们属于同一 family。

`phase context` 表示对象在 workload 时间结构中的位置。

两个对象即使硬件模板相似，如果它们出现在不同稳定 phase，也不能默认合并。

原因是 phase 不同，说明它们在 workload 执行路线中的上下文不同，后续 simulator reasoning lane 也可能不同。

判断：

```text
不同稳定 phase -> 默认拆开
过渡 phase -> 单独观察或 provisional
同 phase -> 继续看 family
```

## 5. 第 2 层：在同一个 phase 内看 family

Family 在 phase 内讨论。

如果两个 anchors 已经属于不同 family，说明它们的硬件执行机制大类不同，默认不应进入同一个 regime。

例如：

```text
dense_compute
streaming_memory
reduction_normalization
irregular_traversal
```

这些 family 之间不直接合并 regime。

判断：

```text
不同 family -> 不同 regime
同 family -> 继续看 Route Primitive
```

## 6. 第 3 层：看 Route Primitive

`Route Primitive` 是 regime 区分里最重要的算法结构层。

它回答：

```text
这个对象在 workload 主计算路径中扮演什么算法角色？
```

注意，它不是 raw operator string，也不是 kernel name。它是归一化后的算法路径角色。

当前 B 线里常见的 Route Primitive 包括：

| Route Primitive | 含义 |
|---|---|
| `Dense Projection/Transform` | dense projection / transform 主计算路径 |
| `Pairwise Score` | 两两关系分数计算 |
| `Reduction / Normalize` | reduction、normalization |
| `Weighted Aggregation` | 加权聚合 |
| `Elementwise Fusion` | elementwise 融合 |

这一步保留算法结构，但不是按算子名字做分类。

例如，不能因为 kernel 名字叫 `softmax_kernel` 就直接建 regime；正确做法是把它归一化成：

```text
Route Primitive = Reduction / Normalize
```

判断：

```text
Route Primitive 不同 -> 默认拆开，或 weak-share / boundary
Route Primitive 相同 -> 继续看 Hardware Execution Template
```

## 7. 第 4 层：看 Hardware Execution Template

`Hardware Execution Template` 表示这个对象在 GPU 上主要通过什么执行骨架实现。

它回答：

```text
这个算法路径角色在硬件上是怎么跑的？
```

常见 template 包括：

| Hardware Template | 含义 |
|---|---|
| `Dense Tiled Compute` | tile 化 dense compute |
| `Reduction Template` | reduction / synchronization 骨架 |
| `Streaming Aggregation Template` | streaming read / aggregation |
| `Elementwise Template` | elementwise 执行路径 |

同一个上层路线里的对象也可能有不同 hardware template。

例如 attention 路线中：

| Route Primitive | Hardware Template |
|---|---|
| `Pairwise Score` | `Dense Tiled Compute` |
| `Reduction / Normalize` | `Reduction Template` |
| `Weighted Aggregation` | `Streaming Aggregation Template` |

它们同属 attention 上层路线，但不能合成一个 regime。

判断：

```text
Hardware Template 不同 -> 拆开
Hardware Template 相同 -> 继续看 shape / size regime
```

## 8. 第 5 层：看 shape / size regime

Regime 不是一个孤立 shape，而是一段 shape / size 空间里的典型工作方式。

例如 dense compute 中可能有：

```text
projection-like dense region
expansion-like dense region
pairwise-score dense region
```

这些都是 shape / size regime。

这里不能只按单个 grid/block 数字硬切。更合理的是看它是否形成稳定区间或类别，例如：

- small / medium / large sequence length；
- projection-like dense region；
- expansion-like dense region；
- row-wise normalization region；
- streaming aggregation region。

判断：

```text
shape / size regime 明显不同 -> 拆开或 boundary
shape / size regime 相近 -> 继续看 resource signature
```

## 9. 第 6 层：看 resource signature

`resource signature` 是最后一道硬件响应检查。

即使对象已经满足：

```text
同 family
同 phase
同 Route Primitive
同 Hardware Template
shape / size 相近
```

如果 resource signature 明显不同，也不能强行合并。

常见 resource signature 包括：

| Resource Signature | 含义 |
|---|---|
| `register / occupancy sensitive` | register 或 occupancy 影响明显 |
| `shared-memory coupled` | shared memory 参与强 |
| `DRAM pressure` | 受 DRAM 带宽或延迟影响 |
| `cache-capacity sensitive` | cache 容量或命中行为重要 |
| `locality / L1-resident` | locality 强，L1 行为重要 |
| `reduction / synchronization sensitive` | reduction / sync path 重要 |
| `irregular control / atomic sensitive` | divergence、atomic 或不规则控制重要 |

判断：

```text
resource signature 不同 -> 拆开或 boundary
resource signature 相容 -> 可以考虑合并
```

## 10. 第 7 层：看 weights / validation role

最后还要看对象在决策层的角色是否相容。

主要包括：

- coverage weight；
- time weight；
- decision weight；
- validation role；
- primary lane。

例如：

```text
high-time main-object
low-time constraint-object
```

这两个不能合成一个 stable regime。

原因是 C 线对它们提出的问题不同：

- main-object：是否能带来有效 tuning gain？
- constraint-object：是否保证 regression / correctness 不坏？

判断：

```text
weights / validation role 强冲突 -> 拆开
primary lane 不同 -> 拆开或 boundary
weights / validation role 相容 -> 可以合并
```

## 11. 完整判断流程

可以把 regime 区分理解成下面这棵决策树：

```text
phase context 相容吗？
  否 -> 拆开或 provisional
  是 -> 同 family 吗？
    否 -> 不同 regime
    是 -> Route Primitive 相同或相容吗？
      否 -> 拆开、weak-share 或 boundary
      是 -> Hardware Execution Template 相同或相容吗？
        否 -> 拆开
        是 -> shape / size regime 相近吗？
          否 -> 拆开或 boundary
          是 -> resource signature 相容吗？
            否 -> 拆开或 boundary
            是 -> weights / validation role 相容吗？
              否 -> 拆开
              是 -> 可以合并为同一个 stable regime
```

## 12. 一个具体例子

假设有三个 anchors 都在 dense family 中：

| Anchor | Phase | Route Primitive | Hardware Template | Shape / Size | Resource Signature | 结论 |
|---|---|---|---|---|---|---|
| A1 | Phase A | Dense Projection/Transform | Dense Tiled Compute | projection-like | register-limited | regime R1 |
| A2 | Phase B | Pairwise Score | Dense Tiled Compute | pairwise-score | shared-memory-coupled | regime R2 |
| A3 | Phase C | Dense Projection/Transform | Dense Tiled Compute | expansion-like | register-sensitive | regime R3 |

它们同属 dense family，说明底层有共享机制。

但它们不应直接合成一个 regime，因为：

- phase 不同；
- Route Primitive 不同或上下文不同；
- shape / size regime 不同；
- resource signature 不同；
- C 线验证问题不同。

所以 B 线应保留多个 regimes。

这不是按 kernel name 拆，也不是按网络模块名拆，而是按：

```text
phase + route primitive + hardware template + shape/size + resource signature
```

拆。

## 13. 和 PKA cluster 的关系

PKA cluster 只说明：

```text
这些 records 在 PKA feature space 中相似
```

它不等于 regime。

如果一个 PKA cluster 内部包含不同 phase、不同 Route Primitive、不同 Hardware Template 或不同 resource signature 的 records，B 线必须拆开或标 boundary。

换句话说：

```text
PKA cluster = A 线 representative compression 结果
regime = B 线 simulator-side execution object
```

两者可以相关，但不能直接等同。

## 14. 最短总结

Regime 的区分不是：

```text
按 kernel name
按 operator string
按网络模块名
按 PKA cluster
按单个 shape 数字
```

而是：

```text
在同一个 family 内，
按 phase context、
Route Primitive、
Hardware Execution Template、
shape / size regime、
resource signature、
weights / validation role
逐层判断。
```

其中 `Route Primitive` 是算法结构进入 regime 的正式方式。

所以我们不是排斥算法拆分，而是把算法拆分从 raw operator name 提升成更稳定的 workload route 结构。
