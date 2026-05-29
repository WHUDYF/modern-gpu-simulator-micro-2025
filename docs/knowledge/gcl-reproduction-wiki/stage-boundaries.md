# Stage Boundaries

这篇文档定义每个 GCL stage 可以声称什么、不能声称什么。

## GCL-M0 Boundary

[[gcl-m0-offline-embedding-selector]] 可以声称：

```text
embedding-based selector interface 已经跑通。
GCL selector 能输出 cluster、anchor 和 structural compression artifacts。
M0 artifacts 可以和 PKA selector artifacts 在结构层面对齐。
```

M0 不能声称：

```text
embedding 来自真实 RGCN。
embedding 学到了真实 kernel mechanism。
GCL 比 PKA 更准。
sampled simulation accuracy 已验证。
```

## GCL-M1 Boundary

[[gcl-m1-trace-graph-construction]] 可以声称：

```text
trace-like inputs 能被稳定转换为 canonical heterogeneous graph artifacts。
graph artifacts 可以 replay、validate、audit。
```

M1 不能声称：

```text
RGCN 已训练。
graph embedding 已可用。
cluster 已合理。
simulator accuracy 已验证。
```

## GCL-M2 Boundary

[[gcl-m2-rgcn-embedding-and-selector]] 可以声称：

```text
embedding 来自 graph encoder。
learned embedding table 满足 M0 selector contract。
RGCN-derived embeddings 能驱动 M0-style anchor export。
```

M2 不能单独声称：

```text
sampled simulation error 已降低。
measured speedup 已成立。
cross-architecture transfer 已验证。
```

这些属于 [[gcl-m3-simulator-evaluation]]。

## GCL-M3 Boundary

[[gcl-m3-simulator-evaluation]] 可以在有 full-vs-sampled measured evidence 时声称：

```text
sampled reconstruction error
microarchitectural metric error
measured simulator speedup
cross-architecture transfer
```

M3 也必须把 structural compression ratio 和 measured speedup 分开。

## Forbidden Shortcut

所有 selector-side 阶段都不得使用以下字段偷做 grouping：

```text
kernel_name
source_path
family
regime
shape_hint
simulator outcome fields
full-workload cycle totals
B-line semantic metadata
```

这些字段可以用于 audit 或 explanation，但不能进入 selector 或 graph topology decision。

