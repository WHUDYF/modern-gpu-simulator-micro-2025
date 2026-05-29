# Implementation Roadmap

这篇文档把 GCL reproduction 拆成可执行顺序。

## Step 1: GCL-M0

先实现 [[gcl-m0-offline-embedding-selector]]。

目标：

```text
fixture/offline embedding table
  -> selector
  -> anchor/evaluation artifacts
```

验收：

- 读取 embedding table；
- 校验 schema；
- 拒绝 forbidden fields；
- 执行 z-score normalization；
- 默认 `silhouette_k`；
- 支持 `deterministic_fixed_k`；
- 输出 M0 formal artifacts；
- 不影响 PKA baseline tests。

## Step 2: GCL-M1

然后实现 [[gcl-m1-trace-graph-construction]]。

目标：

```text
trace records
  -> canonical heterogeneous graph artifacts
```

验收：

- 读取 trace manifest；
- 支持 fixture trace JSON；
- 构建 instruction nodes；
- 构建 control-flow edges；
- 构建至少一种 variable node；
- 构建至少一种 data-flow edge；
- 输出 graph bundle 和 audit；
- 相同输入 replay 出相同 graph hash。

## Step 3: GCL-M2

再实现 [[gcl-m2-rgcn-embedding-and-selector]]。

目标：

```text
graph artifacts
  -> RGCN embeddings
  -> M0-style selector outputs
```

验收：

- graph tensorization 可复现；
- augmentation manifest 可审计；
- RGCN training report 和 model manifest 可 replay；
- embedding table 满足 M0 contract；
- selector-side outputs 复用 M0 artifact semantics。

## Step 4: GCL-M3

最后实现 [[gcl-m3-simulator-evaluation]]。

目标：

```text
representative anchors
  -> sampled/full simulator metric comparison
```

验收：

- full workload metrics 存在；
- sampled representative metrics 存在；
- 可以计算 cycles 和 selected microarchitectural metric error；
- 可以和 PKA 在同一 workload set 上比较；
- measured speedup 与 structural compression ratio 分开报告。

## Practical Rule

如果某个阶段失败，先修该阶段的 artifact contract，不要直接进入下一阶段。

跨阶段连接只通过 [[artifact-contracts]] 中定义的 artifacts 和 hash chain 完成。

