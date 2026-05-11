# 任务计划：M1.5 Photon Workload Suite 源码获取

## 目标
为 Photon 大规模真实 workload 压缩实验下载并登记论文相关 benchmark / workload 源码，优先覆盖大量 kernel instances、大 kernel、不规则 workload 三类压力。

## 当前阶段
阶段 5

## 各阶段

### 阶段 1：需求与边界
- [x] 确认目标 worktree：`/home/dyf/worktrees/trace-compressions-industrial-codex-workload`
- [x] 确认源码下载位置在 git worktree 外部
- [x] 确认第一阶段只下载源码，不下载大模型权重或完整数据集
- **状态：** complete

### 阶段 2：源码获取
- [x] 创建 workload source manifest
- [x] 创建可重复运行的 clone 脚本
- [x] 下载第一批 workload 源码
- [x] 记录成功/失败和 commit hash
- **状态：** complete

### 阶段 3：盘点与筛选
- [x] 将 workload suite 目标升级为可复用 GPU workload trace corpus
- [x] 定义 source / registry / measured feature / trace artifact / training dataset 分层
- [x] 写入 GPU Workload Trace Corpus 设计文档
- **状态：** complete

### 阶段 4：交付
- [x] 检查 git 状态
- [x] 汇总下载路径、成功项、失败项和下一步
- **状态：** complete

### 阶段 5：Gate C0/C1 Registry
- [x] 生成 `registry/source_registry.json`
- [x] 生成 `registry/source_registry.md`
- [x] 生成 `registry/workload_registry.json`
- [x] 生成 `registry/workload_registry.md`
- [x] 运行 registry tests
- [x] 运行 baseline tests
- **状态：** complete

## 关键问题
1. MLPerf / HPC full app 是否后续下载完整数据集与模型权重？
2. 第一轮 acquisition 先从哪些 suite 中挑选可 build/run 的 CUDA workload？

## 已做决策
| 决策 | 理由 |
|------|------|
| 源码放到 `/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources` | 避免把大仓库塞进研究 repo 的 git 状态 |
| 第一阶段只下载源码 | 模型权重、数据集和 full app inputs 体量大且许可复杂 |
| 使用 shallow clone | 快速建立本地候选池，后续需要历史时再加深 |
| MLPerf / HeCBench 使用 sparse/partial clone | 普通 shallow clone 下载停滞，sparse/partial clone 能先落地源码结构 |
| 将 workload suite 升级为 trace corpus | 完整网络和训练集资产对 Photon 与后续工作都有复用价值 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| `mlperf-inference` 普通 shallow clone 停在约 257M | 1 | 改用 sparse/partial clone，并暂不拉数据/权重 |
| `hecbench` 普通 shallow clone 停在约 257M | 1 | 改用 sparse/partial clone，并展开 `src/tools/cmake` |
