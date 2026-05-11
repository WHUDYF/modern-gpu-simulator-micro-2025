# 发现与决策

## 需求
- 用户希望下载论文相关、数量足够多、足够典型的 workload 源码，用于后续验证 Photon 对大量 kernel workload、大 kernel workload、不规则 workload 的压缩效率。
- 用户进一步确认需要完整网络级 workload，并希望详细训练集资产能支持后续其他工作。

## 研究发现
- 候选来源应优先覆盖 Rodinia、Parboil、SHOC、Altis、DeepBench、MLPerf Inference、CUTLASS、Gunrock / graph workloads、Pannotia、HPC full apps。
- Accel-Sim 相关工作强调 kernel-instance 数量，适合作为 Photon workload suite 的规模叙事参考。
- 第一阶段应下载源码与 benchmark harness；模型权重、数据集和完整 benchmark inputs 后续按 workload registry 分批获取。
- 当前 `2.8G` 左右主要是源码 / benchmark harness，不代表最终 corpus 规模；Photon claim-bearing 规模应由 full-network kernel launches、dynamic traces、measured metadata 和 shape grid 决定。

## 技术决策
| 决策 | 理由 |
|------|------|
| 外部下载目录与 repo 分离 | 避免大量第三方源码进入 git status |
| 每个第三方仓库记录 URL、target path、commit hash | 保证后续论文实验可复现 |
| 下载脚本可重复运行且跳过已存在目录 | 支持中断恢复 |
| 先写 corpus 设计再下载权重/数据集 | 防止形成不可复现、许可不清、schema 不一致的大文件堆 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| 部分 workload 源码可能很大或依赖数据集 | 本阶段只 shallow clone 源码，后续单独处理 inputs |

## 资源
- `/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources`
- `docs/superpowers/specs/2026-05-11-gpu-workload-trace-corpus-design.md`

## 视觉/浏览器发现
- 未使用视觉输入。
