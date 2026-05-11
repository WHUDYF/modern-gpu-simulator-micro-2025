# M1.5 Photon Workload Source Manifest

日期：2026-05-08

本文件记录第一阶段下载的 workload 源码候选池。源码本体不放入本仓库，而是放在：

```text
/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources
```

## Phase 1：源码候选池

| 名称 | 来源 | 主要价值 | 预期 Photon 压力类型 |
|------|------|----------|----------------------|
| `gpu-rodinia` | https://github.com/yuhc/gpu-rodinia.git | 传统 GPU architecture / heterogeneous benchmark | mixed, irregular, large-kernel |
| `gpu-parboil` | https://github.com/yuhc/gpu-parboil.git | throughput computing benchmark | large-kernel, sparse, stencil |
| `shoc` | https://github.com/vetter/shoc.git | SHOC benchmark suite | kernel diversity, GEMM/FFT/MD/Stencil |
| `altis` | https://github.com/utcs-scea/altis.git | modern GPU benchmark suite | modern CUDA, DNN/graph/crypto mix |
| `deepbench` | https://github.com/baidu-research/DeepBench.git | DNN primitive benchmark | high kernel count, large kernels |
| `cutlass` | https://github.com/NVIDIA/cutlass.git | GEMM/conv/attention-like kernel generator | large kernels, many kernel variants |
| `mlperf-inference` | https://github.com/mlcommons/inference.git | MLPerf-style modern AI workload definitions | high kernel count, model pipeline |
| `gunrock` | https://github.com/gunrock/gunrock.git | GPU graph analytics | irregular graph kernels |
| `pannotia` | https://github.com/pannotia/pannotia.git | GPU graph benchmark | irregular graph kernels |
| `hecbench` | https://github.com/zjin-lcf/HeCBench.git | heterogeneous computing benchmark collection | broad suite coverage |
| `lammps` | https://github.com/lammps/lammps.git | molecular dynamics full application | full app, large kernels |
| `gromacs` | https://github.com/gromacs/gromacs.git | molecular dynamics full application | full app, large kernels |

## 不在 Phase 1 下载的内容

- MLPerf / model datasets。
- LLM / diffusion model weights。
- SPEC ACCEL licensed packages。
- Large simulation input decks。

这些内容后续按 workload registry 分批下载，并记录许可、大小、checksum 和 acquisition 状态。

## 下载记录

脚本运行后生成：

```text
/home/dyf/workloads/trace-compressions-industrial-codex-workload/clone_status.tsv
```

