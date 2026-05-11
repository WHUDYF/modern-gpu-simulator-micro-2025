# M1.5 Photon Workload Source Download Status

日期：2026-05-08

源码根目录：

```text
/home/dyf/workloads/trace-compressions-industrial-codex-workload/sources
```

状态表：

| 名称 | 状态 | Commit | 大小 | 备注 |
|------|------|--------|------|------|
| `gpu-rodinia` | downloaded | `9c10d3e` | 124M | Rodinia GPU mirror |
| `gpu-parboil` | downloaded | `ccf3d31` | 19M | Parboil GPU mirror |
| `shoc` | downloaded | `00b25e2` | 9.1M | SHOC benchmark suite |
| `altis` | downloaded | `042e292` | 254M | modern GPU benchmark suite |
| `deepbench` | downloaded | `da81ba7` | 3.5M | DNN primitive benchmark |
| `cutlass` | downloaded | `ae6bccf` | 203M | GEMM/conv/kernel generator |
| `mlperf-inference` | sparse downloaded | `7b11eeb` | 197M | sparse/partial clone; no datasets or model weights |
| `gunrock` | downloaded | `748f79e` | 4.0M | graph analytics |
| `pannotia` | downloaded | `16c1f09` | 199M | graph workload suite |
| `hecbench` | sparse downloaded | `48992b6` | 891M | sparse/partial clone with source directories expanded |
| `lammps` | downloaded | `a793f27` | 617M | molecular dynamics full application |
| `gromacs` | downloaded | `cecaeca` | 283M | molecular dynamics full application |

自动生成的 clone 状态文件：

```text
/home/dyf/workloads/trace-compressions-industrial-codex-workload/clone_status.tsv
```

## Registry Artifacts

- `registry/source_registry.json`
- `registry/source_registry.md`
- `registry/workload_registry.json`
- `registry/workload_registry.md`

`workload_registry.*` is a draft. Build/run/input/license statuses remain `pending` or `needs_review` until Gate C2/C3.

## 注意事项

- 本阶段只下载源码 / benchmark harness。
- 未下载 MLPerf 数据集、模型权重、SPEC ACCEL licensed packages 或大型 simulation input decks。
- `mlperf-inference` 和 `hecbench` 使用 sparse/partial clone，后续应按具体 workload 目录逐个展开和验证。
- 已生成 M1.5 workload registry 草案；下一步是 Gate C2/C3：为完整网络、graph、HPC input assets 写下载/许可计划，并验证 build/run/trace smoke。
