# Trace Bottleneck Cost Map

## Summary

- Total records: 28
- Measured records: 18
- Estimated records: 7
- Appendix/excluded records: 3

## Measured Bottleneck Counts

- `balanced / mixed`: 6
- `capture / fixed overhead`: 2
- `simulator throughput`: 4
- `trace export / I/O`: 6

## Cost Map

| Suite | Case | Category | Trace Size | Export | Sim Proxy | Status | Dominant Bottleneck | Evidence |
|---|---|---|---:|---:|---:|---|---|---|
| GPU_Microbenchmark | MaxFlops | compute | 8.715 | 3.52 | 1.53 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | atomic_add_bw | atomic / bandwidth | 5.423 | 2.27 | 10.13 | measured | simulator throughput | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | atomic_add_bw_conflict | atomic / bandwidth | 3.112 | 2.13 | 10.09 | measured | simulator throughput | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | atomic_add_lat | latency | 0.288 | 2.11 | 1.26 | measured | capture / fixed overhead | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_bw_128 | bandwidth | 24.038 | 6.59 | 2.13 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_bw_32f | bandwidth | 2.962 | 2.24 | 1.46 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_bw_32f_unroll | bandwidth | 4.544 | 2.56 | 1.49 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_bw_32f_unroll_large | bandwidth | 7.511 | 2.79 | 1.69 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_bw_64f | bandwidth | 3.026 | 2.29 | 1.49 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_lat | latency | 37.759 | 31.05 | 1.57 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l1_shared_bw | bandwidth | 40.936 | 7.81 | 2.75 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l2_bw_128 | bandwidth | 147.053 | 23.78 | 17.01 | measured | balanced / mixed | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l2_bw_32f | bandwidth | 568.192 | 69.41 | 17.0 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | l2_lat | latency | 37.791 | 33.14 | 1.58 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | mem_bw | bandwidth | 47.975 | 8.43 | 14.92 | measured | simulator throughput | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | mem_lat | latency | 5.267 | 2.91 | 11.12 | measured | simulator throughput | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | shared_bw | bandwidth | 123.217 | 34.55 | 4.89 | measured | trace export / I/O | docs/trace-benchmark-2026-04-03.md |
| GPU_Microbenchmark | shared_lat | latency | 0.34 | 2.18 | 1.29 | measured | capture / fixed overhead | docs/trace-benchmark-2026-04-03.md |
| BabelStream | copy/scale/add/triad/dot | bandwidth | - | - | 20.0 | estimated | trace export or simulator depending on array size | https://github.com/UoB-HPC/BabelStream |
| nvbandwidth | memcpy and link bandwidth patterns | bandwidth / link copy | - | - | 20.0 | estimated | export / I/O for large sweeps; communication path for multi-link modes | https://github.com/NVIDIA/nvbandwidth |
| nvbench | runtime and compile-time parameter sweeps | generic kernel benchmark | - | - | 120.0 | estimated | benchmark sweep explosion | https://github.com/NVIDIA/nvbench |
| CUTLASS profiler | GEMM and convolution configs | dense compute | - | - | 120.0 | estimated | simulator throughput and parameter sweep explosion | https://github.com/NVIDIA/cutlass/wiki/Performance-Profiling |
| Rodinia | nn/backprop/bfs/lud/nw | irregular / mixed | - | - | 30.0 | estimated | mixed control / trace depth | experiments/baseline_diagnosis/results/rodinia |
| Parboil | sgemm/stencil/cutcp/mri-q/histo/bfs | mixed dense / irregular | - | - | 30.0 | estimated | trace export plus irregularity | APEs/*/parboil.md |
| PolyBench/GPU | gemm/3mm/3DConvolution/atax/bicg/syrk | dense / regular | - | - | 5.0 | estimated | simulator throughput for compute-heavy configs | APEs/*/polybench.md |
| NCCL-tests | all_reduce_perf and related collectives | multi-GPU communication | - | - | - | excluded | different problem class | https://github.com/NVIDIA/nccl-tests |
| OSU micro-benchmarks | MPI latency and bandwidth microbenchmarks | communication / network | - | - | - | excluded | different problem class | https://github.com/forresti/osu-micro-benchmarks |
| MLPerf Inference / Training | BERT/ResNet/DLRM/Llama2/Mixtral | full workload anchor | - | - | 200000.0 | appendix_only | full workload scale / trace explosion | https://docs.mlcommons.org/inference/index_gh/ |
