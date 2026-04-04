# Trace 基准测试汇总

日期：2026-04-03

## 范围

这份报告汇总了 `modern-gpu-simulator-micro-2025` 中可用 `GPU_Microbenchmark` workload 的三类测量结果：

1. Trace 大小
2. Trace 导出耗时与导出吞吐
3. Trace 驱动模拟器的运行速度

## 测试环境

- 主机 GPU：NVIDIA GeForce RTX 5090
- 驱动版本：580.105.08
- CUDA Toolkit：12.8
- Trace 输出根目录：
  `/home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/benchstudy-20260403`
- 聚合后的原始结果表：
  `/tmp/bench_summary.tsv`

## 测量方法

### Trace 导出

对以下目录中的每个 benchmark：

`/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-app-collection/src/cuda/GPU_Microbenchmark/bin`

都在 GPU 0 上使用 NVBit tracer 运行，并记录这些指标：

- 整个 trace 目录总大小
- `threadblocks/` 大小
- `extra_info/enhanced_execution_info.json` 大小
- `dynamic_trace.pb` 大小
- trace 导出墙钟时间
- 根据大小和耗时计算出的导出吞吐，单位 MiB/s

### 模拟器速度

对每个成功生成的 trace，再用以下程序进行消费：

`/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out`

统一参数如下：

- 配置：`SM120_RTX5090`
- `OMP_NUM_THREADS=1`
- `-gpgpu_max_cycle 10000`

因此，下表中的模拟器速度是受控 `10000 cycle` 窗口内的结果，不是整条 trace 自然跑完的端到端总耗时。

## 结果表

| benchmark | trace 状态 | trace 大小 (MiB) | 导出耗时 (s) | 导出速度 (MiB/s) | 动态指令数 | 仿真状态 | 仿真耗时 (s) | 10k cycle 内模拟指令数 | 仿真 inst/s | 仿真 cycle/s |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| MaxFlops | ok | 8.715 | 3.52 | 2.476 | 525696 | ok | 1.53 | 402880 | 201440 | 5000 |
| atomic_add_bw | ok | 5.423 | 2.27 | 2.389 | 378880 | ok | 10.13 | 2211200 | 221120 | 1000 |
| atomic_add_bw_conflict | ok | 3.112 | 2.13 | 1.461 | 271360 | ok | 10.09 | 4545846 | 505094 | 1111 |
| atomic_add_lat | ok | 0.288 | 2.11 | 0.137 | 30378 | ok | 1.26 | 46 | 46 | 10000 |
| l1_bw_128 | ok | 24.038 | 6.59 | 3.648 | 3712320 | ok | 2.13 | 289696 | 144848 | 5000 |
| l1_bw_32f | ok | 2.962 | 2.24 | 1.322 | 292640 | ok | 1.46 | 72928 | 36464 | 5000 |
| l1_bw_32f_unroll | ok | 4.544 | 2.56 | 1.775 | 644480 | ok | 1.49 | 72928 | 36464 | 5000 |
| l1_bw_32f_unroll_large | ok | 7.511 | 2.79 | 2.692 | 744000 | ok | 1.69 | 109664 | 109664 | 10000 |
| l1_bw_64f | ok | 3.026 | 2.29 | 1.321 | 259264 | ok | 1.49 | 104320 | 104320 | 10000 |
| l1_lat | ok | 37.759 | 31.05 | 1.216 | 82664 | ok | 1.57 | 1553 | 776 | 5000 |
| l1_shared_bw | ok | 40.936 | 7.81 | 5.241 | 4167232 | ok | 2.75 | 251680 | 125840 | 5000 |
| l2_bw_128 | ok | 147.053 | 23.78 | 6.184 | 24393728 | ok | 17.01 | 6419552 | 377620 | 588 |
| l2_bw_32f | ok | 568.192 | 69.41 | 8.186 | 56820736 | ok | about 17 | 2961792 | 174223 | 588 |
| l2_bw_64f | partial_or_timeout | 0.000 | 90.18 | 0.000 | 0 | missing |  |  |  |  |
| l2_lat | ok | 37.791 | 33.14 | 1.140 | 82664 | ok | 1.58 | 1558 | 1558 | 10000 |
| mem_bw | ok | 47.975 | 8.43 | 5.691 | 6189056 | ok | 14.92 | 2074976 | 148212 | 714 |
| mem_lat | ok | 5.267 | 2.91 | 1.810 | 633968 | ok | 11.12 | 7620000 | 762000 | 1000 |
| shared_bw | ok | 123.217 | 34.55 | 3.566 | 15639424 | ok | 4.89 | 321696 | 64339 | 2000 |
| shared_lat | ok | 0.340 | 2.18 | 0.156 | 17344 | ok | 1.29 | 62056 | 62056 | 10000 |

## 关键结论

- 这一批数据中最大的 trace 是 `l2_bw_32f`，大小为 `568.192 MiB`。
- 观测到的最高 trace 导出吞吐也是 `l2_bw_32f`，为 `8.186 MiB/s`。
- 在这次受控仿真中，最慢的两个 workload 是 `l2_bw_128` 和 `l2_bw_32f`，都只有 `588 cycle/s`。
- 模拟器指令吞吐最高的是 `mem_lat`，达到 `762000 inst/s`。
- 最小的两个 trace 是 `atomic_add_lat`（`0.288 MiB`）和 `shared_lat`（`0.340 MiB`）。

## 说明

- 对大多数 workload 来说，trace 的主要体积来自 `threadblocks/`，而不是 `dynamic_trace.pb`。
- `l1_lat` 和 `l2_lat` 的动态指令数并不高，但 trace 仍然偏大，主要是因为静态增强信息和按 threadblock 组织的 trace 结构占了主要空间。
- `l2_bw_64f` 在设定的超时窗口内没有完成 trace 导出，所以这一批没有对应的仿真数据。
- 对 `l2_bw_32f`，模拟器日志里明确出现了 `gpgpu_simulation_time = 17 sec`，但单独的 `/usr/bin/time` 输出文件为空，所以表里记作 `about 17`。
