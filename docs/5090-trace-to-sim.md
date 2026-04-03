# RTX 5090 从 Trace 到模拟

这份文档记录了这台服务器上已经验证通过的一条工作流：

1. 编译 tracer 和 CUDA workload
2. 在 RTX 5090 上生成真实 trace
3. 用 `accel-sim.out` 消费这份 trace

当前文档里验证通过的 workload 是 `GPU_Microbenchmark/l1_bw_32f`。之所以先用它，是因为它在这台机器上运行稳定。`backprop-rodinia-2.0-ft` 虽然可以编译，但在这台服务器上会在 kernel 结束后发生段错误，不适合作为第一条验证路径。

## 1. 前置条件

这台服务器已经提供：

- GPU：RTX 5090
- CUDA Toolkit：`/usr/local/cuda-12.8`
- NVIDIA 驱动：`580.105.08`

用户目录侧默认通过 `~/.bashrc` 提供这些环境：

- `CUDA_INSTALL_PATH=/usr/local/cuda-12.8`
- `PROTOBUF_PREFIX=$HOME/opt/protobuf-3.21.12`
- `PATH`
- `LD_LIBRARY_PATH`
- `LIBRARY_PATH`
- `CPATH`

开始之前建议先执行：

```bash
source ~/.bashrc
```

## 2. 编译模拟器

这一步只需要做一次，或者在模拟器源码变更后重做。

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled

export IS_SERT=0
source ./gpu-simulator/setup_environment_no_git.sh

make -j 2 -C ./gpu-simulator \
  SKIP_OPENCL=1 \
  SKIP_CUOBJDUMP_TO_PTXPLUS=1 \
  PROTOBUF_INCLUDES="-I$PROTOBUF_PREFIX/include" \
  PROTOBUF_LIBS="-L$PROTOBUF_PREFIX/lib64 -lprotobuf"
```

生成的模拟器主程序在：

```bash
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/bin/release/accel-sim.out
```

## 3. 编译 tracer

这一步只需要做一次，或者在 tracer 相关源码修改后重做。

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
make -j 2 -C ./util/tracer_nvbit
```

生成的 tracer 动态库在：

```bash
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.so
```

## 4. 准备 GPU App Collection 兼容目录

仓库里的部分老 benchmark Makefile 仍然假设存在一个 CUDA SDK 4.2 风格的目录结构。下面这组兼容目录就足够支撑本文档中的工作流：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-app-collection/src

mkdir -p C 4.2
ln -sfn ../cuda/common ./C/common
ln -sfn ../cuda/common/lib ./C/lib
mkdir -p ./C/shared/inc ./C/shared/lib
ln -sfn ../C ./4.2/C
```

## 5. 编译已验证的 workload

这里使用已经验证通过的 `GPU_Microbenchmark`。

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled

export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export NVIDIA_COMPUTE_SDK_LOCATION=$PWD/gpu-app-collection/src/C

source ./gpu-app-collection/src/setup_environment
eval "make -j 2 $MAKE_ARGS -C ./gpu-app-collection/src GPU_Microbenchmark"
```

本文档中验证通过的可执行文件是：

```bash
/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-app-collection/src/cuda/GPU_Microbenchmark/bin/l1_bw_32f
```

如果你想先确认 workload 自己是否能跑，可以直接执行：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
./gpu-app-collection/src/cuda/GPU_Microbenchmark/bin/l1_bw_32f
```

## 6. 在 RTX 5090 上生成 trace

下面这条命令已经在 GPU 0 上验证通过。如果你想用第二张 5090，把 `CUDA_VISIBLE_DEVICES=0` 改成 `1`，同时改一下输出目录即可。

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled

export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=$CUDA_INSTALL_PATH/bin:$PATH

export CUDA_VISIBLE_DEVICES=0
export CUDA_VERSION=12.8
export USER_DEFINED_FOLDERS=1
export TRACES_FOLDER=/home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces
export CUDA_INJECTION64_PATH=$PWD/util/tracer_nvbit/tracer_tool/tracer_tool.so
export LD_PRELOAD=$PWD/util/tracer_nvbit/tracer_tool/tracer_tool.so

mkdir -p "$TRACES_FOLDER"

./gpu-app-collection/src/cuda/GPU_Microbenchmark/bin/l1_bw_32f
```

执行成功后，目标目录下会同时出现：

- `dynamic_trace.pb`
- `stats.csv`
- `extra_info/enhanced_execution_info.json`

在这台机器上的已验证路径是：

- `/home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/dynamic_trace.pb`
- `/home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/stats.csv`
- `/home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/extra_info/enhanced_execution_info.json`

查看 trace 统计可以执行：

```bash
tail -n 20 /home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/stats.csv
```

如果你想查看 protobuf trace 的结构化内容，可以执行：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
./util/tracer_nvbit/tracer_tool/trace_printer \
  /home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces
```

## 7. 用生成的 trace 跑模拟器

这里使用本地新增的 `SM120_RTX5090` 配置。它是从仓库现有的 `SM120_RTX5070_TI` 复制出来的 Blackwell 同代起点，已经比 `SM86_RTX3080` 更适合拿来承接 5090 生成的 trace，但它当前仍然只是一个“未校准的 5090 草案配置”。

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled

export IS_SERT=0
source ./gpu-simulator/setup_environment_no_git.sh

./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM120_RTX5090/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM120_RTX5090/trace.config \
  -trace /home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/dynamic_trace.pb
```

推荐的日志保存形式：

```bash
./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM120_RTX5090/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM120_RTX5090/trace.config \
  -trace /home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-fixed/traces/dynamic_trace.pb \
  | tee /tmp/l1_bw_32f_5090.sim.log
```

然后从日志里抽关键指标：

```bash
rg "gpu_tot_sim_cycle|gpu_tot_ipc|gpu_tot_sim_insn|gpu_tot_occupancy|gpgpu_simulation_time" /tmp/l1_bw_32f_5090.sim.log
```

## 8. 这台服务器上已验证的“可收敛”模拟方式

完整的 `l1_bw_32f` trace 在当前 `SM120_RTX5090` 草案配置下可以被模拟器接受，但作为“等待完整跑完”的验证目标不适合这台共享服务器。更实际的方式是做一个有上限的模拟运行。

下面这条命令已经验证通过：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled

export IS_SERT=0
source ./gpu-simulator/setup_environment_no_git.sh

OMP_NUM_THREADS=1 OMP_PROC_BIND=spread \
./gpu-simulator/bin/release/accel-sim.out \
  -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM120_RTX5090/gpgpusim.config \
  -config ./gpu-simulator/configs/tested-cfgs/SM120_RTX5090/trace.config \
  -gpgpu_max_cycle 10000 \
  -trace /home/dyf/modern-gpu-simulator-micro-2025/hw_run/traces/device-0/12.8/manual-l1_bw_32f-5090-short/traces/dynamic_trace.pb \
  > /tmp/l1_bw_32f_5090_sm120.log 2>&1
```

这次已验证的日志文件是：

```bash
/tmp/l1_bw_32f_5090_sm120.log
```

这次运行里已经实际拿到的关键输出有：

- `launching kernel name: _Z5l1_bwPjS_PfS0____0 uid: 1`
- `gpgpu_simulation_time = 0 days, 0 hrs, 0 min, 1 sec (1 sec)`
- `gpgpu_simulation_rate = 72960 (inst/sec)`
- `gpgpu_simulation_rate = 10000 (cycle/sec)`
- `GPGPU-Sim: ** break due to reaching the maximum cycles (or instructions) **`

这一条受控运行，是当前这台机器上已经验证完成的端到端闭环：编译、5090 生成 trace、模拟器消费 trace 并给出结果。

## 9. 重要说明

- 仓库目前没有官方校准过的 `RTX 5090` 模拟配置，所以本文档中的模拟命令使用的是本地新增的 `SM120_RTX5090`。它是从 `SM120_RTX5070_TI` 复制出来的同代起点，只适合做链路验证和后续调参起点，不应直接视为 5090 的高精度建模结果。
- 在这台机器上，tracer 侧已经对 `GPU_Microbenchmark/l1_bw_32f` 验证通过。
- `rodinia_2.0-ft` 可以编译，但并不是每个 benchmark 在这台服务器上都同样稳定，尤其是 `backprop-rodinia-2.0-ft` 不适合当第一条验证路径。
- 如果新开的 shell 找不到 `nvcc`、`protoc` 或用户目录里的 protobuf 库，请重新执行 `source ~/.bashrc`。
