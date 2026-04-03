# `nv_trace` / NVBit Tracer 说明

## 它是什么

这个项目里常说的 `nv_trace`，实际对应的是基于 NVBit 的取迹器 `simulator-remodeled/util/tracer_nvbit/`。

它不是最终负责仿真的模块，而是 trace-driven 流程的前端：把 CUDA 程序在真实 NVIDIA GPU 上执行时产生的 SASS 动态行为和静态指令信息采集下来，转换成后端模拟器可以重放的输入。

从仓库结构上看，整体链路是：

```text
CUDA 应用
  -> NVBit tracer (`util/tracer_nvbit`)
  -> trace 文件 (`dynamic_trace.pb`, `threadblocks/`, `enhanced_execution_info.json`)
  -> trace parser (`gpu-simulator/trace-parser`)
  -> simulator (`gpu-simulator/bin/release/accel-sim.out`)
```

## 它的作用

`nv_trace` 的核心作用有四个：

1. 在真实 GPU 上拦截和记录 kernel 执行过程，而不是仅依赖 PTX 或静态反汇编。
2. 记录 trace-driven 仿真需要的动态信息，例如 kernel 启动顺序、stream 内事件顺序、warp 指令流、active mask、访存地址等。
3. 抽取静态指令元数据，例如寄存器使用、谓词信息、调用目标以及这个仓库重点使用的 control bits。
4. 把这些信息组织成当前仓库自定义的 trace 格式，交给后续 parser 和 simulator 消费。

因此，没有这一步，后端模拟器就没有“真实执行轨迹”可用，也就无法进入这个项目的 SASS trace-driven 模式。

## 为什么这个项目特别依赖它

根 README 明确列出了两个和 tracer 直接相关的增强点：

- Tracer that parses control bits
- Simulator that interprets control bits

这说明本仓库的 tracer 不只是做“指令流采样”，还会从指令编码里解析 control bits，并把它们带到仿真阶段。这样后端才能更接近现代 GPU 的调度与依赖行为，而不只是做传统的 scoreboard 近似。

## 它怎么被运行

通常通过下面的脚本入口运行：

```bash
cd simulator-remodeled
./util/tracer_nvbit/run_hw_trace.py -B GPU_Microbenchmark -D 0
```

`run_hw_trace.py` 会：

1. 为待运行应用创建 trace 输出目录。
2. 设置 `CUDA_VISIBLE_DEVICES`、`TRACES_FOLDER` 等环境变量。
3. 通过 `CUDA_INJECTION64_PATH` 和 `LD_PRELOAD` 注入 `tracer_tool.so`。
4. 在真实 GPU 上执行目标 CUDA 程序。
5. 让 NVBit tracer 在程序退出时把 trace 写盘。

也就是说，`nv_trace` 的本质是“边运行应用，边在硬件侧收集执行轨迹”。

## 它生成什么

当前仓库的主输出是三部分。

### 1. `dynamic_trace.pb`

这是全局动态 trace 的主入口文件，使用 Protocol Buffers 存储。

它主要包含：

- GPU device 信息
- stream 信息
- CUDA 事件顺序
- kernel 基本信息
- trace 版本信息

这个文件相当于“目录索引 + 高层执行时间线”。后端 parser 会先从这里读出有哪些 stream、kernel、memcpy，以及它们的顺序关系。

### 2. `threadblocks/`

这个目录保存按 `device / stream / kernel / threadblock` 组织的细粒度动态 trace。

这里面记录的是更接近执行本体的数据，例如：

- 某个 thread block 中每个 warp 的指令序列
- 每条指令对应的 PC
- active mask
- memory instruction 的地址信息
- 可选的寄存器值

如果说 `dynamic_trace.pb` 负责告诉模拟器“有什么要执行”，那 `threadblocks/` 负责告诉模拟器“具体执行了什么”。

### 3. `extra_info/enhanced_execution_info.json`

这是增强版静态元数据。

它包含的内容包括：

- traced kernel 集合
- 指令级静态解析结果
- 寄存器使用信息
- 谓词和操作数信息
- 调用目标
- control bits

这个 JSON 的价值在于：后端不需要重新从 cubin/SASS 做一遍完整解析，而是可以直接使用 tracer 已经提取好的静态信息。

## control bits 在这里为什么重要

这个仓库和原始 Accel-Sim 的一个关键区别，就是它把 control bits 当成一等公民。

在 `util/traces_enhanced/src/control_bits.cc` 中，可以看到 tracer 从编码后的指令字段中解析出：

- stall count
- yield 位
- new write barrier id
- new read barrier id
- wait barrier bits

这些信息随后会挂到 `traced_instruction` 上，并被序列化到增强 trace 信息中。

这意味着 simulator 不是只看到“这是一条 load / store / ALU 指令”，还会看到现代 NVIDIA SASS 指令里和调度、等待、barrier 相关的编码信息。这正是该仓库能够实现：

- control-bits driven dependence handling
- 更细的子核心流水模型
- 比传统 scoreboard 更贴近硬件的执行约束

的基础之一。

## 模拟器如何消费这些 trace

后端的入口在 `gpu-simulator/trace-parser/trace_parser.cc`。

它的工作方式大致是：

1. 打开 `dynamic_trace.pb`，解析 stream 和 kernel 的顺序。
2. 定位 `extra_info/enhanced_execution_info.json`。
3. 定位 `threadblocks/` 目录。
4. 逐个 thread block 读取对应的 protobuf 文件。
5. 将动态执行信息和静态元数据拼接成模拟器内部的指令与 kernel 结构。

因此，这个项目不是直接拿一个单一 trace 文件做仿真，而是把：

- 全局动态信息
- 分 threadblock 的动态信息
- 增强静态信息

三者组合起来，一起驱动仿真。

## 和传统“只记录 trace”有什么不同

这个仓库里的 `nv_trace` 不只是一个“导出指令列表”的工具，而是一个更完整的前端。

它额外做了这些事情：

- 解析 control bits
- 保存静态指令元数据
- 建立 kernel 与 unique function id 的对应关系
- 纠正按 kernel/function 区分的地址空间
- 为后端 modern GPU core model 提供更完整的输入

所以它更适合这个项目的研究目标：不是只回放一个粗略的 SASS 指令流，而是尽可能还原现代 GPU SM 的真实执行约束。

## 一句话总结

`nv_trace` 在这个项目里的作用可以概括为：

> 它是 trace-driven 仿真的硬件取迹前端，负责把真实 GPU 上执行出来的 SASS 动态行为和静态控制信息转换成模拟器可重放、可解释的输入。

## 相关文件

- `simulator-remodeled/util/tracer_nvbit/run_hw_trace.py`
- `simulator-remodeled/util/tracer_nvbit/tracer_tool/tracer_tool.cu`
- `simulator-remodeled/util/traces_enhanced/src/control_bits.cc`
- `simulator-remodeled/util/traces_enhanced/src/traced_instruction.cc`
- `simulator-remodeled/gpu-simulator/trace-parser/trace_parser.cc`
