# GPU Trace 压缩 — 剩余任务（需 CUDA 环境）

> 本文档包含需要在有 CUDA 环境的机器（如 5090）上执行的任务。
> 编码/解码核心逻辑已在 `util/trace-compress/` 中实现并测试（9/9 roundtrip 测试通过）。
> 以下任务负责将解码逻辑集成到模拟器 parser 中，并做端到端验证。

**前置条件：**
- 分支：`feat/trace-compression`
- CUDA toolkit 已安装（需要 `vector_types.h`）
- 项目可正常编译：`source setup_environment.sh && make -j -C gpu-simulator`
- 离线转换工具可用：`make -C util/trace-compress`

**已完成的工作：**
- Proto 定义（compressed_instruction.proto, compressed_threadblock.proto）：L1-L4 所有 message 已定义
- 编码器/解码器（trace_compress.h/cc）：encode_v4_to_v5, decode_v5_to_v4, encode_v5_to_v6, decode_v6_to_v5, encode_v5_to_v7, decode_v7_to_v5, encode_kernel_to_v8, decode_v8_to_v7s 全部实现
- 离线转换工具（main.cc）：支持 --input/--output/--from-version/--to-version/--func-id
- 测试（test_roundtrip.cc）：9 个 roundtrip 测试全部通过

---

## Task 3：L1 Parser 解码 — 模拟器集成

**目标：** 让模拟器 parser 能读取 v5 格式 .pb 文件。

### 需要修改的文件

- `gpu-simulator/trace-parser/trace_parser.h`
- `gpu-simulator/trace-parser/trace_parser.cc`

### 实现步骤

- [ ] **Step 1：添加 v5 相关头文件和常量**

在 `trace_parser.h` 中添加：

```cpp
#include "../../util/traces_enhanced/pb_trace/include/compressed_threadblock.pb.h"
#include "../../util/traces_enhanced/pb_trace/include/compressed_instruction.pb.h"

// Compressed trace flags (v5+)
constexpr uint32_t COMPRESS_FLAG_FULL_ACTIVE    = 1 << 0;
constexpr uint32_t COMPRESS_FLAG_PRED_EQ_ACTIVE = 1 << 1;
constexpr uint32_t COMPRESS_FLAG_PC_DELTA       = 1 << 2;
```

- [ ] **Step 2：在 `inst_trace_t` 中添加压缩格式解析方法**

```cpp
bool parse_from_compressed_pb(dynamic_trace::compressed_instruction cinst,
                              uint32_t flags, uint32_t prev_pc,
                              int function_unique_id,
                              unsigned tracer_version, gpgpu_sim *gpu,
                              std::string kernel_name,
                              traced_execution &static_trace_info);
```

实现逻辑：
- 若 `flags & COMPRESS_FLAG_PC_DELTA`：`m_pc = prev_pc + cinst.pc()`，否则 `m_pc = cinst.pc()`
- 若 `flags & COMPRESS_FLAG_FULL_ACTIVE`：`active = 0xFFFFFFFF`，否则 `active = cinst.active_mask()`
- 若 `flags & COMPRESS_FLAG_PRED_EQ_ACTIVE`：`predicate = active`，否则 `predicate = cinst.predicate_mask()`
- `mask = active & predicate`
- `m_unique_function_id = function_unique_id`（从 threadblock 级获取）
- opcode 通过 `static_trace_info` 查询（与 `parse_from_pb` 方式相同）
- 地址解析复用已有的 `parse_memref` 方法

- [ ] **Step 3：添加 `get_next_threadblock_traces_v5` 方法**

在 `trace_parser` 类中添加新方法，结构与现有 `get_next_threadblock_traces` 一致，区别在于：
- 读取 `compressed_threadblock` 而非 `threadblock`
- 从 `ctb.function_unique_id()` 获取 func_id
- 遍历 `compressed_warp` 中的 `compressed_instruction`
- 调用 `parse_from_compressed_pb` 而非 `parse_from_pb`
- 跟踪 `prev_pc` 用于 delta 解码

- [ ] **Step 4：添加版本分发**

在 `trace_driven.cc` 中调用 `get_next_threadblock_traces` 的地方（`trace_kernel_info_t::get_next_threadblock_traces` 方法内），添加版本判断：

```cpp
if (m_kernel_trace_info->trace_verion >= 5) {
  m_parser->get_next_threadblock_traces_v5(
      threadblock_traces, threadblock_traced_pcs,
      m_kernel_trace_info->gpu_device_id, m_kernel_trace_info->cuda_stream_id,
      m_kernel_trace_info->kernel_id, m_kernel_trace_info->trace_verion,
      m_kernel_trace_info->next_tb_to_parse_x, m_kernel_trace_info->next_tb_to_parse_y,
      m_kernel_trace_info->next_tb_to_parse_z, gpu, get_name(), static_trace_info,
      m_kernel_trace_info->func_unique_id);
} else {
  m_parser->get_next_threadblock_traces(/* 原有参数不变 */);
}
```

- [ ] **Step 5：编译验证**

```bash
source setup_environment.sh
make -j -C gpu-simulator
```

- [ ] **Step 6：提交**

```bash
git add gpu-simulator/trace-parser/trace_parser.h \
        gpu-simulator/trace-parser/trace_parser.cc \
        gpu-simulator/trace-driven/trace_driven.cc
git commit -m "Add v5 compressed trace decoder to simulator parser"
```

---

## Task 4：L1 端到端验证

**目标：** 用 rodinia2 trace 验证 v4→v5→模拟器 路径的正确性。

- [ ] **Step 1：批量转换 rodinia2 trace**

```bash
cd simulator-remodeled/util/trace-compress

# 解压 rodinia2
WORK_DIR=$(mktemp -d)
tar xzf ../../exampleTraces/rodinia2Ampere.tar.gz -C $WORK_DIR

# 转换所有 .pb 文件为 v5
find $WORK_DIR -name "*.pb" | while read F; do
  ./trace-compress --input "$F" --output "${F}.v5" --from-version 4 --to-version 5 --func-id 1
  mv "${F}.v5" "$F"  # 覆盖原文件
done
```

- [ ] **Step 2：用转换后的 v5 trace 跑模拟**

修改 trace 的版本号（在 `dynamic_trace.pb` 文件的 `accelsim_version` 字段中设为 5），然后跑模拟器：

```bash
# 在模拟器配置中指向转换后的 trace 目录
# 运行某个 rodinia2 benchmark，对比 v4 和 v5 的 cycle count
```

- [ ] **Step 3：对比 APE 结果**

v5 trace 的 APE 和 cycle count 必须与 v4 完全一致。

- [ ] **Step 4：提交验证脚本**

```bash
git commit -m "Validate L1 compression with rodinia2 end-to-end"
```

---

## Task 7：L2 Parser 解码 + 验证

**目标：** 模拟器 parser 支持 v6（run-length squash）格式。

### 实现方式

最简方案：在 parser 中将 v6 展开为 v5，然后复用 v5 解码路径。

- [ ] **Step 1：添加 `get_next_threadblock_traces_v6` 方法**

读取 `compressed_threadblock_v6`，按 `sequence` 数组展开 runs 为独立指令，得到等效的 `compressed_threadblock`（v5 格式），然后调用 v5 的解码逻辑。

展开逻辑（与 `decode_v6_to_v5` 相同）：
- 遍历 `sequence` 数组
- `tag & SEQ_TAG_RUN_BIT == 0`：取 `instructions[index]`
- `tag & SEQ_TAG_RUN_BIT == 1`：展开 `runs[index]`，生成 `count` 条独立 `compressed_instruction`

`SEQ_TAG_RUN_BIT = 1u << 31`

- [ ] **Step 2：更新版本分发**

```cpp
if (trace_version >= 6) {
  get_next_threadblock_traces_v6(/* ... */);
} else if (trace_version >= 5) {
  get_next_threadblock_traces_v5(/* ... */);
} else {
  get_next_threadblock_traces(/* ... */);
}
```

- [ ] **Step 3：编译 + 端到端验证**

与 Task 4 相同流程，但使用 v6 trace。

- [ ] **Step 4：提交**

```bash
git commit -m "Add v6 run-length squash decoder to simulator parser"
```

---

## Task 9：L3 Parser 解码 + 验证

**目标：** 模拟器 parser 支持 v7（跨 warp PC 去重）格式。

### 实现方式

同理，将 v7 展开为 v5 后复用 v5 解码路径。

- [ ] **Step 1：添加 `get_next_threadblock_traces_v7` 方法**

读取 `compressed_threadblock_v7`，展开逻辑（与 `decode_v7_to_v5` 相同）：
- 从 `shared_pc_sequence` 获取共享 PC 序列
- 为每个 warp 构建 override map（`pc_overrides`）
- 逐指令：PC 来自 shared sequence 或 override，mask 和 addresses 来自 `warp_instruction`
- 重新编码为 v5 delta 格式
- 调用 v5 解码路径

- [ ] **Step 2：更新版本分发，添加 v7 分支**

- [ ] **Step 3：编译 + 端到端验证**

- [ ] **Step 4：提交**

```bash
git commit -m "Add v7 cross-warp PC dedup decoder to simulator parser"
```

---

## Task 11：L4 Parser 解码 + 离线工具完善

**目标：** 模拟器 parser 支持 v8（per-kernel 文件），离线工具支持全链路转换。

### v8 的关键变化

v8 改变了文件组织：**从 per-threadblock 变为 per-kernel**。一个 `.pb` 文件包含整个 kernel 的所有 threadblock（base + deltas）。

### 实现步骤

- [ ] **Step 1：Parser v8 解码**

需要在 `trace_parser` 中添加 per-kernel 文件缓存机制：

```cpp
// 在 trace_parser 类中添加：
std::map<std::string, dynamic_trace::compressed_kernel_v8> m_v8_kernel_cache;
```

`get_next_threadblock_traces_v8` 方法逻辑：
1. 构建 kernel 文件路径（一个 kernel 一个文件，而非一个 threadblock 一个文件）
2. 若 kernel 未缓存，读取并缓存 `compressed_kernel_v8`
3. 根据请求的 block_id，从 base 或 delta 中找到对应 threadblock
4. 若是 delta：克隆 base，应用 `global_address_offset`，应用 `address_overrides`
5. 将恢复后的 v7 threadblock 展开为 v5，调用 v5 解码路径

- [ ] **Step 2：更新离线转换工具**

在 `main.cc` 中添加 `--batch-dir` 参数：

```bash
# 将一个 kernel 目录下所有 threadblock .pb 文件合并为单个 v8 文件
trace-compress --batch-dir /path/to/kernel_N/ --output kernel_N.v8.pb \
  --from-version 4 --to-version 8 --func-id 42
```

实现：
1. 读取目录下所有 .pb 文件为 v4 threadblock
2. 链式调用 v4→v5→v7（跳过 v6，直接用 v5→v7）
3. 收集所有 v7 threadblock，调用 `encode_kernel_to_v8`
4. 写出 `compressed_kernel_v8` 到输出文件

- [ ] **Step 3：更新版本分发**

```cpp
if (trace_version >= 8) {
  get_next_threadblock_traces_v8(/* ... */);
} else if (trace_version >= 7) {
  ...
}
```

- [ ] **Step 4：编译 + 端到端验证**

- [ ] **Step 5：提交**

```bash
git commit -m "Add v8 cross-TB delta decoder to parser, full-chain converter"
```

---

## Task 12：最终验证 + 压缩比报告

**目标：** 对全套 trace 做完整验证，输出各层压缩比报告。

- [ ] **Step 1：编写基准测试脚本**

创建 `util/trace-compress/benchmark_compression.sh`：

```bash
#!/bin/bash
# Measure compression ratios across all layers for rodinia2
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

tar xzf "$SCRIPT_DIR/../../exampleTraces/rodinia2Ampere.tar.gz" -C "$WORK_DIR"

echo "| Layer | Total Size | Ratio vs v4 |"
echo "|-------|-----------|-------------|"

V4_SIZE=$(find "$WORK_DIR/rodinia2" -name "*.pb" -exec du -cb {} + | tail -1 | awk '{print $1}')
echo "| v4 (baseline) | ${V4_SIZE}B | 1.0x |"

# v5 conversion
find "$WORK_DIR/rodinia2" -name "*.pb" | while read F; do
  "$SCRIPT_DIR/trace-compress" --input "$F" --output "${F}.v5" \
    --from-version 4 --to-version 5 --func-id 1 2>/dev/null
done
V5_SIZE=$(find "$WORK_DIR/rodinia2" -name "*.v5" -exec du -cb {} + | tail -1 | awk '{print $1}')
echo "| v5 (L1) | ${V5_SIZE}B | $(echo "scale=1; $V4_SIZE / $V5_SIZE" | bc)x |"

# Additional layers can be added similarly
```

- [ ] **Step 2：跑 AI 模型 trace 的压缩比测试**

使用从 5090 传回的 ResNet-50 / BERT / GPT-2 trace，测试各层压缩比。

- [ ] **Step 3：跑模拟器端到端验证**

用压缩后的 trace 跑模拟器，确认 APE 与 v4 一致。

- [ ] **Step 4：提交**

```bash
git commit -m "Add compression benchmark script with per-layer ratio report"
```

---

## 快速参考

### 关键函数签名（已实现，可在 parser 中复用）

```
trace_compress.h 中的解码函数：
- decode_v5_to_v4(compressed_threadblock → threadblock)
- decode_v6_to_v5(compressed_threadblock_v6 → compressed_threadblock)
- decode_v7_to_v5(compressed_threadblock_v7 → compressed_threadblock)
- decode_v8_to_v7s(compressed_kernel_v8 → vector<compressed_threadblock_v7>)
```

### Flags 常量

```
FLAG_FULL_ACTIVE    = 1 << 0  // active_mask == 0xFFFFFFFF
FLAG_PRED_EQ_ACTIVE = 1 << 1  // predicate_mask == active_mask
FLAG_PC_DELTA       = 1 << 2  // pc is delta from previous
SEQ_TAG_RUN_BIT     = 1 << 31 // sequence tag for run entries
```

### 版本号映射

```
v4 = 原始格式（per-threadblock .pb，threadblock message）
v5 = L1 flags + delta + funcid lift（per-threadblock .pb，compressed_threadblock message）
v6 = L2 run-length squash（per-threadblock .pb，compressed_threadblock_v6 message）
v7 = L3 cross-warp PC dedup（per-threadblock .pb，compressed_threadblock_v7 message）
v8 = L4 cross-TB delta（per-kernel .pb，compressed_kernel_v8 message）
```
