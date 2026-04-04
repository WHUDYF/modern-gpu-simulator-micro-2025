# GPU Trace 压缩设计

将 DiffTest 的 Squash 与 Delta 压缩技术应用于 GPU 模拟器 trace 文件，在保持仿真精度不变的前提下大幅缩减 trace 体积。

## 问题

NVBit 生成的 GPU trace 在复杂 workload 下体积过大。一个 GEMM 级别的 AI workload 产生约 11GB 的 trace 数据。当前基于 protobuf 的格式仅压缩了内存地址字段（base_stride / base_delta），其余逐指令字段均未压缩。

### 当前 Trace 结构

```
threadblock (.pb 文件)
  └── warps (map<int32, warp>)
       └── instructions (repeated instruction)
            ├── pc: uint32
            ├── active_mask: uint32
            ├── predicate_mask: uint32
            ├── function_unique_id: int32
            └── addresses: repeated address
                 ├── compression_format (list_all / base_stride / base_delta)
                 ├── base_address, stride, addrs
                 ├── data_width
                 └── udesc_value
```

### 冗余分析（GEMM 类 workload）

| 字段 | 大小 | 冗余情况 |
|------|------|---------|
| `function_unique_id` | 4B/指令 | 每个 kernel 内恒定，但在每条指令中重复存储 |
| `active_mask` | 4B/指令 | 约 90% 为 0xFFFFFFFF（全线程活跃，无分支发散） |
| `predicate_mask` | 4B/指令 | 约 90% 等于 active_mask |
| `pc` | 4B/指令 | 顺序执行时 next_pc = current_pc + 4，约占 80% |
| `addresses` | 变长 | 已通过 base_stride / base_delta 压缩 |

非地址字段开销：每条指令 16 字节，其中约 90% 在 GEMM workload 中是冗余的。

### 实测压缩潜力（rodinia2Ampere）

- `.pb` 动态 trace：2.0 MB（5,525 个文件）
- `enhanced_execution_info.json`：14.7 MB
- 文件系统目录开销：约 70 MB（数千个小文件的目录结构）
- gzip 对原始 `.pb` 可达 4.5:1 压缩比；xz 可达 6:1
- JSON 经 gzip 压缩至原大小的 3.6%（高度冗余的文本）

## 设计

四层递进式压缩，每层基于前一层构建。所有压缩层对模拟器透明：无论 trace 版本如何，parser 输出的 `inst_trace_t` 结构体与原始版本完全一致。

### 不变量

```
任意版本 .pb --> 版本分发 --> 解码器 --> inst_trace_t（与 v4 bit-exact 一致）
                                              |
                                    模拟器执行逻辑（不改动）
```

验证标准：同一 benchmark，压缩前后的 cycle count 和 APE 必须完全一致。

### Layer 1：Flags + Delta + FuncId 提升（v4 -> v5）

**技术来源**：DiffTest Delta（PC 编码）、DiffTest Squash（mask 折叠）、字段提升。

#### Proto 变更

新增 `compressed_instruction.proto`：

```protobuf
syntax = "proto3";
package dynamic_trace;
import "address.proto";

message compressed_instruction {
  uint32 pc = 1;           // 绝对值或 delta，取决于 flags.bit2
  uint32 flags = 2;        // 位域（见下表）
  uint32 active_mask = 3;  // 仅当 flags.bit0 = 0 时存在
  uint32 predicate_mask = 4; // 仅当 flags.bit1 = 0 时存在
  repeated address addresses = 5;
}
```

新增 `compressed_warp.proto`：

```protobuf
syntax = "proto3";
package dynamic_trace;
import "compressed_instruction.proto";

message compressed_warp {
  int32 id = 1;
  repeated compressed_instruction instructions = 2;
}
```

新增 `compressed_threadblock.proto`：

```protobuf
syntax = "proto3";
package dynamic_trace;
import "compressed_warp.proto";
import "dim3d.proto";

message compressed_threadblock {
  dim3d block_id = 1;
  map<int32, compressed_warp> warps = 2;
  int32 function_unique_id = 3;  // 从逐指令提升至 threadblock 级别
}
```

#### Flags 位域布局

| Bit | 名称 | =1 | =0 |
|-----|------|----|----|
| 0 | `FULL_ACTIVE` | active_mask == 0xFFFFFFFF，字段省略 | active_mask 字段存在 |
| 1 | `PRED_EQ_ACTIVE` | predicate_mask == active_mask，字段省略 | predicate_mask 字段存在 |
| 2 | `PC_DELTA` | pc 字段为与前一条指令的差值 | pc 字段为绝对地址 |

常见情况（顺序执行 + 全活跃）：`flags = 0b111 = 7`，pc_delta = 4。

每条指令存储：16B -> 2B（flags=7 为 1B varint + pc_delta=4 为 1B varint）。

#### 编码规则

- 每个 warp 的首条指令：`flags.bit2 = 0`（绝对 PC）。
- 后续指令：`flags.bit2 = 1`，pc = current_pc - previous_pc。
- `function_unique_id` 在 `compressed_threadblock` 中存储一次，从指令中移除。
- proto3 默认值省略特性：当 active_mask=0（proto3 默认值）时不会序列化。由于我们仅在 flags.bit0=1 时省略该字段，解码器知道应恢复为 0xFFFFFFFF。

#### 预估压缩效果

非地址字段 5-8x 压缩。

### Layer 2：Run-Length Squash（v5 -> v6）

**技术来源**：DiffTest Squash — 合并连续的同构事件。

#### Proto 变更

在 `compressed_instruction.proto` 中新增 `instruction_run`：

```protobuf
message instruction_run {
  uint32 pc_start = 1;    // 首条指令的绝对 PC
  uint32 pc_delta = 2;    // 每条指令的 PC 步长（通常为 4）
  uint32 flags = 3;       // 所有指令共享的 flags
  uint32 count = 4;       // 连续指令数量
}
```

更新 `compressed_warp`（v6）：

```protobuf
message compressed_warp_v6 {
  int32 id = 1;
  // 指令与 run 按执行顺序交错排列
  repeated compressed_instruction instructions = 2;
  repeated instruction_run runs = 3;
  // 执行顺序：每个条目是一个 tag。
  // Tag 格式：bit 31 = 0 表示索引到 instructions[]，bit 31 = 1 表示索引到 runs[]。
  // 低 31 位 = 对应数组内的索引。
  // 示例：[0x00000000, 0x80000000, 0x00000001] 表示 instructions[0], runs[0], instructions[1]。
  repeated uint32 sequence = 4;
}
```

#### 编码规则

- 仅**无内存访问**的指令（ALU、FMA、branch 等）可合并为 run。
- 连续指令必须共享相同的 `flags` 和恒定的 `pc_delta`。
- 最小 run 长度为 3（低于此值时单独存储指令更紧凑）。

#### 预估压缩效果

在 GEMM workload 上额外 2-3x（内循环中大量无地址的 FMA 指令）。

### Layer 3：跨 Warp PC 去重（v6 -> v7）

**技术来源**：DiffTest Squash 跨不同来源 — 合并多个 warp 的相同 PC 序列。

#### 观察

在 SIMT 执行模型中，同一 threadblock 内所有 warp 执行相同的程序。除分支发散外（在 GEMM 中很少出现），它们的 PC 序列完全一致。

#### Proto 变更

```protobuf
message compressed_threadblock_v7 {
  dim3d block_id = 1;
  int32 function_unique_id = 2;

  // 共享 PC 序列（所有 warp 共用，只存一份）
  repeated uint32 shared_pc_sequence = 3;

  // 每个 warp 只存地址和 mask 覆盖
  map<int32, warp_diff> warps = 4;
}

message warp_diff {
  int32 id = 1;
  repeated warp_instruction instructions = 2;
  // 该 warp 与共享序列不同之处的稀疏覆盖
  repeated pc_override pc_overrides = 3;
}

message warp_instruction {
  // 仅地址数据，无 PC/mask（从共享序列继承）
  repeated address addresses = 1;
  // mask 覆盖标志（仅当非全活跃时）
  uint32 flags = 2;
  uint32 active_mask = 3;
  uint32 predicate_mask = 4;
}

message pc_override {
  uint32 instruction_index = 1;  // 序列中的位置
  uint32 actual_pc = 2;          // 该 warp 的实际 PC（与共享不同时）
}
```

#### 编码规则

- 编码器：从 warp 0 提取 PC 序列作为共享参考。
- 对每个其他 warp：比较 PC 序列。若一致，仅存储地址。
- 若某个 warp 在位置 `i` 发散，记录一条 `pc_override`。
- 若某个 warp 的 PC 序列与共享序列超过 10% 不同，回退到 per-warp 完整编码（L2 格式）。

#### 预估压缩效果

额外 2-4x（31 个 warp 的 PC 字段被消除）。

### Layer 4：跨 Threadblock Delta（v7 -> v8）

**技术来源**：DiffTest Delta — 仅编码相邻 threadblock 之间的差异。

#### 观察

同一 kernel 的不同 threadblock 在不同数据 tile 上执行相同代码。它们的指令序列完全一致，仅内存地址不同，通常差异为一个与 tile 位置相关的常量偏移。

#### Proto 变更

```protobuf
message tb_delta {
  dim3d block_id = 1;
  // 参考：与哪个 threadblock 做 delta
  dim3d reference_block_id = 2;
  // 全局地址偏移（覆盖大多数情况：所有地址偏移相同的量）
  int64 global_address_offset = 3;
  // 当全局偏移不足时的逐指令稀疏覆盖
  repeated address_override address_overrides = 4;
}

message address_override {
  uint32 warp_id = 1;
  uint32 instruction_index = 2;
  uint32 address_index = 3;     // 指令内第几个地址
  int64 address_delta = 4;       // 与 (reference_address + global_offset) 的差值
}

message compressed_kernel_v8 {
  // 首个 threadblock：完整编码（v7 格式）
  compressed_threadblock_v7 base_threadblock = 1;
  // 后续 threadblock：与 base 的 delta
  repeated tb_delta delta_threadblocks = 2;
}
```

#### 编码规则

- 每个 kernel 的首个 threadblock：完整 v7 编码（作为"基准"）。
- 后续 threadblock：计算与基准的地址差异。
- 若所有地址差异为常量偏移，仅存储 `global_address_offset`（整个 threadblock 仅 8 字节）。
- 若部分地址偏离全局偏移模式，记录稀疏的 `address_overrides`。
- 若某个 threadblock 偏差过大（超过 20% 的地址需要覆盖），回退到完整 v7 编码。
- 文件组织变更：每个 kernel 一个 `.pb` 文件（而非每个 threadblock 一个），包含基准 + delta。这同时消除了数千个小文件带来的文件系统开销。

#### 预估压缩效果

规则型 workload（GEMM、卷积）额外 5-10x。非规则型 workload（BFS、图算法）收益较小。

### 压缩效果汇总

| 层 | 版本 | 技术 | GEMM 压缩比 | 累积效果（11GB） |
|----|------|------|------------|-----------------|
| 基线 | v4 | - | 1x | 11 GB |
| L1 | v5 | Flags + PC delta + FuncId 提升 | 5-8x | ~1.5-2 GB |
| L2 | v6 | Run-length squash | 2-3x | ~0.6-0.8 GB |
| L3 | v7 | 跨 Warp PC 去重 | 2-4x | ~200-400 MB |
| L4 | v8 | 跨 TB delta | 5-10x | ~30-80 MB |
| + gzip | - | 通用压缩 | 2-3x | ~10-30 MB |

保守总体估计：GEMM 类 AI workload **50-100x** 压缩。

## 需要修改的文件

### Proto 定义（新文件）

- `util/traces_enhanced/dynamic_trace/compressed_instruction.proto`
- `util/traces_enhanced/dynamic_trace/compressed_warp.proto`
- `util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`

### Tracer 端

- `util/tracer_nvbit/tracer_tool/tracer_tool.cu` — 各层编码逻辑

### Parser 端

- `gpu-simulator/trace-parser/trace_parser.h` — 版本分发、新结构体
- `gpu-simulator/trace-parser/trace_parser.cc` — 各层解码逻辑

### 离线转换工具（新增）

- `util/trace-compress/trace_compress.cc` — 独立的 v4->v5->...->v8 转换器

### 现有文件（仅更新版本号字段）

- `util/traces_enhanced/dynamic_trace/trace.proto` — 版本号文档

## 向后兼容

- 原有的 `threadblock`、`warp`、`instruction` message 不做修改。
- Parser 按版本号分发：v4 走原有路径，v5+ 走压缩解码器。
- 离线转换器可将已有 v4 trace 转换为任意目标版本，无需重新运行 NVBit tracer。

## 验证策略

| 层 | 方法 |
|----|------|
| L1 | 将 rodinia2 v4 转换为 v5，解码后逐字段比对 `inst_trace_t`，确认与 v4 解码 bit-exact 一致 |
| L2 | 同上 + 验证 run-length 命中率非零；确认无地址的指令才被合并 |
| L3 | 同上 + 报告跨 warp PC 一致性比例；验证 `pc_overrides` 正确处理发散 |
| L4 | 同上 + 验证 delta 恢复结果；报告 global_address_offset 命中率 |
| 端到端 | 每层完成后跑 rodinia2 全套 benchmark，确认 APE 和 cycle count 与 v4 一致 |
| 规模测试 | 生成 CUTLASS/DeepBench trace，验证压缩比达到 50x+ 目标 |

## 风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 跨 Warp PC 去重假设在分支密集代码中不成立 | L3 压缩率下降 | `pc_overrides` 处理发散；超过 10% 发散阈值时回退到 per-warp 编码 |
| 跨 TB Delta 对非规则 workload（BFS、图算法）无效 | L4 压缩率下降 | 成本收益判断：超过 20% 覆盖阈值时回退到完整 TB 编码 |
| NVBit tracer 修改需要 GPU 环境 | 开发调试不便 | 先构建离线转换工具（纯 C++），验证所有层后再移植到 tracer |
| Proto message 大小限制（默认 64MB） | 大 kernel 可能触发限制 | 使用 `CodedInputStream::SetTotalBytesLimit()` 或按 kernel 拆分 |

## 实施顺序

1. **L1**（flags + PC delta + func_id 提升）：改动最小，立即获得 5-8x 收益
2. **L2**（run-length squash）：基于 L1 proto 扩展，中等改动
3. **L3**（跨 warp PC 去重）：重构 threadblock message，较大改动
4. **L4**（跨 TB delta）：重构文件组织（per-kernel 替代 per-TB），最大改动

每层可独立交付和验证。
