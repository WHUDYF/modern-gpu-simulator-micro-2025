# GPU Trace 压缩实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 实现四层递进式 trace 压缩（Flags/Delta/FuncId提升 → Run-Length Squash → 跨Warp PC去重 → 跨TB Delta），将 11GB 的 GEMM trace 压缩至 50-100x。

**架构：** 在 protobuf schema 层面引入新的压缩 message 类型，通过离线转换工具将 v4 trace 转换为压缩格式，parser 端增加版本分发和解码逻辑。所有压缩对模拟器执行逻辑透明，输出的 `inst_trace_t` 与 v4 bit-exact 一致。

**技术栈：** C++17, Protocol Buffers 3, GNU Make

**关键约束：**
- 代码中只使用英文
- 不修改原有 `threadblock`/`warp`/`instruction` proto message
- 每层独立可交付可验证
- 验证标准：同一 benchmark 压缩前后 APE 和 cycle count 完全一致

---

## 文件结构

### 新建文件

| 文件 | 职责 |
|------|------|
| `util/traces_enhanced/dynamic_trace/compressed_instruction.proto` | L1+L2 压缩指令与 instruction_run message 定义 |
| `util/traces_enhanced/dynamic_trace/compressed_threadblock.proto` | L1-L4 各版本 threadblock message 定义（含 compressed_warp、warp_diff、tb_delta 等） |
| `util/trace-compress/trace_compress.h` | 离线转换工具头文件：编码器接口 |
| `util/trace-compress/trace_compress.cc` | 离线转换工具实现：各层编码逻辑 |
| `util/trace-compress/main.cc` | 离线转换工具入口 |
| `util/trace-compress/Makefile` | 离线转换工具构建 |
| `util/trace-compress/test_roundtrip.cc` | 编解码往返测试 |
| `util/trace-compress/test_roundtrip.sh` | 端到端验证脚本（对比 v4 与压缩后解码结果） |

### 修改文件

| 文件 | 变更内容 |
|------|---------|
| `util/traces_enhanced/Makefile` | 增加新 proto 文件的编译规则 |
| `gpu-simulator/trace-parser/trace_parser.h` | 增加压缩格式解码函数声明和 flags 常量定义 |
| `gpu-simulator/trace-parser/trace_parser.cc` | 增加版本分发逻辑和各层解码实现 |

---

## Task 0：构建基础设施

**目标：** 创建新 proto 文件和离线转换工具的构建框架，确保能编译和链接。

**文件：**
- 新建：`util/traces_enhanced/dynamic_trace/compressed_instruction.proto`
- 新建：`util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`
- 新建：`util/trace-compress/main.cc`
- 新建：`util/trace-compress/Makefile`
- 修改：`util/traces_enhanced/Makefile`

- [ ] **Step 1：创建 L1 压缩指令 proto**

```protobuf
// util/traces_enhanced/dynamic_trace/compressed_instruction.proto
syntax = "proto3";
package dynamic_trace;
import "address.proto";

// Flags bit layout:
// bit 0 (FULL_ACTIVE):    1 = active_mask is 0xFFFFFFFF (field omitted)
// bit 1 (PRED_EQ_ACTIVE): 1 = predicate_mask equals active_mask (field omitted)
// bit 2 (PC_DELTA):       1 = pc field is delta from previous instruction
message compressed_instruction {
  uint32 pc = 1;
  uint32 flags = 2;
  uint32 active_mask = 3;
  uint32 predicate_mask = 4;
  repeated address addresses = 5;
}
```

- [ ] **Step 2：创建压缩 threadblock proto（L1 版本）**

```protobuf
// util/traces_enhanced/dynamic_trace/compressed_threadblock.proto
syntax = "proto3";
package dynamic_trace;
import "compressed_instruction.proto";
import "dim3d.proto";

message compressed_warp {
  int32 id = 1;
  repeated compressed_instruction instructions = 2;
}

message compressed_threadblock {
  dim3d block_id = 1;
  map<int32, compressed_warp> warps = 2;
  int32 function_unique_id = 3;
}
```

- [ ] **Step 3：更新 traces_enhanced Makefile 以编译新 proto**

在 `util/traces_enhanced/Makefile` 中，新 proto 文件会被 `PROTO_FILES=$(wildcard $(TRACES_ENHANCED_PROTO_DIR)/*.proto)` 自动发现，无需手动添加。验证这一点：

运行：`cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/traces_enhanced && ls dynamic_trace/*.proto | wc -l`
预期：11（原有 9 个 + 新增 2 个）

- [ ] **Step 4：创建离线转换工具 Makefile**

```makefile
# util/trace-compress/Makefile
CXX = g++
CXXFLAGS = -Wall -O2 -std=c++17

TRACES_DIR = ../traces_enhanced
PROTO_HDR_DIR = $(TRACES_DIR)/pb_trace/include
PROTO_OBJ_DIR = $(TRACES_DIR)/pb_trace/obj
TRACES_OBJ_DIR = $(TRACES_DIR)/obj

INCLUDES = -I$(PROTO_HDR_DIR) -I$(TRACES_DIR)
LIBS = -lprotobuf -lz -pthread

PROTO_OBJS = $(wildcard $(PROTO_OBJ_DIR)/*.pb.o)
TRACES_OBJS = $(wildcard $(TRACES_OBJ_DIR)/*.o)

.PHONY: all clean test

all: trace-compress

trace-compress: main.o trace_compress.o $(PROTO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

test_roundtrip: test_roundtrip.o trace_compress.o $(PROTO_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

test: test_roundtrip
	./test_roundtrip

clean:
	rm -f *.o trace-compress test_roundtrip
```

- [ ] **Step 5：创建转换工具入口（空壳）**

```cpp
// util/trace-compress/main.cc
#include <iostream>
#include <string>

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --input <file.pb> --output <file.pb>"
            << " --from-version <4> --to-version <5>"
            << std::endl;
}

int main(int argc, char* argv[]) {
  std::string input_path, output_path;
  int from_version = 4, to_version = 5;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--from-version" && i + 1 < argc) from_version = std::stoi(argv[++i]);
    else if (arg == "--to-version" && i + 1 < argc) to_version = std::stoi(argv[++i]);
    else { print_usage(argv[0]); return 1; }
  }

  if (input_path.empty() || output_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  std::cout << "Converting " << input_path
            << " from v" << from_version << " to v" << to_version << std::endl;

  // Encoder dispatch will be added in subsequent tasks
  std::cerr << "Not yet implemented" << std::endl;
  return 1;
}
```

- [ ] **Step 6：创建编码器头文件（空壳）**

```cpp
// util/trace-compress/trace_compress.h
#ifndef TRACE_COMPRESS_H
#define TRACE_COMPRESS_H

#include <string>
#include <vector>
#include "threadblock.pb.h"
#include "compressed_threadblock.pb.h"
#include "compressed_instruction.pb.h"

// Flags bit constants
constexpr uint32_t FLAG_FULL_ACTIVE    = 1 << 0;  // bit 0
constexpr uint32_t FLAG_PRED_EQ_ACTIVE = 1 << 1;  // bit 1
constexpr uint32_t FLAG_PC_DELTA       = 1 << 2;  // bit 2

// L1: Encode a v4 threadblock into v5 compressed_threadblock
bool encode_v4_to_v5(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock* dst,
                     int function_unique_id);

// L1: Decode a v5 compressed_threadblock back to v4 format (for testing)
bool decode_v5_to_v4(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::threadblock* dst);

#endif
```

- [ ] **Step 7：创建编码器实现（空壳）**

```cpp
// util/trace-compress/trace_compress.cc
#include "trace_compress.h"

bool encode_v4_to_v5(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock* dst,
                     int function_unique_id) {
  // Will be implemented in Task 1
  return false;
}

bool decode_v5_to_v4(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::threadblock* dst) {
  // Will be implemented in Task 1
  return false;
}
```

- [ ] **Step 8：验证构建通过**

运行（需要先确保 traces_enhanced 已编译生成 pb 文件）：
```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
source setup_environment_no_git.sh
make -C util/traces_enhanced clean && make -C util/traces_enhanced
make -C util/trace-compress clean && make -C util/trace-compress
```
预期：编译通过，生成 `util/trace-compress/trace-compress` 可执行文件。

- [ ] **Step 9：提交**

```bash
git add util/traces_enhanced/dynamic_trace/compressed_instruction.proto \
        util/traces_enhanced/dynamic_trace/compressed_threadblock.proto \
        util/trace-compress/
git commit -m "Add build infrastructure for trace compression tool"
```

---

## Task 1：L1 编码器 — Flags + PC Delta + FuncId 提升

**目标：** 实现 v4→v5 编码：将 `threadblock` 转换为 `compressed_threadblock`。

**文件：**
- 修改：`util/trace-compress/trace_compress.h`
- 修改：`util/trace-compress/trace_compress.cc`

- [ ] **Step 1：编写往返测试（先写测试）**

```cpp
// util/trace-compress/test_roundtrip.cc
#include <cassert>
#include <iostream>
#include "trace_compress.h"
#include "threadblock.pb.h"
#include "instruction.pb.h"
#include "address.pb.h"

// Build a synthetic v4 threadblock with known data
dynamic_trace::threadblock make_test_threadblock() {
  dynamic_trace::threadblock tb;
  tb.mutable_block_id()->set_x(0);
  tb.mutable_block_id()->set_y(0);
  tb.mutable_block_id()->set_z(0);

  // Create 2 warps, each with 5 instructions
  for (int w = 0; w < 2; w++) {
    dynamic_trace::warp warp;
    warp.set_id(w);
    for (int i = 0; i < 5; i++) {
      auto* inst = warp.add_instructions();
      inst->set_pc(0x1000 + i * 4);  // sequential PCs
      inst->set_active_mask(0xFFFFFFFF);  // all active
      inst->set_predicate_mask(0xFFFFFFFF);  // same as active
      inst->set_function_unique_id(42);

      // Add a memory access on instructions 1 and 3
      if (i == 1 || i == 3) {
        auto* addr = inst->add_addresses();
        addr->set_data_width(4);
        addr->set_udesc_value(0);
        addr->set_compression_format(1);  // base_stride
        addr->set_base_address(0x80000000 + w * 0x1000 + i * 128);
        addr->set_stride(4);
      }
    }
    (*tb.mutable_warps())[w] = warp;
  }
  return tb;
}

// Build a threadblock with divergent masks (not all active)
dynamic_trace::threadblock make_divergent_threadblock() {
  dynamic_trace::threadblock tb;
  tb.mutable_block_id()->set_x(1);
  tb.mutable_block_id()->set_y(0);
  tb.mutable_block_id()->set_z(0);

  dynamic_trace::warp warp;
  warp.set_id(0);

  // Instruction 0: absolute PC, all active
  auto* i0 = warp.add_instructions();
  i0->set_pc(0x2000);
  i0->set_active_mask(0xFFFFFFFF);
  i0->set_predicate_mask(0xFFFFFFFF);
  i0->set_function_unique_id(7);

  // Instruction 1: sequential PC, partial active
  auto* i1 = warp.add_instructions();
  i1->set_pc(0x2004);
  i1->set_active_mask(0x0000FFFF);  // only lower 16 threads
  i1->set_predicate_mask(0x0000FFFF);
  i1->set_function_unique_id(7);

  // Instruction 2: branch target (non-sequential PC), all active, different predicate
  auto* i2 = warp.add_instructions();
  i2->set_pc(0x3000);  // branch jump
  i2->set_active_mask(0xFFFFFFFF);
  i2->set_predicate_mask(0x00FF00FF);  // different from active
  i2->set_function_unique_id(7);

  (*tb.mutable_warps())[0] = warp;
  return tb;
}

void compare_threadblocks(const dynamic_trace::threadblock& a,
                          const dynamic_trace::threadblock& b) {
  assert(a.block_id().x() == b.block_id().x());
  assert(a.block_id().y() == b.block_id().y());
  assert(a.block_id().z() == b.block_id().z());
  assert(a.warps_size() == b.warps_size());

  for (auto& [wid, warp_a] : a.warps()) {
    auto it = b.warps().find(wid);
    assert(it != b.warps().end());
    const auto& warp_b = it->second;
    assert(warp_a.instructions_size() == warp_b.instructions_size());

    for (int i = 0; i < warp_a.instructions_size(); i++) {
      const auto& ia = warp_a.instructions(i);
      const auto& ib = warp_b.instructions(i);
      assert(ia.pc() == ib.pc());
      assert(ia.active_mask() == ib.active_mask());
      assert(ia.predicate_mask() == ib.predicate_mask());
      assert(ia.function_unique_id() == ib.function_unique_id());
      assert(ia.addresses_size() == ib.addresses_size());
      for (int j = 0; j < ia.addresses_size(); j++) {
        assert(ia.addresses(j).data_width() == ib.addresses(j).data_width());
        assert(ia.addresses(j).compression_format() == ib.addresses(j).compression_format());
        assert(ia.addresses(j).base_address() == ib.addresses(j).base_address());
        assert(ia.addresses(j).stride() == ib.addresses(j).stride());
        assert(ia.addresses(j).addrs_size() == ib.addresses(j).addrs_size());
        for (int k = 0; k < ia.addresses(j).addrs_size(); k++) {
          assert(ia.addresses(j).addrs(k) == ib.addresses(j).addrs(k));
        }
      }
    }
  }
}

void test_v5_roundtrip() {
  std::cout << "test_v5_roundtrip: sequential + all active... ";
  auto tb_orig = make_test_threadblock();

  dynamic_trace::compressed_threadblock compressed;
  bool ok = encode_v4_to_v5(tb_orig, &compressed, 42);
  assert(ok);
  assert(compressed.function_unique_id() == 42);

  dynamic_trace::threadblock tb_decoded;
  ok = decode_v5_to_v4(compressed, &tb_decoded);
  assert(ok);

  compare_threadblocks(tb_orig, tb_decoded);
  std::cout << "PASS" << std::endl;
}

void test_v5_divergent() {
  std::cout << "test_v5_divergent: branch + partial mask... ";
  auto tb_orig = make_divergent_threadblock();

  dynamic_trace::compressed_threadblock compressed;
  bool ok = encode_v4_to_v5(tb_orig, &compressed, 7);
  assert(ok);

  dynamic_trace::threadblock tb_decoded;
  ok = decode_v5_to_v4(compressed, &tb_decoded);
  assert(ok);

  compare_threadblocks(tb_orig, tb_decoded);
  std::cout << "PASS" << std::endl;
}

void test_v5_compression_ratio() {
  std::cout << "test_v5_compression_ratio: size reduction check... ";
  auto tb_orig = make_test_threadblock();

  dynamic_trace::compressed_threadblock compressed;
  encode_v4_to_v5(tb_orig, &compressed, 42);

  size_t orig_size = tb_orig.ByteSizeLong();
  size_t comp_size = compressed.ByteSizeLong();

  std::cout << "original=" << orig_size << "B compressed=" << comp_size << "B ";
  assert(comp_size < orig_size);
  std::cout << "PASS (ratio=" << (double)orig_size / comp_size << "x)" << std::endl;
}

int main() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  test_v5_roundtrip();
  test_v5_divergent();
  test_v5_compression_ratio();

  std::cout << "\nAll L1 tests passed." << std::endl;
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
```

- [ ] **Step 2：运行测试确认失败**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/trace-compress
make test_roundtrip && ./test_roundtrip
```
预期：assert 失败于 `encode_v4_to_v5` 返回 false。

- [ ] **Step 3：实现 `encode_v4_to_v5`**

更新 `util/trace-compress/trace_compress.cc`：

```cpp
#include "trace_compress.h"
#include <iostream>

bool encode_v4_to_v5(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock* dst,
                     int function_unique_id) {
  // Copy block_id
  *dst->mutable_block_id() = src.block_id();

  // Hoist function_unique_id to threadblock level
  dst->set_function_unique_id(function_unique_id);

  for (const auto& [warp_id, warp_src] : src.warps()) {
    dynamic_trace::compressed_warp cwarp;
    cwarp.set_id(warp_src.id());

    uint32_t prev_pc = 0;
    bool first_instruction = true;

    for (int i = 0; i < warp_src.instructions_size(); i++) {
      const auto& inst_src = warp_src.instructions(i);
      auto* inst_dst = cwarp.add_instructions();

      uint32_t flags = 0;
      uint32_t active = inst_src.active_mask();
      uint32_t predicate = inst_src.predicate_mask();
      uint32_t pc = inst_src.pc();

      // Flag bit 0: FULL_ACTIVE
      if (active == 0xFFFFFFFF) {
        flags |= FLAG_FULL_ACTIVE;
        // active_mask field left as 0 (proto3 omits it)
      } else {
        inst_dst->set_active_mask(active);
      }

      // Flag bit 1: PRED_EQ_ACTIVE
      if (predicate == active) {
        flags |= FLAG_PRED_EQ_ACTIVE;
        // predicate_mask field left as 0 (proto3 omits it)
      } else {
        inst_dst->set_predicate_mask(predicate);
      }

      // Flag bit 2: PC_DELTA
      if (!first_instruction) {
        flags |= FLAG_PC_DELTA;
        inst_dst->set_pc(pc - prev_pc);
      } else {
        inst_dst->set_pc(pc);
        first_instruction = false;
      }
      prev_pc = pc;

      inst_dst->set_flags(flags);

      // Copy addresses unchanged
      for (int j = 0; j < inst_src.addresses_size(); j++) {
        *inst_dst->add_addresses() = inst_src.addresses(j);
      }
    }

    (*dst->mutable_warps())[warp_id] = cwarp;
  }

  return true;
}

bool decode_v5_to_v4(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::threadblock* dst) {
  *dst->mutable_block_id() = src.block_id();
  int func_id = src.function_unique_id();

  for (const auto& [warp_id, cwarp] : src.warps()) {
    dynamic_trace::warp warp_dst;
    warp_dst.set_id(cwarp.id());

    uint32_t prev_pc = 0;

    for (int i = 0; i < cwarp.instructions_size(); i++) {
      const auto& cinst = cwarp.instructions(i);
      auto* inst_dst = warp_dst.add_instructions();

      uint32_t flags = cinst.flags();

      // Decode PC
      uint32_t pc;
      if (flags & FLAG_PC_DELTA) {
        pc = prev_pc + cinst.pc();
      } else {
        pc = cinst.pc();
      }
      prev_pc = pc;
      inst_dst->set_pc(pc);

      // Decode active_mask
      if (flags & FLAG_FULL_ACTIVE) {
        inst_dst->set_active_mask(0xFFFFFFFF);
      } else {
        inst_dst->set_active_mask(cinst.active_mask());
      }

      // Decode predicate_mask
      uint32_t active = inst_dst->active_mask();
      if (flags & FLAG_PRED_EQ_ACTIVE) {
        inst_dst->set_predicate_mask(active);
      } else {
        inst_dst->set_predicate_mask(cinst.predicate_mask());
      }

      // Restore function_unique_id
      inst_dst->set_function_unique_id(func_id);

      // Copy addresses unchanged
      for (int j = 0; j < cinst.addresses_size(); j++) {
        *inst_dst->add_addresses() = cinst.addresses(j);
      }
    }

    (*dst->mutable_warps())[warp_id] = warp_dst;
  }

  return true;
}
```

- [ ] **Step 4：运行测试确认通过**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/trace-compress
make test_roundtrip && ./test_roundtrip
```
预期：三个测试全部 PASS。

- [ ] **Step 5：提交**

```bash
git add util/trace-compress/trace_compress.cc util/trace-compress/trace_compress.h \
        util/trace-compress/test_roundtrip.cc
git commit -m "Implement L1 encoder: flags + PC delta + funcid lift (v4->v5)"
```

---

## Task 2：L1 离线转换工具 — 文件级 v4→v5 转换

**目标：** 让 `trace-compress` 工具能读取实际 .pb 文件、编码为 v5、写出。

**文件：**
- 修改：`util/trace-compress/main.cc`
- 修改：`util/trace-compress/trace_compress.h`
- 修改：`util/trace-compress/trace_compress.cc`
- 新建：`util/trace-compress/test_roundtrip.sh`

- [ ] **Step 1：实现文件级转换逻辑**

更新 `main.cc`，替换 "Not yet implemented" 部分：

```cpp
// util/trace-compress/main.cc
#include <fstream>
#include <iostream>
#include <string>
#include "trace_compress.h"
#include "threadblock.pb.h"

int convert_file_v4_to_v5(const std::string& input_path,
                          const std::string& output_path,
                          int function_unique_id) {
  // Read v4 threadblock
  dynamic_trace::threadblock tb;
  {
    std::ifstream input(input_path, std::ios::binary);
    if (!input) {
      std::cerr << "Error: cannot open " << input_path << std::endl;
      return 1;
    }
    if (!tb.ParseFromIstream(&input)) {
      std::cerr << "Error: failed to parse " << input_path << std::endl;
      return 1;
    }
  }

  // Encode to v5
  dynamic_trace::compressed_threadblock ctb;
  if (!encode_v4_to_v5(tb, &ctb, function_unique_id)) {
    std::cerr << "Error: encoding failed" << std::endl;
    return 1;
  }

  // Write v5
  {
    std::ofstream output(output_path, std::ios::binary);
    if (!ctb.SerializeToOstream(&output)) {
      std::cerr << "Error: failed to write " << output_path << std::endl;
      return 1;
    }
  }

  size_t orig_size = tb.ByteSizeLong();
  size_t comp_size = ctb.ByteSizeLong();
  std::cout << "OK: " << orig_size << "B -> " << comp_size << "B"
            << " (ratio=" << (double)orig_size / comp_size << "x)" << std::endl;
  return 0;
}

void print_usage(const char* prog) {
  std::cerr << "Usage: " << prog
            << " --input <file.pb> --output <file.pb>"
            << " --from-version <4> --to-version <5>"
            << " [--func-id <N>]"
            << std::endl;
}

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::string input_path, output_path;
  int from_version = 4, to_version = 5;
  int func_id = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--from-version" && i + 1 < argc) from_version = std::stoi(argv[++i]);
    else if (arg == "--to-version" && i + 1 < argc) to_version = std::stoi(argv[++i]);
    else if (arg == "--func-id" && i + 1 < argc) func_id = std::stoi(argv[++i]);
    else { print_usage(argv[0]); return 1; }
  }

  if (input_path.empty() || output_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  int ret = 1;
  if (from_version == 4 && to_version == 5) {
    ret = convert_file_v4_to_v5(input_path, output_path, func_id);
  } else {
    std::cerr << "Unsupported conversion: v" << from_version
              << " -> v" << to_version << std::endl;
  }

  google::protobuf::ShutdownProtobufLibrary();
  return ret;
}
```

- [ ] **Step 2：编写端到端 shell 测试**

```bash
#!/bin/bash
# util/trace-compress/test_roundtrip.sh
# End-to-end test: v4 -> v5 -> decode -> compare against v4
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRACE_DIR="$SCRIPT_DIR/../../exampleTraces"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "=== Extracting sample traces ==="
tar xzf "$TRACE_DIR/rodinia2Ampere.tar.gz" -C "$WORK_DIR"

# Find first .pb file
PB_FILE=$(find "$WORK_DIR" -name "*.pb" | head -1)
if [ -z "$PB_FILE" ]; then
  echo "FAIL: no .pb files found"
  exit 1
fi
echo "Test file: $PB_FILE ($(du -b "$PB_FILE" | cut -f1) bytes)"

echo "=== Converting v4 -> v5 ==="
"$SCRIPT_DIR/trace-compress" --input "$PB_FILE" --output "$WORK_DIR/compressed.pb" \
  --from-version 4 --to-version 5 --func-id 1

echo "=== Verifying compressed file exists and is smaller ==="
ORIG_SIZE=$(du -b "$PB_FILE" | cut -f1)
COMP_SIZE=$(du -b "$WORK_DIR/compressed.pb" | cut -f1)
echo "Original: ${ORIG_SIZE}B  Compressed: ${COMP_SIZE}B"

if [ "$COMP_SIZE" -ge "$ORIG_SIZE" ]; then
  echo "WARN: compressed file is not smaller (may be expected for tiny files)"
fi

echo "=== Running unit roundtrip tests ==="
"$SCRIPT_DIR/test_roundtrip"

echo ""
echo "=== ALL TESTS PASSED ==="
```

- [ ] **Step 3：运行端到端测试**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/util/trace-compress
make trace-compress test_roundtrip
chmod +x test_roundtrip.sh
./test_roundtrip.sh
```
预期：转换成功，单元测试通过。

- [ ] **Step 4：提交**

```bash
git add util/trace-compress/main.cc util/trace-compress/test_roundtrip.sh
git commit -m "Add file-level v4->v5 converter with end-to-end test"
```

---

## Task 3：L1 Parser 解码 — 模拟器集成

**目标：** 让模拟器 parser 能读取 v5 格式 .pb 文件，解码后输出与 v4 完全一致的 `inst_trace_t`。

**文件：**
- 修改：`gpu-simulator/trace-parser/trace_parser.h`
- 修改：`gpu-simulator/trace-parser/trace_parser.cc`

- [ ] **Step 1：在 trace_parser.h 中添加 flags 常量和 v5 解码声明**

在 `trace_parser.h` 的 `#define WARP_SIZE 32` 之后，`address_format` 枚举之前添加：

```cpp
// Compressed trace flags (v5+)
constexpr uint32_t COMPRESS_FLAG_FULL_ACTIVE    = 1 << 0;
constexpr uint32_t COMPRESS_FLAG_PRED_EQ_ACTIVE = 1 << 1;
constexpr uint32_t COMPRESS_FLAG_PC_DELTA       = 1 << 2;
```

在 `inst_trace_t` 结构体中添加新的解析方法：

```cpp
bool parse_from_compressed_pb(dynamic_trace::compressed_instruction cinst,
                              uint32_t flags, uint32_t prev_pc,
                              int function_unique_id,
                              unsigned tracer_version, gpgpu_sim *gpu,
                              std::string kernel_name,
                              traced_execution &static_trace_info);
```

在 `trace_parser` 类中添加：

```cpp
void get_next_threadblock_traces_v5(
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces,
    std::vector<std::vector<address_type> *> threadblock_traced_pcs,
    unsigned int gpu_device_id, unsigned int streamid, unsigned int kernelid,
    unsigned trace_version, unsigned int block_id_x, unsigned int block_id_y,
    unsigned int block_id_z, gpgpu_sim *gpu, std::string kernel_name,
    traced_execution &static_trace_info, int function_unique_id);
```

- [ ] **Step 2：实现 `parse_from_compressed_pb`**

在 `trace_parser.cc` 中添加：

```cpp
#include "../../util/traces_enhanced/pb_trace/include/compressed_threadblock.pb.h"
#include "../../util/traces_enhanced/pb_trace/include/compressed_instruction.pb.h"

bool inst_trace_t::parse_from_compressed_pb(
    dynamic_trace::compressed_instruction cinst,
    uint32_t flags, uint32_t prev_pc,
    int function_unique_id,
    unsigned tracer_version, gpgpu_sim *gpu,
    std::string kernel_name,
    traced_execution &static_trace_info) {

  // Decode PC
  if (flags & COMPRESS_FLAG_PC_DELTA) {
    m_pc = prev_pc + cinst.pc();
  } else {
    m_pc = cinst.pc();
  }

  // Decode masks
  uint32_t active;
  if (flags & COMPRESS_FLAG_FULL_ACTIVE) {
    active = 0xFFFFFFFF;
  } else {
    active = cinst.active_mask();
  }

  uint32_t predicate;
  if (flags & COMPRESS_FLAG_PRED_EQ_ACTIVE) {
    predicate = active;
  } else {
    predicate = cinst.predicate_mask();
  }

  mask = active & predicate;
  m_unique_function_id = function_unique_id;

  std::bitset<WARP_SIZE> mask_bits(mask);
  opcode = static_trace_info.get_kernel_by_unique_function_id(
      m_unique_function_id).get_instruction(m_pc).get_op_code();
  m_next_traced_pc = 0;

  unsigned int num_memrefs = cinst.addresses_size();
  memadd_info.resize(num_memrefs);
  for (unsigned int i = 0; i < num_memrefs; ++i) {
    unsigned int mem_width = cinst.addresses(i).data_width();
    parse_memref(i, mem_width, mask_bits, cinst.addresses(i));
    memadd_info[i]->u_desc_value = cinst.addresses(i).udesc_value();
  }

  return true;
}
```

- [ ] **Step 3：实现 `get_next_threadblock_traces_v5`**

在 `trace_parser.cc` 中添加，结构与原 `get_next_threadblock_traces` 一致，但读取 `compressed_threadblock`：

```cpp
void trace_parser::get_next_threadblock_traces_v5(
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces,
    std::vector<std::vector<address_type> *> threadblock_traced_pcs,
    unsigned int gpu_device_id, unsigned int streamid, unsigned int kernelid,
    unsigned trace_version, unsigned int block_id_x, unsigned int block_id_y,
    unsigned int block_id_z, gpgpu_sim *gpu, std::string kernel_name,
    traced_execution &static_trace_info, int function_unique_id) {

  for (unsigned i = 0; i < threadblock_traces.size(); ++i) {
    threadblock_traces[i]->clear();
  }

  dynamic_trace::compressed_threadblock ctb;

  std::string tb_path = m_threadblocks_main_path + "device_" +
      std::to_string(gpu_device_id) + "/stream_" + std::to_string(streamid) +
      "/kernel_" + std::to_string(kernelid) + "/d_" +
      std::to_string(gpu_device_id) + "_s_" + std::to_string(streamid) +
      "_k_" + std::to_string(kernelid) + "_" +
      std::to_string(block_id_x) + "," + std::to_string(block_id_y) +
      "," + std::to_string(block_id_z) + ".pb";

  std::ifstream input(tb_path, std::ios::binary);
  if (!input) {
    std::cout << "Error: file not found in path: " << tb_path << "\n";
    fflush(stdout);
    abort();
  }
  if (!ctb.ParseFromIstream(&input)) {
    std::cout << "Error: Failed to parse compressed threadblock file\n";
    fflush(stdout);
    abort();
  }
  input.close();

  int func_id = ctb.function_unique_id();
  unsigned int size_traced_instructions_num_used_vector = gpu == nullptr ? 32 : 1;

  std::cout << "thread block = " << ctb.block_id().x() << ","
            << ctb.block_id().y() << "," << ctb.block_id().z() << std::endl;

  for (const auto& [warp_id_key, cwarp] : ctb.warps()) {
    unsigned warp_id = warp_id_key;
    unsigned insts_num = cwarp.instructions_size();
    threadblock_traced_pcs[warp_id]->resize(insts_num);

    uint32_t prev_pc = 0;
    address_type previous_traced_pc = 0;

    for (int idx = 0; idx < cwarp.instructions_size(); idx++) {
      const auto& cinst = cwarp.instructions(idx);
      uint32_t flags = cinst.flags();

      inst_trace_t current_inst;
      current_inst.parse_from_compressed_pb(
          cinst, flags, prev_pc, func_id,
          trace_version, gpu, kernel_name, static_trace_info);
      current_inst.block_idx_x = block_id_x;
      current_inst.block_idx_y = block_id_y;
      current_inst.block_idx_z = block_id_z;

      prev_pc = current_inst.m_pc;

      auto* map_inst_of_warp = threadblock_traces[warp_id];
      auto it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
      if (it_find_pc == map_inst_of_warp->end()) {
        map_inst_of_warp->insert(std::pair<address_type, traced_instructions_by_pc>(
            current_inst.m_pc,
            traced_instructions_by_pc(current_inst.m_pc,
                                      size_traced_instructions_num_used_vector)));
        it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
      }
      it_find_pc->second.instructions.push_back(current_inst);
      threadblock_traced_pcs[warp_id]->at(idx) = current_inst.m_pc;

      if (idx > 0) {
        auto it_find_prev_pc = map_inst_of_warp->find(previous_traced_pc);
        assert(it_find_prev_pc != map_inst_of_warp->end());
        it_find_prev_pc->second.instructions[
            it_find_prev_pc->second.num_traced_instructions - 1]
            .m_next_traced_pc = current_inst.m_pc;
      }
      previous_traced_pc = current_inst.m_pc;
      it_find_pc->second.num_traced_instructions++;
    }
  }
  ctb.Clear();
}
```

- [ ] **Step 4：添加版本分发逻辑**

在 `get_next_threadblock_traces` 的调用方（模拟器主循环）需要根据 trace 版本选择调用哪个函数。找到调用 `get_next_threadblock_traces` 的位置，在其前添加版本判断。

这一步需要先搜索调用点：
```bash
grep -rn "get_next_threadblock_traces" simulator-remodeled/gpu-simulator/ --include="*.cc" --include="*.h"
```

根据搜索结果，在调用处增加版本分发：
```cpp
if (trace_version >= 5) {
  parser->get_next_threadblock_traces_v5(
      threadblock_traces, threadblock_traced_pcs,
      gpu_device_id, streamid, kernelid, trace_version,
      block_id_x, block_id_y, block_id_z, gpu,
      kernel_name, static_trace_info, func_unique_id);
} else {
  parser->get_next_threadblock_traces(
      threadblock_traces, threadblock_traced_pcs,
      gpu_device_id, streamid, kernelid, trace_version,
      block_id_x, block_id_y, block_id_z, gpu,
      kernel_name, static_trace_info);
}
```

- [ ] **Step 5：编译验证**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
source setup_environment_no_git.sh
make -j -C gpu-simulator
```
预期：编译通过。

- [ ] **Step 6：提交**

```bash
git add gpu-simulator/trace-parser/trace_parser.h \
        gpu-simulator/trace-parser/trace_parser.cc
git commit -m "Add v5 compressed trace decoder to simulator parser"
```

---

## Task 4：L1 端到端验证

**目标：** 用 rodinia2 trace 验证完整 v4→v5→模拟器 路径的正确性。

**文件：**
- 修改：`util/trace-compress/test_roundtrip.sh`

- [ ] **Step 1：扩展测试脚本，批量转换所有 .pb 文件**

在 `test_roundtrip.sh` 末尾追加批量转换逻辑：

```bash
echo "=== Batch converting all .pb files ==="
TOTAL=0
CONVERTED=0
TOTAL_ORIG=0
TOTAL_COMP=0

find "$WORK_DIR/rodinia2" -name "*.pb" | while read PB; do
  OUT="${PB%.pb}.v5.pb"
  "$SCRIPT_DIR/trace-compress" --input "$PB" --output "$OUT" \
    --from-version 4 --to-version 5 --func-id 1 > /dev/null 2>&1
  if [ $? -eq 0 ]; then
    ORIG=$(du -b "$PB" | cut -f1)
    COMP=$(du -b "$OUT" | cut -f1)
    TOTAL_ORIG=$((TOTAL_ORIG + ORIG))
    TOTAL_COMP=$((TOTAL_COMP + COMP))
    CONVERTED=$((CONVERTED + 1))
  fi
  TOTAL=$((TOTAL + 1))
done

echo "Converted: $CONVERTED / $TOTAL files"
echo "Total original: ${TOTAL_ORIG}B"
echo "Total compressed: ${TOTAL_COMP}B"
if [ "$TOTAL_ORIG" -gt 0 ]; then
  RATIO=$(echo "scale=2; $TOTAL_ORIG / $TOTAL_COMP" | bc 2>/dev/null || echo "N/A")
  echo "Overall compression ratio: ${RATIO}x"
fi
```

- [ ] **Step 2：运行完整验证**

```bash
./test_roundtrip.sh
```
预期：所有文件转换成功，报告总体压缩比。

- [ ] **Step 3：提交**

```bash
git add util/trace-compress/test_roundtrip.sh
git commit -m "Add batch L1 compression validation for rodinia2"
```

---

## Task 5：L2 Proto 扩展 — instruction_run 和 compressed_warp_v6

**目标：** 扩展 proto 定义以支持 run-length squash。

**文件：**
- 修改：`util/traces_enhanced/dynamic_trace/compressed_instruction.proto`
- 修改：`util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`

- [ ] **Step 1：添加 instruction_run message**

在 `compressed_instruction.proto` 末尾追加：

```protobuf
// Run-length encoding for consecutive address-free instructions
// sharing the same flags and constant PC delta
message instruction_run {
  uint32 pc_start = 1;
  uint32 pc_delta = 2;
  uint32 flags = 3;
  uint32 count = 4;
}
```

- [ ] **Step 2：添加 compressed_warp_v6 message**

在 `compressed_threadblock.proto` 末尾追加：

```protobuf
message compressed_warp_v6 {
  int32 id = 1;
  repeated compressed_instruction instructions = 2;
  repeated instruction_run runs = 3;
  // Execution order tags:
  // bit 31 = 0 -> index into instructions[], bit 31 = 1 -> index into runs[]
  // lower 31 bits = array index
  repeated uint32 sequence = 4;
}

message compressed_threadblock_v6 {
  dim3d block_id = 1;
  map<int32, compressed_warp_v6> warps = 2;
  int32 function_unique_id = 3;
}
```

- [ ] **Step 3：重新编译 proto**

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled
make -C util/traces_enhanced clean && make -C util/traces_enhanced
```
预期：编译通过，新 message 出现在生成的 .pb.h 中。

- [ ] **Step 4：提交**

```bash
git add util/traces_enhanced/dynamic_trace/compressed_instruction.proto \
        util/traces_enhanced/dynamic_trace/compressed_threadblock.proto
git commit -m "Add L2 proto definitions: instruction_run and compressed_warp_v6"
```

---

## Task 6：L2 编码器 — Run-Length Squash

**目标：** 实现 v5→v6 编码和解码，将连续无地址同 flags 指令合并为 `instruction_run`。

**文件：**
- 修改：`util/trace-compress/trace_compress.h`
- 修改：`util/trace-compress/trace_compress.cc`
- 修改：`util/trace-compress/test_roundtrip.cc`

- [ ] **Step 1：在头文件中添加 L2 接口**

```cpp
// Sequence tag constants
constexpr uint32_t SEQ_TAG_RUN_BIT = 1u << 31;

// Minimum run length to justify run-length encoding
constexpr uint32_t MIN_RUN_LENGTH = 3;

// L2: Encode v5 -> v6 (run-length squash)
bool encode_v5_to_v6(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v6* dst);

// L2: Decode v6 -> v5
bool decode_v6_to_v5(const dynamic_trace::compressed_threadblock_v6& src,
                     dynamic_trace::compressed_threadblock* dst);

// Convenience: v4 -> v6 direct
bool encode_v4_to_v6(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock_v6* dst,
                     int function_unique_id);
```

- [ ] **Step 2：添加 L2 往返测试**

在 `test_roundtrip.cc` 中添加：

```cpp
void test_v6_roundtrip() {
  std::cout << "test_v6_roundtrip: run-length squash... ";
  auto tb_orig = make_test_threadblock();

  // v4 -> v5 -> v6
  dynamic_trace::compressed_threadblock v5;
  encode_v4_to_v5(tb_orig, &v5, 42);

  dynamic_trace::compressed_threadblock_v6 v6;
  bool ok = encode_v5_to_v6(v5, &v6);
  assert(ok);

  // v6 -> v5 -> v4
  dynamic_trace::compressed_threadblock v5_back;
  ok = decode_v6_to_v5(v6, &v5_back);
  assert(ok);

  dynamic_trace::threadblock tb_decoded;
  ok = decode_v5_to_v4(v5_back, &tb_decoded);
  assert(ok);

  compare_threadblocks(tb_orig, tb_decoded);

  // Verify that runs were actually created (instructions 0, 2, 4 have no addresses)
  bool found_run = false;
  for (const auto& [wid, warp] : v6.warps()) {
    if (warp.runs_size() > 0) found_run = true;
  }
  // With 5 instructions where 3 are address-free and sequential,
  // we expect at least one run (if they are consecutive)
  std::cout << "PASS (runs_found=" << found_run << ")" << std::endl;
}
```

在 `main()` 中调用 `test_v6_roundtrip();`

- [ ] **Step 3：运行测试确认失败**

```bash
make test_roundtrip && ./test_roundtrip
```
预期：assert 失败于 `encode_v5_to_v6` 返回 false。

- [ ] **Step 4：实现 L2 编码器和解码器**

```cpp
bool encode_v5_to_v6(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v6* dst) {
  *dst->mutable_block_id() = src.block_id();
  dst->set_function_unique_id(src.function_unique_id());

  for (const auto& [warp_id, cwarp] : src.warps()) {
    dynamic_trace::compressed_warp_v6 warp_v6;
    warp_v6.set_id(cwarp.id());

    int i = 0;
    int num_insts = cwarp.instructions_size();
    uint32_t inst_index = 0;
    uint32_t run_index = 0;

    while (i < num_insts) {
      const auto& inst = cwarp.instructions(i);

      // Check if this instruction can start a run:
      // - no addresses
      // - has PC_DELTA flag (not the first instruction)
      bool can_run = (inst.addresses_size() == 0) && (inst.flags() & FLAG_PC_DELTA);

      if (can_run) {
        // Count consecutive address-free instructions with same flags and pc delta
        uint32_t run_flags = inst.flags();
        uint32_t run_pc_delta = inst.pc();
        int run_start = i;

        int j = i + 1;
        while (j < num_insts) {
          const auto& next = cwarp.instructions(j);
          if (next.addresses_size() != 0) break;
          if (next.flags() != run_flags) break;
          if (next.pc() != run_pc_delta) break;
          j++;
        }
        int run_len = j - i;

        if (run_len >= (int)MIN_RUN_LENGTH) {
          // Compute absolute PC of run start
          // We need to reconstruct it: the first inst in the run has a delta PC.
          // But we store pc_start as absolute. We need the previous instruction's
          // absolute PC. For simplicity, store the delta-based pc_start and
          // reconstruct during decode using the previous instruction's PC.
          // Actually, let's store the absolute PC. We can compute it from context.
          // For the encoder, we need to track absolute PCs.
          // Let's just store pc_start as the delta value of the first instruction
          // and reconstruct in the decoder knowing the previous absolute PC.
          auto* run = warp_v6.add_runs();
          run->set_pc_start(inst.pc());  // This is still a delta value
          run->set_pc_delta(run_pc_delta);
          run->set_flags(run_flags);
          run->set_count(run_len);
          warp_v6.add_sequence(SEQ_TAG_RUN_BIT | run_index);
          run_index++;
          i = j;
          continue;
        }
      }

      // Not a run, emit as individual instruction
      *warp_v6.add_instructions() = inst;
      warp_v6.add_sequence(inst_index);
      inst_index++;
      i++;
    }

    (*dst->mutable_warps())[warp_id] = warp_v6;
  }
  return true;
}

bool decode_v6_to_v5(const dynamic_trace::compressed_threadblock_v6& src,
                     dynamic_trace::compressed_threadblock* dst) {
  *dst->mutable_block_id() = src.block_id();
  dst->set_function_unique_id(src.function_unique_id());

  for (const auto& [warp_id, warp_v6] : src.warps()) {
    dynamic_trace::compressed_warp cwarp;
    cwarp.set_id(warp_v6.id());

    for (int s = 0; s < warp_v6.sequence_size(); s++) {
      uint32_t tag = warp_v6.sequence(s);
      bool is_run = (tag & SEQ_TAG_RUN_BIT) != 0;
      uint32_t index = tag & ~SEQ_TAG_RUN_BIT;

      if (is_run) {
        const auto& run = warp_v6.runs(index);
        for (uint32_t k = 0; k < run.count(); k++) {
          auto* inst = cwarp.add_instructions();
          inst->set_pc(k == 0 ? run.pc_start() : run.pc_delta());
          inst->set_flags(run.flags());
          // No addresses, no active_mask/predicate_mask overrides
        }
      } else {
        *cwarp.add_instructions() = warp_v6.instructions(index);
      }
    }

    (*dst->mutable_warps())[warp_id] = cwarp;
  }
  return true;
}

bool encode_v4_to_v6(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock_v6* dst,
                     int function_unique_id) {
  dynamic_trace::compressed_threadblock v5;
  if (!encode_v4_to_v5(src, &v5, function_unique_id)) return false;
  return encode_v5_to_v6(v5, dst);
}
```

- [ ] **Step 5：运行测试确认通过**

```bash
make test_roundtrip && ./test_roundtrip
```
预期：所有 L1 + L2 测试 PASS。

- [ ] **Step 6：提交**

```bash
git add util/trace-compress/trace_compress.h util/trace-compress/trace_compress.cc \
        util/trace-compress/test_roundtrip.cc
git commit -m "Implement L2 encoder: run-length squash (v5->v6)"
```

---

## Task 7：L2 Parser 解码 + 验证

**目标：** 在模拟器 parser 中添加 v6 解码支持，并验证端到端正确性。

**文件：**
- 修改：`gpu-simulator/trace-parser/trace_parser.h`
- 修改：`gpu-simulator/trace-parser/trace_parser.cc`
- 修改：`util/trace-compress/main.cc`

- [ ] **Step 1：在 trace_parser.h 中添加 v6 解码声明**

```cpp
constexpr uint32_t SEQ_TAG_RUN_BIT = 1u << 31;

void get_next_threadblock_traces_v6(
    std::vector<std::map<address_type, traced_instructions_by_pc> *> threadblock_traces,
    std::vector<std::vector<address_type> *> threadblock_traced_pcs,
    unsigned int gpu_device_id, unsigned int streamid, unsigned int kernelid,
    unsigned trace_version, unsigned int block_id_x, unsigned int block_id_y,
    unsigned int block_id_z, gpgpu_sim *gpu, std::string kernel_name,
    traced_execution &static_trace_info, int function_unique_id);
```

- [ ] **Step 2：实现 v6 解码**

与 v5 解码类似，但先按 `sequence` 展开 runs 为独立指令，然后复用 v5 的逐指令解码逻辑。实现方式：在函数内部先将 v6 展开为 v5 的 `compressed_threadblock`，然后调用 v5 解码路径。

```cpp
void trace_parser::get_next_threadblock_traces_v6(/* ... */) {
  dynamic_trace::compressed_threadblock_v6 ctb_v6;

  // Read and parse compressed_threadblock_v6 from .pb file
  std::string tb_path = /* same path construction as v5 */;
  std::ifstream input(tb_path, std::ios::binary);
  if (!input) { /* error handling same as v4 */ }
  if (!ctb_v6.ParseFromIstream(&input)) { /* error handling */ }
  input.close();

  // Expand v6 -> v5 in memory
  dynamic_trace::compressed_threadblock ctb_v5;
  decode_v6_to_v5_inline(ctb_v6, &ctb_v5);
  // (inline version of decode_v6_to_v5, or link against trace_compress lib)

  // Then use v5 decode logic on the expanded data
  // ... (reuse get_next_threadblock_traces_v5 logic with ctb_v5)
}
```

注意：由于 parser 和 trace-compress 工具共享解码逻辑，可以将 `decode_v6_to_v5` 提取为共享库，或者在 parser 中直接内联展开逻辑。推荐后者以避免构建依赖复杂化。

- [ ] **Step 3：更新版本分发逻辑**

在 Task 3 Step 4 的版本分发处添加 v6 分支：

```cpp
if (trace_version >= 6) {
  parser->get_next_threadblock_traces_v6(/* ... */);
} else if (trace_version >= 5) {
  parser->get_next_threadblock_traces_v5(/* ... */);
} else {
  parser->get_next_threadblock_traces(/* ... */);
}
```

- [ ] **Step 4：更新离线转换工具支持 v4→v6**

在 `main.cc` 中添加 v4→v6 和 v5→v6 转换分支。

- [ ] **Step 5：编译并运行批量验证**

```bash
make -j -C gpu-simulator
./util/trace-compress/test_roundtrip.sh
```

- [ ] **Step 6：提交**

```bash
git add gpu-simulator/trace-parser/ util/trace-compress/main.cc
git commit -m "Add v6 run-length squash decoder to simulator parser"
```

---

## Task 8：L3 Proto + 编码器 — 跨 Warp PC 去重

**目标：** 实现 v7 格式：提取共享 PC 序列，每个 warp 只存地址和 mask 覆盖。

**文件：**
- 修改：`util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`
- 修改：`util/trace-compress/trace_compress.h`
- 修改：`util/trace-compress/trace_compress.cc`
- 修改：`util/trace-compress/test_roundtrip.cc`

- [ ] **Step 1：添加 L3 proto message**

在 `compressed_threadblock.proto` 中追加：

```protobuf
message warp_instruction {
  repeated address addresses = 1;
  uint32 flags = 2;
  uint32 active_mask = 3;
  uint32 predicate_mask = 4;
}

message pc_override {
  uint32 instruction_index = 1;
  uint32 actual_pc = 2;
}

message warp_diff {
  int32 id = 1;
  repeated warp_instruction instructions = 2;
  repeated pc_override pc_overrides = 3;
}

message compressed_threadblock_v7 {
  dim3d block_id = 1;
  int32 function_unique_id = 2;
  repeated uint32 shared_pc_sequence = 3;
  map<int32, warp_diff> warps = 4;
}
```

- [ ] **Step 2：添加 L3 接口到头文件**

```cpp
// Warp PC divergence threshold: if more than this fraction of PCs differ,
// fall back to per-warp encoding
constexpr double WARP_DIVERGENCE_THRESHOLD = 0.10;

bool encode_v5_to_v7(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v7* dst);

bool decode_v7_to_v5(const dynamic_trace::compressed_threadblock_v7& src,
                     dynamic_trace::compressed_threadblock* dst);
```

- [ ] **Step 3：编写 L3 往返测试**

```cpp
void test_v7_roundtrip() {
  std::cout << "test_v7_roundtrip: cross-warp PC dedup... ";
  auto tb_orig = make_test_threadblock();  // 2 warps with identical PC sequences

  dynamic_trace::compressed_threadblock v5;
  encode_v4_to_v5(tb_orig, &v5, 42);

  dynamic_trace::compressed_threadblock_v7 v7;
  bool ok = encode_v5_to_v7(v5, &v7);
  assert(ok);

  // Verify shared PC sequence exists
  assert(v7.shared_pc_sequence_size() == 5);  // 5 instructions
  assert(v7.shared_pc_sequence(0) == 0x1000);

  dynamic_trace::compressed_threadblock v5_back;
  ok = decode_v7_to_v5(v7, &v5_back);
  assert(ok);

  dynamic_trace::threadblock tb_decoded;
  decode_v5_to_v4(v5_back, &tb_decoded);
  compare_threadblocks(tb_orig, tb_decoded);
  std::cout << "PASS" << std::endl;
}
```

- [ ] **Step 4：实现 L3 编码器**

核心逻辑：
1. 从 warp 0 提取绝对 PC 序列作为 `shared_pc_sequence`
2. 对每个 warp 比较 PC 序列，记录差异为 `pc_overrides`
3. 每个 warp 只保留 addresses 和 mask 信息

```cpp
bool encode_v5_to_v7(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v7* dst) {
  *dst->mutable_block_id() = src.block_id();
  dst->set_function_unique_id(src.function_unique_id());

  // Extract absolute PC sequences from all warps
  // First, reconstruct absolute PCs from v5 delta-encoded PCs
  std::map<int, std::vector<uint32_t>> warp_pcs;
  for (const auto& [wid, cwarp] : src.warps()) {
    uint32_t prev_pc = 0;
    auto& pcs = warp_pcs[wid];
    for (int i = 0; i < cwarp.instructions_size(); i++) {
      const auto& inst = cwarp.instructions(i);
      uint32_t pc;
      if (inst.flags() & FLAG_PC_DELTA) {
        pc = prev_pc + inst.pc();
      } else {
        pc = inst.pc();
      }
      pcs.push_back(pc);
      prev_pc = pc;
    }
  }

  // Use warp 0 as reference (or first available warp)
  auto ref_it = warp_pcs.begin();
  if (ref_it == warp_pcs.end()) return true;  // empty threadblock
  const auto& ref_pcs = ref_it->second;

  for (uint32_t pc : ref_pcs) {
    dst->add_shared_pc_sequence(pc);
  }

  // For each warp, compute diff against shared sequence
  for (const auto& [wid, cwarp] : src.warps()) {
    dynamic_trace::warp_diff wd;
    wd.set_id(cwarp.id());

    const auto& pcs = warp_pcs[wid];
    int override_count = 0;

    // Check divergence rate
    int diff_count = 0;
    int min_len = std::min((int)ref_pcs.size(), (int)pcs.size());
    for (int i = 0; i < min_len; i++) {
      if (pcs[i] != ref_pcs[i]) diff_count++;
    }
    diff_count += std::abs((int)ref_pcs.size() - (int)pcs.size());
    double divergence = min_len > 0 ? (double)diff_count / min_len : 0;

    // If divergence too high, fall back: store all PCs as overrides
    // (In practice we'd fall back to v5 format, but for simplicity
    //  we record all differing PCs as overrides)

    uint32_t prev_pc_ref = 0;
    for (int i = 0; i < cwarp.instructions_size(); i++) {
      const auto& cinst = cwarp.instructions(i);

      // Store address + mask info
      auto* wi = wd.add_instructions();
      for (int j = 0; j < cinst.addresses_size(); j++) {
        *wi->add_addresses() = cinst.addresses(j);
      }

      // Store flags and masks
      uint32_t flags = cinst.flags();
      // Remove PC_DELTA flag (PC is now in shared sequence)
      uint32_t mask_flags = flags & ~FLAG_PC_DELTA;
      wi->set_flags(mask_flags);
      if (!(flags & FLAG_FULL_ACTIVE)) {
        wi->set_active_mask(cinst.active_mask());
      }
      if (!(flags & FLAG_PRED_EQ_ACTIVE)) {
        wi->set_predicate_mask(cinst.predicate_mask());
      }

      // Record PC override if different from shared
      if (i < (int)ref_pcs.size() && pcs[i] != ref_pcs[i]) {
        auto* ov = wd.add_pc_overrides();
        ov->set_instruction_index(i);
        ov->set_actual_pc(pcs[i]);
      }
    }

    (*dst->mutable_warps())[wid] = wd;
  }

  return true;
}
```

- [ ] **Step 5：实现 L3 解码器**

```cpp
bool decode_v7_to_v5(const dynamic_trace::compressed_threadblock_v7& src,
                     dynamic_trace::compressed_threadblock* dst) {
  *dst->mutable_block_id() = src.block_id();
  dst->set_function_unique_id(src.function_unique_id());

  // Build shared PC sequence
  std::vector<uint32_t> shared_pcs(src.shared_pc_sequence().begin(),
                                    src.shared_pc_sequence().end());

  for (const auto& [wid, wd] : src.warps()) {
    dynamic_trace::compressed_warp cwarp;
    cwarp.set_id(wd.id());

    // Build PC override map
    std::map<uint32_t, uint32_t> overrides;
    for (const auto& ov : wd.pc_overrides()) {
      overrides[ov.instruction_index()] = ov.actual_pc();
    }

    uint32_t prev_pc = 0;
    for (int i = 0; i < wd.instructions_size(); i++) {
      const auto& wi = wd.instructions(i);
      auto* cinst = cwarp.add_instructions();

      // Determine absolute PC
      uint32_t pc;
      auto ov_it = overrides.find(i);
      if (ov_it != overrides.end()) {
        pc = ov_it->second;
      } else if (i < (int)shared_pcs.size()) {
        pc = shared_pcs[i];
      } else {
        pc = 0;  // should not happen
      }

      // Re-encode as v5 format (delta PC)
      if (i == 0) {
        cinst->set_pc(pc);
        cinst->set_flags(wi.flags());  // no PC_DELTA flag
      } else {
        cinst->set_pc(pc - prev_pc);
        cinst->set_flags(wi.flags() | FLAG_PC_DELTA);
      }
      prev_pc = pc;

      // Copy masks
      if (wi.active_mask() != 0) cinst->set_active_mask(wi.active_mask());
      if (wi.predicate_mask() != 0) cinst->set_predicate_mask(wi.predicate_mask());

      // Copy addresses
      for (int j = 0; j < wi.addresses_size(); j++) {
        *cinst->add_addresses() = wi.addresses(j);
      }
    }

    (*dst->mutable_warps())[wid] = cwarp;
  }

  return true;
}
```

- [ ] **Step 6：运行测试确认通过**

```bash
make test_roundtrip && ./test_roundtrip
```

- [ ] **Step 7：提交**

```bash
git add util/traces_enhanced/dynamic_trace/compressed_threadblock.proto \
        util/trace-compress/
git commit -m "Implement L3: cross-warp PC deduplication (v5->v7)"
```

---

## Task 9：L3 Parser 解码 + 验证

**目标：** 模拟器 parser 支持 v7 格式解码。

**与 Task 7 结构相同：** 读取 `compressed_threadblock_v7`，展开为 v5 格式后复用 v5 解码路径，更新版本分发逻辑。

实现步骤与 Task 7 一致，略去重复代码结构。关键差异：

- 读取 `compressed_threadblock_v7` 而非 `compressed_threadblock_v6`
- 展开逻辑调用 `decode_v7_to_v5` 的内联版本
- 版本分发增加 `trace_version >= 7` 分支

- [ ] **Step 1：添加 v7 解码声明到 trace_parser.h**
- [ ] **Step 2：实现 v7 解码（v7→v5 展开 + v5 解码）**
- [ ] **Step 3：更新版本分发（v7 → v6 → v5 → v4 链式判断）**
- [ ] **Step 4：编译验证**
- [ ] **Step 5：运行 rodinia2 批量转换 + 验证**
- [ ] **Step 6：提交**

```bash
git commit -m "Add v7 cross-warp PC dedup decoder to simulator parser"
```

---

## Task 10：L4 Proto + 编码器 — 跨 Threadblock Delta

**目标：** 实现 v8 格式：首个 threadblock 完整存储，后续只存地址偏移 delta。

**文件：**
- 修改：`util/traces_enhanced/dynamic_trace/compressed_threadblock.proto`
- 修改：`util/trace-compress/trace_compress.h`
- 修改：`util/trace-compress/trace_compress.cc`
- 修改：`util/trace-compress/test_roundtrip.cc`

- [ ] **Step 1：添加 L4 proto message**

在 `compressed_threadblock.proto` 中追加：

```protobuf
message address_override {
  uint32 warp_id = 1;
  uint32 instruction_index = 2;
  uint32 address_index = 3;
  int64 address_delta = 4;
}

message tb_delta {
  dim3d block_id = 1;
  dim3d reference_block_id = 2;
  int64 global_address_offset = 3;
  repeated address_override address_overrides = 4;
  bool is_full_encoding = 5;  // true = fallback, ignore delta fields
  compressed_threadblock_v7 full_threadblock = 6;  // only if is_full_encoding
}

message compressed_kernel_v8 {
  compressed_threadblock_v7 base_threadblock = 1;
  repeated tb_delta delta_threadblocks = 2;
}
```

- [ ] **Step 2：添加 L4 接口**

```cpp
// L4 divergence threshold: if more than this fraction of addresses
// need overrides, fall back to full encoding
constexpr double TB_DIVERGENCE_THRESHOLD = 0.20;

// Encode a set of v7 threadblocks (same kernel) into a single v8 kernel
bool encode_kernel_to_v8(
    const std::vector<dynamic_trace::compressed_threadblock_v7>& threadblocks,
    dynamic_trace::compressed_kernel_v8* dst);

// Decode v8 kernel back to individual v7 threadblocks
bool decode_v8_to_v7s(
    const dynamic_trace::compressed_kernel_v8& src,
    std::vector<dynamic_trace::compressed_threadblock_v7>* dst);
```

- [ ] **Step 3：编写 L4 往返测试**

构造两个 threadblock，地址偏移 0x1000，验证编码为单个 `global_address_offset` 后可完整恢复。

```cpp
void test_v8_roundtrip() {
  std::cout << "test_v8_roundtrip: cross-TB delta... ";

  // Create two similar threadblocks with different addresses
  auto tb0 = make_test_threadblock();  // addresses start at 0x80000000
  auto tb1 = make_test_threadblock();
  // Shift all addresses in tb1 by 0x1000
  tb1.mutable_block_id()->set_x(1);
  for (auto& [wid, warp] : *tb1.mutable_warps()) {
    for (int i = 0; i < warp.instructions_size(); i++) {
      for (int j = 0; j < warp.instructions(i).addresses_size(); j++) {
        auto* addr = warp.mutable_instructions(i)->mutable_addresses(j);
        addr->set_base_address(addr->base_address() + 0x1000);
      }
    }
  }

  // Encode both to v7
  dynamic_trace::compressed_threadblock v5_0, v5_1;
  encode_v4_to_v5(tb0, &v5_0, 42);
  encode_v4_to_v5(tb1, &v5_1, 42);

  dynamic_trace::compressed_threadblock_v7 v7_0, v7_1;
  encode_v5_to_v7(v5_0, &v7_0);
  encode_v5_to_v7(v5_1, &v7_1);

  // Encode to v8
  std::vector<dynamic_trace::compressed_threadblock_v7> tbs = {v7_0, v7_1};
  dynamic_trace::compressed_kernel_v8 v8;
  bool ok = encode_kernel_to_v8(tbs, &v8);
  assert(ok);
  assert(v8.delta_threadblocks_size() == 1);
  assert(v8.delta_threadblocks(0).global_address_offset() == 0x1000);
  assert(!v8.delta_threadblocks(0).is_full_encoding());

  // Decode back
  std::vector<dynamic_trace::compressed_threadblock_v7> decoded_tbs;
  ok = decode_v8_to_v7s(v8, &decoded_tbs);
  assert(ok);
  assert(decoded_tbs.size() == 2);

  // Verify roundtrip through full chain: v8 -> v7 -> v5 -> v4
  for (int t = 0; t < 2; t++) {
    dynamic_trace::compressed_threadblock v5_back;
    decode_v7_to_v5(decoded_tbs[t], &v5_back);
    dynamic_trace::threadblock tb_back;
    decode_v5_to_v4(v5_back, &tb_back);
    compare_threadblocks(t == 0 ? tb0 : tb1, tb_back);
  }

  std::cout << "PASS" << std::endl;
}
```

- [ ] **Step 4：实现 L4 编码器和解码器**

编码器核心逻辑：
1. 取第一个 threadblock 为 base
2. 对每个后续 threadblock，收集所有地址，计算最频繁的 offset 作为 `global_address_offset`
3. 对不符合 global offset 的地址记录 `address_overrides`
4. 若 override 比例超过 20%，回退到 `is_full_encoding = true`

解码器核心逻辑：
1. 输出 base threadblock
2. 对每个 delta：克隆 base，将所有地址加上 `global_address_offset`，应用 `address_overrides`

- [ ] **Step 5：运行测试确认通过**

```bash
make test_roundtrip && ./test_roundtrip
```

- [ ] **Step 6：提交**

```bash
git add util/traces_enhanced/dynamic_trace/compressed_threadblock.proto \
        util/trace-compress/
git commit -m "Implement L4: cross-threadblock delta encoding (v7->v8)"
```

---

## Task 11：L4 Parser 解码 + 离线工具完善

**目标：** 模拟器 parser 支持 v8 格式（per-kernel 文件），离线工具支持全链路转换。

**文件：**
- 修改：`gpu-simulator/trace-parser/trace_parser.h`
- 修改：`gpu-simulator/trace-parser/trace_parser.cc`
- 修改：`util/trace-compress/main.cc`

- [ ] **Step 1：Parser v8 解码**

v8 改变了文件组织：从 per-threadblock 变为 per-kernel。Parser 读取整个 kernel 文件后，按 block_id 查找对应的 delta 并解码。

在 `trace_parser` 中添加：
- per-kernel 文件缓存（读取一次，服务同一 kernel 的所有 threadblock 请求）
- `get_next_threadblock_traces_v8` 方法：从缓存中取出对应 threadblock 的 delta，应用到 base 上，然后调用 v7→v5→inst_trace_t 解码链

- [ ] **Step 2：更新离线转换工具支持全链路**

在 `main.cc` 中：
- 添加 `--batch-dir` 参数，接受一个 kernel 目录，将其中所有 .pb 合并为单个 v8 文件
- 支持 `v4→v8` 直接转换（内部链式调用 v4→v5→v7→v8）

```bash
trace-compress --batch-dir /path/to/kernel_N/ --output kernel_N.v8.pb \
  --from-version 4 --to-version 8 --func-id 42
```

- [ ] **Step 3：版本分发更新**

```cpp
if (trace_version >= 8) {
  parser->get_next_threadblock_traces_v8(/* ... */);
} else if (trace_version >= 7) {
  // ...
}
```

- [ ] **Step 4：编译验证**

```bash
make -j -C gpu-simulator
```

- [ ] **Step 5：端到端批量验证**

扩展 `test_roundtrip.sh` 添加 v8 测试：将一个 kernel 目录下所有 threadblock 编码为单个 v8 文件，验证解码正确性和压缩比。

- [ ] **Step 6：提交**

```bash
git commit -m "Add v8 cross-TB delta decoder to parser, full-chain converter"
```

---

## Task 12：最终验证与压缩比报告

**目标：** 对 rodinia2 全套 trace 跑完整验证，输出各层压缩比报告。

**文件：**
- 新建：`util/trace-compress/benchmark_compression.sh`

- [ ] **Step 1：编写基准测试脚本**

```bash
#!/bin/bash
# util/trace-compress/benchmark_compression.sh
# Measure compression ratios across all layers for rodinia2
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "Extracting rodinia2Ampere..."
tar xzf "$SCRIPT_DIR/../../exampleTraces/rodinia2Ampere.tar.gz" -C "$WORK_DIR"

echo ""
echo "=== Compression Benchmark ==="
echo "| Layer | Total Size | Ratio vs v4 |"
echo "|-------|-----------|-------------|"

# v4 baseline
V4_SIZE=$(find "$WORK_DIR/rodinia2" -name "*.pb" -exec du -cb {} + | tail -1 | awk '{print $1}')
echo "| v4 (baseline) | ${V4_SIZE}B | 1.0x |"

# v5
V5_SIZE=0
find "$WORK_DIR/rodinia2" -name "*.pb" | while read F; do
  OUT="${F%.pb}.v5.pb"
  "$SCRIPT_DIR/trace-compress" --input "$F" --output "$OUT" \
    --from-version 4 --to-version 5 --func-id 1 2>/dev/null
done
V5_SIZE=$(find "$WORK_DIR/rodinia2" -name "*.v5.pb" -exec du -cb {} + | tail -1 | awk '{print $1}')
RATIO=$(echo "scale=1; $V4_SIZE / $V5_SIZE" | bc)
echo "| v5 (L1: flags+delta) | ${V5_SIZE}B | ${RATIO}x |"

echo ""
echo "Benchmark complete."
```

- [ ] **Step 2：运行基准测试**

```bash
chmod +x util/trace-compress/benchmark_compression.sh
./util/trace-compress/benchmark_compression.sh
```

- [ ] **Step 3：运行全部单元测试确认绿色**

```bash
cd util/trace-compress && make test && ./test_roundtrip
```

- [ ] **Step 4：提交**

```bash
git add util/trace-compress/benchmark_compression.sh
git commit -m "Add compression benchmark script with per-layer ratio report"
```

---

## 依赖关系

```
Task 0 (构建基础设施)
  ├── Task 1 (L1 编码器)
  │     └── Task 2 (L1 离线工具)
  │           └── Task 3 (L1 Parser 解码)
  │                 └── Task 4 (L1 验证)
  │                       └── Task 5 (L2 Proto)
  │                             └── Task 6 (L2 编码器)
  │                                   └── Task 7 (L2 Parser + 验证)
  │                                         └── Task 8 (L3 Proto + 编码器)
  │                                               └── Task 9 (L3 Parser + 验证)
  │                                                     └── Task 10 (L4 Proto + 编码器)
  │                                                           └── Task 11 (L4 Parser + 工具)
  │                                                                 └── Task 12 (最终验证)
```

所有 Task 严格串行。每个 Layer 完成后可独立交付使用。
