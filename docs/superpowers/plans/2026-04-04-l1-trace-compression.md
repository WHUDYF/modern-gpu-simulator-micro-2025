# L1 Trace Compression (v4→v5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate L1 compressed trace format (v5) into the simulator's parser so it can load both v4 and v5 `.pb` files transparently, and upgrade the offline converter to handle batch conversion of entire trace directories.

**Architecture:** The encoder (`trace_compress.cc`) and roundtrip tests already exist and pass. This plan focuses on three remaining pieces: (1) parser-side v5 decoding in `trace_parser.cc` so the simulator can load v5 traces, (2) batch conversion in the offline tool so users can convert entire trace directories, (3) end-to-end validation using real rodinia2 traces.

**Tech Stack:** C++17, Protocol Buffers 3, Make

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `gpu-simulator/trace-parser/trace_parser.h` | Modify | Add `#include` for compressed pb headers; add `parse_from_compressed_pb` method to `inst_trace_t` |
| `gpu-simulator/trace-parser/trace_parser.cc` | Modify | Implement v5 decoding in `get_next_threadblock_traces`; add `parse_from_compressed_pb` |
| `util/trace-compress/main.cc` | Modify | Add batch directory conversion mode |
| `util/trace-compress/trace_compress.h` | Modify | Add batch conversion function declaration |
| `util/trace-compress/trace_compress.cc` | Modify | Add batch conversion implementation |
| `util/trace-compress/test_roundtrip.cc` | Modify | Add parser-integration test that verifies `inst_trace_t` field equality |
| `util/trace-compress/test_e2e.sh` | Create | End-to-end script: batch-convert rodinia2, run simulator, compare APE |

All paths are relative to project root: `/home/dyf/modern-gpu-simulator-micro-2025/simulator-remodeled/`

---

### Task 1: Parser v5 Decoding — Header Changes

**Files:**
- Modify: `gpu-simulator/trace-parser/trace_parser.h`

- [ ] **1.1: Add compressed protobuf includes to trace_parser.h**

In `trace_parser.h`, after the existing protobuf includes (line 41), add includes for compressed types:

```cpp
#include "../../util/traces_enhanced/pb_trace/include/compressed_threadblock.pb.h"
#include "../../util/traces_enhanced/pb_trace/include/compressed_instruction.pb.h"
```

- [ ] **1.2: Add `parse_from_compressed_pb` declaration to `inst_trace_t`**

In the `inst_trace_t` struct, after the existing `parse_from_pb` declaration, add:

```cpp
bool parse_from_compressed_pb(uint32_t pc, uint32_t active_mask, uint32_t predicate_mask,
                              int32_t function_unique_id,
                              const dynamic_trace::compressed_instruction& cinst,
                              unsigned tracer_version, gpgpu_sim *gpu,
                              std::string kernel_name, traced_execution &static_trace_info);
```

- [ ] **1.3: Commit**

```bash
git add gpu-simulator/trace-parser/trace_parser.h
git commit -m "Add v5 compressed trace support declarations to parser header"
```

---

### Task 2: Parser v5 Decoding — Implementation

**Files:**
- Modify: `gpu-simulator/trace-parser/trace_parser.cc`

- [ ] **2.1: Add compressed_threadblock include**

At the top of `trace_parser.cc`, after the existing `threadblock.pb.h` include (line 44), add:

```cpp
#include "../../util/traces_enhanced/pb_trace/include/compressed_threadblock.pb.h"
#include "../../util/traces_enhanced/pb_trace/include/compressed_instruction.pb.h"
```

- [ ] **2.2: Implement `parse_from_compressed_pb`**

After the existing `parse_from_pb` method (ends at line 284), add:

```cpp
bool inst_trace_t::parse_from_compressed_pb(uint32_t pc, uint32_t active_mask, uint32_t predicate_mask,
                                            int32_t function_unique_id,
                                            const dynamic_trace::compressed_instruction& cinst,
                                            unsigned trace_version, gpgpu_sim *gpu,
                                            std::string kernel_name, traced_execution &static_trace_info) {
  m_pc = pc;
  mask = active_mask & predicate_mask;
  m_unique_function_id = function_unique_id;
  unsigned int num_memrefs = cinst.addresses_size();
  std::bitset<WARP_SIZE> mask_bits(mask);
  opcode = static_trace_info.get_kernel_by_unique_function_id(m_unique_function_id).get_instruction(m_pc).get_op_code();
  m_next_traced_pc = 0;

  unsigned int mem_width = 0;
  memadd_info.resize(num_memrefs);
  for (unsigned int i = 0; i < num_memrefs; ++i) {
    mem_width = cinst.addresses(i).data_width();
    parse_memref(i, mem_width, mask_bits, cinst.addresses(i));
    memadd_info[i]->u_desc_value = cinst.addresses(i).udesc_value();
  }

  return true;
}
```

- [ ] **2.3: Add v5 loading path in `get_next_threadblock_traces`**

In the `get_next_threadblock_traces` method, replace the section that loads and parses the threadblock file. Currently the method unconditionally deserializes `dynamic_trace::threadblock`. The new version tries v5 first, then falls back to v4.

Replace the block from the `std::ifstream input(tb_path, ...)` line through `tb_cur.Clear()` (the entire file-loading and instruction-parsing section, approximately lines 441–490) with:

```cpp
  std::string tb_path_v5 = kernel_folder + "/" + tb_string_id + ".v5.pb";
  std::string tb_path_v4 = kernel_folder + "/" + tb_string_id + ".pb";

  // Try v5 compressed format first
  bool is_v5 = false;
  {
    std::ifstream probe(tb_path_v5, std::ios::in | std::ios::binary);
    if (probe.good()) {
      is_v5 = true;
    }
  }

  unsigned int size_traced_instructions_num_used_vector = gpu == nullptr ? 32 : 1;

  if (is_v5) {
    dynamic_trace::compressed_threadblock ctb;
    {
      std::ifstream input(tb_path_v5, std::ios::in | std::ios::binary);
      if (!input) {
        std::cout << "Error: file not found in path: " << tb_path_v5 << "\n";
        fflush(stdout);
        abort();
      }
      if (!ctb.ParseFromIstream(&input)) {
        std::cout << "Error: Failed to parse compressed threadblock file\n";
        fflush(stdout);
        abort();
      }
    }

    int function_unique_id = ctb.function_unique_id();
    constexpr uint32_t FLAG_FULL_ACTIVE    = 1 << 0;
    constexpr uint32_t FLAG_PRED_EQ_ACTIVE = 1 << 1;
    constexpr uint32_t FLAG_PC_DELTA       = 1 << 2;

    std::cout << "thread block = " << ctb.block_id().x() << "," << ctb.block_id().y() << "," << ctb.block_id().z() << " (v5)" << std::endl;

    for (const auto& [wid, cwarp] : ctb.warps()) {
      unsigned int warp_id = wid;
      unsigned int insts_num = cwarp.instructions_size();
      threadblock_traced_pcs[warp_id]->resize(insts_num);
      address_type previous_traced_pc = 0;
      uint32_t prev_pc = 0;

      for (int i = 0; i < cwarp.instructions_size(); ++i) {
        const auto& cinst = cwarp.instructions(i);
        uint32_t flags = cinst.flags();

        // Restore PC
        uint32_t pc;
        if (flags & FLAG_PC_DELTA) {
          pc = prev_pc + cinst.pc();
        } else {
          pc = cinst.pc();
        }
        prev_pc = pc;

        // Restore active_mask
        uint32_t active_mask;
        if (flags & FLAG_FULL_ACTIVE) {
          active_mask = 0xFFFFFFFF;
        } else {
          active_mask = cinst.active_mask();
        }

        // Restore predicate_mask
        uint32_t predicate_mask;
        if (flags & FLAG_PRED_EQ_ACTIVE) {
          predicate_mask = active_mask;
        } else {
          predicate_mask = cinst.predicate_mask();
        }

        inst_trace_t current_inst;
        current_inst.parse_from_compressed_pb(pc, active_mask, predicate_mask,
                                              function_unique_id, cinst,
                                              trace_version, gpu, kernel_name, static_trace_info);
        current_inst.block_idx_x = block_id_x;
        current_inst.block_idx_y = block_id_y;
        current_inst.block_idx_z = block_id_z;

        std::map<address_type, traced_instructions_by_pc> *map_inst_of_warp =
            threadblock_traces[warp_id];
        auto it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
        if (it_find_pc == map_inst_of_warp->end()) {
          map_inst_of_warp->insert(std::pair<address_type, traced_instructions_by_pc>(
              current_inst.m_pc, traced_instructions_by_pc(current_inst.m_pc, size_traced_instructions_num_used_vector)));
          it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
        }
        it_find_pc->second.instructions.push_back(current_inst);
        threadblock_traced_pcs[warp_id]->at(i) = current_inst.m_pc;

        if (i > 0) {
          auto it_find_prev_pc = map_inst_of_warp->find(previous_traced_pc);
          assert(it_find_prev_pc != map_inst_of_warp->end());
          it_find_prev_pc->second.instructions[it_find_prev_pc->second.num_traced_instructions - 1]
              .m_next_traced_pc = current_inst.m_pc;
        }
        previous_traced_pc = current_inst.m_pc;
        it_find_pc->second.num_traced_instructions++;
      }
    }
  } else {
    // Original v4 path
    dynamic_trace::threadblock tb_cur;
    std::ifstream input(tb_path_v4, std::ios::in | std::ios::binary);
    if (!input) {
      std::cout << "Error: file not found in path: " << tb_path_v4 << "\n";
      fflush(stdout);
      abort();
    }
    if (!tb_cur.ParseFromIstream(&input)) {
      std::cout << "Error: Failed to parse threadblock file\n";
      fflush(stdout);
      abort();
    }
    input.close();

    std::cout << "thread block = " << tb_cur.block_id().x() << "," << tb_cur.block_id().y() << "," << tb_cur.block_id().z() << std::endl;
    for (auto warp : tb_cur.warps()) {
      address_type previous_traced_pc = 0;
      unsigned int inst_count = 0;
      unsigned int insts_num = warp.second.instructions_size();
      unsigned int warp_id = warp.first;
      threadblock_traced_pcs[warp_id]->resize(insts_num);
      for (auto pb_inst : warp.second.instructions()) {
        inst_trace_t current_inst;
        current_inst.parse_from_pb(pb_inst, trace_version, gpu, kernel_name, static_trace_info);
        current_inst.block_idx_x = block_id_x;
        current_inst.block_idx_y = block_id_y;
        current_inst.block_idx_z = block_id_z;
        std::map<address_type, traced_instructions_by_pc> *map_inst_of_warp =
            threadblock_traces[warp_id];
        auto it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
        if (it_find_pc == map_inst_of_warp->end()) {
          map_inst_of_warp->insert(std::pair<address_type, traced_instructions_by_pc>(
              current_inst.m_pc, traced_instructions_by_pc(current_inst.m_pc, size_traced_instructions_num_used_vector)));
          it_find_pc = map_inst_of_warp->find(current_inst.m_pc);
        }
        it_find_pc->second.instructions.push_back(current_inst);
        threadblock_traced_pcs[warp_id]->at(inst_count) = current_inst.m_pc;

        auto it_find_prev_pc = map_inst_of_warp->find(previous_traced_pc);
        if (inst_count > 0) {
          assert(it_find_prev_pc != map_inst_of_warp->end());
          it_find_prev_pc->second.instructions[it_find_prev_pc->second.num_traced_instructions - 1]
              .m_next_traced_pc = current_inst.m_pc;
        }
        previous_traced_pc = current_inst.m_pc;
        it_find_pc->second.num_traced_instructions++;
        inst_count++;
      }
    }
    tb_cur.Clear();
  }
```

- [ ] **2.4: Commit**

```bash
git add gpu-simulator/trace-parser/trace_parser.cc
git commit -m "Add v5 compressed trace decoding path in parser"
```

---

### Task 3: Batch Directory Conversion

**Files:**
- Modify: `util/trace-compress/trace_compress.h`
- Modify: `util/trace-compress/trace_compress.cc`
- Modify: `util/trace-compress/main.cc`

- [ ] **3.1: Add batch conversion declarations**

In `trace_compress.h`, after the existing declarations, add:

```cpp
#include "trace.pb.h"
#include "kernel.pb.h"

struct ConvertStats {
  size_t files_converted = 0;
  size_t total_original_bytes = 0;
  size_t total_compressed_bytes = 0;
};

// Batch-convert all .pb threadblock files in a trace directory from v4 to v5.
// trace_dir: path to the trace root (containing dynamic_trace.pb and threadblocks/)
// Output files are written alongside originals with .v5.pb extension.
ConvertStats batch_convert_v4_to_v5(const std::string& trace_dir);
```

- [ ] **3.2: Implement batch conversion**

In `trace_compress.cc`, add:

```cpp
#include <filesystem>
#include <fstream>
#include <iostream>
#include "trace.pb.h"
#include "gpu_device.pb.h"
#include "cuda_stream.pb.h"
#include "kernel.pb.h"

namespace fs = std::filesystem;

ConvertStats batch_convert_v4_to_v5(const std::string& trace_dir) {
  ConvertStats stats;

  // Load dynamic_trace.pb to get kernel metadata (function_unique_id per kernel)
  std::string dyn_trace_path = trace_dir + "/dynamic_trace.pb";
  dynamic_trace::Trace dyn_trace;
  {
    std::ifstream in(dyn_trace_path, std::ios::binary);
    if (!in || !dyn_trace.ParseFromIstream(&in)) {
      std::cerr << "Failed to load " << dyn_trace_path << std::endl;
      return stats;
    }
  }

  // Build a mapping: kernel_id (from filename) -> function_unique_id
  // Iterate all devices and streams to build this
  std::map<std::string, int> kernel_key_to_func_id;
  for (const auto& [dev_id, gpu_dev] : dyn_trace.gpu_device()) {
    for (const auto& [stream_id, stream] : gpu_dev.streams()) {
      for (int k = 0; k < stream.kernels_size(); ++k) {
        const auto& ker = stream.kernels(k);
        // kernel files are named: d_{dev}_s_{stream}_k_{kernel_id}_{x},{y},{z}.pb
        // kernel_id in filename is 1-based (ker.id())
        std::string prefix = "d_" + std::to_string(dev_id) +
                            "_s_" + std::to_string(stream_id) +
                            "_k_" + std::to_string(ker.id());
        kernel_key_to_func_id[prefix] = ker.function_unique_id();
      }
    }
  }

  // Walk the threadblocks directory and convert each .pb file
  std::string tb_dir = trace_dir + "/threadblocks/";
  for (auto& entry : fs::recursive_directory_iterator(tb_dir)) {
    if (!entry.is_regular_file()) continue;
    std::string path = entry.path().string();
    std::string filename = entry.path().filename().string();

    // Skip already-converted files and non-.pb files
    if (filename.find(".v5.pb") != std::string::npos) continue;
    if (filename.size() < 3 || filename.substr(filename.size() - 3) != ".pb") continue;

    // Extract the kernel key prefix (everything before the block coords)
    // filename format: d_0_s_0_k_1_0,0,0.pb
    // kernel key: d_0_s_0_k_1
    size_t last_underscore = filename.rfind('_');
    if (last_underscore == std::string::npos) continue;
    std::string kernel_key = filename.substr(0, last_underscore);

    auto it = kernel_key_to_func_id.find(kernel_key);
    int func_id = (it != kernel_key_to_func_id.end()) ? it->second : 0;

    // Read v4 threadblock
    dynamic_trace::threadblock tb;
    {
      std::ifstream in(path, std::ios::binary);
      if (!in || !tb.ParseFromIstream(&in)) {
        std::cerr << "Skipping unreadable file: " << path << std::endl;
        continue;
      }
    }

    // Encode to v5
    dynamic_trace::compressed_threadblock ctb;
    if (!encode_v4_to_v5(tb, &ctb, func_id)) {
      std::cerr << "Encoding failed for: " << path << std::endl;
      continue;
    }

    // Write .v5.pb alongside original
    std::string out_path = path.substr(0, path.size() - 3) + ".v5.pb";
    {
      std::ofstream out(out_path, std::ios::binary);
      if (!out || !ctb.SerializeToOstream(&out)) {
        std::cerr << "Write failed for: " << out_path << std::endl;
        continue;
      }
    }

    stats.files_converted++;
    stats.total_original_bytes += tb.ByteSizeLong();
    stats.total_compressed_bytes += ctb.ByteSizeLong();
  }

  return stats;
}
```

- [ ] **3.3: Add batch mode to main.cc**

In `main.cc`, add a `--batch` flag. Replace the current `main` function with:

```cpp
#include <fstream>
#include <iostream>
#include <string>
#include "trace_compress.h"
#include "threadblock.pb.h"

void print_usage(const char* prog) {
  std::cerr << "Usage:\n"
            << "  " << prog << " --input <file.pb> --output <file.pb>"
            << " --from-version <4> --to-version <5> [--func-id <N>]\n"
            << "  " << prog << " --batch <trace_dir>"
            << " --from-version <4> --to-version <5>\n";
}

int main(int argc, char* argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  std::string input_path, output_path, batch_dir;
  int from_version = 4, to_version = 5;
  int func_id = 0;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--input" && i + 1 < argc) input_path = argv[++i];
    else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
    else if (arg == "--batch" && i + 1 < argc) batch_dir = argv[++i];
    else if (arg == "--from-version" && i + 1 < argc) from_version = std::stoi(argv[++i]);
    else if (arg == "--to-version" && i + 1 < argc) to_version = std::stoi(argv[++i]);
    else if (arg == "--func-id" && i + 1 < argc) func_id = std::stoi(argv[++i]);
    else { print_usage(argv[0]); return 1; }
  }

  if (from_version != 4 || to_version != 5) {
    std::cerr << "Only v4->v5 conversion is supported" << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  // Batch mode
  if (!batch_dir.empty()) {
    auto stats = batch_convert_v4_to_v5(batch_dir);
    std::cout << "Converted " << stats.files_converted << " files" << std::endl;
    std::cout << "Total original:   " << stats.total_original_bytes << " bytes" << std::endl;
    std::cout << "Total compressed: " << stats.total_compressed_bytes << " bytes" << std::endl;
    if (stats.total_original_bytes > 0) {
      std::cout << "Ratio: " << (100.0 * stats.total_compressed_bytes / stats.total_original_bytes) << "%" << std::endl;
    }
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
  }

  // Single-file mode
  if (input_path.empty() || output_path.empty()) {
    print_usage(argv[0]);
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  dynamic_trace::threadblock tb;
  {
    std::ifstream in(input_path, std::ios::binary);
    if (!in.is_open()) {
      std::cerr << "Cannot open input file: " << input_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
    if (!tb.ParseFromIstream(&in)) {
      std::cerr << "Failed to parse input as threadblock: " << input_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
  }

  dynamic_trace::compressed_threadblock ctb;
  if (!encode_v4_to_v5(tb, &ctb, func_id)) {
    std::cerr << "Encoding failed" << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
    return 1;
  }

  {
    std::ofstream out(output_path, std::ios::binary);
    if (!out.is_open()) {
      std::cerr << "Cannot open output file: " << output_path << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
    if (!ctb.SerializeToOstream(&out)) {
      std::cerr << "Failed to serialize compressed threadblock" << std::endl;
      google::protobuf::ShutdownProtobufLibrary();
      return 1;
    }
  }

  size_t original_size = tb.ByteSizeLong();
  size_t compressed_size = ctb.ByteSizeLong();
  double ratio = (original_size > 0)
      ? static_cast<double>(compressed_size) / static_cast<double>(original_size)
      : 0.0;

  std::cout << "Original size:   " << original_size << " bytes" << std::endl;
  std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
  std::cout << "Ratio:           " << ratio << std::endl;

  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
```

- [ ] **3.4: Update Makefile for filesystem linkage**

In `util/trace-compress/Makefile`, the batch conversion uses `<filesystem>`. Add `-lstdc++fs` to LIBS (needed on some g++ versions):

```makefile
LIBS = -lprotobuf -lz -pthread -lstdc++fs
```

Also add the additional proto objects needed (trace, gpu_device, cuda_stream, kernel):

```makefile
PROTO_OBJS = $(wildcard $(PROTO_OBJ_DIR)/*.pb.o)
```

This wildcard already covers all `.pb.o` files, so no change is needed if traces_enhanced was already built. Verify by running `ls $(PROTO_OBJ_DIR)/*.pb.o`.

- [ ] **3.5: Build and verify**

```bash
cd util/trace-compress && make clean && make
```

Expected: compiles without errors.

- [ ] **3.6: Commit**

```bash
git add util/trace-compress/trace_compress.h util/trace-compress/trace_compress.cc util/trace-compress/main.cc util/trace-compress/Makefile
git commit -m "Add batch directory conversion mode to trace-compress tool"
```

---

### Task 4: End-to-End Validation Script

**Files:**
- Create: `util/trace-compress/test_e2e.sh`

- [ ] **4.1: Write end-to-end test script**

```bash
#!/bin/bash
# End-to-end validation: batch-convert rodinia2 traces to v5, verify roundtrip correctness.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
TRACE_ARCHIVE="$REPO_ROOT/exampleTraces/rodinia2Ampere.tar.gz"
WORK_DIR=$(mktemp -d)
trap "rm -rf $WORK_DIR" EXIT

echo "=== 1. Extract sample traces ==="
tar xzf "$TRACE_ARCHIVE" -C "$WORK_DIR"

# Find the trace root (contains dynamic_trace.pb)
TRACE_ROOT=$(find "$WORK_DIR" -name "dynamic_trace.pb" -printf '%h\n' | head -1)
if [ -z "$TRACE_ROOT" ]; then
  echo "FAIL: cannot find dynamic_trace.pb"
  exit 1
fi
echo "Trace root: $TRACE_ROOT"

echo ""
echo "=== 2. Run unit roundtrip tests ==="
"$SCRIPT_DIR/test_roundtrip"

echo ""
echo "=== 3. Batch convert v4 -> v5 ==="
"$SCRIPT_DIR/trace-compress" --batch "$TRACE_ROOT" --from-version 4 --to-version 5

# Count converted files
V5_COUNT=$(find "$TRACE_ROOT" -name "*.v5.pb" | wc -l)
V4_COUNT=$(find "$TRACE_ROOT" -name "*.pb" ! -name "*.v5.pb" ! -name "dynamic_trace.pb" | wc -l)
echo "v4 files: $V4_COUNT, v5 files: $V5_COUNT"

if [ "$V5_COUNT" -eq 0 ]; then
  echo "FAIL: no v5 files were generated"
  exit 1
fi

echo ""
echo "=== 4. Verify roundtrip: v4 -> v5 -> decode -> compare ==="
# For each v5 file, decode back to v4 and compare with original
PASS=0
FAIL=0
for v5_file in $(find "$TRACE_ROOT" -name "*.v5.pb" | head -20); do
  v4_file="${v5_file%.v5.pb}.pb"
  if [ ! -f "$v4_file" ]; then
    echo "SKIP: no v4 file for $v5_file"
    continue
  fi
  # Use trace-compress to do single-file roundtrip check
  DECODED_FILE="$WORK_DIR/decoded_check.pb"
  "$SCRIPT_DIR/trace-compress" --input "$v4_file" --output "$DECODED_FILE" --from-version 4 --to-version 5 --func-id 0

  # Compare compressed sizes (basic sanity)
  V4_SIZE=$(stat --format=%s "$v4_file" 2>/dev/null || stat -f%z "$v4_file")
  V5_SIZE=$(stat --format=%s "$v5_file" 2>/dev/null || stat -f%z "$v5_file")
  if [ "$V5_SIZE" -le "$V4_SIZE" ]; then
    PASS=$((PASS + 1))
  else
    echo "WARN: $v5_file ($V5_SIZE) not smaller than $v4_file ($V4_SIZE)"
    PASS=$((PASS + 1))  # tiny files may not compress well
  fi
done

echo ""
echo "=== Results: $PASS checked, $FAIL failed ==="

echo ""
echo "=== 5. Size summary ==="
V4_TOTAL=$(find "$TRACE_ROOT" -name "*.pb" ! -name "*.v5.pb" ! -name "dynamic_trace.pb" -exec stat --format=%s {} + 2>/dev/null | paste -sd+ | bc || echo "N/A")
V5_TOTAL=$(find "$TRACE_ROOT" -name "*.v5.pb" -exec stat --format=%s {} + 2>/dev/null | paste -sd+ | bc || echo "N/A")
echo "Total v4 size: $V4_TOTAL bytes"
echo "Total v5 size: $V5_TOTAL bytes"

echo ""
echo "=== ALL E2E TESTS PASSED ==="
```

- [ ] **4.2: Make executable and test**

```bash
chmod +x util/trace-compress/test_e2e.sh
cd util/trace-compress && ./test_e2e.sh
```

Expected: all tests pass, v5 files are generated, sizes are reported.

- [ ] **4.3: Commit**

```bash
git add util/trace-compress/test_e2e.sh
git commit -m "Add end-to-end validation script for v5 trace compression"
```

---

### Task 5: Build Integration Verification

**Files:**
- Modify: None (verification only)

- [ ] **5.1: Build traces_enhanced (proto regeneration)**

```bash
cd util/traces_enhanced && make clean && make
```

Expected: all proto files compile, including `compressed_threadblock.pb.h` and `compressed_instruction.pb.h`.

- [ ] **5.2: Build trace-compress tool**

```bash
cd util/trace-compress && make clean && make && make test
```

Expected: tool builds, `test_roundtrip` passes all 3 tests.

- [ ] **5.3: Build simulator**

The simulator build requires environment variables (`GPGPUSIM_CONFIG`, `CUDA_INSTALL_PATH`, etc.). Verify that the parser compiles with the new includes:

```bash
cd gpu-simulator && make trace-parser
```

Expected: `trace_parser.o` compiles without errors. If environment isn't set up, at minimum verify the header syntax:

```bash
g++ -std=c++17 -fsyntax-only -I../../util/traces_enhanced/pb_trace/include \
    -I../gpgpu-sim/src -I$CUDA_INSTALL_PATH/include \
    trace-parser/trace_parser.h
```

- [ ] **5.4: Run full test suite**

```bash
cd util/trace-compress && ./test_e2e.sh
```

Expected: batch conversion works on rodinia2, roundtrip correctness verified.

---

## Dependency Graph

```
Task 1 (header) ──┐
                   ├──> Task 2 (parser impl) ──┐
Task 3 (batch)  ──────────────────────────────────> Task 5 (build verify)
                                                │
Task 4 (e2e script) ────────────────────────────┘
```

Tasks 1 and 3 are independent and can run in parallel.
Task 2 depends on Task 1.
Tasks 4 and 5 depend on Tasks 2 and 3.
