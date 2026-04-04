#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "trace_compress.h"
#include "threadblock.pb.h"
#include "compressed_threadblock.pb.h"
#include "warp.pb.h"
#include "instruction.pb.h"
#include "address.pb.h"
#include "dim3d.pb.h"

static const int kFuncId = 42;

// Build a synthetic v4 threadblock with 2 warps, 5 sequential instructions each.
// Instructions 1 and 3 carry memory addresses (base_address + stride).
dynamic_trace::threadblock make_test_threadblock() {
  dynamic_trace::threadblock tb;
  auto* bid = tb.mutable_block_id();
  bid->set_x(1);
  bid->set_y(2);
  bid->set_z(0);

  for (int w = 0; w < 2; ++w) {
    dynamic_trace::warp warp;
    warp.set_id(w);
    for (int i = 0; i < 5; ++i) {
      auto* inst = warp.add_instructions();
      inst->set_pc(0x1000 + i * 4);
      inst->set_active_mask(0xFFFFFFFF);
      inst->set_predicate_mask(0xFFFFFFFF);
      inst->set_function_unique_id(kFuncId);
      if (i == 1 || i == 3) {
        auto* addr = inst->add_addresses();
        addr->set_base_address(0xDEAD0000 + i * 0x100);
        addr->set_stride(4);
        addr->set_data_width(4);
      }
    }
    (*tb.mutable_warps())[w] = warp;
  }
  return tb;
}

// Build a threadblock with divergent behavior: varying active masks,
// non-sequential PC, differing predicate mask.
dynamic_trace::threadblock make_divergent_threadblock() {
  dynamic_trace::threadblock tb;
  auto* bid = tb.mutable_block_id();
  bid->set_x(3);
  bid->set_y(0);
  bid->set_z(1);

  dynamic_trace::warp warp;
  warp.set_id(0);

  // Instruction 0: all active, sequential start
  {
    auto* inst = warp.add_instructions();
    inst->set_pc(0x2000);
    inst->set_active_mask(0xFFFFFFFF);
    inst->set_predicate_mask(0xFFFFFFFF);
    inst->set_function_unique_id(kFuncId);
  }
  // Instruction 1: partial active mask, predicate equals active
  {
    auto* inst = warp.add_instructions();
    inst->set_pc(0x2004);
    inst->set_active_mask(0x0000FFFF);
    inst->set_predicate_mask(0x0000FFFF);
    inst->set_function_unique_id(kFuncId);
  }
  // Instruction 2: branch jump (non-sequential PC) + different predicate
  {
    auto* inst = warp.add_instructions();
    inst->set_pc(0x3000);
    inst->set_active_mask(0xFFFFFFFF);
    inst->set_predicate_mask(0x00FF00FF);
    inst->set_function_unique_id(kFuncId);
  }

  (*tb.mutable_warps())[0] = warp;
  return tb;
}

// Field-by-field comparison of two threadblocks.
// Returns true if they match; prints first mismatch on failure.
bool compare_threadblocks(const dynamic_trace::threadblock& a,
                          const dynamic_trace::threadblock& b) {
  // block_id
  if (a.has_block_id() != b.has_block_id()) {
    fprintf(stderr, "FAIL: block_id presence mismatch\n");
    return false;
  }
  if (a.has_block_id()) {
    if (a.block_id().x() != b.block_id().x() ||
        a.block_id().y() != b.block_id().y() ||
        a.block_id().z() != b.block_id().z()) {
      fprintf(stderr, "FAIL: block_id mismatch (%d,%d,%d) vs (%d,%d,%d)\n",
              a.block_id().x(), a.block_id().y(), a.block_id().z(),
              b.block_id().x(), b.block_id().y(), b.block_id().z());
      return false;
    }
  }

  // warp count
  if (a.warps_size() != b.warps_size()) {
    fprintf(stderr, "FAIL: warps_size %d vs %d\n", a.warps_size(), b.warps_size());
    return false;
  }

  for (const auto& [wid, wa] : a.warps()) {
    auto it = b.warps().find(wid);
    if (it == b.warps().end()) {
      fprintf(stderr, "FAIL: warp %d missing in second threadblock\n", wid);
      return false;
    }
    const auto& wb = it->second;

    if (wa.instructions_size() != wb.instructions_size()) {
      fprintf(stderr, "FAIL: warp %d instructions_size %d vs %d\n",
              wid, wa.instructions_size(), wb.instructions_size());
      return false;
    }

    for (int i = 0; i < wa.instructions_size(); ++i) {
      const auto& ia = wa.instructions(i);
      const auto& ib = wb.instructions(i);

      if (ia.pc() != ib.pc()) {
        fprintf(stderr, "FAIL: warp %d inst %d pc 0x%x vs 0x%x\n",
                wid, i, ia.pc(), ib.pc());
        return false;
      }
      if (ia.active_mask() != ib.active_mask()) {
        fprintf(stderr, "FAIL: warp %d inst %d active_mask 0x%x vs 0x%x\n",
                wid, i, ia.active_mask(), ib.active_mask());
        return false;
      }
      if (ia.predicate_mask() != ib.predicate_mask()) {
        fprintf(stderr, "FAIL: warp %d inst %d predicate_mask 0x%x vs 0x%x\n",
                wid, i, ia.predicate_mask(), ib.predicate_mask());
        return false;
      }
      if (ia.function_unique_id() != ib.function_unique_id()) {
        fprintf(stderr, "FAIL: warp %d inst %d function_unique_id %d vs %d\n",
                wid, i, ia.function_unique_id(), ib.function_unique_id());
        return false;
      }
      if (ia.addresses_size() != ib.addresses_size()) {
        fprintf(stderr, "FAIL: warp %d inst %d addresses_size %d vs %d\n",
                wid, i, ia.addresses_size(), ib.addresses_size());
        return false;
      }
      for (int j = 0; j < ia.addresses_size(); ++j) {
        const auto& aa = ia.addresses(j);
        const auto& ab = ib.addresses(j);
        if (aa.base_address() != ab.base_address()) {
          fprintf(stderr, "FAIL: warp %d inst %d addr %d base_address mismatch\n",
                  wid, i, j);
          return false;
        }
        if (aa.stride() != ab.stride()) {
          fprintf(stderr, "FAIL: warp %d inst %d addr %d stride mismatch\n",
                  wid, i, j);
          return false;
        }
        if (aa.data_width() != ab.data_width()) {
          fprintf(stderr, "FAIL: warp %d inst %d addr %d data_width mismatch\n",
                  wid, i, j);
          return false;
        }
      }
    }
  }
  return true;
}

bool test_v5_roundtrip() {
  auto original = make_test_threadblock();
  dynamic_trace::compressed_threadblock compressed;
  if (!encode_v4_to_v5(original, &compressed, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false\n");
    return false;
  }
  dynamic_trace::threadblock decoded;
  if (!decode_v5_to_v4(compressed, &decoded)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false\n");
    return false;
  }
  return compare_threadblocks(original, decoded);
}

bool test_v5_divergent() {
  auto original = make_divergent_threadblock();
  dynamic_trace::compressed_threadblock compressed;
  if (!encode_v4_to_v5(original, &compressed, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false (divergent)\n");
    return false;
  }
  dynamic_trace::threadblock decoded;
  if (!decode_v5_to_v4(compressed, &decoded)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false (divergent)\n");
    return false;
  }
  return compare_threadblocks(original, decoded);
}

bool test_v5_compression_ratio() {
  auto original = make_test_threadblock();
  dynamic_trace::compressed_threadblock compressed;
  if (!encode_v4_to_v5(original, &compressed, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false (ratio test)\n");
    return false;
  }
  size_t orig_size = original.ByteSizeLong();
  size_t comp_size = compressed.ByteSizeLong();
  printf("  original=%zu bytes, compressed=%zu bytes, ratio=%.2f%%\n",
         orig_size, comp_size, 100.0 * comp_size / orig_size);
  if (comp_size >= orig_size) {
    fprintf(stderr, "FAIL: compressed (%zu) not smaller than original (%zu)\n",
            comp_size, orig_size);
    return false;
  }
  return true;
}

bool test_v6_roundtrip() {
  auto tb_orig = make_test_threadblock();

  // v4 -> v5 -> v6
  dynamic_trace::compressed_threadblock v5;
  if (!encode_v4_to_v5(tb_orig, &v5, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false\n");
    return false;
  }

  dynamic_trace::compressed_threadblock_v6 v6;
  if (!encode_v5_to_v6(v5, &v6)) {
    fprintf(stderr, "FAIL: encode_v5_to_v6 returned false\n");
    return false;
  }

  // v6 -> v5 -> v4
  dynamic_trace::compressed_threadblock v5_back;
  if (!decode_v6_to_v5(v6, &v5_back)) {
    fprintf(stderr, "FAIL: decode_v6_to_v5 returned false\n");
    return false;
  }

  dynamic_trace::threadblock tb_decoded;
  if (!decode_v5_to_v4(v5_back, &tb_decoded)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false\n");
    return false;
  }

  return compare_threadblocks(tb_orig, tb_decoded);
}

bool test_v6_with_long_run() {
  // Create a threadblock with 20 sequential, all-active, no-address instructions
  dynamic_trace::threadblock tb;
  tb.mutable_block_id()->set_x(0);
  tb.mutable_block_id()->set_y(0);
  tb.mutable_block_id()->set_z(0);

  dynamic_trace::warp warp;
  warp.set_id(0);
  for (int i = 0; i < 20; i++) {
    auto* inst = warp.add_instructions();
    inst->set_pc(0x1000 + i * 4);
    inst->set_active_mask(0xFFFFFFFF);
    inst->set_predicate_mask(0xFFFFFFFF);
    inst->set_function_unique_id(1);
    // No addresses: ALU-type instructions
  }
  (*tb.mutable_warps())[0] = warp;

  // Encode v4 -> v5 -> v6
  dynamic_trace::compressed_threadblock v5;
  if (!encode_v4_to_v5(tb, &v5, 1)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false\n");
    return false;
  }

  dynamic_trace::compressed_threadblock_v6 v6;
  if (!encode_v5_to_v6(v5, &v6)) {
    fprintf(stderr, "FAIL: encode_v5_to_v6 returned false\n");
    return false;
  }

  // Verify runs were created
  bool found_run = false;
  for (const auto& [wid, w] : v6.warps()) {
    if (w.runs_size() > 0) {
      found_run = true;
      if (w.runs(0).count() < MIN_RUN_LENGTH) {
        fprintf(stderr, "FAIL: run count %u < MIN_RUN_LENGTH %u\n",
                w.runs(0).count(), MIN_RUN_LENGTH);
        return false;
      }
    }
  }
  if (!found_run) {
    fprintf(stderr, "FAIL: no runs created for 20 sequential instructions\n");
    return false;
  }

  // Verify roundtrip
  dynamic_trace::compressed_threadblock v5_back;
  if (!decode_v6_to_v5(v6, &v5_back)) {
    fprintf(stderr, "FAIL: decode_v6_to_v5 returned false\n");
    return false;
  }
  dynamic_trace::threadblock tb_back;
  if (!decode_v5_to_v4(v5_back, &tb_back)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false\n");
    return false;
  }
  return compare_threadblocks(tb, tb_back);
}

int main() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  int passed = 0, failed = 0;

  auto run = [&](const char* name, bool (*fn)()) {
    printf("Running %s ... ", name);
    if (fn()) {
      printf("PASSED\n");
      ++passed;
    } else {
      printf("FAILED\n");
      ++failed;
    }
  };

  run("test_v5_roundtrip", test_v5_roundtrip);
  run("test_v5_divergent", test_v5_divergent);
  run("test_v5_compression_ratio", test_v5_compression_ratio);
  run("test_v6_roundtrip", test_v6_roundtrip);
  run("test_v6_with_long_run", test_v6_with_long_run);

  printf("\n%d passed, %d failed\n", passed, failed);

  google::protobuf::ShutdownProtobufLibrary();
  return failed == 0 ? 0 : 1;
}
