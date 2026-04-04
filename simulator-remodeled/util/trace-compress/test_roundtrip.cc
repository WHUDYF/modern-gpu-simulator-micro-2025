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

bool test_v7_roundtrip() {
  auto tb_orig = make_test_threadblock();

  // v4 -> v5
  dynamic_trace::compressed_threadblock v5;
  if (!encode_v4_to_v5(tb_orig, &v5, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false\n");
    return false;
  }

  // v5 -> v7
  dynamic_trace::compressed_threadblock_v7 v7;
  if (!encode_v5_to_v7(v5, &v7)) {
    fprintf(stderr, "FAIL: encode_v5_to_v7 returned false\n");
    return false;
  }

  // Verify shared_pc_sequence has 5 entries starting at 0x1000
  if (v7.shared_pc_sequence_size() != 5) {
    fprintf(stderr, "FAIL: shared_pc_sequence size %d, expected 5\n",
            v7.shared_pc_sequence_size());
    return false;
  }
  if (v7.shared_pc_sequence(0) != 0x1000) {
    fprintf(stderr, "FAIL: shared_pc_sequence[0] = 0x%x, expected 0x1000\n",
            v7.shared_pc_sequence(0));
    return false;
  }

  // v7 -> v5 -> v4
  dynamic_trace::compressed_threadblock v5_back;
  if (!decode_v7_to_v5(v7, &v5_back)) {
    fprintf(stderr, "FAIL: decode_v7_to_v5 returned false\n");
    return false;
  }

  dynamic_trace::threadblock tb_decoded;
  if (!decode_v5_to_v4(v5_back, &tb_decoded)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false\n");
    return false;
  }

  return compare_threadblocks(tb_orig, tb_decoded);
}

bool test_v7_divergent_warps() {
  // Create a threadblock with 2 warps where warp 1 diverges at instruction 2
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
      // Warp 1 branches to 0x5000 at instruction 2 instead of 0x1008
      if (w == 1 && i == 2) {
        inst->set_pc(0x5000);
      } else {
        inst->set_pc(0x1000 + i * 4);
      }
      inst->set_active_mask(0xFFFFFFFF);
      inst->set_predicate_mask(0xFFFFFFFF);
      inst->set_function_unique_id(kFuncId);
    }
    (*tb.mutable_warps())[w] = warp;
  }

  // v4 -> v5
  dynamic_trace::compressed_threadblock v5;
  if (!encode_v4_to_v5(tb, &v5, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 returned false\n");
    return false;
  }

  // v5 -> v7
  dynamic_trace::compressed_threadblock_v7 v7;
  if (!encode_v5_to_v7(v5, &v7)) {
    fprintf(stderr, "FAIL: encode_v5_to_v7 returned false\n");
    return false;
  }

  // Verify warp 1 has pc_overrides
  auto it = v7.warps().find(1);
  if (it == v7.warps().end()) {
    fprintf(stderr, "FAIL: warp 1 not found in v7\n");
    return false;
  }
  if (it->second.pc_overrides_size() == 0) {
    fprintf(stderr, "FAIL: warp 1 should have pc_overrides for divergent PC\n");
    return false;
  }

  // Verify roundtrip: v7 -> v5 -> v4
  dynamic_trace::compressed_threadblock v5_back;
  if (!decode_v7_to_v5(v7, &v5_back)) {
    fprintf(stderr, "FAIL: decode_v7_to_v5 returned false\n");
    return false;
  }

  dynamic_trace::threadblock tb_decoded;
  if (!decode_v5_to_v4(v5_back, &tb_decoded)) {
    fprintf(stderr, "FAIL: decode_v5_to_v4 returned false\n");
    return false;
  }

  return compare_threadblocks(tb, tb_decoded);
}

bool test_v8_roundtrip() {
  // Create two threadblocks; the second has block_id.x=1 and all addresses
  // shifted by +0x1000
  auto tb1_v4 = make_test_threadblock();
  auto tb2_v4 = make_test_threadblock();
  tb2_v4.mutable_block_id()->set_x(1);

  // Shift all addresses in tb2 by +0x1000
  for (auto& [wid, warp] : *tb2_v4.mutable_warps()) {
    for (int i = 0; i < warp.instructions_size(); ++i) {
      auto* inst = warp.mutable_instructions(i);
      for (int j = 0; j < inst->addresses_size(); ++j) {
        auto* addr = inst->mutable_addresses(j);
        addr->set_base_address(addr->base_address() + 0x1000);
      }
    }
  }

  // Encode both to v7 via v4 -> v5 -> v7
  dynamic_trace::compressed_threadblock v5_1, v5_2;
  if (!encode_v4_to_v5(tb1_v4, &v5_1, kFuncId) ||
      !encode_v4_to_v5(tb2_v4, &v5_2, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 failed\n");
    return false;
  }

  dynamic_trace::compressed_threadblock_v7 v7_1, v7_2;
  if (!encode_v5_to_v7(v5_1, &v7_1) || !encode_v5_to_v7(v5_2, &v7_2)) {
    fprintf(stderr, "FAIL: encode_v5_to_v7 failed\n");
    return false;
  }

  // Encode to v8
  std::vector<dynamic_trace::compressed_threadblock_v7> v7s = {v7_1, v7_2};
  dynamic_trace::compressed_kernel_v8 v8;
  if (!encode_kernel_to_v8(v7s, &v8)) {
    fprintf(stderr, "FAIL: encode_kernel_to_v8 failed\n");
    return false;
  }

  // Verify structure
  if (v8.delta_threadblocks_size() != 1) {
    fprintf(stderr, "FAIL: expected 1 delta_threadblock, got %d\n",
            v8.delta_threadblocks_size());
    return false;
  }
  const auto& d = v8.delta_threadblocks(0);
  if (d.global_address_offset() != 0x1000) {
    fprintf(stderr, "FAIL: expected global_address_offset=0x1000, got %ld\n",
            (long)d.global_address_offset());
    return false;
  }
  if (d.is_full_encoding()) {
    fprintf(stderr, "FAIL: expected is_full_encoding=false\n");
    return false;
  }

  // Decode v8 -> v7 vector, then v7 -> v5 -> v4 and compare
  std::vector<dynamic_trace::compressed_threadblock_v7> decoded_v7s;
  if (!decode_v8_to_v7s(v8, &decoded_v7s)) {
    fprintf(stderr, "FAIL: decode_v8_to_v7s failed\n");
    return false;
  }
  if (decoded_v7s.size() != 2) {
    fprintf(stderr, "FAIL: expected 2 decoded v7s, got %zu\n", decoded_v7s.size());
    return false;
  }

  // Decode each v7 back to v4 and compare
  dynamic_trace::threadblock originals[2] = {tb1_v4, tb2_v4};
  for (int k = 0; k < 2; ++k) {
    dynamic_trace::compressed_threadblock v5_back;
    if (!decode_v7_to_v5(decoded_v7s[k], &v5_back)) {
      fprintf(stderr, "FAIL: decode_v7_to_v5 failed for tb %d\n", k);
      return false;
    }
    dynamic_trace::threadblock v4_back;
    if (!decode_v5_to_v4(v5_back, &v4_back)) {
      fprintf(stderr, "FAIL: decode_v5_to_v4 failed for tb %d\n", k);
      return false;
    }
    if (!compare_threadblocks(originals[k], v4_back)) {
      fprintf(stderr, "FAIL: threadblock %d mismatch after v8 roundtrip\n", k);
      return false;
    }
  }

  return true;
}

bool test_v8_fallback() {
  // Create two threadblocks where the second has completely different addresses
  auto tb1_v4 = make_test_threadblock();
  auto tb2_v4 = make_test_threadblock();
  tb2_v4.mutable_block_id()->set_x(5);

  // Replace all addresses in tb2 with completely different values
  int addr_idx = 0;
  for (auto& [wid, warp] : *tb2_v4.mutable_warps()) {
    for (int i = 0; i < warp.instructions_size(); ++i) {
      auto* inst = warp.mutable_instructions(i);
      for (int j = 0; j < inst->addresses_size(); ++j) {
        auto* addr = inst->mutable_addresses(j);
        // Each address gets a unique, wildly different value
        addr->set_base_address(0xBEEF0000 + addr_idx * 0x77777);
        ++addr_idx;
      }
    }
  }

  // Encode both to v7
  dynamic_trace::compressed_threadblock v5_1, v5_2;
  if (!encode_v4_to_v5(tb1_v4, &v5_1, kFuncId) ||
      !encode_v4_to_v5(tb2_v4, &v5_2, kFuncId)) {
    fprintf(stderr, "FAIL: encode_v4_to_v5 failed\n");
    return false;
  }

  dynamic_trace::compressed_threadblock_v7 v7_1, v7_2;
  if (!encode_v5_to_v7(v5_1, &v7_1) || !encode_v5_to_v7(v5_2, &v7_2)) {
    fprintf(stderr, "FAIL: encode_v5_to_v7 failed\n");
    return false;
  }

  // Encode to v8
  std::vector<dynamic_trace::compressed_threadblock_v7> v7s = {v7_1, v7_2};
  dynamic_trace::compressed_kernel_v8 v8;
  if (!encode_kernel_to_v8(v7s, &v8)) {
    fprintf(stderr, "FAIL: encode_kernel_to_v8 failed\n");
    return false;
  }

  // The second threadblock should fall back to full encoding
  if (v8.delta_threadblocks_size() != 1) {
    fprintf(stderr, "FAIL: expected 1 delta_threadblock, got %d\n",
            v8.delta_threadblocks_size());
    return false;
  }
  if (!v8.delta_threadblocks(0).is_full_encoding()) {
    fprintf(stderr, "FAIL: expected is_full_encoding=true for divergent tb\n");
    return false;
  }

  // Verify roundtrip still works
  std::vector<dynamic_trace::compressed_threadblock_v7> decoded_v7s;
  if (!decode_v8_to_v7s(v8, &decoded_v7s)) {
    fprintf(stderr, "FAIL: decode_v8_to_v7s failed\n");
    return false;
  }

  dynamic_trace::threadblock originals[2] = {tb1_v4, tb2_v4};
  for (int k = 0; k < 2; ++k) {
    dynamic_trace::compressed_threadblock v5_back;
    if (!decode_v7_to_v5(decoded_v7s[k], &v5_back)) {
      fprintf(stderr, "FAIL: decode_v7_to_v5 failed for tb %d\n", k);
      return false;
    }
    dynamic_trace::threadblock v4_back;
    if (!decode_v5_to_v4(v5_back, &v4_back)) {
      fprintf(stderr, "FAIL: decode_v5_to_v4 failed for tb %d\n", k);
      return false;
    }
    if (!compare_threadblocks(originals[k], v4_back)) {
      fprintf(stderr, "FAIL: threadblock %d mismatch after v8 fallback roundtrip\n", k);
      return false;
    }
  }

  return true;
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
  run("test_v7_roundtrip", test_v7_roundtrip);
  run("test_v7_divergent_warps", test_v7_divergent_warps);
  run("test_v8_roundtrip", test_v8_roundtrip);
  run("test_v8_fallback", test_v8_fallback);

  printf("\n%d passed, %d failed\n", passed, failed);

  google::protobuf::ShutdownProtobufLibrary();
  return failed == 0 ? 0 : 1;
}
