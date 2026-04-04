#include "trace_compress.h"
#include "warp.pb.h"
#include "instruction.pb.h"
#include "address.pb.h"

bool encode_v4_to_v5(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock* dst,
                     int function_unique_id) {
  if (!dst) return false;
  dst->Clear();

  // Copy block_id
  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }

  // Hoist function_unique_id to threadblock level
  dst->set_function_unique_id(function_unique_id);

  // Encode each warp
  for (const auto& [wid, src_warp] : src.warps()) {
    dynamic_trace::compressed_warp cwarp;
    cwarp.set_id(src_warp.id());

    uint32_t prev_pc = 0;

    for (int i = 0; i < src_warp.instructions_size(); ++i) {
      const auto& inst = src_warp.instructions(i);
      auto* cinst = cwarp.add_instructions();

      uint32_t flags = 0;

      // PC delta encoding: first instruction uses absolute PC
      if (i > 0) {
        flags |= FLAG_PC_DELTA;
        cinst->set_pc(inst.pc() - prev_pc);
      } else {
        cinst->set_pc(inst.pc());
      }
      prev_pc = inst.pc();

      // Full active mask elision
      if (inst.active_mask() == 0xFFFFFFFF) {
        flags |= FLAG_FULL_ACTIVE;
      } else {
        cinst->set_active_mask(inst.active_mask());
      }

      // Predicate == active mask elision
      if (inst.predicate_mask() == inst.active_mask()) {
        flags |= FLAG_PRED_EQ_ACTIVE;
      } else {
        cinst->set_predicate_mask(inst.predicate_mask());
      }

      cinst->set_flags(flags);

      // Copy addresses unchanged
      for (int j = 0; j < inst.addresses_size(); ++j) {
        *cinst->add_addresses() = inst.addresses(j);
      }
    }

    (*dst->mutable_warps())[wid] = cwarp;
  }

  return true;
}

bool decode_v5_to_v4(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::threadblock* dst) {
  if (!dst) return false;
  dst->Clear();

  // Copy block_id
  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }

  int function_unique_id = src.function_unique_id();

  // Decode each warp
  for (const auto& [wid, cwarp] : src.warps()) {
    dynamic_trace::warp warp;
    warp.set_id(cwarp.id());

    uint32_t prev_pc = 0;

    for (int i = 0; i < cwarp.instructions_size(); ++i) {
      const auto& cinst = cwarp.instructions(i);
      auto* inst = warp.add_instructions();

      uint32_t flags = cinst.flags();

      // Restore PC
      uint32_t pc;
      if (flags & FLAG_PC_DELTA) {
        pc = prev_pc + cinst.pc();
      } else {
        pc = cinst.pc();
      }
      inst->set_pc(pc);
      prev_pc = pc;

      // Restore active_mask
      uint32_t active_mask;
      if (flags & FLAG_FULL_ACTIVE) {
        active_mask = 0xFFFFFFFF;
      } else {
        active_mask = cinst.active_mask();
      }
      inst->set_active_mask(active_mask);

      // Restore predicate_mask
      if (flags & FLAG_PRED_EQ_ACTIVE) {
        inst->set_predicate_mask(active_mask);
      } else {
        inst->set_predicate_mask(cinst.predicate_mask());
      }

      // Restore function_unique_id from threadblock level
      inst->set_function_unique_id(function_unique_id);

      // Copy addresses unchanged
      for (int j = 0; j < cinst.addresses_size(); ++j) {
        *inst->add_addresses() = cinst.addresses(j);
      }
    }

    (*dst->mutable_warps())[wid] = warp;
  }

  return true;
}
