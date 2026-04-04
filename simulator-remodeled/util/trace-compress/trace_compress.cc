#include "trace_compress.h"
#include "warp.pb.h"
#include "instruction.pb.h"
#include "address.pb.h"
#include <unordered_map>
#include <vector>

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

// ---------------------------------------------------------------------------
// L2: v5 -> v6 run-length squash
// ---------------------------------------------------------------------------

static bool can_run_encode(const dynamic_trace::compressed_instruction& ci) {
  // Must have PC_DELTA flag and no addresses
  return (ci.flags() & FLAG_PC_DELTA) && ci.addresses_size() == 0;
}

bool encode_v5_to_v6(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v6* dst) {
  if (!dst) return false;
  dst->Clear();

  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }
  dst->set_function_unique_id(src.function_unique_id());

  for (const auto& [wid, src_warp] : src.warps()) {
    dynamic_trace::compressed_warp_v6 dwarp;
    dwarp.set_id(src_warp.id());

    int n = src_warp.instructions_size();
    int i = 0;
    uint32_t abs_pc = 0;  // running absolute PC for the warp

    while (i < n) {
      const auto& ci = src_warp.instructions(i);

      // Compute absolute PC of this instruction from its v5 encoding
      uint32_t cur_abs_pc;
      if (ci.flags() & FLAG_PC_DELTA) {
        cur_abs_pc = abs_pc + ci.pc();
      } else {
        cur_abs_pc = ci.pc();
      }

      // Try to start a run if this instruction qualifies
      if (can_run_encode(ci)) {
        uint32_t run_flags = ci.flags();
        uint32_t run_delta = ci.pc();  // delta step between consecutive instructions
        int run_len = 1;

        while (i + run_len < n) {
          const auto& next = src_warp.instructions(i + run_len);
          if (can_run_encode(next) &&
              next.flags() == run_flags &&
              next.pc() == run_delta) {
            ++run_len;
          } else {
            break;
          }
        }

        if (run_len >= static_cast<int>(MIN_RUN_LENGTH)) {
          // Emit as a run; pc_start is the absolute PC of the first instruction
          auto* run = dwarp.add_runs();
          run->set_pc_start(cur_abs_pc);
          run->set_pc_delta(run_delta);
          run->set_flags(run_flags);
          run->set_count(run_len);
          uint32_t run_idx = dwarp.runs_size() - 1;
          dwarp.add_sequence(SEQ_TAG_RUN_BIT | run_idx);
          // Advance abs_pc past all instructions in this run
          abs_pc = cur_abs_pc + run_delta * (run_len - 1);
          i += run_len;
          continue;
        }
        // Fall through: emit individually
      }

      // Emit as individual instruction
      uint32_t inst_idx = dwarp.instructions_size();
      *dwarp.add_instructions() = ci;
      dwarp.add_sequence(inst_idx);
      abs_pc = cur_abs_pc;
      ++i;
    }

    (*dst->mutable_warps())[wid] = dwarp;
  }

  return true;
}

bool decode_v6_to_v5(const dynamic_trace::compressed_threadblock_v6& src,
                     dynamic_trace::compressed_threadblock* dst) {
  if (!dst) return false;
  dst->Clear();

  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }
  dst->set_function_unique_id(src.function_unique_id());

  for (const auto& [wid, src_warp] : src.warps()) {
    dynamic_trace::compressed_warp dwarp;
    dwarp.set_id(src_warp.id());

    uint32_t abs_pc = 0;          // running absolute PC for the warp
    bool first_inst = true;       // tracks whether any instruction has been emitted

    for (int s = 0; s < src_warp.sequence_size(); ++s) {
      uint32_t tag = src_warp.sequence(s);
      if (tag & SEQ_TAG_RUN_BIT) {
        // Expand a run; run.pc_start() is the absolute PC of the first instruction
        uint32_t run_idx = tag & ~SEQ_TAG_RUN_BIT;
        const auto& run = src_warp.runs(run_idx);
        uint32_t cur_abs_pc = run.pc_start();

        for (uint32_t k = 0; k < run.count(); ++k) {
          auto* ci = dwarp.add_instructions();
          if (k == 0) {
            // Emit delta from prev abs_pc, or absolute if this is the first instruction
            if (first_inst) {
              ci->set_pc(cur_abs_pc);
              ci->set_flags(run.flags() & ~static_cast<uint32_t>(FLAG_PC_DELTA));
            } else {
              ci->set_pc(cur_abs_pc - abs_pc);
              ci->set_flags(run.flags() | FLAG_PC_DELTA);
            }
          } else {
            // Subsequent instructions in run retain the original delta step
            ci->set_pc(run.pc_delta());
            ci->set_flags(run.flags());
          }
          abs_pc = cur_abs_pc;
          first_inst = false;
          cur_abs_pc += run.pc_delta();
        }
      } else {
        // Individual instruction — copy as-is, then update abs_pc
        const auto& ci_src = src_warp.instructions(tag);
        *dwarp.add_instructions() = ci_src;
        if (ci_src.flags() & FLAG_PC_DELTA) {
          abs_pc += ci_src.pc();
        } else {
          abs_pc = ci_src.pc();
        }
        first_inst = false;
      }
    }

    (*dst->mutable_warps())[wid] = dwarp;
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

// ---------------------------------------------------------------------------
// L1: v5 -> v4 decode
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// L3: v5 -> v7 cross-warp PC deduplication
// ---------------------------------------------------------------------------

// Reconstruct absolute PC sequence from v5 delta-encoded instructions
static std::vector<uint32_t> reconstruct_pc_sequence(
    const dynamic_trace::compressed_warp& warp) {
  std::vector<uint32_t> pcs;
  uint32_t prev_pc = 0;
  for (int i = 0; i < warp.instructions_size(); ++i) {
    const auto& ci = warp.instructions(i);
    uint32_t abs_pc;
    if (ci.flags() & FLAG_PC_DELTA) {
      abs_pc = prev_pc + ci.pc();
    } else {
      abs_pc = ci.pc();
    }
    pcs.push_back(abs_pc);
    prev_pc = abs_pc;
  }
  return pcs;
}

bool encode_v5_to_v7(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v7* dst) {
  if (!dst) return false;
  dst->Clear();

  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }
  dst->set_function_unique_id(src.function_unique_id());

  if (src.warps().empty()) return true;

  // Reconstruct absolute PC sequences for all warps
  std::vector<std::pair<int, std::vector<uint32_t>>> warp_pcs;
  for (const auto& [wid, warp] : src.warps()) {
    warp_pcs.emplace_back(wid, reconstruct_pc_sequence(warp));
  }

  // Use first warp's PC sequence as the shared reference
  const auto& ref_pcs = warp_pcs[0].second;
  for (uint32_t pc : ref_pcs) {
    dst->add_shared_pc_sequence(pc);
  }

  // Encode each warp
  for (const auto& [wid, src_warp] : src.warps()) {
    auto pcs = reconstruct_pc_sequence(src_warp);
    dynamic_trace::warp_diff wdiff;
    wdiff.set_id(src_warp.id());

    for (int i = 0; i < src_warp.instructions_size(); ++i) {
      const auto& ci = src_warp.instructions(i);

      // Record PC override if different from shared sequence
      if (i < static_cast<int>(ref_pcs.size()) && pcs[i] != ref_pcs[i]) {
        auto* ov = wdiff.add_pc_overrides();
        ov->set_instruction_index(i);
        ov->set_actual_pc(pcs[i]);
      } else if (i >= static_cast<int>(ref_pcs.size())) {
        auto* ov = wdiff.add_pc_overrides();
        ov->set_instruction_index(i);
        ov->set_actual_pc(pcs[i]);
      }

      // Create warp_instruction with addresses and mask flags only
      auto* wi = wdiff.add_instructions();
      for (int j = 0; j < ci.addresses_size(); ++j) {
        *wi->add_addresses() = ci.addresses(j);
      }

      // Keep only mask-related flags, strip FLAG_PC_DELTA
      uint32_t flags = ci.flags() & (FLAG_FULL_ACTIVE | FLAG_PRED_EQ_ACTIVE);
      wi->set_flags(flags);

      // Store masks when not elided by flags
      if (!(flags & FLAG_FULL_ACTIVE)) {
        wi->set_active_mask(ci.active_mask());
      }
      if (!(flags & FLAG_PRED_EQ_ACTIVE)) {
        wi->set_predicate_mask(ci.predicate_mask());
      }
    }

    (*dst->mutable_warps())[wid] = wdiff;
  }

  return true;
}

bool decode_v7_to_v5(const dynamic_trace::compressed_threadblock_v7& src,
                     dynamic_trace::compressed_threadblock* dst) {
  if (!dst) return false;
  dst->Clear();

  if (src.has_block_id()) {
    *dst->mutable_block_id() = src.block_id();
  }
  dst->set_function_unique_id(src.function_unique_id());

  // Build shared PC vector
  std::vector<uint32_t> shared_pcs(src.shared_pc_sequence().begin(),
                                    src.shared_pc_sequence().end());

  for (const auto& [wid, wdiff] : src.warps()) {
    dynamic_trace::compressed_warp cwarp;
    cwarp.set_id(wdiff.id());

    // Build override map: instruction_index -> actual_pc
    std::unordered_map<uint32_t, uint32_t> overrides;
    for (const auto& ov : wdiff.pc_overrides()) {
      overrides[ov.instruction_index()] = ov.actual_pc();
    }

    uint32_t prev_pc = 0;

    for (int i = 0; i < wdiff.instructions_size(); ++i) {
      const auto& wi = wdiff.instructions(i);
      auto* ci = cwarp.add_instructions();

      // Determine absolute PC for this instruction
      uint32_t abs_pc;
      auto ov_it = overrides.find(i);
      if (ov_it != overrides.end()) {
        abs_pc = ov_it->second;
      } else if (i < static_cast<int>(shared_pcs.size())) {
        abs_pc = shared_pcs[i];
      } else {
        abs_pc = prev_pc;  // fallback (should not happen in well-formed data)
      }

      // Re-encode as v5 delta
      uint32_t flags = wi.flags();  // mask-related flags from v7
      if (i > 0) {
        flags |= FLAG_PC_DELTA;
        ci->set_pc(abs_pc - prev_pc);
      } else {
        ci->set_pc(abs_pc);
      }
      prev_pc = abs_pc;

      ci->set_flags(flags);

      // Copy masks when not elided
      if (!(flags & FLAG_FULL_ACTIVE)) {
        ci->set_active_mask(wi.active_mask());
      }
      if (!(flags & FLAG_PRED_EQ_ACTIVE)) {
        ci->set_predicate_mask(wi.predicate_mask());
      }

      // Copy addresses
      for (int j = 0; j < wi.addresses_size(); ++j) {
        *ci->add_addresses() = wi.addresses(j);
      }
    }

    (*dst->mutable_warps())[wid] = cwarp;
  }

  return true;
}
