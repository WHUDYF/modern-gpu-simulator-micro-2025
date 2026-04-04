#ifndef TRACE_COMPRESS_H
#define TRACE_COMPRESS_H

#include <string>
#include <vector>
#include "threadblock.pb.h"
#include "compressed_threadblock.pb.h"
#include "compressed_instruction.pb.h"

// Flags bit constants
constexpr uint32_t FLAG_FULL_ACTIVE    = 1 << 0;
constexpr uint32_t FLAG_PRED_EQ_ACTIVE = 1 << 1;
constexpr uint32_t FLAG_PC_DELTA       = 1 << 2;

// Encode a v4 threadblock into v5 compressed_threadblock
bool encode_v4_to_v5(const dynamic_trace::threadblock& src,
                     dynamic_trace::compressed_threadblock* dst,
                     int function_unique_id);

// Decode a v5 compressed_threadblock back to v4 format (for testing)
bool decode_v5_to_v4(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::threadblock* dst);

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

// Warp PC divergence threshold
constexpr double WARP_DIVERGENCE_THRESHOLD = 0.10;

// L3: Encode v5 -> v7 (cross-warp PC dedup)
bool encode_v5_to_v7(const dynamic_trace::compressed_threadblock& src,
                     dynamic_trace::compressed_threadblock_v7* dst);

// L3: Decode v7 -> v5
bool decode_v7_to_v5(const dynamic_trace::compressed_threadblock_v7& src,
                     dynamic_trace::compressed_threadblock* dst);

#endif
