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

#endif
