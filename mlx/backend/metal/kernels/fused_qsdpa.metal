// Copyright © 2026 (WS1 fused_qsdpa).
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/fused_qsdpa.h"

// Pass1 template arguments: (T, D, GROUP_SIZE, BITS, GQA_FACTOR)
#define instantiate_fused_qsdpa_pass1(type, d, gs, bits, gqa)                  \
  instantiate_kernel(                                                          \
      "fused_qsdpa_pass1_" #type "_d_" #d "_gs_" #gs "_b_" #bits "_gqa_" #gqa, \
      fused_qsdpa_pass1, type, d, gs, bits, gqa)

#define instantiate_fused_qsdpa_pass2(type, d)             \
  instantiate_kernel(                                      \
      "fused_qsdpa_pass2_" #type "_d_" #d,                 \
      fused_qsdpa_pass2, type, d)

// Limit instantiations to the cross-product needed by Qwen3.6:
//   D=256, GROUP_SIZE in {32,64}, BITS in {4,8}, GQA_FACTOR=8.
#define instantiate_fused_qsdpa_for_type(type)                  \
  instantiate_fused_qsdpa_pass1(type, 256, 32, 4, 8)            \
  instantiate_fused_qsdpa_pass1(type, 256, 32, 8, 8)            \
  instantiate_fused_qsdpa_pass1(type, 256, 64, 4, 8)            \
  instantiate_fused_qsdpa_pass1(type, 256, 64, 8, 8)            \
  instantiate_fused_qsdpa_pass2(type, 256)

instantiate_fused_qsdpa_for_type(bfloat16_t)
instantiate_fused_qsdpa_for_type(float16_t)
instantiate_fused_qsdpa_for_type(float)
    // clang-format on
