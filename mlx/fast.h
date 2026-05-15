// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>
#include <variant>

#include "mlx/api.h"
#include "mlx/utils.h"

namespace mlx::core::fast {

MLX_API array rms_norm(
    const array& x,
    const std::optional<array>& weight,
    float eps,
    StreamOrDevice s = {});

MLX_API array layer_norm(
    const array& x,
    const std::optional<array>& weight,
    const std::optional<array>& bias,
    float eps,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    int offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

MLX_API array rope(
    const array& x,
    int dims,
    bool traditional,
    std::optional<float> base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs = std::nullopt,
    StreamOrDevice s = {});

/** Computes: O = softmax(Q @ K.T) @ V **/
MLX_API array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::string& mask_mode = "",
    std::optional<array> mask_arr = {},
    const std::optional<array>& sinks = {},
    StreamOrDevice s = {});

using TemplateArg = std::variant<int, bool, Dtype>;
using ScalarArg = std::variant<bool, int, float>;

using CustomKernelFunction = std::function<std::vector<array>(
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::vector<std::pair<std::string, TemplateArg>>,
    std::optional<float>,
    bool,
    StreamOrDevice)>;

MLX_API CustomKernelFunction metal_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    bool atomic_outputs = false);

MLX_API CustomKernelFunction cuda_kernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header = "",
    bool ensure_row_contiguous = true,
    int shared_memory = 0);

MLX_API std::vector<array> precompiled_cuda_kernel(
    const std::string& name,
    const std::string& compiled_source,
    const std::vector<array>& inputs,
    const std::vector<Shape>& output_shapes,
    const std::vector<Dtype>& output_dtypes,
    const std::vector<ScalarArg>& scalars,
    std::tuple<int, int, int> grid,
    std::tuple<int, int, int> threadgroup,
    int shared_memory = 0,
    std::optional<float> init_value = std::nullopt,
    bool ensure_row_contiguous = false,
    StreamOrDevice s = {});

/**
 * Fused SiLU(gate) * up + gather quantized matvec.
 *
 * Computes:  out = (silu(gate) * up) @ w[rhs_indices[k]]^T
 * where the matmul is the same as gather_qmm in matvec (M=1) mode.
 *
 * Requires:
 *   - gate, up: same shape (..., M, K_in), same dtype
 *   - w: (E, N_out, K_in / pack_factor) uint32 (4-bit packed)
 *   - scales, biases: (E, N_out, K_in / group_size) dtype matching gate
 *   - rhs_indices: int32, broadcastable to gate's batch dims
 *
 * Currently only the fast path is implemented (N % 8 == 0 && K % 512 == 0).
 */
MLX_API array fused_swiglu_gather_qmv(
    const array& gate,
    const array& up,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& rhs_indices,
    int group_size = 64,
    int bits = 4,
    const std::string& mode = "affine",
    StreamOrDevice s = {});

} // namespace mlx::core::fast
