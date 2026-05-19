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
 * Fused quantized 2-pass SDPA with GQA-shared K/V loads.
 *
 * Decode-time scaled-dot-product attention against an already quantized KV
 * cache. Equivalent to:
 *
 *   K = dequantize(q_keys_packed, q_keys_scales, q_keys_biases, group_size,
 * bits) V = dequantize(q_values_packed, q_values_scales, q_values_biases,
 * group_size, bits) out = softmax(Q @ K^T * scale + mask) @ V
 *
 * but executed as a single fused 2-pass kernel that (a) avoids materializing
 * dequantized K/V tensors and (b) cooperatively loads each K/V tile into
 * threadgroup memory once for all gqa_factor query heads.
 *
 * Currently supported: T_q=1 (decode), head_dim=256, bits in {4,8},
 * group_size in {32,64}, gqa_factor=8, causal-only mask.
 */
MLX_API array fused_qsdpa(
    const array& queries,
    const array& q_keys_packed,
    const array& q_keys_scales,
    const array& q_keys_biases,
    const array& q_values_packed,
    const array& q_values_scales,
    const array& q_values_biases,
    float scale,
    const std::optional<array>& mask = std::nullopt,
    int group_size = 64,
    int bits = 4,
    int head_dim = 256,
    int gqa_factor = 8,
    bool do_causal = false,
    const std::optional<array>& left_padding = std::nullopt,
    const std::string& mode = "affine",
    StreamOrDevice s = {});

} // namespace mlx::core::fast
