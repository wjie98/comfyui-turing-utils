/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified for ComfyUI Turing Utils: fold SwiGLU into the first pass of the staged
 * ConvRot INT8 activation quantizer.  The FHT, scale, and rounding order
 * intentionally match comfy-kitchen's CUDA implementation.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <cmath>
#include <type_traits>

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

constexpr int kWarpThreads = 32;
constexpr int kConvRotGroup = 256;
constexpr int kGroupThreads = 64;
constexpr int kGroupsPerBlock = 8;
constexpr int kRotateThreads = kGroupsPerBlock * kGroupThreads;
constexpr int kQuantThreads = 512;

template <typename T>
__device__ __forceinline__ float to_float(T value);

template <>
__device__ __forceinline__ float to_float<half>(half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ half from_float<half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float finite_max();

template <>
__device__ __forceinline__ float finite_max<half>() {
    return 65504.0f;
}

template <>
__device__ __forceinline__ float finite_max<nv_bfloat16>() {
    return 3.38953139e38f;
}

template <typename T>
__device__ __forceinline__ float quant_div_to_float(T value, float scale) {
    const float rounded_scale = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(value) / rounded_scale));
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = kWarpThreads / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

template <int Warps>
__device__ __forceinline__ float block_reduce_max(
    float value, float *warp_values, float *block_value) {
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const int warp = threadIdx.x / kWarpThreads;
    value = warp_reduce_max(value);
    if (lane == 0) {
        warp_values[warp] = value;
    }
    __syncthreads();
    if (warp == 0) {
        float total = lane < Warps ? warp_values[lane] : 0.0f;
        total = warp_reduce_max(total);
        if (lane == 0) {
            *block_value = total;
        }
    }
    __syncthreads();
    return *block_value;
}

template <int S>
__device__ __forceinline__ void fht_stage(
    const float *__restrict__ src, float *__restrict__ dst, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    dst[base] = 0.5f * (x0 + x1 + x2 - x3);
    dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
    dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
    dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}

// Preserve eager's FP16 rotation; fuse only the following row quantization.
__global__ void fp16_int8_rowwise_kernel(
    const half *__restrict__ input, int8_t *__restrict__ output,
    float *__restrict__ scales, int k) {
    __shared__ float warp_values[8];
    __shared__ float block_value;
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * k;
    float abs_max = 0.0f;
    bool has_nan = false;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const float value = __half2float(input[offset + col]);
        has_nan |= isnan(value);
        abs_max = fmaxf(abs_max, fabsf(value));
    }
    const bool row_nan = __syncthreads_or(has_nan);
    abs_max = block_reduce_max<8>(abs_max, warp_values, &block_value);
    const float scale = row_nan ? CUDART_NAN_F : fmaxf(abs_max * (1.0f / 127.0f), 1.0e-30f);
    if (threadIdx.x == 0) scales[blockIdx.x] = scale;
    float math_scale = __half2float(__float2half_rn(scale));
    if (math_scale == 0.0f) math_scale = 0.00006103515625f;
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
        const float divided = __half2float(__float2half_rn(__fdiv_rn(__half2float(input[offset + col]), math_scale)));
        output[offset + col] = isnan(divided) ? 0 : static_cast<int8_t>(
            fminf(127.0f, fmaxf(-128.0f, nearbyintf(divided))));
    }
}

template <int S, typename OutputType>
__device__ __forceinline__ float fht_store_absmax(
    const float *__restrict__ src, OutputType *__restrict__ output, int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = from_float<OutputType>(y0);
    output[base + S] = from_float<OutputType>(y1);
    output[base + 2 * S] = from_float<OutputType>(y2);
    output[base + 3 * S] = from_float<OutputType>(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template <int S, typename InputType>
__device__ __forceinline__ void fht_store_scaled_int8(
    const float *__restrict__ src,
    int8_t *__restrict__ output,
    int lane,
    float scale) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float values[4] = {
        0.5f * (x0 + x1 + x2 - x3),
        0.5f * (x0 + x1 - x2 + x3),
        0.5f * (x0 - x1 + x2 + x3),
        0.5f * (-x0 + x1 + x2 + x3),
    };
#pragma unroll
    for (int index = 0; index < 4; ++index) {
        const InputType rounded = from_float<InputType>(values[index]);
        float quantized = nearbyintf(quant_div_to_float(rounded, scale));
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        output[base + index * S] = static_cast<int8_t>(quantized);
    }
}

template <typename InputType>
__device__ __forceinline__ float load_swiglu(
    const InputType *__restrict__ input, int64_t row_offset, int col, int k) {
    const float gate = to_float(input[row_offset + col]);
    const float up = to_float(input[row_offset + k + col]);
    return (gate / (1.0f + expf(-gate))) * up;
}

template <typename InputType>
__device__ __forceinline__ float load_gelu_tanh(
    const InputType *__restrict__ input, int64_t row_offset, int col) {
    const float x = to_float(input[row_offset + col]);
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float activated =
        0.5f * x * (1.0f + tanhf(kAlpha * (x + kBeta * x * x * x)));
    // Match nn.GELU's output tensor boundary before ConvRot. Keeping this
    // explicit makes the fused path numerically equivalent to eager BF16/FP16.
    return to_float(from_float<InputType>(activated));
}

template <typename InputType, bool Gelu = false>
__global__ void swiglu_rotate_amax_kernel(
    const InputType *__restrict__ input,
    InputType *__restrict__ rotated,
    float *__restrict__ partial_absmax,
    int k) {
    extern __shared__ float smem[];
    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * kGroupsPerBlock + sub;
    const int row = static_cast<int>(blockIdx.x);
    const int groups = k / kConvRotGroup;
    const bool active = group < groups;
    const int group_col = group * kConvRotGroup;
    const int base = lane * 4;
    const int col = group_col + base;
    constexpr int kInputWidth = Gelu ? 1 : 2;
    const int64_t input_row = static_cast<int64_t>(row) * (kInputWidth * k);
    const int64_t output_row = static_cast<int64_t>(row) * k;

    float *buf0 = smem + sub * (2 * kConvRotGroup);
    float *buf1 = buf0 + kConvRotGroup;

    const float x0 = active ? (Gelu ? load_gelu_tanh(input, input_row, col) : load_swiglu(input, input_row, col, k)) : 0.0f;
    const float x1 = active ? (Gelu ? load_gelu_tanh(input, input_row, col + 1) : load_swiglu(input, input_row, col + 1, k)) : 0.0f;
    const float x2 = active ? (Gelu ? load_gelu_tanh(input, input_row, col + 2) : load_swiglu(input, input_row, col + 2, k)) : 0.0f;
    const float x3 = active ? (Gelu ? load_gelu_tanh(input, input_row, col + 3) : load_swiglu(input, input_row, col + 3, k)) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    fht_stage<4>(buf1, buf0, lane);
    __syncthreads();
    fht_stage<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = fht_store_absmax<64>(
            buf1, rotated + output_row + group_col, lane);
    }
    buf0[lane] = local_max;
    __syncthreads();

    if (lane < 32) {
        float value = fmaxf(buf0[lane], buf0[lane + 32]);
        value = warp_reduce_max(value);
        if (lane == 0 && active) {
            partial_absmax[static_cast<int64_t>(row) * groups + group] = value;
        }
    }
}

template <typename InputType>
__global__ void swiglu_rotate_scaled_quantize_kernel(
    const InputType *__restrict__ input,
    const float *__restrict__ scales,
    int8_t *__restrict__ output,
    int k,
    int64_t output_stride) {
    extern __shared__ float smem[];
    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int group = static_cast<int>(blockIdx.y) * kGroupsPerBlock + sub;
    const int row = static_cast<int>(blockIdx.x);
    const int groups = k / kConvRotGroup;
    const bool active = group < groups;
    const int group_col = group * kConvRotGroup;
    const int base = lane * 4;
    const int col = group_col + base;
    const int64_t input_row = static_cast<int64_t>(row) * (2 * k);
    const int64_t output_row = static_cast<int64_t>(row) * output_stride;
    float *buf0 = smem + sub * (2 * kConvRotGroup);
    float *buf1 = buf0 + kConvRotGroup;

    const float x0 = active ? load_swiglu(input, input_row, col, k) : 0.0f;
    const float x1 = active ? load_swiglu(input, input_row, col + 1, k) : 0.0f;
    const float x2 = active ? load_swiglu(input, input_row, col + 2, k) : 0.0f;
    const float x3 = active ? load_swiglu(input, input_row, col + 3, k) : 0.0f;
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();

    fht_stage<4>(buf1, buf0, lane);
    __syncthreads();
    fht_stage<16>(buf0, buf1, lane);
    __syncthreads();
    if (active) {
        fht_store_scaled_int8<64, InputType>(
            buf1, output + output_row + group_col, lane, scales[row]);
    }
}

template <typename InputType>
__global__ void swiglu_rotate_shard_inplace_kernel(
    InputType *__restrict__ gate,
    const InputType *__restrict__ up,
    float *__restrict__ partial_absmax,
    int hidden,
    int shard_width,
    int channel_offset) {
    extern __shared__ float smem[];
    const int sub = threadIdx.x / kGroupThreads;
    const int lane = threadIdx.x % kGroupThreads;
    const int local_group = static_cast<int>(blockIdx.y) * kGroupsPerBlock + sub;
    const int shard_groups = shard_width / kConvRotGroup;
    const bool active = local_group < shard_groups;
    const int row = static_cast<int>(blockIdx.x);
    const int local_group_column = local_group * kConvRotGroup;
    const int global_group_column = channel_offset + local_group_column;
    const int base = lane * 4;
    const int64_t gate_row = static_cast<int64_t>(row) * hidden;
    const int64_t up_row = static_cast<int64_t>(row) * shard_width;
    float *buf0 = smem + sub * (2 * kConvRotGroup);
    float *buf1 = buf0 + kConvRotGroup;

    const auto activated = [&](int delta) {
        if (!active) {
            return 0.0f;
        }
        const float gate_value = to_float(
            gate[gate_row + global_group_column + base + delta]);
        const float up_value = to_float(
            up[up_row + local_group_column + base + delta]);
        return (gate_value / (1.0f + expf(-gate_value))) * up_value;
    };
    const float x0 = activated(0);
    const float x1 = activated(1);
    const float x2 = activated(2);
    const float x3 = activated(3);
    buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
    buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
    buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
    buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    fht_stage<4>(buf1, buf0, lane);
    __syncthreads();
    fht_stage<16>(buf0, buf1, lane);
    __syncthreads();

    float local_max = 0.0f;
    if (active) {
        local_max = fht_store_absmax<64>(
            buf1, gate + gate_row + global_group_column, lane);
    }
    buf0[lane] = local_max;
    __syncthreads();
    if (lane < 32) {
        float value = fmaxf(buf0[lane], buf0[lane + 32]);
        value = warp_reduce_max(value);
        if (lane == 0 && active) {
            const int global_group = global_group_column / kConvRotGroup;
            partial_absmax[
                static_cast<int64_t>(row) * (hidden / kConvRotGroup) +
                global_group] = value;
        }
    }
}

template <typename InputType>
__global__ void quantize_from_partials_kernel(
    const InputType *__restrict__ rotated,
    const float *__restrict__ partial_absmax,
    int8_t *__restrict__ output,
    float *__restrict__ scales,
    int k) {
    constexpr int kWarps = kQuantThreads / kWarpThreads;
    __shared__ float warp_values[kWarps];
    __shared__ float block_value;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int groups = k / kConvRotGroup;
    const int64_t row_offset = static_cast<int64_t>(row) * k;
    const float *row_partials = partial_absmax + static_cast<int64_t>(row) * groups;

    float abs_max = 0.0f;
    for (int group = tid; group < groups; group += kQuantThreads) {
        abs_max = fmaxf(abs_max, row_partials[group]);
    }
    abs_max = block_reduce_max<kWarps>(abs_max, warp_values, &block_value);
    const float scale = fmaxf(
        fminf(abs_max, finite_max<InputType>()) * (1.0f / 127.0f),
        1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int col = tid; col < k; col += kQuantThreads) {
        const int64_t index = row_offset + col;
        float quantized = nearbyintf(quant_div_to_float(rotated[index], scale));
        quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
        output[index] = static_cast<int8_t>(quantized);
    }
}

template <typename InputType>
__global__ void quantize_int4_from_partials_kernel(
    const InputType *__restrict__ rotated,
    const float *__restrict__ partial_absmax,
    int8_t *__restrict__ output,
    float *__restrict__ scales,
    int k) {
    constexpr int kWarps = kQuantThreads / kWarpThreads;
    __shared__ float warp_values[kWarps];
    __shared__ float block_value;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int groups = k / kConvRotGroup;
    const int64_t input_row = static_cast<int64_t>(row) * k;
    const int64_t output_row = static_cast<int64_t>(row) * (k / 2);
    const float *row_partials = partial_absmax + static_cast<int64_t>(row) * groups;

    float abs_max = 0.0f;
    for (int group = tid; group < groups; group += kQuantThreads) {
        abs_max = fmaxf(abs_max, row_partials[group]);
    }
    abs_max = block_reduce_max<kWarps>(abs_max, warp_values, &block_value);
    const float scale = fmaxf(
        fminf(abs_max, finite_max<InputType>()) * (1.0f / 7.0f),
        1.0e-10f);
    if (tid == 0) {
        scales[row] = scale;
    }

    for (int packed_col = tid; packed_col < k / 2; packed_col += kQuantThreads) {
        const int col = packed_col * 2;
        const float inv_scale = 1.0f / scale;
        float low = nearbyintf(to_float(rotated[input_row + col]) * inv_scale);
        float high = nearbyintf(to_float(rotated[input_row + col + 1]) * inv_scale);
        low = fminf(7.0f, fmaxf(-7.0f, low));
        high = fminf(7.0f, fmaxf(-7.0f, high));
        const uint8_t low_bits = static_cast<uint8_t>(static_cast<int8_t>(low)) & 0x0f;
        const uint8_t high_bits = static_cast<uint8_t>(static_cast<int8_t>(high)) & 0x0f;
        output[output_row + packed_col] = static_cast<int8_t>(low_bits | (high_bits << 4));
    }
}

__device__ __forceinline__ uint16_t float_to_bf16_rn_bits(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return 0x7fffU;
    }
    uint16_t upper = static_cast<uint16_t>(bits >> 16U);
    const uint32_t remainder = bits << 16U;
    if (remainder > 0x80000000U ||
        (remainder == 0x80000000U && (upper & 1U) != 0U)) {
        ++upper;
    }
    return upper;
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16U);
}

template <int S>
__device__ __forceinline__ float fht_store_bf16_absmax(
    const float *__restrict__ src,
    uint16_t *__restrict__ output,
    int lane) {
    const int base = (lane % S) + (lane / S) * (4 * S);
    const float x0 = src[base];
    const float x1 = src[base + S];
    const float x2 = src[base + 2 * S];
    const float x3 = src[base + 3 * S];
    const float y0 = 0.5f * (x0 + x1 + x2 - x3);
    const float y1 = 0.5f * (x0 + x1 - x2 + x3);
    const float y2 = 0.5f * (x0 - x1 + x2 + x3);
    const float y3 = 0.5f * (-x0 + x1 + x2 + x3);
    output[base] = float_to_bf16_rn_bits(y0);
    output[base + S] = float_to_bf16_rn_bits(y1);
    output[base + 2 * S] = float_to_bf16_rn_bits(y2);
    output[base + 3 * S] = float_to_bf16_rn_bits(y3);
    return fmaxf(fmaxf(fabsf(y0), fabsf(y1)), fmaxf(fabsf(y2), fabsf(y3)));
}

template <bool SwiGLU, bool Gelu = false>
__device__ __forceinline__ float load_bf16_input(
    const nv_bfloat16 *__restrict__ input,
    int64_t row_offset,
    int column,
    int k) {
    if constexpr (SwiGLU) {
        return load_swiglu(input, row_offset, column, k);
    } else if constexpr (Gelu) {
        return load_gelu_tanh(input, row_offset, column);
    }
    return __bfloat162float(input[row_offset + column]);
}

// Whole-row ConvRot. The inactive row is stored as BF16 while each active
// group retains FP32 scratch.
// This preserves the staged path's BF16 rounding but removes its global-memory
// intermediate and second kernel launch.
template <int BlockThreads, bool SwiGLU, bool Int4, bool Gelu = false>
__global__ void bf16_rowbuffer_convrot_quantize_kernel(
    const nv_bfloat16 *__restrict__ input,
    int8_t *__restrict__ output,
    float *__restrict__ scales,
    int k) {
    constexpr int kGroupsInFlight = BlockThreads / kGroupThreads;
    constexpr int kWarps = BlockThreads / kWarpThreads;

    extern __shared__ unsigned char shared_bytes[];
    uint16_t *row_buffer = reinterpret_cast<uint16_t *>(shared_bytes);
    float *scratch = reinterpret_cast<float *>(row_buffer + k);
    __shared__ float warp_values[kWarps];
    __shared__ float block_value;

    const int row = static_cast<int>(blockIdx.x);
    const int tid = threadIdx.x;
    const int sub = tid / kGroupThreads;
    const int lane = tid % kGroupThreads;
    const int groups = k / kConvRotGroup;
    constexpr int kInputWidth = SwiGLU ? 2 : 1;
    const int64_t input_row = static_cast<int64_t>(row) * k * kInputWidth;
    const int64_t output_row = static_cast<int64_t>(row) * k;
    float *buf0 = scratch + sub * (2 * kConvRotGroup);
    float *buf1 = buf0 + kConvRotGroup;
    float abs_max = 0.0f;

    const int iterations = (groups + kGroupsInFlight - 1) / kGroupsInFlight;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int group = iteration * kGroupsInFlight + sub;
        const bool active = group < groups;
        const int base = lane * 4;
        const int group_column = group * kConvRotGroup;
        const int column = group_column + base;
        const float x0 = active ? load_bf16_input<SwiGLU, Gelu>(input, input_row, column, k) : 0.0f;
        const float x1 = active ? load_bf16_input<SwiGLU, Gelu>(input, input_row, column + 1, k) : 0.0f;
        const float x2 = active ? load_bf16_input<SwiGLU, Gelu>(input, input_row, column + 2, k) : 0.0f;
        const float x3 = active ? load_bf16_input<SwiGLU, Gelu>(input, input_row, column + 3, k) : 0.0f;
        buf1[base] = 0.5f * (x0 + x1 + x2 - x3);
        buf1[base + 1] = 0.5f * (x0 + x1 - x2 + x3);
        buf1[base + 2] = 0.5f * (x0 - x1 + x2 + x3);
        buf1[base + 3] = 0.5f * (-x0 + x1 + x2 + x3);
        __syncthreads();

        fht_stage<4>(buf1, buf0, lane);
        __syncthreads();
        fht_stage<16>(buf0, buf1, lane);
        __syncthreads();

        if (active) {
            abs_max = fmaxf(
                abs_max,
                fht_store_bf16_absmax<64>(
                    buf1, row_buffer + group_column, lane));
        }
        __syncthreads();
    }

    abs_max = block_reduce_max<kWarps>(abs_max, warp_values, &block_value);
    constexpr float quant_max = Int4 ? 7.0f : 127.0f;
    const float scale = fmaxf(
        fminf(abs_max, finite_max<nv_bfloat16>()) / quant_max,
        Int4 ? 1.0e-10f : 1.0e-30f);
    if (tid == 0) {
        scales[row] = scale;
    }

    const float rounded_scale = bf16_bits_to_float(float_to_bf16_rn_bits(scale));
    if constexpr (Int4) {
        const float inv_scale = 1.0f / scale;
        const int64_t packed_output_row = static_cast<int64_t>(row) * (k / 2);
        for (int packed_column = tid; packed_column < k / 2; packed_column += BlockThreads) {
            const int column = packed_column * 2;
            const float low_value = bf16_bits_to_float(row_buffer[column]);
            const float high_value = bf16_bits_to_float(row_buffer[column + 1]);
            float low = nearbyintf(low_value * inv_scale);
            float high = nearbyintf(high_value * inv_scale);
            low = fminf(7.0f, fmaxf(-7.0f, low));
            high = fminf(7.0f, fmaxf(-7.0f, high));
            const uint8_t low_bits = static_cast<uint8_t>(static_cast<int8_t>(low)) & 0x0f;
            const uint8_t high_bits = static_cast<uint8_t>(static_cast<int8_t>(high)) & 0x0f;
            output[packed_output_row + packed_column] =
                static_cast<int8_t>(low_bits | (high_bits << 4));
        }
    } else {
        for (int column = tid; column < k; column += BlockThreads) {
            const float value = bf16_bits_to_float(row_buffer[column]);
            const float divided = bf16_bits_to_float(
                float_to_bf16_rn_bits(value / rounded_scale));
            float quantized = nearbyintf(divided);
            quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
            output[output_row + column] = static_cast<int8_t>(quantized);
        }
    }
}

template <int BlockThreads, bool SwiGLU, bool Int4 = false, bool Gelu = false>
void launch_bf16_rowbuffer(Tensor input, Tensor output, Tensor scales) {
    const int rows = input.size(0);
    const int k = Int4 ? output.size(1) * 2 : output.size(1);
    constexpr int kGroupsInFlight = BlockThreads / kGroupThreads;
    const size_t shared_bytes =
        static_cast<size_t>(k) * sizeof(uint16_t) +
        kGroupsInFlight * 2 * kConvRotGroup * sizeof(float);
    auto kernel =
        bf16_rowbuffer_convrot_quantize_kernel<BlockThreads, SwiGLU, Int4, Gelu>;
    checkCUDA(cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shared_bytes)));
    kernel
        <<<rows, BlockThreads, shared_bytes, getCurrentCUDAStream()>>>(
            static_cast<const nv_bfloat16 *>(input.ptr),
            static_cast<int8_t *>(output.ptr),
            static_cast<float *>(scales.ptr),
            k);
    checkCUDA(cudaGetLastError());
}

template <typename InputType, bool Int4 = false, bool Gelu = false>
void launch_swiglu_quantize(Tensor input,
                            Tensor rotated,
                            Tensor partial_absmax,
                            Tensor output,
                            Tensor scales) {
    const int rows = input.size(0);
    const int k = rotated.size(1);
    const int group_blocks = ceilDiv(k / kConvRotGroup, kGroupsPerBlock);
    const dim3 grid(static_cast<unsigned int>(rows), static_cast<unsigned int>(group_blocks));
    constexpr size_t smem_bytes =
        kGroupsPerBlock * 2 * kConvRotGroup * sizeof(float);
    swiglu_rotate_amax_kernel<InputType, Gelu><<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
        static_cast<const InputType *>(input.ptr),
        static_cast<InputType *>(rotated.ptr),
        static_cast<float *>(partial_absmax.ptr),
        k);
    checkCUDA(cudaGetLastError());

    if constexpr (Int4) {
        quantize_int4_from_partials_kernel<InputType>
            <<<rows, kQuantThreads, 0, getCurrentCUDAStream()>>>(
                static_cast<const InputType *>(rotated.ptr),
                static_cast<const float *>(partial_absmax.ptr),
                static_cast<int8_t *>(output.ptr),
                static_cast<float *>(scales.ptr),
                k);
    } else {
        quantize_from_partials_kernel<InputType>
            <<<rows, kQuantThreads, 0, getCurrentCUDAStream()>>>(
                static_cast<const InputType *>(rotated.ptr),
                static_cast<const float *>(partial_absmax.ptr),
                static_cast<int8_t *>(output.ptr),
                static_cast<float *>(scales.ptr),
                k);
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace

void turing_swiglu_int8_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_swiglu_quantize<nv_bfloat16>(input, rotated, partial_absmax, output, scales);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_swiglu_quantize<half>(input, rotated, partial_absmax, output, scales);
    } else {
        throw std::runtime_error("SwiGLU staged ConvRot requires float16 or bfloat16 input");
    }
}

void turing_swiglu_int8_convrot_quantize_scaled(Tensor input,
                                                 Tensor scales,
                                                 Tensor output) {
    const int rows = input.size(0);
    const int k = output.size(1);
    const int group_blocks = ceilDiv(k / kConvRotGroup, kGroupsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(group_blocks));
    constexpr size_t smem_bytes =
        kGroupsPerBlock * 2 * kConvRotGroup * sizeof(float);
    if (input.scalar_type() == Tensor::BF16) {
        swiglu_rotate_scaled_quantize_kernel<nv_bfloat16>
            <<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
                static_cast<const nv_bfloat16 *>(input.ptr),
                static_cast<const float *>(scales.ptr),
                static_cast<int8_t *>(output.ptr),
                k,
                output.stride(0));
    } else if (input.scalar_type() == Tensor::FP16) {
        swiglu_rotate_scaled_quantize_kernel<half>
            <<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
                static_cast<const half *>(input.ptr),
                static_cast<const float *>(scales.ptr),
                static_cast<int8_t *>(output.ptr),
                k,
                output.stride(0));
    } else {
        throw std::runtime_error(
            "scaled SwiGLU ConvRot requires float16 or bfloat16 input");
    }
    checkCUDA(cudaGetLastError());
}

void turing_swiglu_convrot_shard_inplace(Tensor gate,
                                          Tensor up,
                                          Tensor partial_absmax,
                                          int channel_offset) {
    const int rows = gate.size(0);
    const int hidden = gate.size(1);
    const int shard_width = up.size(1);
    const int group_blocks = ceilDiv(
        shard_width / kConvRotGroup, kGroupsPerBlock);
    const dim3 grid(
        static_cast<unsigned int>(rows),
        static_cast<unsigned int>(group_blocks));
    constexpr size_t smem_bytes =
        kGroupsPerBlock * 2 * kConvRotGroup * sizeof(float);
    if (gate.scalar_type() == Tensor::BF16) {
        swiglu_rotate_shard_inplace_kernel<nv_bfloat16>
            <<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
                static_cast<nv_bfloat16 *>(gate.ptr),
                static_cast<const nv_bfloat16 *>(up.ptr),
                static_cast<float *>(partial_absmax.ptr),
                hidden,
                shard_width,
                channel_offset);
    } else if (gate.scalar_type() == Tensor::FP16) {
        swiglu_rotate_shard_inplace_kernel<half>
            <<<grid, kRotateThreads, smem_bytes, getCurrentCUDAStream()>>>(
                static_cast<half *>(gate.ptr),
                static_cast<const half *>(up.ptr),
                static_cast<float *>(partial_absmax.ptr),
                hidden,
                shard_width,
                channel_offset);
    } else {
        throw std::runtime_error(
            "sharded SwiGLU ConvRot requires float16 or bfloat16 input");
    }
    checkCUDA(cudaGetLastError());
}

void turing_int8_convrot_quantize_from_partials(Tensor rotated,
                                                 Tensor partial_absmax,
                                                 Tensor output,
                                                 Tensor scales) {
    const int rows = rotated.size(0);
    const int hidden = rotated.size(1);
    if (rotated.scalar_type() == Tensor::BF16) {
        quantize_from_partials_kernel<nv_bfloat16>
            <<<rows, kQuantThreads, 0, getCurrentCUDAStream()>>>(
                static_cast<const nv_bfloat16 *>(rotated.ptr),
                static_cast<const float *>(partial_absmax.ptr),
                static_cast<int8_t *>(output.ptr),
                static_cast<float *>(scales.ptr),
                hidden);
    } else if (rotated.scalar_type() == Tensor::FP16) {
        quantize_from_partials_kernel<half>
            <<<rows, kQuantThreads, 0, getCurrentCUDAStream()>>>(
                static_cast<const half *>(rotated.ptr),
                static_cast<const float *>(partial_absmax.ptr),
                static_cast<int8_t *>(output.ptr),
                static_cast<float *>(scales.ptr),
                hidden);
    } else {
        throw std::runtime_error(
            "ConvRot partial reduction requires float16 or bfloat16 input");
    }
    checkCUDA(cudaGetLastError());
}

void turing_swiglu_int4_convrot_quantize(Tensor input,
                                          Tensor rotated,
                                          Tensor partial_absmax,
                                          Tensor output,
                                          Tensor scales) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_swiglu_quantize<nv_bfloat16, true>(input, rotated, partial_absmax, output, scales);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_swiglu_quantize<half, true>(input, rotated, partial_absmax, output, scales);
    } else {
        throw std::runtime_error("SwiGLU staged INT4 ConvRot requires float16 or bfloat16 input");
    }
}

void turing_gelu_int8_convrot_quantize(Tensor input,
                                        Tensor rotated,
                                        Tensor partial_absmax,
                                        Tensor output,
                                        Tensor scales) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_swiglu_quantize<nv_bfloat16, false, true>(input, rotated, partial_absmax, output, scales);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_swiglu_quantize<half, false, true>(input, rotated, partial_absmax, output, scales);
    } else {
        throw std::runtime_error("GELU staged ConvRot requires float16 or bfloat16 input");
    }
}

void turing_gelu_int4_convrot_quantize(Tensor input,
                                        Tensor rotated,
                                        Tensor partial_absmax,
                                        Tensor output,
                                        Tensor scales) {
    if (input.scalar_type() == Tensor::BF16) {
        launch_swiglu_quantize<nv_bfloat16, true, true>(input, rotated, partial_absmax, output, scales);
    } else if (input.scalar_type() == Tensor::FP16) {
        launch_swiglu_quantize<half, true, true>(input, rotated, partial_absmax, output, scales);
    } else {
        throw std::runtime_error("GELU staged INT4 ConvRot requires float16 or bfloat16 input");
    }
}

void turing_bf16_int8_convrot_quantize(Tensor input,
                                        Tensor output,
                                        Tensor scales,
                                        bool swiglu,
                                        int block_threads) {
    if (input.scalar_type() != Tensor::BF16) {
        throw std::runtime_error("BF16 row-buffer ConvRot requires bfloat16 input");
    }
    if (swiglu) {
        if (block_threads == 1024) {
            launch_bf16_rowbuffer<1024, true>(input, output, scales);
        } else if (block_threads == 768) {
            launch_bf16_rowbuffer<768, true>(input, output, scales);
        } else {
            launch_bf16_rowbuffer<512, true>(input, output, scales);
        }
    } else if (block_threads == 1024) {
        launch_bf16_rowbuffer<1024, false>(input, output, scales);
    } else if (block_threads == 768) {
        launch_bf16_rowbuffer<768, false>(input, output, scales);
    } else {
        launch_bf16_rowbuffer<512, false>(input, output, scales);
    }
}

void turing_bf16_int4_convrot_quantize(Tensor input,
                                        Tensor output,
                                        Tensor scales,
                                        bool swiglu,
                                        int block_threads) {
    if (input.scalar_type() != Tensor::BF16) {
        throw std::runtime_error("BF16 row-buffer INT4 ConvRot requires bfloat16 input");
    }
    if (swiglu) {
        if (block_threads == 1024) {
            launch_bf16_rowbuffer<1024, true, true>(input, output, scales);
        } else if (block_threads == 768) {
            launch_bf16_rowbuffer<768, true, true>(input, output, scales);
        } else {
            launch_bf16_rowbuffer<512, true, true>(input, output, scales);
        }
    } else if (block_threads == 1024) {
        launch_bf16_rowbuffer<1024, false, true>(input, output, scales);
    } else if (block_threads == 768) {
        launch_bf16_rowbuffer<768, false, true>(input, output, scales);
    } else {
        launch_bf16_rowbuffer<512, false, true>(input, output, scales);
    }
}

void turing_fp16_int8_quantize(Tensor input, Tensor output, Tensor scales) {
    fp16_int8_rowwise_kernel<<<input.size(0), 256, 0, getCurrentCUDAStream()>>>(
        static_cast<const half *>(input.ptr), static_cast<int8_t *>(output.ptr),
        static_cast<float *>(scales.ptr), input.size(1));
    checkCUDA(cudaGetLastError());
}

void turing_bf16_gelu_int8_convrot_quantize(Tensor input,
                                             Tensor output,
                                             Tensor scales,
                                             int block_threads) {
    if (input.scalar_type() != Tensor::BF16) {
        throw std::runtime_error("BF16 GELU row-buffer ConvRot requires bfloat16 input");
    }
    if (block_threads == 1024) {
        launch_bf16_rowbuffer<1024, false, false, true>(input, output, scales);
    } else if (block_threads == 768) {
        launch_bf16_rowbuffer<768, false, false, true>(input, output, scales);
    } else {
        launch_bf16_rowbuffer<512, false, false, true>(input, output, scales);
    }
}

void turing_bf16_gelu_int4_convrot_quantize(Tensor input,
                                             Tensor output,
                                             Tensor scales,
                                             int block_threads) {
    if (input.scalar_type() != Tensor::BF16) {
        throw std::runtime_error("BF16 GELU row-buffer INT4 ConvRot requires bfloat16 input");
    }
    if (block_threads == 1024) {
        launch_bf16_rowbuffer<1024, false, true, true>(input, output, scales);
    } else if (block_threads == 768) {
        launch_bf16_rowbuffer<768, false, true, true>(input, output, scales);
    } else {
        launch_bf16_rowbuffer<512, false, true, true>(input, output, scales);
    }
}

}  // namespace comfyui_turing_utils::kernels
