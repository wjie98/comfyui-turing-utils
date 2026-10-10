// FP16 storage, FP32 activation/ConvRot/quantization, no global float row buffer.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

constexpr unsigned kFullWarp = 0xffffffffu;
constexpr int kGroupSize = 256;

__device__ __forceinline__ float hadamard4(int digit, float a, float b, float c, float d) {
    switch (digit) {
        case 0: return 0.5f * (a + b + c - d);
        case 1: return 0.5f * (a + b - c + d);
        case 2: return 0.5f * (a - b + c + d);
        default: return 0.5f * (-a + b + c + d);
    }
}

// Lane l owns elements l + 32*j. The first two radix-4 stages exchange
// lanes, the third exchanges both lanes and registers, and the last is local.
__device__ __forceinline__ void rotate_group(float (&v)[8], int lane) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int base0 = lane & ~3;
        v[j] = hadamard4(lane & 3,
            __shfl_sync(kFullWarp, v[j], base0),
            __shfl_sync(kFullWarp, v[j], base0 + 1),
            __shfl_sync(kFullWarp, v[j], base0 + 2),
            __shfl_sync(kFullWarp, v[j], base0 + 3));
        const int base1 = lane & ~12;
        v[j] = hadamard4((lane >> 2) & 3,
            __shfl_sync(kFullWarp, v[j], base1),
            __shfl_sync(kFullWarp, v[j], base1 + 4),
            __shfl_sync(kFullWarp, v[j], base1 + 8),
            __shfl_sync(kFullWarp, v[j], base1 + 12));
    }
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const float a = __shfl_sync(kFullWarp, v[j], lane & 15);
        const float b = __shfl_sync(kFullWarp, v[j], (lane & 15) + 16);
        const float c = __shfl_sync(kFullWarp, v[j + 1], lane & 15);
        const float d = __shfl_sync(kFullWarp, v[j + 1], (lane & 15) + 16);
        v[j] = hadamard4(lane >> 4, a, b, c, d);
        v[j + 1] = hadamard4(2 + (lane >> 4), a, b, c, d);
    }
#pragma unroll
    for (int parity = 0; parity < 2; ++parity) {
        const float a = v[parity];
        const float b = v[parity + 2];
        const float c = v[parity + 4];
        const float d = v[parity + 6];
        v[parity] = hadamard4(0, a, b, c, d);
        v[parity + 2] = hadamard4(1, a, b, c, d);
        v[parity + 4] = hadamard4(2, a, b, c, d);
        v[parity + 6] = hadamard4(3, a, b, c, d);
    }
}

template <bool Sum, int Threads>
__device__ __forceinline__ float reduce_row(float value, float *scratch) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(kFullWarp, value, offset);
        value = Sum ? value + other : fmaxf(value, other);
    }
    const int lane = threadIdx.x & 31;
    if (lane == 0) scratch[threadIdx.x / 32] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = lane < Threads / 32 ? scratch[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other = __shfl_down_sync(kFullWarp, value, offset);
            value = Sum ? value + other : fmaxf(value, other);
        }
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

template <int Activation>
__device__ __forceinline__ void load_group(
    const half *input, const void *norm_weight, bool norm_is_float,
    float norm_scale, int k, int base, int lane, float (&v)[8]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = base + lane + 32 * j;
        float value = __half2float(input[col]);
        if constexpr (Activation == 1) {
            value = 0.5f * value * (1.0f + tanhf(
                0.7978845608f * (value + 0.044715f * value * value * value)));
        } else if constexpr (Activation == 2) {
            value = value / (1.0f + expf(-value)) * __half2float(input[k + col]);
        } else if constexpr (Activation == 3) {
            const float weight = norm_is_float
                ? static_cast<const float *>(norm_weight)[col]
                : __half2float(static_cast<const half *>(norm_weight)[col]);
            value = value * norm_scale * weight;
        }
        v[j] = value;
    }
    rotate_group(v, lane);
}

template <int Activation, int Threads, bool CacheRotation>
__global__ void fp16_convrot_quantize_kernel(
    const half *__restrict__ input, int8_t *__restrict__ output,
    float *__restrict__ scales, const void *__restrict__ norm_weight,
    bool norm_is_float, float eps, int k) {
    __shared__ float scratch[Threads / 32];
    extern __shared__ float rotated[];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x / 32;
    const int64_t row = blockIdx.x;
    const half *row_input = input + row * (Activation == 2 ? 2LL * k : k);
    int8_t *row_output = output + row * k;
    float norm_scale = 1.0f;
    if constexpr (Activation == 3) {
        float sum = 0.0f;
        for (int col = threadIdx.x; col < k; col += Threads) {
            const float value = __half2float(row_input[col]);
            sum += value * value;
        }
        norm_scale = rsqrtf(reduce_row<true, Threads>(sum, scratch) / k + eps);
        // Every warp must consume the reduction before scratch is reused.
        __syncthreads();
    }

    float absmax = 0.0f;
    for (int base = warp * kGroupSize; base < k; base += (Threads / 32) * kGroupSize) {
        float values[8];
        load_group<Activation>(row_input, norm_weight, norm_is_float,
            norm_scale, k, base, lane, values);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            absmax = fmaxf(absmax, isfinite(values[j]) ? fabsf(values[j]) : CUDART_INF_F);
            if constexpr (CacheRotation) rotated[base + lane + 32 * j] = values[j];
        }
    }
    absmax = reduce_row<false, Threads>(absmax, scratch);
    const float scale = isfinite(absmax)
        ? fmaxf(__fdiv_rn(absmax, 127.0f), 1.0e-30f) : CUDART_NAN_F;
    if (threadIdx.x == 0) scales[row] = scale;

    // Common widths reuse FP32 values in shared memory. Very wide rows
    // recompute the same group instead; neither path writes a float row to HBM.
    for (int base = warp * kGroupSize; base < k; base += (Threads / 32) * kGroupSize) {
        float values[8];
        if constexpr (CacheRotation) {
#pragma unroll
            for (int j = 0; j < 8; ++j) values[j] = rotated[base + lane + 32 * j];
        } else {
            load_group<Activation>(row_input, norm_weight, norm_is_float,
                norm_scale, k, base, lane, values);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float divided = __fdiv_rn(values[j], scale);
            row_output[base + lane + 32 * j] = isfinite(divided)
                ? static_cast<int8_t>(fminf(127.0f, fmaxf(-127.0f, nearbyintf(divided)))) : 0;
        }
    }
}

template <int Activation, int Threads>
void launch(Tensor input, Tensor output, Tensor scales, Tensor norm_weight, float eps) {
    // A bounded 32-KiB row cache fits the default shared-memory limit on SM75+.
    // Leave wide-row support independent of architecture-specific opt-in limits.
    const bool cache_rotation = output.size(1) <= 8192;
    const int shared_bytes = cache_rotation ? output.size(1) * sizeof(float) : 0;
    auto kernel = cache_rotation
        ? fp16_convrot_quantize_kernel<Activation, Threads, true>
        : fp16_convrot_quantize_kernel<Activation, Threads, false>;
    kernel<<<input.size(0), Threads, shared_bytes, getCurrentCUDAStream()>>>(
            static_cast<const half *>(input.ptr), static_cast<int8_t *>(output.ptr),
            static_cast<float *>(scales.ptr), norm_weight.ptr,
            norm_weight.scalar_type() == Tensor::FP32, eps, output.size(1));
}

template <int Activation>
void launch_for_width(Tensor input, Tensor output, Tensor scales, Tensor norm_weight, float eps) {
    const int k = output.size(1);
    if (k == 256) launch<Activation, 32>(input, output, scales, norm_weight, eps);
    else if (k <= 3072) launch<Activation, 128>(input, output, scales, norm_weight, eps);
    else launch<Activation, 256>(input, output, scales, norm_weight, eps);
}

}  // namespace

void turing_fp16_int8_convrot_quantize(
    Tensor input, Tensor output, Tensor scales, int input_act, Tensor norm_weight, float eps) {
    switch (input_act) {
        case 0: launch_for_width<0>(input, output, scales, norm_weight, eps); break;
        case 1: launch_for_width<1>(input, output, scales, norm_weight, eps); break;
        case 2: launch_for_width<2>(input, output, scales, norm_weight, eps); break;
        case 3: launch_for_width<3>(input, output, scales, norm_weight, eps); break;
        default: throw std::runtime_error("Unsupported FP16 ConvRot activation");
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace comfyui_turing_utils::kernels
