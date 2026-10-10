#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "kernel_api.h"

namespace {

class TorchOpContext {
public:
    TorchOpContext() {
        stackCUDAStreams.push_back(at::cuda::getCurrentCUDAStream().stream());
    }

    TorchOpContext(const TorchOpContext &) = delete;
    TorchOpContext(TorchOpContext &&) = delete;

    ~TorchOpContext() {
        assert(!stackCUDAStreams.empty());
        assert(stackCUDAStreams.back() == at::cuda::getCurrentCUDAStream().stream());
        stackCUDAStreams.pop_back();
    }
};

template <typename To, typename From>
To int_cast(From value) {
    TORCH_CHECK(value >= static_cast<From>(std::numeric_limits<To>::min()) &&
                    value <= static_cast<From>(std::numeric_limits<To>::max()),
                "integer overflow while converting tensor metadata");
    return static_cast<To>(value);
}

Tensor from_torch(at::Tensor input) {
    Tensor result;

    const int ndims = int_cast<int>(input.dim());
    for (int i = 0; i < ndims; ++i) {
        result.shape.dataExtent.push_back(int_cast<int>(input.size(i)));
        result.shape.dataStride.push_back(int_cast<int>(input.stride(i)));
    }

    switch (input.scalar_type()) {
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
        result.scalarType = Tensor::INT8;
        break;
    case at::ScalarType::Short:
        result.scalarType = Tensor::INT16;
        break;
    case at::ScalarType::Int:
        result.scalarType = Tensor::INT32;
        break;
    case at::ScalarType::Long:
        result.scalarType = Tensor::INT64;
        break;
    case at::ScalarType::Half:
        result.scalarType = Tensor::FP16;
        break;
    case at::ScalarType::Float:
        result.scalarType = Tensor::FP32;
        break;
    case at::ScalarType::BFloat16:
        result.scalarType = Tensor::BF16;
        break;
    case at::ScalarType::Float8_e4m3fn:
        result.scalarType = Tensor::FP8_E4M3;
        break;
    case at::ScalarType::Float8_e5m2:
        result.scalarType = Tensor::FP8_E5M2;
        break;
    default:
        TORCH_CHECK(false, "unsupported tensor dtype for Turing Utils kernel");
    }

    result.ptr = input.data_ptr();
    result.dev = Device{input.is_cuda() ? Device::CUDA : Device::CPU, input.is_cuda() ? input.get_device() : 0};
    result.owner = std::make_shared<at::Tensor>(std::move(input));
    return result;
}

void check_cuda_2d(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dim() == 2, name, " must be 2D");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

void check_half_like(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kHalf || t.scalar_type() == at::kBFloat16,
                name,
                " must be float16 or bfloat16");
}

void check_float_like(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.scalar_type() == at::kHalf ||
                    t.scalar_type() == at::kBFloat16 ||
                    t.scalar_type() == at::kFloat,
                name,
                " must be float16, bfloat16, or float32");
}

int select_bf16_convrot_threads(const cudaDeviceProp *properties,
                                int64_t hidden,
                                int64_t rows,
                                int forced_threads = 0) {
    const int64_t shared_limit = properties->sharedMemPerBlockOptin > 0
        ? static_cast<int64_t>(properties->sharedMemPerBlockOptin)
        : static_cast<int64_t>(properties->sharedMemPerBlock);
    int best_threads = 0;
    int64_t best_active_warps = -1;
    for (const int threads : {512, 768, 1024}) {
        if (forced_threads != 0 && threads != forced_threads) {
            continue;
        }
        const int64_t groups_in_flight = threads / 64;
        const int64_t dynamic_bytes =
            hidden * static_cast<int64_t>(sizeof(uint16_t)) +
            groups_in_flight * 2 * 256 * static_cast<int64_t>(sizeof(float));
        const int64_t static_bytes =
            (threads / 32 + 4) * static_cast<int64_t>(sizeof(float));
        const int64_t shared_bytes = dynamic_bytes + static_bytes;
        if (shared_bytes > shared_limit) {
            continue;
        }
        if (forced_threads != 0) {
            return threads;
        }

        const int64_t shared_per_sm = properties->sharedMemPerMultiprocessor;
        const int64_t resident_by_shared = shared_per_sm / shared_bytes;
        const int64_t resident_by_threads =
            properties->maxThreadsPerMultiProcessor / threads;
        const int64_t grid_ctas_per_sm =
            (rows + properties->multiProcessorCount - 1) /
            properties->multiProcessorCount;
        const int64_t resident_ctas = std::max<int64_t>(
            1, std::min({resident_by_shared, resident_by_threads, grid_ctas_per_sm}));
        const int64_t active_warps = resident_ctas * (threads / 32);
        if (active_warps > best_active_warps) {
            best_active_warps = active_warps;
            best_threads = threads;
        }
    }
    if (best_threads != 0) {
        return best_threads;
    }
    TORCH_CHECK(false,
                "BF16 row-buffer ConvRot requires more shared memory than device ",
                properties->name, " provides (opt-in limit ", shared_limit, " bytes)");
    return 0;
}

Tensor maybe_tensor(const std::optional<at::Tensor> &value) {
    if (!value.has_value()) {
        return Tensor{};
    }
    return from_torch(value.value());
}

at::Tensor turing_w4a8_linear(at::Tensor activation,
                              at::Tensor weight,
                              at::Tensor activation_scale,
                              at::Tensor weight_scale,
                              std::optional<at::Tensor> bias) {
    activation = activation.contiguous();
    weight = weight.contiguous();
    activation_scale = activation_scale.to(at::kFloat).contiguous();
    weight_scale = weight_scale.to(at::kFloat).contiguous();
    if (bias.has_value()) {
        bias = bias.value().to(at::kFloat).contiguous();
    }

    check_cuda_2d(activation, "activation");
    check_cuda_2d(weight, "weight");
    TORCH_CHECK(activation.scalar_type() == at::kChar, "activation must be int8");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be packed int8 storage");
    TORCH_CHECK(activation.device() == weight.device(), "activation and weight must be on the same CUDA device");
    TORCH_CHECK(activation_scale.is_cuda() && weight_scale.is_cuda(), "scales must be CUDA tensors");
    TORCH_CHECK(activation_scale.device() == activation.device() && weight_scale.device() == activation.device(),
                "scales must be on the activation device");
    TORCH_CHECK(activation.size(1) % 4 == 0, "activation K must be divisible by 4");
    TORCH_CHECK(weight.size(1) * 2 == activation.size(1), "packed weight K must match activation K");
    TORCH_CHECK(activation_scale.numel() == activation.size(0), "activation_scale must contain one value per row");
    TORCH_CHECK(weight_scale.numel() == weight.size(0), "weight_scale must contain one value per output channel");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda() && bias.value().device() == activation.device(),
                    "bias must be on the activation device");
        TORCH_CHECK(bias.value().numel() == weight.size(0), "bias must contain one value per output channel");
    }

    const at::cuda::CUDAGuard device_guard(activation.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "turing_w4a8_linear requires sm75 or newer");

    const int64_t output_channels = weight.size(0);
    const int64_t padded_output_channels =
        ((output_channels + 7) / 8) * 8;
    at::Tensor output = at::empty(
        {activation.size(0), padded_output_channels},
        activation.options().dtype(at::kBFloat16));
    TorchOpContext ctx;
    const int64_t tensor_core_channels = (output_channels / 8) * 8;
    if (tensor_core_channels > 0) {
        auto bulk_weight = weight.narrow(0, 0, tensor_core_channels);
        auto bulk_scale = weight_scale.narrow(0, 0, tensor_core_channels);
        auto bulk_bias = bias.has_value()
            ? std::optional<at::Tensor>(bias.value().narrow(0, 0, tensor_core_channels))
            : std::nullopt;
        auto bulk_output = output.narrow(1, 0, tensor_core_channels);
        comfyui_turing_utils::kernels::turing_w4a8_linear(
            from_torch(activation),
            from_torch(bulk_weight),
            from_torch(activation_scale),
            from_torch(bulk_scale),
            maybe_tensor(bulk_bias),
            from_torch(bulk_output));
    }
    if (tensor_core_channels != output_channels) {
        const int64_t tail_channels = output_channels - tensor_core_channels;
        auto tail_weight = at::constant_pad_nd(
            weight.narrow(0, tensor_core_channels, tail_channels),
            {0, 0, 0, 8 - tail_channels},
            0);
        auto tail_scale = at::constant_pad_nd(
            weight_scale.narrow(0, tensor_core_channels, tail_channels),
            {0, 8 - tail_channels},
            0);
        std::optional<at::Tensor> tail_bias = std::nullopt;
        if (bias.has_value()) {
            tail_bias = at::constant_pad_nd(
                bias.value().narrow(0, tensor_core_channels, tail_channels),
                {0, 8 - tail_channels},
                0);
        }
        auto tail_output = output.narrow(1, tensor_core_channels, 8);
        comfyui_turing_utils::kernels::turing_w4a8_linear(
            from_torch(activation),
            from_torch(tail_weight),
            from_torch(activation_scale),
            from_torch(tail_scale),
            maybe_tensor(tail_bias),
            from_torch(tail_output));
    }
    if (padded_output_channels == output_channels) {
        return output;
    }
    return output.narrow(1, 0, output_channels).contiguous();
}

at::Tensor turing_codebook_w4a8_linear(at::Tensor activation,
                                       at::Tensor weight,
                                       at::Tensor activation_scale,
                                       at::Tensor group_scale,
                                       at::Tensor channel_scale,
                                       std::optional<at::Tensor> codebook,
                                       std::optional<at::Tensor> bias,
                                       int64_t group_size,
                                       int64_t chunk_rows) {
    activation = activation.contiguous();
    weight = weight.contiguous();
    activation_scale = activation_scale.reshape({-1}).to(at::kFloat).contiguous();
    group_scale = group_scale.contiguous();
    channel_scale = channel_scale.reshape({-1}).to(at::kFloat).contiguous();
    if (codebook.has_value()) {
        codebook = codebook.value().reshape({-1}).to(at::kFloat).contiguous();
    }
    if (bias.has_value()) {
        bias = bias.value().reshape({-1}).to(at::kFloat).contiguous();
    }

    check_cuda_2d(activation, "activation");
    check_cuda_2d(weight, "weight");
    check_cuda_2d(group_scale, "group_scale");
    TORCH_CHECK(activation.scalar_type() == at::kChar, "activation must be int8");
    TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must use packed int8 storage");
    TORCH_CHECK(group_scale.scalar_type() == at::kByte ||
                    group_scale.scalar_type() == at::kFloat,
                "group_scale must contain raw float8_e4m3fn bytes or float32 scales");
    TORCH_CHECK(activation.device() == weight.device() &&
                    activation.device() == group_scale.device() &&
                    activation.device() == activation_scale.device() &&
                    activation.device() == channel_scale.device() &&
                    (!codebook.has_value() || activation.device() == codebook.value().device()),
                "all codebook W4A8 tensors must use the same CUDA device");

    const int64_t m = activation.size(0);
    const int64_t k = activation.size(1);
    const int64_t n = weight.size(0);
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "codebook W4A8 dimensions must be positive");
    TORCH_CHECK(m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
                "codebook W4A8 dimensions exceed the CUDA kernel range");
    TORCH_CHECK(k % 16 == 0 && n % 8 == 0,
                "codebook W4A8 requires K divisible by 16 and N divisible by 8");
    const bool w6 = weight.size(1) * 8 == k * 6;
    TORCH_CHECK(w6 || weight.size(1) * 2 == k, "packed weight must store W4 or W6 at activation K");
    TORCH_CHECK(!w6 || (k % 32 == 0 && group_size >= 16 && group_size % 16 == 0),
                "W6A8 requires K%32=0 and group_size a multiple of 16");
    TORCH_CHECK(group_size >= 4 && k % group_size == 0 &&
                    (16 % group_size == 0 || group_size % 16 == 0),
                "unsupported codebook W4A8 group_size");
    TORCH_CHECK(group_scale.size(0) == n && group_scale.size(1) == k / group_size,
                "group_scale must have shape [N, K/group_size]");
    TORCH_CHECK(activation_scale.numel() == m,
                "activation_scale must contain one value per activation row");
    TORCH_CHECK(channel_scale.numel() == n,
                "channel_scale must contain one value per output channel");
    TORCH_CHECK(w6 ? !codebook.has_value() : (codebook.has_value() && codebook.value().numel() == 16),
                "W4A8 requires a 16-entry codebook; uniform W6A8 requires no codebook");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda() && bias.value().device() == activation.device(),
                    "bias must use the activation CUDA device");
        TORCH_CHECK(bias.value().numel() == n,
                    "bias must contain one value per output channel");
    }

    const at::cuda::CUDAGuard device_guard(activation.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "turing_codebook_w4a8_linear requires sm75 or newer");

    // chunk_rows == 0 selects the production path, a positive value forces
    // the bounded staged path, and -1 forces the inline path for diagnostics.
    // Auto only selects inline decode for the long-sequence tile, whose W8A8
    // contraction is already limited to one resident CTA on SM75.
    const bool force_inline = chunk_rows == -1;
    TORCH_CHECK(chunk_rows >= -1, "chunk_rows must be -1, 0, or a positive multiple of 8");
    // W6 staging reuses the asynchronous Ampere INT8 schedule. A40 measurements
    // favor that path over register decode; keep SM75's bounded inline tile,
    // and retain explicit inline diagnostics on every supported architecture.
    const bool prefer_inline = !w6 || properties->major == 7;
    const bool inline_decode = group_size == 16 && m > 8192 && chunk_rows <= 0
        && (force_inline || prefer_inline);
    constexpr int64_t default_chunk_rows = 4096;
    if (inline_decode) {
        chunk_rows = 0;
    } else if (chunk_rows == 0) {
        chunk_rows = std::min<int64_t>(n, default_chunk_rows);
    } else {
        TORCH_CHECK(chunk_rows % 8 == 0, "chunk_rows must be divisible by 8");
        chunk_rows = std::min<int64_t>(n, chunk_rows);
    }
    TORCH_CHECK(!force_inline || inline_decode,
                "forced inline codebook W4A8 requires group_size=16 and M>8192");

    at::Tensor workspace;
    if (!inline_decode) {
        TORCH_CHECK(chunk_rows > 0, "codebook W4A8 could not select a valid chunk size");
        workspace = at::empty({chunk_rows, k}, activation.options().dtype(at::kChar));
    }
    at::Tensor output = at::empty(
        {m, n}, activation.options().dtype(at::kBFloat16));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_codebook_w4a8_linear(
        from_torch(activation),
        from_torch(weight),
        from_torch(activation_scale),
        from_torch(group_scale),
        from_torch(channel_scale),
        maybe_tensor(codebook),
        maybe_tensor(bias),
        workspace.defined() ? from_torch(workspace) : Tensor{},
        from_torch(output),
        int_cast<int>(group_size),
        inline_decode);
    return output;
}

std::tuple<at::Tensor, at::Tensor> turing_nvfp4_convrot_quantize(
    at::Tensor weight, at::Tensor blocks, at::Tensor tensor_scale, int64_t first_row, int64_t rows, int64_t input_columns,
    c10::optional<at::Tensor> output_buffer = c10::nullopt, c10::optional<at::Tensor> scale_buffer = c10::nullopt) {
    check_cuda_2d(weight, "weight");
    check_cuda_2d(blocks, "block_scale");
    TORCH_CHECK(weight.scalar_type() == at::kByte && blocks.scalar_type() == at::kByte,
                "NVFP4 ConvRot requires packed uint8 weight and raw E4M3 scale bytes");
    TORCH_CHECK(weight.device() == blocks.device() && weight.device() == tensor_scale.device(),
                "NVFP4 ConvRot tensors must share a CUDA device");
    TORCH_CHECK(weight.is_contiguous() && blocks.is_contiguous(), "NVFP4 storage must be contiguous");
    TORCH_CHECK(tensor_scale.numel() == 1, "tensor_scale must be scalar");
    const int64_t n = weight.size(0), stored_k = weight.size(1) * 2;
    const int64_t logical_k = input_columns < 0 ? stored_k : input_columns;
    TORCH_CHECK(logical_k > 0 && logical_k <= stored_k, "invalid NVFP4 input_columns");
    const int64_t k = ((logical_k + 255) / 256) * 256;
    rows = rows < 0 ? n - first_row : rows;
    TORCH_CHECK(first_row >= 0 && rows > 0 && first_row <= n - rows && n <= INT_MAX,
                "NVFP4 ConvRot row range is invalid");
    TORCH_CHECK(k > 0 && k <= INT_MAX && k % 256 == 0, "NVFP4 ConvRot requires K divisible by 256");
    TORCH_CHECK(blocks.size(0) >= ((n + 127) / 128) * 128 &&
                    blocks.size(1) >= (stored_k + 15) / 16 && blocks.size(1) % 4 == 0,
                "NVFP4 ConvRot requires padded cuBLAS scale storage");
    const at::cuda::CUDAGuard guard(weight.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "NVFP4 ConvRot requires sm75 or newer");
    tensor_scale = tensor_scale.to(at::kFloat).contiguous();
    auto output = output_buffer.has_value() ? *output_buffer : at::empty({rows, k}, weight.options().dtype(at::kChar));
    auto scales = scale_buffer.has_value() ? *scale_buffer : at::empty({rows}, weight.options().dtype(at::kFloat));
    TORCH_CHECK(output.device() == weight.device() && output.scalar_type() == at::kChar &&
                    output.is_contiguous() && output.dim() == 2 && output.size(0) == rows && output.size(1) == k,
                "NVFP4 output buffer must be contiguous CUDA int8 [rows, padded_k]");
    TORCH_CHECK(scales.device() == weight.device() && scales.scalar_type() == at::kFloat &&
                    scales.is_contiguous() && scales.dim() == 1 && scales.numel() == rows,
                "NVFP4 scale buffer must be contiguous CUDA float32 [rows]");
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_nvfp4_convrot_quantize(
        from_torch(weight), from_torch(blocks), from_torch(tensor_scale),
        from_torch(output), from_torch(scales), first_row, logical_k);
    return {output, scales};
}

at::Tensor int8_linear_impl(at::Tensor activation,
                              at::Tensor weight,
                              at::Tensor activation_scale,
                              at::Tensor weight_scale,
                              std::optional<at::Tensor> bias,
                              at::ScalarType output_dtype) {
    activation = activation.contiguous();
    weight = weight.contiguous();
    activation_scale = activation_scale.reshape({-1}).to(at::kFloat).contiguous();
    weight_scale = weight_scale.reshape({-1}).to(at::kFloat).contiguous();
    if (bias.has_value()) {
        if (output_dtype == at::kHalf) {
            bias = bias.value().to(at::kHalf);
        }
        bias = bias.value().reshape({-1}).to(at::kFloat).contiguous();
    }
    check_cuda_2d(activation, "activation");
    check_cuda_2d(weight, "weight");
    TORCH_CHECK(activation.scalar_type() == at::kChar && weight.scalar_type() == at::kChar,
                "Turing INT8 linear activation and weight must be int8");
    TORCH_CHECK(activation.device() == weight.device() &&
                    activation.device() == activation_scale.device() &&
                    activation.device() == weight_scale.device(),
                "Turing INT8 linear tensors must use the same CUDA device");
    TORCH_CHECK(weight.size(1) == activation.size(1), "INT8 linear K dimensions must match");
    TORCH_CHECK(activation.size(1) % 16 == 0 && weight.size(0) % 8 == 0,
                "Turing INT8 linear requires K%16=0 and N%8=0");
    TORCH_CHECK(activation_scale.numel() == activation.size(0),
                "activation_scale must contain one value per row");
    TORCH_CHECK(weight_scale.numel() == weight.size(0),
                "weight_scale must contain one value per output channel");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda() && bias.value().device() == activation.device() &&
                        bias.value().numel() == weight.size(0),
                    "bias must contain one value per output channel on the same CUDA device");
    }
    const at::cuda::CUDAGuard device_guard(activation.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "turing_int8_linear requires sm75 or newer");
    at::Tensor output = at::empty(
        {activation.size(0), weight.size(0)}, activation.options().dtype(output_dtype));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_int8_linear(
        from_torch(activation),
        from_torch(weight),
        from_torch(activation_scale),
        from_torch(weight_scale),
        maybe_tensor(bias),
        from_torch(output));
    return output;
}

at::Tensor turing_int8_batched_residual(at::Tensor x, at::Tensor w,
                                      at::Tensor xs, at::Tensor ws, at::Tensor residual) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 3 && w.dim() == 3 && residual.dim() == 3,
                "batched INT8 residual GEMM expects CUDA [H,M,K], [H,N,K], [H,M,R]");
    TORCH_CHECK(x.scalar_type() == at::kChar && w.scalar_type() == at::kChar &&
                residual.scalar_type() == at::kBFloat16 && xs.scalar_type() == at::kFloat &&
                ws.scalar_type() == at::kFloat, "batched INT8 residual GEMM dtype mismatch");
    for (const auto &t : {w, xs, ws, residual})
        TORCH_CHECK(t.device() == x.device(), "batched INT8 residual GEMM device mismatch");
    const int64_t h=x.size(0), m=x.size(1), k=x.size(2), n=w.size(1);
    TORCH_CHECK(h>0 && h<=65535 && m>0 && n>0 && k>0 && k%16==0 && n%8==0 &&
                w.size(0)==h && w.size(2)==k && xs.numel()==h*m && ws.numel()==h*n &&
                residual.size(0)==h && residual.size(1)==m && residual.size(2)>=n &&
                residual.size(2)%8==0, "batched INT8 residual GEMM shape/alignment mismatch");
    const c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(getCurrentDeviceProperties()->major >= 8 ||
                (getCurrentDeviceProperties()->major == 7 && getCurrentDeviceProperties()->minor >= 5),
                "batched INT8 residual GEMM requires SM75 or newer");
    x=x.contiguous(); w=w.contiguous(); residual=residual.contiguous();
    if (reinterpret_cast<uintptr_t>(x.data_ptr())%16) x=x.clone();
    if (reinterpret_cast<uintptr_t>(w.data_ptr())%16) w=w.clone();
    if (reinterpret_cast<uintptr_t>(residual.data_ptr())%16) residual=residual.clone();
    xs=xs.contiguous(); ws=ws.contiguous();
    auto out=at::empty({h,m,n}, residual.options());
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_int8_batched_residual(
        from_torch(x), from_torch(w), from_torch(xs), from_torch(ws), from_torch(residual), from_torch(out));
    return out;
}

at::Tensor turing_int8_linear(at::Tensor activation, at::Tensor weight,
                              at::Tensor activation_scale, at::Tensor weight_scale,
                              std::optional<at::Tensor> bias) {
    return int8_linear_impl(activation, weight, activation_scale, weight_scale, bias, at::kBFloat16);
}

at::Tensor turing_fp16_int8_linear(at::Tensor activation, at::Tensor weight,
                                   at::Tensor activation_scale, at::Tensor weight_scale,
                                   std::optional<at::Tensor> bias) {
    return int8_linear_impl(activation, weight, activation_scale, weight_scale, bias, at::kHalf);
}

at::Tensor turing_int8_linear_out(at::Tensor activation,
                                  at::Tensor weight,
                                  at::Tensor activation_scale,
                                  at::Tensor weight_scale,
                                  std::optional<at::Tensor> bias,
                                  at::Tensor output) {
    activation = activation.contiguous();
    weight = weight.contiguous();
    activation_scale = activation_scale.reshape({-1}).to(at::kFloat).contiguous();
    weight_scale = weight_scale.reshape({-1}).to(at::kFloat).contiguous();
    if (bias.has_value()) {
        bias = bias.value().reshape({-1}).to(at::kFloat).contiguous();
    }
    check_cuda_2d(activation, "activation");
    check_cuda_2d(weight, "weight");
    TORCH_CHECK(activation.scalar_type() == at::kChar && weight.scalar_type() == at::kChar,
                "Turing INT8 linear activation and weight must be int8");
    TORCH_CHECK(output.is_cuda() && output.dim() == 2 &&
                    (output.scalar_type() == at::kBFloat16 || output.scalar_type() == at::kHalf || output.scalar_type() == at::kFloat) && output.stride(1) == 1,
                "INT8 direct output must be a row-major FP16/BF16/FP32 CUDA matrix");
    TORCH_CHECK(output.size(0) == activation.size(0) &&
                    output.size(1) == weight.size(0) &&
                    output.stride(0) >= output.size(1),
                "INT8 direct output shape/stride is incompatible");
    TORCH_CHECK(activation.device() == weight.device() &&
                    activation.device() == activation_scale.device() &&
                    activation.device() == weight_scale.device() &&
                    activation.device() == output.device(),
                "Turing INT8 linear tensors must use the same CUDA device");
    TORCH_CHECK(weight.size(1) == activation.size(1), "INT8 linear K dimensions must match");
    TORCH_CHECK(activation.size(1) % 16 == 0 && weight.size(0) % 8 == 0,
                "Turing INT8 linear requires K%16=0 and N%8=0");
    TORCH_CHECK(activation_scale.numel() == activation.size(0),
                "activation_scale must contain one value per row");
    TORCH_CHECK(weight_scale.numel() == weight.size(0),
                "weight_scale must contain one value per output channel");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda() && bias.value().device() == activation.device() &&
                        bias.value().numel() == weight.size(0),
                    "bias must contain one value per output channel on the same CUDA device");
    }
    const at::cuda::CUDAGuard device_guard(activation.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "turing_int8_linear_out requires sm75 or newer");
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_int8_linear(
        from_torch(activation),
        from_torch(weight),
        from_torch(activation_scale),
        from_torch(weight_scale),
        maybe_tensor(bias),
        from_torch(output), true);
    return output;
}

at::Tensor turing_dequantize_int8_bf16(at::Tensor accumulator,
                                        at::Tensor activation_scale,
                                        at::Tensor weight_scale,
                                        int64_t output_columns) {
    accumulator = accumulator.contiguous();
    activation_scale = activation_scale.reshape({-1}).to(at::kFloat).contiguous();
    weight_scale = weight_scale.reshape({-1}).to(at::kFloat).contiguous();
    check_cuda_2d(accumulator, "accumulator");
    TORCH_CHECK(accumulator.scalar_type() == at::kInt,
                "accumulator must be int32");
    TORCH_CHECK(activation_scale.is_cuda() && weight_scale.is_cuda(),
                "scales must be CUDA tensors");
    TORCH_CHECK(accumulator.device() == activation_scale.device() &&
                    accumulator.device() == weight_scale.device(),
                "accumulator and scales must use the same CUDA device");

    const int64_t rows = accumulator.size(0);
    const int64_t accumulator_columns = accumulator.size(1);
    if (output_columns < 0) {
        output_columns = accumulator_columns;
    }
    TORCH_CHECK(rows > 0 && accumulator_columns > 0,
                "INT8 BF16 epilogue dimensions must be positive");
    TORCH_CHECK(output_columns > 0 && output_columns <= accumulator_columns,
                "output_columns must be positive and no larger than the accumulator width");
    TORCH_CHECK(activation_scale.numel() == rows,
                "activation_scale must contain one value per accumulator row");
    TORCH_CHECK(weight_scale.numel() == 1 || weight_scale.numel() == output_columns,
                "weight_scale must be scalar or contain one value per output column");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    accumulator_columns <= std::numeric_limits<int>::max() &&
                    output_columns <= std::numeric_limits<int>::max(),
                "INT8 BF16 epilogue dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(accumulator.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "INT8 BF16 epilogue requires sm75 or newer");

    at::Tensor output = at::empty(
        {rows, output_columns}, accumulator.options().dtype(at::kBFloat16));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_dequantize_int8_bf16(
        from_torch(accumulator),
        from_torch(activation_scale),
        from_torch(weight_scale),
        from_torch(output));
    return output;
}

std::tuple<at::Tensor, at::Tensor> turing_swiglu_int8_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256, "SwiGLU staged ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0, "SwiGLU input width must be even");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0, "SwiGLU staged ConvRot requires at least one row");
    TORCH_CHECK(hidden > 0 && hidden % group_size == 0,
                "activated SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "SwiGLU staged ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "SwiGLU staged ConvRot requires sm75 or newer");

    at::Tensor rotated = at::empty({rows, hidden}, input.options());
    at::Tensor partial_absmax = at::empty(
        {rows, hidden / group_size}, input.options().dtype(at::kFloat));
    at::Tensor output = at::empty({rows, hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int8_convrot_quantize(
        from_torch(input),
        from_torch(rotated),
        from_torch(partial_absmax),
        from_torch(output),
        from_torch(scales));
    return {output, scales};
}

at::Tensor turing_swiglu_int8_convrot_quantize_scaled(
    at::Tensor input, at::Tensor scales, int64_t group_size) {
    input = input.contiguous();
    scales = scales.reshape({-1}).to(at::kFloat).contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256,
                "scaled SwiGLU ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0,
                "scaled SwiGLU input width must be even");
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "scaled SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(scales.is_cuda() && scales.device() == input.device() &&
                    scales.numel() == rows,
                "scaled SwiGLU requires one FP32 scale per input row");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "scaled SwiGLU dimensions exceed the CUDA kernel range");
    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "scaled SwiGLU ConvRot requires sm75 or newer");
    at::Tensor output = at::empty(
        {rows, hidden}, input.options().dtype(at::kChar));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int8_convrot_quantize_scaled(
        from_torch(input), from_torch(scales), from_torch(output));
    return output;
}

at::Tensor turing_swiglu_int8_convrot_quantize_scaled_out(
    at::Tensor input,
    at::Tensor scales,
    at::Tensor output,
    int64_t group_size) {
    input = input.contiguous();
    scales = scales.reshape({-1}).to(at::kFloat).contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256,
                "scaled SwiGLU ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0,
                "scaled SwiGLU input width must be even");
    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "scaled SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(scales.is_cuda() && scales.device() == input.device() &&
                    scales.numel() == rows,
                "scaled SwiGLU requires one FP32 scale per input row");
    TORCH_CHECK(output.is_cuda() && output.device() == input.device() &&
                    output.dim() == 2 && output.scalar_type() == at::kChar &&
                    output.size(0) == rows && output.size(1) == hidden &&
                    output.stride(1) == 1 && output.stride(0) >= hidden,
                "scaled SwiGLU direct output shape/stride is incompatible");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max() &&
                    output.stride(0) <= std::numeric_limits<int>::max(),
                "scaled SwiGLU dimensions exceed the CUDA kernel range");
    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "scaled SwiGLU ConvRot requires sm75 or newer");
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int8_convrot_quantize_scaled(
        from_torch(input), from_torch(scales), from_torch(output));
    return output;
}

void turing_swiglu_convrot_shard_inplace(
    at::Tensor gate,
    at::Tensor up,
    at::Tensor partial_absmax,
    int64_t channel_offset) {
    check_cuda_2d(gate, "gate");
    check_cuda_2d(up, "up");
    check_cuda_2d(partial_absmax, "partial_absmax");
    check_half_like(gate, "gate");
    check_half_like(up, "up");
    TORCH_CHECK(gate.is_contiguous() && up.is_contiguous() &&
                    partial_absmax.is_contiguous(),
                "sharded SwiGLU ConvRot tensors must be contiguous");
    TORCH_CHECK(gate.device() == up.device() &&
                    gate.device() == partial_absmax.device(),
                "sharded SwiGLU ConvRot tensors must share a CUDA device");
    TORCH_CHECK(gate.scalar_type() == up.scalar_type(),
                "gate and up must use the same dtype");
    TORCH_CHECK(partial_absmax.scalar_type() == at::kFloat,
                "partial_absmax must use float32 storage");
    const int64_t rows = gate.size(0);
    const int64_t hidden = gate.size(1);
    const int64_t shard_width = up.size(1);
    TORCH_CHECK(rows > 0 && up.size(0) == rows,
                "gate and up row counts must match");
    TORCH_CHECK(hidden > 0 && hidden % 256 == 0 &&
                    shard_width > 0 && shard_width % 256 == 0 &&
                    channel_offset >= 0 && channel_offset % 256 == 0 &&
                    channel_offset + shard_width <= hidden,
                "sharded SwiGLU intervals must be 256-aligned and in range");
    TORCH_CHECK(partial_absmax.size(0) == rows &&
                    partial_absmax.size(1) == hidden / 256,
                "partial_absmax must have shape [rows, hidden / 256]");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max() &&
                    shard_width <= std::numeric_limits<int>::max() &&
                    channel_offset <= std::numeric_limits<int>::max(),
                "sharded SwiGLU dimensions exceed the CUDA kernel range");
    const at::cuda::CUDAGuard device_guard(gate.device());
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_convrot_shard_inplace(
        from_torch(gate),
        from_torch(up),
        from_torch(partial_absmax),
        static_cast<int>(channel_offset));
}

std::tuple<at::Tensor, at::Tensor>
turing_int8_convrot_quantize_from_partials(
    at::Tensor rotated,
    at::Tensor partial_absmax) {
    check_cuda_2d(rotated, "rotated");
    check_cuda_2d(partial_absmax, "partial_absmax");
    check_half_like(rotated, "rotated");
    TORCH_CHECK(rotated.is_contiguous() && partial_absmax.is_contiguous(),
                "ConvRot partial reduction tensors must be contiguous");
    TORCH_CHECK(rotated.device() == partial_absmax.device() &&
                    partial_absmax.scalar_type() == at::kFloat,
                "ConvRot partial reduction requires same-device FP32 partials");
    const int64_t rows = rotated.size(0);
    const int64_t hidden = rotated.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % 256 == 0 &&
                    partial_absmax.size(0) == rows &&
                    partial_absmax.size(1) == hidden / 256,
                "ConvRot partial reduction shape is incompatible");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "ConvRot partial reduction dimensions exceed the CUDA kernel range");
    const at::cuda::CUDAGuard device_guard(rotated.device());
    at::Tensor output = at::empty(
        {rows, hidden}, rotated.options().dtype(at::kChar));
    at::Tensor scales = at::empty(
        {rows, 1}, rotated.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_int8_convrot_quantize_from_partials(
        from_torch(rotated),
        from_torch(partial_absmax),
        from_torch(output),
        from_torch(scales));
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_swiglu_int4_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256, "SwiGLU staged INT4 ConvRot only supports group_size=256");
    TORCH_CHECK(input.size(1) % 2 == 0, "SwiGLU input width must be even");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1) / 2;
    TORCH_CHECK(rows > 0, "SwiGLU staged INT4 ConvRot requires at least one row");
    TORCH_CHECK(hidden > 0 && hidden % group_size == 0,
                "activated SwiGLU width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "SwiGLU staged INT4 ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "SwiGLU staged INT4 ConvRot requires sm75 or newer");

    at::Tensor rotated = at::empty({rows, hidden}, input.options());
    at::Tensor partial_absmax = at::empty(
        {rows, hidden / group_size}, input.options().dtype(at::kFloat));
    at::Tensor output = at::empty({rows, hidden / 2}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_swiglu_int4_convrot_quantize(
        from_torch(input),
        from_torch(rotated),
        from_torch(partial_absmax),
        from_torch(output),
        from_torch(scales));
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_gelu_convrot_quantize(
    at::Tensor input, int64_t group_size, bool int4) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    check_half_like(input, "input");
    TORCH_CHECK(group_size == 256,
                "GELU staged ConvRot only supports group_size=256");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0, "GELU staged ConvRot requires at least one row");
    TORCH_CHECK(hidden > 0 && hidden % group_size == 0,
                "activated GELU width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "GELU staged ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "GELU staged ConvRot requires sm75 or newer");

    at::Tensor rotated = at::empty({rows, hidden}, input.options());
    at::Tensor partial_absmax = at::empty(
        {rows, hidden / group_size}, input.options().dtype(at::kFloat));
    at::Tensor output = at::empty(
        {rows, int4 ? hidden / 2 : hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));

    TorchOpContext ctx;
    if (int4) {
        comfyui_turing_utils::kernels::turing_gelu_int4_convrot_quantize(
            from_torch(input), from_torch(rotated), from_torch(partial_absmax),
            from_torch(output), from_torch(scales));
    } else {
        comfyui_turing_utils::kernels::turing_gelu_int8_convrot_quantize(
            from_torch(input), from_torch(rotated), from_torch(partial_absmax),
            from_torch(output), from_torch(scales));
    }
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_gelu_int8_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    return turing_gelu_convrot_quantize(std::move(input), group_size, false);
}

std::tuple<at::Tensor, at::Tensor> turing_gelu_int4_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    return turing_gelu_convrot_quantize(std::move(input), group_size, true);
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_int8_convrot_quantize(
    at::Tensor input, int64_t group_size, bool swiglu, int64_t forced_threads) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "BF16 row-buffer ConvRot input must be bfloat16");
    TORCH_CHECK(group_size == 256,
                "BF16 row-buffer ConvRot only supports group_size=256");
    TORCH_CHECK(forced_threads == 0 || forced_threads == 512 ||
                    forced_threads == 768 || forced_threads == 1024,
                "forced_threads must be 0, 512, 768, or 1024");

    const int64_t rows = input.size(0);
    const int64_t input_columns = input.size(1);
    TORCH_CHECK(!swiglu || input_columns % 2 == 0,
                "SwiGLU BF16 row-buffer ConvRot input width must be even");
    const int64_t hidden = swiglu ? input_columns / 2 : input_columns;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "BF16 row-buffer ConvRot width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "BF16 row-buffer ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "BF16 row-buffer ConvRot requires sm75 or newer");
    const int block_threads = select_bf16_convrot_threads(
        properties, hidden, rows, static_cast<int>(forced_threads));

    at::Tensor output = at::empty(
        {rows, hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty(
        {rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_bf16_int8_convrot_quantize(
        from_torch(input),
        from_torch(output),
        from_torch(scales),
        swiglu,
        block_threads);
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_fp16_int8_convrot_quantize(
    at::Tensor input, int64_t group_size, int64_t input_act,
    std::optional<at::Tensor> input_act_weight, double input_act_eps) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kHalf, "FP16 ConvRot input must be float16");
    TORCH_CHECK(group_size == 256, "FP16 ConvRot requires group_size=256");
    TORCH_CHECK(input_act >= 0 && input_act <= 3, "Unsupported FP16 ConvRot activation");
    TORCH_CHECK(input_act != 2 || input.size(1) % 2 == 0, "SwiGLU input width must be even");
    const int64_t rows = input.size(0);
    const int64_t hidden = input_act == 2 ? input.size(1) / 2 : input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0 &&
                rows <= INT_MAX && hidden <= INT_MAX && hidden % 256 == 0,
                "FP16 ConvRot requires positive dimensions and K divisible by 256");
    if (input_act == 3) {
        TORCH_CHECK(input_act_weight.has_value(), "RMSNorm requires input_act_weight");
        auto &weight = *input_act_weight;
        TORCH_CHECK(weight.device() == input.device() && weight.dim() == 1 && weight.numel() == hidden,
                    "RMSNorm weight must be a same-device vector of length K");
        TORCH_CHECK(weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kFloat,
                    "RMSNorm weight must be float16 or float32");
        TORCH_CHECK(std::isfinite(input_act_eps) && input_act_eps >= 0,
                    "RMSNorm epsilon must be finite and nonnegative");
        weight = weight.contiguous();
    }
    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "FP16 quantization requires sm75 or newer");
    auto output = at::empty({rows, hidden}, input.options().dtype(at::kChar));
    auto scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_fp16_int8_convrot_quantize(
        from_torch(input), from_torch(output), from_torch(scales), input_act,
        maybe_tensor(input_act_weight), input_act_eps);
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_int4_convrot_quantize(
    at::Tensor input, int64_t group_size, bool swiglu) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "BF16 row-buffer INT4 ConvRot input must be bfloat16");
    TORCH_CHECK(group_size == 256,
                "BF16 row-buffer INT4 ConvRot only supports group_size=256");

    const int64_t rows = input.size(0);
    const int64_t input_columns = input.size(1);
    TORCH_CHECK(!swiglu || input_columns % 2 == 0,
                "SwiGLU BF16 row-buffer INT4 ConvRot input width must be even");
    const int64_t hidden = swiglu ? input_columns / 2 : input_columns;
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "BF16 row-buffer INT4 ConvRot width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "BF16 row-buffer INT4 ConvRot dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "BF16 row-buffer INT4 ConvRot requires sm75 or newer");
    const int block_threads = select_bf16_convrot_threads(properties, hidden, rows);

    at::Tensor output = at::empty({rows, hidden / 2}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_bf16_int4_convrot_quantize(
        from_torch(input),
        from_torch(output),
        from_torch(scales),
        swiglu,
        block_threads);
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_gelu_convrot_quantize(
    at::Tensor input, int64_t group_size, bool int4) {
    input = input.contiguous();
    check_cuda_2d(input, "input");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16,
                "BF16 GELU row-buffer ConvRot input must be bfloat16");
    TORCH_CHECK(group_size == 256,
                "BF16 GELU row-buffer ConvRot only supports group_size=256");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0 && hidden % group_size == 0,
                "BF16 GELU row-buffer width must be positive and divisible by 256");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max(),
                "BF16 GELU row-buffer dimensions exceed the CUDA kernel range");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "BF16 GELU row-buffer ConvRot requires sm75 or newer");
    const int block_threads = select_bf16_convrot_threads(properties, hidden, rows);

    at::Tensor output = at::empty(
        {rows, int4 ? hidden / 2 : hidden}, input.options().dtype(at::kChar));
    at::Tensor scales = at::empty({rows, 1}, input.options().dtype(at::kFloat));
    TorchOpContext ctx;
    if (int4) {
        comfyui_turing_utils::kernels::turing_bf16_gelu_int4_convrot_quantize(
            from_torch(input), from_torch(output), from_torch(scales), block_threads);
    } else {
        comfyui_turing_utils::kernels::turing_bf16_gelu_int8_convrot_quantize(
            from_torch(input), from_torch(output), from_torch(scales), block_threads);
    }
    return {output, scales};
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_gelu_int8_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    return turing_bf16_gelu_convrot_quantize(std::move(input), group_size, false);
}

std::tuple<at::Tensor, at::Tensor> turing_bf16_gelu_int4_convrot_quantize(
    at::Tensor input, int64_t group_size) {
    return turing_bf16_gelu_convrot_quantize(std::move(input), group_size, true);
}

void check_modulation_segments(const at::Tensor &segments, int64_t rows) {
    TORCH_CHECK(segments.is_cuda() && segments.is_contiguous(),
                "modulation indices must be contiguous on CUDA");
    TORCH_CHECK((segments.dim() == 2 && segments.scalar_type() == at::kInt && segments.size(1) == 3) ||
                    (segments.dim() == 1 && segments.scalar_type() == at::kLong && segments.size(0) == rows),
                "segments must be int32 [S, 3] triples or int64 [M] per-token indices");
    TORCH_CHECK(segments.size(0) > 0, "modulation indices must not be empty");
}

at::Tensor turing_segmented_rms_adaln(at::Tensor input,
                                       at::Tensor weight,
                                       at::Tensor scale,
                                       at::Tensor shift,
                                       at::Tensor segments,
                                       double epsilon) {
    input = input.contiguous();
    weight = weight.contiguous();
    segments = segments.contiguous();
    check_cuda_2d(input, "input");
    check_float_like(input, "input");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.is_contiguous(),
                "weight must be a contiguous 1D CUDA tensor");
    TORCH_CHECK(scale.is_cuda() && scale.dim() == 2 && scale.stride(1) == 1,
                "scale must be a CUDA matrix with contiguous rows");
    TORCH_CHECK(shift.is_cuda() && shift.dim() == 2 && shift.stride(1) == 1,
                "shift must be a CUDA matrix with contiguous rows");
    check_modulation_segments(segments, input.size(0));
    TORCH_CHECK(input.device() == weight.device() &&
                    input.device() == scale.device() &&
                    input.device() == shift.device() &&
                    input.device() == segments.device(),
                "all segmented RMSNorm+AdaLN tensors must use the same CUDA device");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type() &&
                    scale.scalar_type() == input.scalar_type() &&
                    shift.scalar_type() == input.scalar_type(),
                "weight, scale, and shift dtypes must match input");

    const int64_t rows = input.size(0);
    const int64_t hidden = input.size(1);
    TORCH_CHECK(rows > 0 && hidden > 0,
                "segmented RMSNorm+AdaLN input dimensions must be positive");
    TORCH_CHECK(weight.numel() == hidden,
                "RMSNorm weight length must match the input hidden dimension");
    TORCH_CHECK(scale.size(0) > 0 && scale.size(1) == hidden,
                "scale must be [modulation_rows, hidden]");
    TORCH_CHECK(shift.sizes() == scale.sizes(), "shift shape must match scale shape");
    TORCH_CHECK(segments.size(0) > 0, "segments must contain at least one row");
    TORCH_CHECK(rows <= std::numeric_limits<int>::max() &&
                    hidden <= std::numeric_limits<int>::max() &&
                    scale.size(0) <= std::numeric_limits<int>::max() &&
                    scale.stride(0) <= std::numeric_limits<int>::max() &&
                    shift.stride(0) <= std::numeric_limits<int>::max() &&
                    segments.size(0) <= std::numeric_limits<int>::max(),
                "segmented RMSNorm+AdaLN dimensions exceed the CUDA kernel range");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
                "RMSNorm epsilon must be finite and positive");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "segmented RMSNorm+AdaLN requires sm75 or newer");

    at::Tensor output = at::empty_like(input);
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_segmented_rms_adaln(
        from_torch(input),
        from_torch(weight),
        from_torch(scale),
        from_torch(shift),
        from_torch(segments),
        from_torch(output),
        static_cast<float>(epsilon));
    return output;
}

void turing_segmented_mod_gate(at::Tensor input,
                               at::Tensor gate,
                               at::Tensor residual,
                               at::Tensor segments) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(),
                "input must be a contiguous 2D CUDA tensor");
    check_float_like(input, "input");
    TORCH_CHECK(gate.is_cuda() && gate.dim() == 2 && gate.stride(1) == 1,
                "gate must be a CUDA matrix with contiguous rows");
    TORCH_CHECK(residual.is_cuda() && residual.dim() == 2 && residual.is_contiguous(),
                "residual must be a contiguous 2D CUDA tensor");
    check_modulation_segments(segments, input.size(0));
    TORCH_CHECK(input.device() == gate.device() && input.device() == residual.device() &&
                    input.device() == segments.device(),
                "all segmented mod-gate tensors must use the same CUDA device");
    TORCH_CHECK(input.scalar_type() == gate.scalar_type() &&
                    input.scalar_type() == residual.scalar_type(),
                "input, gate, and residual dtypes must match");
    TORCH_CHECK(input.sizes() == residual.sizes() && gate.size(1) == input.size(1),
                "residual must match input and gate must match hidden width");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && gate.size(0) > 0 &&
                    segments.size(0) > 0,
                "segmented mod-gate dimensions must be positive");
    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 || (properties->major == 7 && properties->minor >= 5),
                "segmented mod-gate requires sm75 or newer");
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_segmented_mod_gate(
        from_torch(input), from_torch(gate), from_torch(residual), from_torch(segments));
}

at::Tensor turing_segmented_mod_gate_rms_adaln(at::Tensor input,
                                                at::Tensor gate,
                                                at::Tensor residual,
                                                at::Tensor weight,
                                                at::Tensor scale,
                                                at::Tensor shift,
                                                at::Tensor segments,
                                                double epsilon) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(),
                "input must be a contiguous 2D CUDA tensor");
    check_float_like(input, "input");
    TORCH_CHECK(gate.is_cuda() && gate.dim() == 2 && gate.stride(1) == 1 &&
                    residual.is_cuda() && residual.dim() == 2 && residual.is_contiguous(),
                "gate and residual must be CUDA matrices with contiguous rows");
    check_modulation_segments(segments, input.size(0));
    TORCH_CHECK(input.device() == gate.device() && input.device() == residual.device() &&
                    input.device() == segments.device(),
                "all fused segmented tensors must use the same CUDA device");
    TORCH_CHECK(input.scalar_type() == gate.scalar_type() &&
                    input.scalar_type() == residual.scalar_type() &&
                    input.sizes() == residual.sizes() && gate.size(1) == input.size(1),
                "gate/residual dtype and shape must match input");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 1 && weight.is_contiguous() &&
                    weight.numel() == input.size(1),
                "weight must be a contiguous CUDA vector matching hidden width");
    TORCH_CHECK(scale.is_cuda() && scale.dim() == 2 && scale.stride(1) == 1 &&
                    shift.is_cuda() && shift.dim() == 2 && shift.stride(1) == 1 &&
                    scale.sizes() == shift.sizes() && scale.size(1) == input.size(1),
                "scale and shift must be matching CUDA matrices");
    TORCH_CHECK(weight.scalar_type() == input.scalar_type() &&
                    scale.scalar_type() == input.scalar_type() &&
                    shift.scalar_type() == input.scalar_type(),
                "normalization parameter dtypes must match input");
    TORCH_CHECK(input.device() == weight.device() && input.device() == scale.device() &&
                    input.device() == shift.device(),
                "normalization parameters must use the input CUDA device");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
                "RMSNorm epsilon must be finite and positive");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && scale.size(0) > 0 &&
                    gate.sizes() == scale.sizes(), "gate and scale must have matching positive dimensions");
    const at::cuda::CUDAGuard device_guard(input.device());
    at::Tensor output = at::empty_like(input);
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_segmented_mod_gate_rms_adaln(
        from_torch(input), from_torch(gate), from_torch(residual),
        from_torch(weight), from_torch(scale), from_torch(shift),
        from_torch(segments), from_torch(output), static_cast<float>(epsilon));
    return output;
}

at::Tensor turing_layer_norm_adaln(at::Tensor input,
                                    at::Tensor scale,
                                    at::Tensor shift,
                                    double epsilon) {
    input = input.contiguous();
    scale = scale.contiguous();
    shift = shift.contiguous();
    TORCH_CHECK(input.is_cuda() && input.dim() == 3 && input.is_contiguous(),
                "input must be contiguous CUDA [batch, sequence, hidden]");
    check_float_like(input, "input");
    TORCH_CHECK(scale.is_cuda() && scale.dim() == 3 && scale.is_contiguous(),
                "scale must be contiguous CUDA [batch, modulation_steps, hidden]");
    TORCH_CHECK(shift.is_cuda() && shift.dim() == 3 && shift.is_contiguous(),
                "shift must be contiguous CUDA [batch, modulation_steps, hidden]");
    TORCH_CHECK(input.device() == scale.device() && input.device() == shift.device(),
                "LayerNorm+AdaLN tensors must use the same CUDA device");
    TORCH_CHECK(input.scalar_type() == scale.scalar_type() &&
                    input.scalar_type() == shift.scalar_type(),
                "LayerNorm+AdaLN tensor dtypes must match");
    TORCH_CHECK(scale.sizes() == shift.sizes(), "shift shape must match scale shape");
    TORCH_CHECK(scale.size(0) == input.size(0) && scale.size(2) == input.size(2),
                "scale must match input batch and hidden dimensions");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) > 0 && input.size(2) > 0 &&
                    scale.size(1) > 0,
                "LayerNorm+AdaLN dimensions must be positive");
    TORCH_CHECK(input.size(0) * input.size(1) <= std::numeric_limits<int>::max() &&
                    input.size(1) <= std::numeric_limits<int>::max() &&
                    input.size(2) <= std::numeric_limits<int>::max() &&
                    scale.size(1) <= std::numeric_limits<int>::max(),
                "LayerNorm+AdaLN dimensions exceed the CUDA kernel range");
    TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0,
                "LayerNorm epsilon must be finite and positive");

    const at::cuda::CUDAGuard device_guard(input.device());
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    TORCH_CHECK(properties->major > 7 ||
                    (properties->major == 7 && properties->minor >= 5),
                "LayerNorm+AdaLN requires sm75 or newer");
    at::Tensor output = at::empty_like(input);
    TorchOpContext ctx;
    comfyui_turing_utils::kernels::turing_layer_norm_adaln(
        from_torch(input), from_torch(scale), from_torch(shift),
        from_torch(output), static_cast<float>(epsilon));
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("segmented_modulation_schema") = 2;
    m.def("turing_w4a8_linear",
          &turing_w4a8_linear,
          pybind11::arg("activation"),
          pybind11::arg("weight"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias") = std::nullopt);
    m.def("turing_codebook_w4a8_linear",
          &turing_codebook_w4a8_linear,
          pybind11::arg("activation"),
          pybind11::arg("weight"),
          pybind11::arg("activation_scale"),
          pybind11::arg("group_scale"),
          pybind11::arg("channel_scale"),
          pybind11::arg("codebook"),
          pybind11::arg("bias") = std::nullopt,
          pybind11::arg("group_size") = 16,
          pybind11::arg("chunk_rows") = 0);
    m.def("turing_nvfp4_convrot_quantize", [](at::Tensor w, at::Tensor b, at::Tensor s,
            int64_t first, int64_t rows, int64_t k) {
        return turing_nvfp4_convrot_quantize(w, b, s, first, rows, k);
    });
    m.def("turing_nvfp4_convrot_quantize_out", [](at::Tensor w, at::Tensor b, at::Tensor s,
            at::Tensor out, at::Tensor scales, int64_t first, int64_t k) {
        turing_nvfp4_convrot_quantize(w, b, s, first, out.size(0), k, out, scales);
    });
    m.attr("nvfp4_runtime_schema") = 6;
    m.def("turing_int8_batched_residual", &turing_int8_batched_residual,
          pybind11::arg("activation"), pybind11::arg("weight"),
          pybind11::arg("activation_scale"), pybind11::arg("weight_scale"), pybind11::arg("residual"));
    m.def("turing_int8_linear",
          &turing_int8_linear,
          pybind11::arg("activation"),
          pybind11::arg("weight"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias") = std::nullopt);
    m.def("turing_int8_linear_out",
          &turing_int8_linear_out,
          pybind11::arg("activation"),
          pybind11::arg("weight"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("bias"),
          pybind11::arg("output"));
    m.def("turing_fp16_int8_linear", &turing_fp16_int8_linear,
          pybind11::arg("activation"), pybind11::arg("weight"),
          pybind11::arg("activation_scale"), pybind11::arg("weight_scale"),
          pybind11::arg("bias") = std::nullopt);
    m.def("turing_fp16_int8_convrot_quantize", &turing_fp16_int8_convrot_quantize,
          pybind11::arg("input"), pybind11::arg("group_size") = 256,
          pybind11::arg("input_act") = 0,
          pybind11::arg("input_act_weight") = std::nullopt,
          pybind11::arg("input_act_eps") = 0.0);
    m.def("turing_dequantize_int8_bf16",
          &turing_dequantize_int8_bf16,
          pybind11::arg("accumulator"),
          pybind11::arg("activation_scale"),
          pybind11::arg("weight_scale"),
          pybind11::arg("output_columns") = -1);
    m.def("turing_swiglu_int8_convrot_quantize",
          &turing_swiglu_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_swiglu_int8_convrot_quantize_scaled",
          &turing_swiglu_int8_convrot_quantize_scaled,
          pybind11::arg("input"),
          pybind11::arg("scales"),
          pybind11::arg("group_size") = 256);
    m.def("turing_swiglu_int8_convrot_quantize_scaled_out",
          &turing_swiglu_int8_convrot_quantize_scaled_out,
          pybind11::arg("input"),
          pybind11::arg("scales"),
          pybind11::arg("output"),
          pybind11::arg("group_size") = 256);
    m.def("turing_swiglu_convrot_shard_inplace",
          &turing_swiglu_convrot_shard_inplace,
          pybind11::arg("gate"),
          pybind11::arg("up"),
          pybind11::arg("partial_absmax"),
          pybind11::arg("channel_offset"));
    m.def("turing_int8_convrot_quantize_from_partials",
          &turing_int8_convrot_quantize_from_partials,
          pybind11::arg("rotated"),
          pybind11::arg("partial_absmax"));
    m.def("turing_swiglu_int4_convrot_quantize",
          &turing_swiglu_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_gelu_int8_convrot_quantize",
          &turing_gelu_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_gelu_int4_convrot_quantize",
          &turing_gelu_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_bf16_int8_convrot_quantize",
          &turing_bf16_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256,
          pybind11::arg("swiglu") = false,
          pybind11::arg("forced_threads") = 0);
    m.def("turing_bf16_int4_convrot_quantize",
          &turing_bf16_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256,
          pybind11::arg("swiglu") = false);
    m.def("turing_bf16_gelu_int8_convrot_quantize",
          &turing_bf16_gelu_int8_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_bf16_gelu_int4_convrot_quantize",
          &turing_bf16_gelu_int4_convrot_quantize,
          pybind11::arg("input"),
          pybind11::arg("group_size") = 256);
    m.def("turing_segmented_rms_adaln",
          &turing_segmented_rms_adaln,
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("scale"),
          pybind11::arg("shift"),
          pybind11::arg("segments"),
          pybind11::arg("epsilon") = 1.0e-5);
    m.def("turing_segmented_mod_gate",
          &turing_segmented_mod_gate,
          pybind11::arg("input"),
          pybind11::arg("gate"),
          pybind11::arg("residual"),
          pybind11::arg("segments"));
    m.def("turing_segmented_mod_gate_rms_adaln",
          &turing_segmented_mod_gate_rms_adaln,
          pybind11::arg("input"),
          pybind11::arg("gate"),
          pybind11::arg("residual"),
          pybind11::arg("weight"),
          pybind11::arg("scale"),
          pybind11::arg("shift"),
          pybind11::arg("segments"),
          pybind11::arg("epsilon") = 1.0e-5);
    m.def("turing_layer_norm_adaln",
          &turing_layer_norm_adaln,
          pybind11::arg("input"),
          pybind11::arg("scale"),
          pybind11::arg("shift"),
          pybind11::arg("epsilon") = 1.0e-5);
}
