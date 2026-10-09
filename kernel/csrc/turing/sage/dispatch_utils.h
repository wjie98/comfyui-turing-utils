/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "torch_compat.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_MSC_VER)
#define SAGE_FUNC_NAME __FUNCSIG__
#else
#define SAGE_FUNC_NAME __PRETTY_FUNCTION__
#endif

template <typename Kernel>
inline void configure_dynamic_shared_memory(
    Kernel kernel, size_t requested_bytes, const char *kernel_name) {
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  TORCH_CHECK(error == cudaSuccess,
              kernel_name, " could not query the CUDA device: ",
              cudaGetErrorString(error));
  int optin_limit = 0;
  error = cudaDeviceGetAttribute(
      &optin_limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
  TORCH_CHECK(error == cudaSuccess,
              kernel_name, " could not query dynamic shared memory: ",
              cudaGetErrorString(error));
  TORCH_CHECK(requested_bytes <= static_cast<size_t>(optin_limit),
              kernel_name, " requests ", requested_bytes,
              " bytes of dynamic shared memory, but device ", device,
              " supports ", optin_limit);
  error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(requested_bytes));
  TORCH_CHECK(error == cudaSuccess,
              kernel_name, " could not opt in to ", requested_bytes,
              " bytes of dynamic shared memory: ", cudaGetErrorString(error));
}

inline bool attention_kernel_profile_enabled()
{
  static const bool enabled = []() {
    const char *raw = std::getenv("COMFYUI_TURING_UTILS_PROFILE");
    if (raw == nullptr || *raw == '\0')
      return false;
    // Resource queries are reserved for explicit detail mode.
    return raw[0] == '2' && raw[1] == '\0';
  }();
  return enabled;
}

template <typename Kernel>
inline void report_cuda_kernel_profile(
    Kernel kernel,
    const char *operation,
    const std::string &schedule,
    int block_threads,
    size_t dynamic_shared_bytes,
    int grid_x,
    int grid_y,
    int grid_z)
{
  int device = 0;
  cudaDeviceProp properties{};
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  const cudaError_t device_error = cudaGetDevice(&device);
  const cudaError_t properties_error = device_error == cudaSuccess
      ? cudaGetDeviceProperties(&properties, device)
      : device_error;
  const cudaError_t attributes_error = cudaFuncGetAttributes(&attributes, kernel);
  const cudaError_t occupancy_error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, block_threads, dynamic_shared_bytes);
  if (properties_error != cudaSuccess || attributes_error != cudaSuccess ||
      occupancy_error != cudaSuccess)
  {
    std::clog << "[Turing kernel profile] op=" << operation
              << " query_failed device=" << cudaGetErrorString(properties_error)
              << " attributes=" << cudaGetErrorString(attributes_error)
              << " occupancy=" << cudaGetErrorString(occupancy_error)
              << std::endl;
    // Diagnostic queries must not poison the subsequent launch check.
    cudaGetLastError();
    return;
  }

  const int active_warps = active_blocks * block_threads / 32;
  const int maximum_warps = properties.maxThreadsPerMultiProcessor / 32;
  const double occupancy = maximum_warps > 0
      ? 100.0 * static_cast<double>(active_warps) /
            static_cast<double>(maximum_warps)
      : 0.0;
  std::clog << "[Turing kernel profile] op=" << operation
            << " device=" << properties.name
            << " device_sm=sm" << properties.major << properties.minor
            << " binary_sm=sm" << attributes.binaryVersion
            << " ptx_compute=compute_" << attributes.ptxVersion
            << " schedule={" << schedule << "}"
            << " grid=" << grid_x << "x" << grid_y << "x" << grid_z
            << " block_threads=" << block_threads
            << " registers_per_thread=" << attributes.numRegs
            << " static_shared=" << attributes.sharedSizeBytes
            << " dynamic_shared=" << dynamic_shared_bytes
            << " local_bytes=" << attributes.localSizeBytes
            << " active_ctas_per_sm=" << active_blocks
            << " active_warps_per_sm=" << active_warps
            << " occupancy=" << std::fixed << std::setprecision(1)
            << occupancy << "%" << std::endl;
}

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)              \
  if (head_dim == 64) {                                         \
    constexpr int HEAD_DIM = 64;                                \
    __VA_ARGS__                                                 \
  } else if (head_dim == 128) {                                 \
    constexpr int HEAD_DIM = 128;                               \
    __VA_ARGS__                                                 \
  } else {                                                      \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported head dim: " << int(head_dim);       \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_CAUSAL(is_causal, IS_CAUSAL, ...)              \
  if (is_causal == 1) {                                         \
    constexpr bool IS_CAUSAL = true;                            \
    __VA_ARGS__                                                 \
  } else if (is_causal == 0) {                                  \
    constexpr bool IS_CAUSAL = false;                           \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported causal mode: " << int(is_causal);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_QK_QUANT_GRAN(qk_quant_gran, QK_QUANT_GRAN, ...)              \
  if (qk_quant_gran == 1) {                                         \
    constexpr int QK_QUANT_GRAN = 1;                            \
    __VA_ARGS__                                                 \
  } else if (qk_quant_gran == 2) {                                  \
    constexpr int QK_QUANT_GRAN = 2;                            \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported qk_quant_gran: " << int(qk_quant_gran);   \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_RETURN_LSE(return_lse, RETURN_LSE, ...)             \
  if (return_lse == 1) {                                         \
    constexpr bool RETURN_LSE = true;                            \
    __VA_ARGS__                                                  \
  } else if (return_lse == 0) {                                  \
    constexpr bool RETURN_LSE = false;                           \
    __VA_ARGS__                                                  \
  }  else {                                                      \
    std::ostringstream err_msg;                                  \
    err_msg << "Unsupported causal mode: " << int(return_lse);   \
    throw std::invalid_argument(err_msg.str());                  \
  }

#define DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(pytorch_dtype, c_type, ...)                \
  if (pytorch_dtype == at::ScalarType::Half) {                                          \
    using c_type = half;                                                                \
    __VA_ARGS__                                                                         \
  } else if (pytorch_dtype == at::ScalarType::BFloat16) {                               \
    using c_type = nv_bfloat16;                                                         \
    __VA_ARGS__                                                                         \
  } else {                                                                              \
    std::ostringstream oss;                                                             \
    oss << SAGE_FUNC_NAME << " failed to dispatch data type " << pytorch_dtype;         \
    TORCH_CHECK(false, oss.str());                                                      \
  }

#define DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, ...)        \
  if (block_size == 64) {                                       \
    constexpr int BLOCK_SIZE = 64;                              \
    __VA_ARGS__                                                 \
  } else if (block_size == 128) {                               \
    constexpr int BLOCK_SIZE = 128;                             \
    __VA_ARGS__                                                 \
  } else if (block_size == 32) {                                \
    constexpr int BLOCK_SIZE = 32;                              \
    __VA_ARGS__                                                 \
  }  else {                                                     \
    std::ostringstream err_msg;                                 \
    err_msg << "Unsupported block_size " << int(block_size);    \
    throw std::invalid_argument(err_msg.str());                 \
  }

#define DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, ...)  \
  if (warp_block_size == 16) {                                           \
    constexpr int WARP_BLOCK_SIZE = 16;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 32) {                                    \
    constexpr int WARP_BLOCK_SIZE = 32;                                  \
    __VA_ARGS__                                                          \
  } else if (warp_block_size == 64) {                                    \
    constexpr int WARP_BLOCK_SIZE = 64;                                  \
    __VA_ARGS__                                                          \
  }  else {                                                              \
    std::ostringstream err_msg;                                          \
    err_msg << "Unsupported warp_block_size " << int(warp_block_size);   \
    throw std::invalid_argument(err_msg.str());                          \
  }
