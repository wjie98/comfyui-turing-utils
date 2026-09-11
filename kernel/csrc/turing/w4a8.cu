// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <climits>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/threadblock/default_mma_core.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "kernel_api.h"

namespace comfyui_turing_utils::kernels {
namespace {

using cute::_0;
using cute::_1;

template <int N>
struct TuringW4ToS8 {
    static_assert(N % 8 == 0, "Turing W4 conversion requires groups of eight values");
    using result_type = cutlass::Array<int8_t, N>;
    using source_type = cutlass::Array<cutlass::int4b_t, N>;

    CUTLASS_HOST_DEVICE
    result_type operator()(source_type const &source) const {
#if defined(__CUDA_ARCH__)
        result_type result;
        auto const *packed = reinterpret_cast<uint32_t const *>(&source);
        auto *unpacked = reinterpret_cast<uint32_t *>(&result);

        CUTLASS_PRAGMA_UNROLL
        for (int index = 0; index < N / 8; ++index) {
            uint32_t const value = packed[index];
            uint32_t even = value & 0x0f0f0f0fu;
            uint32_t odd = (value >> 4) & 0x0f0f0f0fu;

            // Each lane is at most 0xf, so multiplying its sign bit by 30
            // cannot carry into the adjacent byte. This sign-extends four
            // nibbles in parallel with one IMAD instead of CUTLASS's PRMT,
            // mask, shift, and merge sequence.
            even |= (even & 0x08080808u) * 30u;
            odd |= (odd & 0x08080808u) * 30u;

            asm volatile(
                "prmt.b32 %0, %2, %3, 0x5140;\n"
                "prmt.b32 %1, %2, %3, 0x7362;\n"
                : "=&r"(unpacked[index * 2]),
                  "=&r"(unpacked[index * 2 + 1])
                : "r"(even), "r"(odd));
        }
        return result;
#else
        return cutlass::NumericArrayConverter<int8_t, cutlass::int4b_t, N>{}(source);
#endif
    }
};

template <typename ScaleT>
__device__ __forceinline__ float load_group_scale(ScaleT value);

template <>
__device__ __forceinline__ float load_group_scale<uint8_t>(uint8_t value) {
    return __half2float(__nv_cvt_fp8_to_halfraw(value, __NV_E4M3));
}

// CUTLASS normally loads the packed operand fragment and applies a stateless
// numeric transform before storing it to shared memory. Grouped-codebook W4
// also needs a per-group E4M3 scale and a 16-entry codebook. This iterator
// wraps the normal predicated packed-W4 load, but returns an S8 fragment after
// decoding in registers. MmaPipelined therefore writes exactly the same S8
// crosswise shared-memory tile as the existing W8A8 kernel: no decoded global
// workspace and no additional shared memory are introduced.
template <typename Shape_, typename ThreadMap_>
class TuringCodebookW4Iterator {
public:
    using Shape = Shape_;
    using Element = cutlass::int4b_t;
    using Layout = cutlass::layout::ColumnMajor;
    static int const kAdvanceRank = 0;
    using ThreadMap = ThreadMap_;
    using Index = typename Layout::Index;
    using LongIndex = typename Layout::LongIndex;
    using TensorRef = cutlass::TensorRef<Element, Layout>;
    using TensorView = cutlass::TensorView<Element, Layout>;
    using TensorCoord = typename Layout::TensorCoord;
    using Pointer = Element *;
    using NonConstPointer = Element *;
    using RawIterator = cutlass::transform::threadblock::PredicatedTileIterator<
        Shape,
        Element,
        Layout,
        kAdvanceRank,
        ThreadMap,
        16>;
    using RawFragment = typename RawIterator::Fragment;
    using AccessType = typename RawIterator::AccessType;
    using Fragment = cutlass::Array<
        int8_t,
        ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;
    using Mask = typename RawIterator::Mask;

    static_assert(ThreadMap::kElementsPerAccess == 16,
                  "inline codebook W4 requires one complete 16-value group per access");

    struct Params {
        typename RawIterator::Params raw;
        uint8_t const *group_scale = nullptr;
        float const *codebook = nullptr;
        int groups_per_row = 0;

        Params() = default;

        CUTLASS_HOST_DEVICE
        Params(Layout const &layout) : raw(layout) {}
    };

private:
    RawIterator raw_;
    uint8_t const *group_scale_ = nullptr;
    float const *codebook_ = nullptr;
    int groups_per_row_ = 0;
    int extent_k_ = 0;
    int extent_n_ = 0;
    int thread_k_ = 0;
    int thread_n_ = 0;
    bool enabled_ = true;

public:
    TuringCodebookW4Iterator() = default;

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator(Params const &params,
                             Pointer pointer,
                             TensorCoord extent,
                             int thread_id,
                             TensorCoord const &threadblock_offset,
                             int const *indices = nullptr)
        : raw_(params.raw, pointer, extent, thread_id, threadblock_offset, indices),
          group_scale_(params.group_scale),
          codebook_(params.codebook),
          groups_per_row_(params.groups_per_row),
          extent_k_(extent.row()),
          extent_n_(extent.column()) {
        auto const thread_offset = ThreadMap::initial_offset(thread_id);
        thread_k_ = threadblock_offset.row() + thread_offset.contiguous();
        thread_n_ = threadblock_offset.column() + thread_offset.strided();
    }

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator(Params const &params,
                             Pointer pointer,
                             TensorCoord extent,
                             int thread_id)
        : TuringCodebookW4Iterator(params, pointer, extent, thread_id, TensorCoord()) {}

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator &operator++() {
        ++raw_;
        thread_k_ += Shape::kRow;
        return *this;
    }

    CUTLASS_HOST_DEVICE
    TuringCodebookW4Iterator operator++(int) {
        TuringCodebookW4Iterator self(*this);
        operator++();
        return self;
    }

    CUTLASS_HOST_DEVICE
    void clear_mask(bool enable = true) {
        raw_.clear_mask(enable);
        if (enable) {
            enabled_ = false;
        }
    }

    CUTLASS_HOST_DEVICE
    void enable_mask() {
        raw_.enable_mask();
        enabled_ = true;
    }

    CUTLASS_HOST_DEVICE
    void set_mask(Mask const &mask) {
        raw_.set_mask(mask);
        enabled_ = true;
    }

    CUTLASS_HOST_DEVICE
    void get_mask(Mask &mask) { raw_.get_mask(mask); }

    CUTLASS_DEVICE
    void load(Fragment &fragment) {
        RawFragment packed;
        packed.clear();
        raw_.load(packed);
        auto const *packed_bytes = reinterpret_cast<uint8_t const *>(&packed);

        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
                int const access = c + s * ThreadMap::Iterations::kContiguous;
                int const base = access * ThreadMap::kElementsPerAccess;
                int const k = thread_k_ + c * ThreadMap::Delta::kContiguous;
                int const n = thread_n_ + s * ThreadMap::Delta::kStrided;
                bool const valid = enabled_ && n < extent_n_ && k < extent_k_;
                float const scale = valid
                    ? load_group_scale<uint8_t>(
                          group_scale_[static_cast<int64_t>(n) * groups_per_row_ + k / 16])
                    : 0.0f;

                CUTLASS_PRAGMA_UNROLL
                for (int element = 0; element < 16; ++element) {
                    uint8_t const byte = packed_bytes[(base + element) / 2];
                    int const code = (byte >> ((element & 1) * 4)) & 0x0f;
                    float const value = valid ? __ldg(codebook_ + code) * scale : 0.0f;
                    int const rounded = __float2int_rn(value);
                    fragment[base + element] = static_cast<int8_t>(
                        max(-127, min(127, rounded)));
                }
            }
        }
    }
};

template <typename BaseKernel>
struct TuringCodebookGemmKernel : BaseKernel {
    using Base = BaseKernel;
    using Arguments = typename Base::Arguments;

    struct Params : Base::Params {
        Params() = default;

        Params(Arguments const &args, int device_sms, int sm_occupancy)
            : Base::Params(args, device_sms, sm_occupancy) {
            set_codebook_params(args);
        }

        void update(Arguments const &args) {
            Base::Params::update(args);
            set_codebook_params(args);
        }

    private:
        void set_codebook_params(Arguments const &args) {
            this->params_B.group_scale = reinterpret_cast<uint8_t const *>(args.ptr_C);
            this->params_B.codebook = reinterpret_cast<float const *>(args.ptr_D);
            this->params_B.groups_per_row = args.problem_size.k() / 16;
        }
    };
};

enum class WeightKind {
    kInt8,
    kSignedW4,
    kCodebookW4,
};

template <typename Output,
          WeightKind Kind,
          int TBM,
          int TBN,
          int WM,
          int WN,
          int InputAlignment = 16,
          int WeightAlignment = 16>
struct TuringW4A8Gemm {
    static constexpr bool PackedWeight = Kind != WeightKind::kInt8;
    static constexpr bool CodebookWeight = Kind == WeightKind::kCodebookW4;
    using ElementA = int8_t;
    using ElementB = std::conditional_t<PackedWeight, cutlass::int4b_t, int8_t>;
    using SharedElementB = int8_t;
    using ElementC = Output;
    using ElementAccumulator = int32_t;
    using ElementCompute = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, 64>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 16>;
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    static constexpr int AlignmentA = InputAlignment;
    static constexpr int AlignmentB = WeightAlignment;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int EpilogueStages = 1;

    using MmaCore = cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        ElementA,
        LayoutA,
        SharedElementB,
        LayoutB,
        ElementAccumulator,
        LayoutC,
        cutlass::arch::OpClassTensorOp,
        2,
        cutlass::arch::OpMultiplyAddSaturate>;

    using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        1,
        typename MmaCore::IteratorThreadMapA,
        AlignmentA>;

    // Reuse the W8A8 tile map but read 16 packed W4 values with one 64-bit
    // access. The Turing transform expands each vector before writing the
    // normal SM75 crosswise S8 shared-memory tile.
    using PredicatedIteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        0,
        typename MmaCore::IteratorThreadMapB,
        AlignmentB>;
    using IteratorB = std::conditional_t<
        CodebookWeight,
        TuringCodebookW4Iterator<
            cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
            typename MmaCore::IteratorThreadMapB>,
        PredicatedIteratorB>;

    using TransformA = cutlass::NumericArrayConverter<
        typename MmaCore::SmemIteratorA::Element,
        typename IteratorA::Element,
        IteratorA::Fragment::kElements>;
    using TransformB = std::conditional_t<
        CodebookWeight,
        cutlass::NumericArrayConverter<
            typename MmaCore::SmemIteratorB::Element,
            int8_t,
            IteratorB::Fragment::kElements>,
        std::conditional_t<
            PackedWeight,
            TuringW4ToS8<IteratorB::Fragment::kElements>,
            cutlass::NumericArrayConverter<
                typename MmaCore::SmemIteratorB::Element,
                typename IteratorB::Element,
                IteratorB::Fragment::kElements>>>;

    using Mma = cutlass::gemm::threadblock::MmaPipelined<
        typename MmaCore::Shape,
        IteratorA,
        typename MmaCore::SmemIteratorA,
        IteratorB,
        typename MmaCore::SmemIteratorB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore::MmaPolicy,
        TransformA,
        TransformB>;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape,
        WarpShape,
        ElementC,
        AlignmentC,
        EpilogueStages>;
    using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
    using ActivationScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_1, _0, int32_t>>;
    using WeightScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_0, _1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap,
        ElementCompute,
        cute::Stride<_0, _1, int32_t>>;
    using ScaleActivation = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledActivation = cutlass::epilogue::threadblock::Sm80EVT<
        ScaleActivation,
        Accumulator,
        ActivationScale>;
    using ScaleWeight = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies,
        ElementCompute,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using BF16ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
        ScaleWeight,
        ScaledActivation,
        WeightScale>;
    // FP16 VAE matches eager's tensor boundaries: combine FP32 scales first,
    // round the scaled accumulator to FP16, then add the FP16-rounded bias.
    using FP16Scale = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementC, ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using FP16ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
        FP16Scale,
        Accumulator,
        cutlass::epilogue::threadblock::Sm80EVT<ScaleWeight, ActivationScale, WeightScale>>;
    using ScaledOutput = std::conditional_t<
        std::is_same_v<ElementC, cutlass::half_t>, FP16ScaledOutput, BF16ScaledOutput>;
    using AddBias = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus,
        ElementC,
        ElementCompute,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using BiasedOutput = cutlass::epilogue::threadblock::Sm80EVT<
        AddBias,
        ScaledOutput,
        Bias>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap,
        ElementC,
        cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
    using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<Store, BiasedOutput>;

    using EpilogueBase = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        AlignmentA,
        SharedElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        16,
        ElementC,
        LayoutC,
        AlignmentC,
        ElementAccumulator,
        ElementCompute,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Callbacks,
        ThreadblockSwizzle,
        2,
        cutlass::arch::OpMultiplyAddSaturate,
        EpilogueStages>;
    using Epilogue = typename EpilogueBase::Epilogue;
    using BaseGemmKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
        Mma,
        Epilogue,
        ThreadblockSwizzle>;
    using GemmKernel = std::conditional_t<
        CodebookWeight,
        TuringCodebookGemmKernel<BaseGemmKernel>,
        BaseGemmKernel>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    static_assert(sizeof(typename GemmKernel::SharedStorage) <= 48 * 1024,
                  "Turing W4A8 must stay within the default 48 KiB shared-memory limit");

    static bool run(const int8_t *activation,
                    const int8_t *weight,
                    const float *activation_scale,
                    const float *weight_scale,
                    const float *bias,
                    Output *output,
                    int m,
                    int n,
                    int k,
                    int output_stride,
                    cudaStream_t stream,
                    const uint8_t *group_scale = nullptr,
                    const float *codebook = nullptr) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        auto scale_arguments = [&]() -> typename ScaledOutput::Arguments {
            if constexpr (std::is_same_v<ElementC, cutlass::half_t>) {
                return {{},
                        {{const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}},
                         {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}}, {}}, {}};
            } else {
                return {{{}, {const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}}, {}},
                        {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}}, {}};
            }
        };
        typename Callbacks::Arguments callbacks{
            {scale_arguments(),
             {const_cast<float *>(bias), 0.0f, {_0{}, _1{}, n}},
             {}},
            {output, {output_stride, _1{}, static_cast<int64_t>(m) * output_stride}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t *>(activation),
            reinterpret_cast<ElementB *>(const_cast<int8_t *>(weight)),
            CodebookWeight ? reinterpret_cast<ElementC const *>(group_scale) : nullptr,
            CodebookWeight
                ? reinterpret_cast<ElementC *>(const_cast<float *>(codebook))
                : nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
            return false;
        }
        if (Gemm::get_workspace_size(arguments) != 0) {
            return false;
        }
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
            return false;
        }
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <int TBM, int TBN, int WM, int WN>
bool run_tile(const int8_t *activation,
              const int8_t *weight,
              const float *activation_scale,
              const float *weight_scale,
              const float *bias,
              __nv_bfloat16 *output,
              int m,
              int n,
              int k,
              int output_stride,
              cudaStream_t stream) {
    return TuringW4A8Gemm<
        cutlass::bfloat16_t, WeightKind::kSignedW4, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        output_stride,
        stream);
}

template <int TBM, int TBN, int WM, int WN>
bool run_k_tail_tile(const int8_t *activation,
                     const int8_t *weight,
                     const float *activation_scale,
                     const float *weight_scale,
                     const float *bias,
                     __nv_bfloat16 *output,
                     int m,
                     int n,
                     int k,
                     int output_stride,
                     cudaStream_t stream) {
    return TuringW4A8Gemm<
        cutlass::bfloat16_t,
        WeightKind::kSignedW4,
        TBM,
        TBN,
        WM,
        WN,
        4,
        4>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        output_stride,
        stream);
}

template <int TBM, int TBN, int WM, int WN, typename Output>
bool run_int8_tile(const int8_t *activation,
                   const int8_t *weight,
                   const float *activation_scale,
                   const float *weight_scale,
                   const float *bias,
                   Output *output,
                   int m,
                   int n,
                   int k,
                   int output_stride,
                   cudaStream_t stream) {
    return TuringW4A8Gemm<
        Output, WeightKind::kInt8, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        weight_scale,
        bias,
        output,
        m,
        n,
        k,
        output_stride,
        stream);
}

// Native SM80 INT8 Tensor Core mainloop.  Keeping the same EVT epilogue as
// the SM75 implementation makes this a schedule substitution only: integer
// accumulation, row/channel scales, bias, and BF16 rounding are unchanged.
template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages, typename Output>
struct AmpereInt8Gemm {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = Output;
    using AccumulatorT = int32_t;
    using ComputeT = float;
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using ThreadblockShape = cutlass::gemm::GemmShape<TBM, TBN, TBK>;
    using WarpShape = cutlass::gemm::GemmShape<WM, WN, WK>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
    using Swizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
    static constexpr int Alignment = 16;
    static constexpr int AlignmentC = 8;
    static constexpr int EpilogueStages = 1;

    using ThreadMap = cutlass::epilogue::threadblock::OutputTileThreadLayout<
        ThreadblockShape, WarpShape, ElementC, AlignmentC, EpilogueStages>;
    using Accumulator = cutlass::epilogue::threadblock::VisitorAccFetch;
    using XScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
        ThreadMap, ComputeT, cute::Stride<_1, _0, int32_t>>;
    using WScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ComputeT, cute::Stride<_0, _1, int32_t>>;
    using Bias = cutlass::epilogue::threadblock::VisitorRowBroadcast<
        ThreadMap, ComputeT, cute::Stride<_0, _1, int32_t>>;
    using Multiply = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ComputeT, ComputeT,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using ScaledActivation = cutlass::epilogue::threadblock::Sm80EVT<
        Multiply, Accumulator, XScale>;
    using BF16ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
        Multiply, ScaledActivation, WScale>;
    using FP16Scale = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::multiplies, ElementC, ComputeT,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using FP16ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
        FP16Scale, Accumulator,
        cutlass::epilogue::threadblock::Sm80EVT<Multiply, XScale, WScale>>;
    using ScaledOutput = std::conditional_t<
        std::is_same_v<ElementC, cutlass::half_t>, FP16ScaledOutput, BF16ScaledOutput>;
    using Add = cutlass::epilogue::threadblock::VisitorCompute<
        cutlass::plus, ElementC, ComputeT,
        cutlass::FloatRoundStyle::round_to_nearest>;
    using BiasedOutput = cutlass::epilogue::threadblock::Sm80EVT<
        Add, ScaledOutput, Bias>;
    using Store = cutlass::epilogue::threadblock::VisitorAuxStore<
        ThreadMap, ElementC, cutlass::FloatRoundStyle::round_to_nearest,
        cute::Stride<int64_t, _1, int64_t>>;
    using Callbacks = cutlass::epilogue::threadblock::Sm80EVT<Store, BiasedOutput>;
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        ElementA, LayoutA, cutlass::ComplexTransform::kNone, Alignment,
        ElementB, LayoutB, cutlass::ComplexTransform::kNone, Alignment,
        ElementC, LayoutC, AlignmentC,
        AccumulatorT, ComputeT,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        ThreadblockShape, WarpShape, InstructionShape, Callbacks, Swizzle,
        Stages, cutlass::arch::OpMultiplyAddSaturate,
        EpilogueStages>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    static bool run(const int8_t *activation,
                    const int8_t *weight,
                    const float *activation_scale,
                    const float *weight_scale,
                    const float *bias,
                    Output *output,
                    int m,
                    int n,
                    int k,
                    int output_stride,
                    cudaStream_t stream) {
        cutlass::gemm::GemmCoord problem(m, n, k);
        auto scale_arguments = [&]() -> typename ScaledOutput::Arguments {
            if constexpr (std::is_same_v<ElementC, cutlass::half_t>) {
                return {{},
                        {{const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}},
                         {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}}, {}}, {}};
            } else {
                return {{{}, {const_cast<float *>(activation_scale), 0.0f, {_1{}, _0{}, m}}, {}},
                        {const_cast<float *>(weight_scale), 0.0f, {_0{}, _1{}, n}}, {}};
            }
        };
        typename Callbacks::Arguments callbacks{
            {scale_arguments(),
             {const_cast<float *>(bias), 0.0f, {_0{}, _1{}, n}},
             {}},
            {reinterpret_cast<ElementC *>(output),
             {output_stride, _1{}, static_cast<int64_t>(m) * output_stride}}};
        typename Gemm::Arguments arguments(
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem,
            1,
            callbacks,
            const_cast<int8_t *>(activation),
            const_cast<int8_t *>(weight),
            nullptr,
            nullptr,
            static_cast<int64_t>(m) * k,
            static_cast<int64_t>(n) * k,
            0,
            0,
            k,
            k,
            0,
            0);
        Gemm gemm;
        if (gemm.can_implement(arguments) != cutlass::Status::kSuccess ||
            Gemm::get_workspace_size(arguments) != 0) {
            return false;
        }
        if (gemm.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
            return false;
        }
        return gemm(stream) == cutlass::Status::kSuccess;
    }
};

template <int TBM, int TBN, int TBK, int WM, int WN, int WK, int Stages, typename Output>
bool run_ampere_int8_tile(const int8_t *activation,
                          const int8_t *weight,
                          const float *activation_scale,
                          const float *weight_scale,
                          const float *bias,
                          Output *output,
                          int m,
                          int n,
                          int k,
                          int output_stride,
                          cudaStream_t stream) {
    return AmpereInt8Gemm<
        TBM, TBN, TBK, WM, WN, WK, Stages, Output>::run(
        activation, weight, activation_scale, weight_scale, bias, output,
        m, n, k, output_stride, stream);
}

template <int TBM, int TBN, int WM, int WN>
bool run_codebook_tile(const int8_t *activation,
                       const int8_t *weight,
                       const float *activation_scale,
                       const uint8_t *group_scale,
                       const float *channel_scale,
                       const float *codebook,
                       const float *bias,
                       __nv_bfloat16 *output,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream) {
    return TuringW4A8Gemm<
        cutlass::bfloat16_t, WeightKind::kCodebookW4, TBM, TBN, WM, WN>::run(
        activation,
        weight,
        activation_scale,
        channel_scale,
        bias,
        reinterpret_cast<cutlass::bfloat16_t *>(output),
        m,
        n,
        k,
        n,
        stream,
        group_scale,
        codebook);
}

bool dispatch(const int8_t *activation,
              const int8_t *weight,
              const float *activation_scale,
              const float *weight_scale,
              const float *bias,
              __nv_bfloat16 *output,
              int m,
              int n,
              int k,
              int output_stride,
              cudaStream_t stream) {
    return run_tile<128, 256, 64, 64>(
        activation, weight, activation_scale, weight_scale, bias,
        output, m, n, k, output_stride, stream);
}

template <typename Output>
bool dispatch_int8(const int8_t *activation,
                   const int8_t *weight,
                   const float *activation_scale,
                   const float *weight_scale,
                   const float *bias,
                   Output *output,
                   int m,
                   int n,
                   int k,
                   int output_stride,
                   cudaStream_t stream) {
    const cudaDeviceProp *properties = getCurrentDeviceProperties();
    if (properties->major >= 8 && n >= 16384) {
        return run_ampere_int8_tile<128, 256, 64, 64, 64, 64, 3>(
            activation, weight, activation_scale, weight_scale, bias,
            output, m, n, k, output_stride, stream);
    }
    return run_int8_tile<128, 256, 64, 64>(
        activation, weight, activation_scale, weight_scale, bias,
        output, m, n, k, output_stride, stream);
}

bool dispatch_codebook(const int8_t *activation,
                       const int8_t *weight,
                       const float *activation_scale,
                       const uint8_t *group_scale,
                       const float *channel_scale,
                       const float *codebook,
                       const float *bias,
                       __nv_bfloat16 *output,
                       int m,
                       int n,
                       int k,
                       cudaStream_t stream) {
    // The long-sequence W8A8 schedule is already register-limited to one CTA per
    // SM75 SM. Inline decoding cannot reduce its CTA residency, while using it
    // for smaller schedules could cross a two-CTA register threshold. Keep the
    // production inline path deliberately scoped to the long tile.
    if (m <= 8192) {
        return false;
    }
    return run_codebook_tile<128, 256, 64, 64>(
        activation, weight, activation_scale, group_scale,
        channel_scale, codebook, bias, output, m, n, k, stream);
}

// Decode one 16-column vector per thread. This intentionally matches Kitchen's
// reference rounding and E4M3 conversion so the staged SM75 path is bit exact
// before the INT8 contraction.
__global__ void decode_codebook_w4_to_s8(
    const int8_t *__restrict__ packed_weight,
    const uint8_t *__restrict__ group_scale,
    const float *__restrict__ codebook,
    int8_t *__restrict__ output,
    int64_t vector_count,
    int packed_k,
    int k,
    int group_size) {
    __shared__ float shared_codebook[16];
    if (threadIdx.x < 16) {
        shared_codebook[threadIdx.x] = codebook[threadIdx.x];
    }
    __syncthreads();

    const int64_t vector = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector >= vector_count) {
        return;
    }
    const int vectors_per_row = packed_k / 8;
    const int row = static_cast<int>(vector / vectors_per_row);
    const int packed_column = static_cast<int>(vector % vectors_per_row) * 8;
    const int output_column = packed_column * 2;
    const int groups_per_row = k / group_size;
    const int64_t scale_row = static_cast<int64_t>(row) * groups_per_row;
    const uint2 packed = *reinterpret_cast<const uint2 *>(
        packed_weight + static_cast<int64_t>(row) * packed_k + packed_column);
    const int base_group = output_column / group_size;
    uint4 decoded;
#pragma unroll
    for (int output_word = 0; output_word < 4; ++output_word) {
        const unsigned bytes = output_word < 2 ? packed.x : packed.y;
        const int input_shift = (output_word & 1) * 16;
        const int local_group = group_size >= 16 ? 0 : (output_word * 4) / group_size;
        const float scale = load_group_scale<uint8_t>(
            group_scale[scale_row + base_group + local_group]);
        unsigned decoded_word = 0;
#pragma unroll
        for (int byte_index = 0; byte_index < 2; ++byte_index) {
            const unsigned value =
                (bytes >> (input_shift + byte_index * 8)) & 0xffu;
            const unsigned low = value & 0x0fu;
            const unsigned high = value >> 4;
            const int low_value = max(
                -127, min(127, __float2int_rn(shared_codebook[low] * scale)));
            const int high_value = max(
                -127, min(127, __float2int_rn(shared_codebook[high] * scale)));
            decoded_word |=
                (static_cast<unsigned>(static_cast<uint8_t>(low_value))
                 << (byte_index * 16));
            decoded_word |=
                (static_cast<unsigned>(static_cast<uint8_t>(high_value))
                 << (byte_index * 16 + 8));
        }
        if (output_word == 0) {
            decoded.x = decoded_word;
        } else if (output_word == 1) {
            decoded.y = decoded_word;
        } else if (output_word == 2) {
            decoded.z = decoded_word;
        } else {
            decoded.w = decoded_word;
        }
    }
    *reinterpret_cast<uint4 *>(output + static_cast<int64_t>(row) * k + output_column) =
        decoded;
}

void launch_codebook_decode(const int8_t *packed_weight,
                            const uint8_t *group_scale,
                            const float *codebook,
                            int8_t *workspace,
                            int rows,
                            int k,
                            int group_size,
                            cudaStream_t stream) {
    constexpr int threads = 256;
    const int packed_k = k / 2;
    const int64_t vector_count = static_cast<int64_t>(rows) * packed_k / 8;
    const int blocks = static_cast<int>(ceilDiv(vector_count, static_cast<int64_t>(threads)));
    decode_codebook_w4_to_s8<<<blocks, threads, 0, stream>>>(
        packed_weight,
        group_scale,
        codebook,
        workspace,
        vector_count,
        packed_k,
        k,
        group_size);
}

}  // namespace

void turing_w4a8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output) {
    const int64_t m64 = activation.size(0);
    const int64_t k64 = activation.size(1);
    const int64_t n64 = weight.size(0);
    if (m64 == 0 || n64 == 0 || k64 == 0) {
        return;
    }
    if (m64 > INT_MAX || n64 > INT_MAX || k64 > INT_MAX) {
        throw std::runtime_error("Turing W4A8 dimensions are unsupported");
    }

    const auto *activation_ptr = static_cast<const int8_t *>(activation.ptr);
    const auto *weight_ptr = static_cast<const int8_t *>(weight.ptr);
    const auto *activation_scale_ptr = static_cast<const float *>(activation_scale.ptr);
    const auto *weight_scale_ptr = static_cast<const float *>(weight_scale.ptr);
    const auto *bias_ptr = bias.valid() ? static_cast<const float *>(bias.ptr) : nullptr;
    auto *output_ptr = static_cast<__nv_bfloat16 *>(output.ptr);
    const int m = static_cast<int>(m64);
    const int n = static_cast<int>(n64);
    const int k = static_cast<int>(k64);
    const cudaStream_t stream = getCurrentCUDAStream();

    // Keep the K-tail guard at the CUDA ABI boundary so direct callers cannot
    // silently enter a scalar full-matrix fallback. The normal dispatcher pads
    // contractions when profitable, while this path remains correct for direct
    // or deliberately unpadded calls.
    if (k % 16 != 0) {
        const bool launched = m <= 2048
            ? run_k_tail_tile<64, 128, 32, 64>(
                activation_ptr,
                weight_ptr,
                activation_scale_ptr,
                weight_scale_ptr,
                bias_ptr,
                output_ptr,
                m,
                n,
                k,
                static_cast<int>(output.stride(0)),
                stream)
            : run_k_tail_tile<256, 128, 64, 64>(
                activation_ptr,
                weight_ptr,
                activation_scale_ptr,
                weight_scale_ptr,
                bias_ptr,
                output_ptr,
                m,
                n,
                k,
                static_cast<int>(output.stride(0)),
                stream);
        if (!launched) {
            throw std::runtime_error("CUTLASS SM75 W4A8 K-tail kernel rejected the problem");
        }
        checkCUDA(cudaGetLastError());
        return;
    }

    if (n % 8 != 0) {
        throw std::runtime_error("Turing W4A8 requires N padded to a multiple of 8");
    }

    const bool launched = dispatch(
        activation_ptr,
        weight_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        m,
        n,
        k,
        static_cast<int>(output.stride(0)),
        stream);
    if (!launched) {
        throw std::runtime_error("CUTLASS SM75 W4A8 kernel rejected the problem shape");
    }
    checkCUDA(cudaGetLastError());
}

void turing_codebook_w4a8_linear(Tensor activation,
                                 Tensor weight,
                                 Tensor activation_scale,
                                 Tensor group_scale,
                                 Tensor channel_scale,
                                 Tensor codebook,
                                 Tensor bias,
                                 Tensor workspace,
                                 Tensor output,
                                 int group_size,
                                 bool inline_decode) {
    const int m = activation.size(0);
    const int k = activation.size(1);
    const int n = weight.size(0);
    const int chunk_rows = workspace.valid() ? workspace.size(0) : 0;
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (k % 16 != 0 || n % 8 != 0 ||
        (!inline_decode && (chunk_rows <= 0 || chunk_rows % 8 != 0))) {
        throw std::runtime_error(
            "Turing codebook W4A8 requires K%16=0, N%8=0, and an 8-row-aligned staged workspace");
    }
    if (group_size < 4 || k % group_size != 0 ||
        (16 % group_size != 0 && group_size % 16 != 0)) {
        throw std::runtime_error("unsupported Turing codebook W4A8 group size");
    }

    const auto *activation_ptr = activation.data_ptr<int8_t>();
    const auto *weight_ptr = weight.data_ptr<int8_t>();
    const auto *activation_scale_ptr = activation_scale.data_ptr<float>();
    const auto *group_scale_ptr = group_scale.data_ptr<uint8_t>();
    const auto *channel_scale_ptr = channel_scale.data_ptr<float>();
    const auto *codebook_ptr = codebook.data_ptr<float>();
    const auto *bias_ptr = bias.valid() ? bias.data_ptr<float>() : nullptr;
    auto *output_ptr = output.data_ptr<__nv_bfloat16>();
    const int packed_k = k / 2;
    const int groups_per_row = k / group_size;
    const cudaStream_t stream = getCurrentCUDAStream();

    if (inline_decode) {
        if (group_size != 16) {
            throw std::runtime_error("inline Turing codebook W4A8 requires group_size=16");
        }
        if (!dispatch_codebook(
                activation_ptr,
                weight_ptr,
                activation_scale_ptr,
                group_scale_ptr,
                channel_scale_ptr,
                codebook_ptr,
                bias_ptr,
                output_ptr,
                m,
                n,
                k,
                stream)) {
            throw std::runtime_error(
                "inline CUTLASS SM75 codebook W4A8 rejected the problem shape");
        }
        checkCUDA(cudaGetLastError());
        return;
    }

    auto *workspace_ptr = workspace.data_ptr<int8_t>();

    for (int row = 0; row < n; row += chunk_rows) {
        const int rows = std::min(chunk_rows, n - row);
        launch_codebook_decode(
            weight_ptr + static_cast<int64_t>(row) * packed_k,
            group_scale_ptr + static_cast<int64_t>(row) * groups_per_row,
            codebook_ptr,
            workspace_ptr,
            rows,
            k,
            group_size,
            stream);
        if (!dispatch_int8(
                activation_ptr,
                workspace_ptr,
                activation_scale_ptr,
                channel_scale_ptr + row,
                bias_ptr == nullptr ? nullptr : bias_ptr + row,
                reinterpret_cast<cutlass::bfloat16_t *>(output_ptr + row),
                m,
                rows,
                k,
                n,
                stream)) {
            throw std::runtime_error("CUTLASS SM75 codebook W4A8 kernel rejected a chunk");
        }
    }
    checkCUDA(cudaGetLastError());
}

void turing_int8_linear(Tensor activation,
                        Tensor weight,
                        Tensor activation_scale,
                        Tensor weight_scale,
                        Tensor bias,
                        Tensor output) {
    const int m = activation.size(0);
    const int k = activation.size(1);
    const int n = weight.size(0);
    if (m == 0 || n == 0 || k == 0) {
        return;
    }
    if (k % 16 != 0 || n % 8 != 0) {
        throw std::runtime_error("Turing INT8 linear requires K%16=0 and N%8=0");
    }
    // Row-major [N,K] weights are already a column-major [K,N] GEMM operand.
    // Never transpose/materialize a second weight buffer.
    const auto launch = [&](auto *output_ptr) {
        return dispatch_int8(
            activation.data_ptr<int8_t>(), weight.data_ptr<int8_t>(),
            activation_scale.data_ptr<float>(), weight_scale.data_ptr<float>(),
            bias.valid() ? bias.data_ptr<float>() : nullptr, output_ptr,
            m, n, k, static_cast<int>(output.stride(0)), getCurrentCUDAStream());
    };
    const bool launched = output.scalar_type() == Tensor::FP16
        ? launch(static_cast<cutlass::half_t *>(output.ptr))
        : launch(static_cast<cutlass::bfloat16_t *>(output.ptr));

    if (!launched) {
        throw std::runtime_error("CUTLASS SM75 INT8 kernel rejected the problem shape");
    }
    checkCUDA(cudaGetLastError());
}

}  // namespace comfyui_turing_utils::kernels
