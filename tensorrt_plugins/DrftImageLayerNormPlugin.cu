#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <new>

#if defined(_WIN32)
#define DRFT_PLUGIN_EXPORT extern "C" __declspec(dllexport)
#else
#define DRFT_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace
{
constexpr char kPluginName[] = "DrftImageLayerNorm_TRT";
constexpr char kFastPluginName[] = "DrftImageLayerNormFast_TRT";
constexpr char kCooperativePluginName[] = "DrftImageLayerNormCooperative_TRT";
constexpr char kPluginVersion[] = "1";
constexpr int32_t kTactic = 1;
constexpr int32_t kThreads = 256;
constexpr int32_t kMaxBlocks = 256;

__device__ __forceinline__ float warpReduceSum(float value)
{
    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xFFFFFFFFU, value, offset);
    }
    return value;
}

template <typename Transform>
__global__ void reduceImage(__nv_bfloat16 const* input, float* partial,
    int64_t elements, Transform transform)
{
    float sum = 0.0F;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        sum += transform(__bfloat162float(input[index]));
    }
    sum = warpReduceSum(sum);
    __shared__ float warpSums[kThreads / 32];
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            partial[blockIdx.x] = sum;
        }
    }
}

struct IdentityTransform
{
    __device__ float operator()(float value) const { return value; }
};

struct CenteredSquareTransform
{
    float const* mean;

    __device__ float operator()(float value) const
    {
        float const centered = __bfloat162float(__float2bfloat16_rn(value - mean[0]));
        return __bfloat162float(__float2bfloat16_rn(centered * centered));
    }
};

__global__ void finalizeMean(float const* partial, float* statistics,
    int32_t blocks, int64_t elements)
{
    float sum = threadIdx.x < blocks ? partial[threadIdx.x] : 0.0F;
    sum = warpReduceSum(sum);
    __shared__ float warpSums[kThreads / 32];
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            float const mean = sum / static_cast<float>(elements);
            statistics[0] = __bfloat162float(__float2bfloat16_rn(mean));
        }
    }
}

__global__ void finalizeStd(float const* partial, float* statistics,
    __nv_bfloat16* stdOutput, int32_t blocks, int64_t elements, float epsilon)
{
    float sum = threadIdx.x < blocks ? partial[threadIdx.x] : 0.0F;
    sum = warpReduceSum(sum);
    __shared__ float warpSums[kThreads / 32];
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            float const variance = __bfloat162float(__float2bfloat16_rn(
                sum / static_cast<float>(elements)));
            float const epsilonRounded
                = __bfloat162float(__float2bfloat16_rn(epsilon));
            float const rootInput = __bfloat162float(__float2bfloat16_rn(
                variance + epsilonRounded));
            __nv_bfloat16 const stdValue = __float2bfloat16_rn(sqrtf(rootInput));
            statistics[1] = __bfloat162float(stdValue);
            stdOutput[0] = stdValue;
        }
    }
}

__global__ void normalizeImage(__nv_bfloat16 const* input,
    __nv_bfloat16 const* weight, __nv_bfloat16 const* bias,
    float const* statistics, __nv_bfloat16* output,
    int64_t elements, int32_t channels)
{
    float const mean = statistics[0];
    float const stdValue = statistics[1];
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        int32_t const channel = static_cast<int32_t>(index % channels);
        float const value = __bfloat162float(input[index]);
        float const centered = __bfloat162float(__float2bfloat16_rn(value - mean));
        float const normalized = __bfloat162float(__float2bfloat16_rn(
            centered / stdValue));
        float const scaled = __bfloat162float(__float2bfloat16_rn(
            normalized * __bfloat162float(weight[channel])));
        output[index] = __float2bfloat16_rn(
            scaled + __bfloat162float(bias[channel]));
    }
}

cudaError_t launchImageLayerNorm(void const* input, void const* weight,
    void const* bias, void* normalizedOutput, void* stdOutput, void* workspace,
    int32_t batch, int64_t tokens, int32_t channels, float epsilon,
    cudaStream_t stream) noexcept
{
    if (input == nullptr || weight == nullptr || bias == nullptr
        || normalizedOutput == nullptr || stdOutput == nullptr || workspace == nullptr
        || batch != 1 || tokens <= 0 || channels <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t elements = tokens * channels;
    int32_t blocks = static_cast<int32_t>(
        (elements + kThreads - 1) / kThreads < kMaxBlocks
            ? (elements + kThreads - 1) / kThreads : kMaxBlocks);
    auto* partial = static_cast<float*>(workspace);
    auto* statistics = partial + kMaxBlocks;
    auto const* inputBf16 = static_cast<__nv_bfloat16 const*>(input);

    reduceImage<<<blocks, kThreads, 0, stream>>>(
        inputBf16, partial, elements, IdentityTransform{});
    finalizeMean<<<1, kThreads, 0, stream>>>(
        partial, statistics, blocks, elements);
    reduceImage<<<blocks, kThreads, 0, stream>>>(
        inputBf16, partial, elements, CenteredSquareTransform{statistics});
    finalizeStd<<<1, kThreads, 0, stream>>>(
        partial, statistics, static_cast<__nv_bfloat16*>(stdOutput),
        blocks, elements, epsilon);
    normalizeImage<<<blocks, kThreads, 0, stream>>>(
        inputBf16,
        static_cast<__nv_bfloat16 const*>(weight),
        static_cast<__nv_bfloat16 const*>(bias),
        statistics,
        static_cast<__nv_bfloat16*>(normalizedOutput),
        elements, channels);
    return cudaPeekAtLastError();
}

__global__ void imageLayerNormCooperative(__nv_bfloat16 const* input,
    __nv_bfloat16 const* weight, __nv_bfloat16 const* bias,
    float* partial, float* statistics, __nv_bfloat16* output,
    __nv_bfloat16* stdOutput, int64_t elements, int32_t channels,
    int32_t blocks, float epsilon)
{
    cooperative_groups::grid_group const grid
        = cooperative_groups::this_grid();
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    __shared__ float warpSums[kThreads / 32];
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    float sum = 0.0F;
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        sum += __bfloat162float(input[index]);
    }
    sum = warpReduceSum(sum);
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            partial[blockIdx.x] = sum;
        }
    }
    grid.sync();

    if (blockIdx.x == 0)
    {
        sum = threadIdx.x < blocks ? partial[threadIdx.x] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            warpSums[warp] = sum;
        }
        __syncthreads();
        if (warp == 0)
        {
            sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
            sum = warpReduceSum(sum);
            if (lane == 0)
            {
                float const mean = sum / static_cast<float>(elements);
                statistics[0]
                    = __bfloat162float(__float2bfloat16_rn(mean));
            }
        }
    }
    grid.sync();

    float const mean = statistics[0];
    sum = 0.0F;
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        float const value = __bfloat162float(input[index]);
        float const centered
            = __bfloat162float(__float2bfloat16_rn(value - mean));
        sum += __bfloat162float(__float2bfloat16_rn(centered * centered));
    }
    sum = warpReduceSum(sum);
    if (lane == 0)
    {
        warpSums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0)
    {
        sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            partial[blockIdx.x] = sum;
        }
    }
    grid.sync();

    if (blockIdx.x == 0)
    {
        sum = threadIdx.x < blocks ? partial[threadIdx.x] : 0.0F;
        sum = warpReduceSum(sum);
        if (lane == 0)
        {
            warpSums[warp] = sum;
        }
        __syncthreads();
        if (warp == 0)
        {
            sum = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
            sum = warpReduceSum(sum);
            if (lane == 0)
            {
                float const variance = __bfloat162float(__float2bfloat16_rn(
                    sum / static_cast<float>(elements)));
                float const epsilonRounded
                    = __bfloat162float(__float2bfloat16_rn(epsilon));
                float const rootInput = __bfloat162float(__float2bfloat16_rn(
                    variance + epsilonRounded));
                __nv_bfloat16 const stdValue
                    = __float2bfloat16_rn(sqrtf(rootInput));
                statistics[1] = __bfloat162float(stdValue);
                stdOutput[0] = stdValue;
            }
        }
    }
    grid.sync();

    float const stdValue = statistics[1];
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        int32_t const channel = static_cast<int32_t>(index % channels);
        float const value = __bfloat162float(input[index]);
        float const centered
            = __bfloat162float(__float2bfloat16_rn(value - mean));
        float const normalized = __bfloat162float(__float2bfloat16_rn(
            centered / stdValue));
        float const scaled = __bfloat162float(__float2bfloat16_rn(
            normalized * __bfloat162float(weight[channel])));
        output[index] = __float2bfloat16_rn(
            scaled + __bfloat162float(bias[channel]));
    }
}

cudaError_t launchImageLayerNormCooperative(void const* input,
    void const* weight, void const* bias, void* normalizedOutput,
    void* stdOutput, void* workspace, int32_t batch, int64_t tokens,
    int32_t channels, float epsilon, cudaStream_t stream) noexcept
{
    if (input == nullptr || weight == nullptr || bias == nullptr
        || normalizedOutput == nullptr || stdOutput == nullptr
        || workspace == nullptr || batch != 1 || tokens <= 0 || channels <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t elements = tokens * channels;
    int32_t blocks = static_cast<int32_t>(
        (elements + kThreads - 1) / kThreads < kMaxBlocks
            ? (elements + kThreads - 1) / kThreads : kMaxBlocks);
    auto* partial = static_cast<float*>(workspace);
    auto* statistics = partial + kMaxBlocks;
    auto const* inputBf16 = static_cast<__nv_bfloat16 const*>(input);
    auto const* weightBf16 = static_cast<__nv_bfloat16 const*>(weight);
    auto const* biasBf16 = static_cast<__nv_bfloat16 const*>(bias);
    auto* outputBf16 = static_cast<__nv_bfloat16*>(normalizedOutput);
    auto* stdBf16 = static_cast<__nv_bfloat16*>(stdOutput);
    void* arguments[] = {&inputBf16, &weightBf16, &biasBf16, &partial,
        &statistics, &outputBf16, &stdBf16, &elements, &channels,
        &blocks, &epsilon};
    return cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(imageLayerNormCooperative), dim3(blocks),
        dim3(kThreads), arguments, 0, stream);
}

struct WelfordState
{
    float mean;
    float m2;
    int32_t count;
};

__device__ __forceinline__ WelfordState combineWelford(
    WelfordState left, WelfordState right)
{
    if (right.count == 0)
    {
        return left;
    }
    if (left.count == 0)
    {
        return right;
    }
    float const delta = right.mean - left.mean;
    int32_t const count = left.count + right.count;
    float const rightWeight
        = static_cast<float>(right.count) / static_cast<float>(count);
    float const crossWeight
        = static_cast<float>(left.count) * static_cast<float>(right.count)
        / static_cast<float>(count);
    return WelfordState{
        left.mean + delta * rightWeight,
        left.m2 + right.m2 + delta * delta * crossWeight,
        count,
    };
}

__device__ __forceinline__ WelfordState warpReduceWelford(WelfordState state)
{
    #pragma unroll
    for (int32_t offset = 16; offset > 0; offset >>= 1)
    {
        WelfordState const other{
            __shfl_down_sync(0xFFFFFFFFU, state.mean, offset),
            __shfl_down_sync(0xFFFFFFFFU, state.m2, offset),
            __shfl_down_sync(0xFFFFFFFFU, state.count, offset),
        };
        state = combineWelford(state, other);
    }
    return state;
}

__global__ void reduceImageWelford(__nv_bfloat16 const* input,
    WelfordState* partial, int64_t elements)
{
    WelfordState state{0.0F, 0.0F, 0};
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        float const value = __bfloat162float(input[index]);
        int32_t const nextCount = state.count + 1;
        float const delta = value - state.mean;
        state.mean += delta / static_cast<float>(nextCount);
        float const delta2 = value - state.mean;
        state.m2 += delta * delta2;
        state.count = nextCount;
    }

    state = warpReduceWelford(state);
    __shared__ WelfordState warpStates[kThreads / 32];
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpStates[warp] = state;
    }
    __syncthreads();
    if (warp == 0)
    {
        state = lane < (kThreads / 32)
            ? warpStates[lane] : WelfordState{0.0F, 0.0F, 0};
        state = warpReduceWelford(state);
        if (lane == 0)
        {
            partial[blockIdx.x] = state;
        }
    }
}

__global__ void finalizeAndNormalizeWelford(__nv_bfloat16 const* input,
    __nv_bfloat16 const* weight, __nv_bfloat16 const* bias,
    WelfordState const* partial, __nv_bfloat16* output,
    __nv_bfloat16* stdOutput, int32_t blocks, int64_t elements,
    int32_t channels, float epsilon)
{
    WelfordState state = threadIdx.x < blocks
        ? partial[threadIdx.x] : WelfordState{0.0F, 0.0F, 0};
    state = warpReduceWelford(state);
    __shared__ WelfordState warpStates[kThreads / 32];
    __shared__ float sharedMean;
    __shared__ float sharedStd;
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpStates[warp] = state;
    }
    __syncthreads();
    if (warp == 0)
    {
        state = lane < (kThreads / 32)
            ? warpStates[lane] : WelfordState{0.0F, 0.0F, 0};
        state = warpReduceWelford(state);
        if (lane == 0)
        {
            float const mean = __bfloat162float(__float2bfloat16_rn(state.mean));
            float const meanDelta = state.mean - mean;
            float const centeredM2 = state.m2
                + static_cast<float>(state.count) * meanDelta * meanDelta;
            float const variance = __bfloat162float(__float2bfloat16_rn(
                centeredM2 / static_cast<float>(state.count)));
            float const epsilonRounded
                = __bfloat162float(__float2bfloat16_rn(epsilon));
            float const rootInput = __bfloat162float(__float2bfloat16_rn(
                variance + epsilonRounded));
            __nv_bfloat16 const stdValue
                = __float2bfloat16_rn(sqrtf(rootInput));
            sharedMean = mean;
            sharedStd = __bfloat162float(stdValue);
            if (blockIdx.x == 0)
            {
                stdOutput[0] = stdValue;
            }
        }
    }
    __syncthreads();

    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        int32_t const channel = static_cast<int32_t>(index % channels);
        float const value = __bfloat162float(input[index]);
        float const centered = __bfloat162float(__float2bfloat16_rn(
            value - sharedMean));
        float const normalized = __bfloat162float(__float2bfloat16_rn(
            centered / sharedStd));
        float const scaled = __bfloat162float(__float2bfloat16_rn(
            normalized * __bfloat162float(weight[channel])));
        output[index] = __float2bfloat16_rn(
            scaled + __bfloat162float(bias[channel]));
    }
}

cudaError_t launchImageLayerNormWelford(void const* input, void const* weight,
    void const* bias, void* normalizedOutput, void* stdOutput, void* workspace,
    int32_t batch, int64_t tokens, int32_t channels, float epsilon,
    cudaStream_t stream) noexcept
{
    if (input == nullptr || weight == nullptr || bias == nullptr
        || normalizedOutput == nullptr || stdOutput == nullptr || workspace == nullptr
        || batch != 1 || tokens <= 0 || channels <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const elements = tokens * channels;
    int32_t const blocks = static_cast<int32_t>(
        (elements + kThreads - 1) / kThreads < kMaxBlocks
            ? (elements + kThreads - 1) / kThreads : kMaxBlocks);
    auto* partial = static_cast<WelfordState*>(workspace);
    auto const* inputBf16 = static_cast<__nv_bfloat16 const*>(input);
    reduceImageWelford<<<blocks, kThreads, 0, stream>>>(
        inputBf16, partial, elements);
    finalizeAndNormalizeWelford<<<blocks, kThreads, 0, stream>>>(
        inputBf16,
        static_cast<__nv_bfloat16 const*>(weight),
        static_cast<__nv_bfloat16 const*>(bias),
        partial,
        static_cast<__nv_bfloat16*>(normalizedOutput),
        static_cast<__nv_bfloat16*>(stdOutput),
        blocks, elements, channels, epsilon);
    return cudaPeekAtLastError();
}

bool isConcreteOrWildcard(int32_t actual, int32_t expected) noexcept
{
    return actual == -1 || actual == expected;
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs, int32_t channels) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 3 || nbOutputs != 2)
    {
        return false;
    }
    auto const& x = inputs[0].dims;
    auto const& weight = inputs[1].dims;
    auto const& bias = inputs[2].dims;
    auto const& normalized = outputs[0].dims;
    auto const& stdOutput = outputs[1].dims;
    if (x.nbDims != 3 || weight.nbDims != 1 || bias.nbDims != 1
        || normalized.nbDims != 3 || stdOutput.nbDims != 3)
    {
        return false;
    }
    return isConcreteOrWildcard(x.d[0], 1)
        && isConcreteOrWildcard(x.d[2], channels)
        && isConcreteOrWildcard(weight.d[0], channels)
        && isConcreteOrWildcard(bias.d[0], channels)
        && isConcreteOrWildcard(normalized.d[0], x.d[0])
        && isConcreteOrWildcard(normalized.d[1], x.d[1])
        && isConcreteOrWildcard(normalized.d[2], x.d[2])
        && isConcreteOrWildcard(stdOutput.d[0], x.d[0])
        && isConcreteOrWildcard(stdOutput.d[1], 1)
        && isConcreteOrWildcard(stdOutput.d[2], 1);
}

class DrftImageLayerNormPlugin final : public nvinfer1::IPluginV3,
                                       public nvinfer1::IPluginV3OneCore,
                                       public nvinfer1::IPluginV3OneBuild,
                                       public nvinfer1::IPluginV3OneRuntime
{
public:
    DrftImageLayerNormPlugin(int32_t channels, float epsilon, bool fast,
        bool cooperative = false)
        : mChannels(channels), mEpsilon(epsilon), mFast(fast),
          mCooperative(cooperative)
    {
        refreshSerializationFields();
    }

    DrftImageLayerNormPlugin(DrftImageLayerNormPlugin const& other)
        : mChannels(other.mChannels), mEpsilon(other.mEpsilon),
          mTactic(other.mTactic), mFast(other.mFast),
          mCooperative(other.mCooperative)
    {
        refreshSerializationFields();
    }

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override
    {
        switch (type)
        {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        default:
            return nullptr;
        }
    }

    nvinfer1::IPluginV3* clone() noexcept override
    {
        return new (std::nothrow) DrftImageLayerNormPlugin(*this);
    }

    char const* getPluginName() const noexcept override
    {
        return mCooperative
            ? kCooperativePluginName : (mFast ? kFastPluginName : kPluginName);
    }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 2; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbOutputs != 2 || nbInputs != 3)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        outputTypes[1] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& builder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 3
            || nbShapeInputs != 0 || nbOutputs != 2)
        {
            return 1;
        }
        outputs[0] = inputs[0];
        outputs[1].nbDims = 3;
        outputs[1].d[0] = inputs[0].d[0];
        outputs[1].d[1] = builder.constant(1);
        outputs[1].d[2] = builder.constant(1);
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 3 || nbOutputs != 2
            || position < 0 || position >= 5)
        {
            return false;
        }
        return inOut[position].desc.format == nvinfer1::TensorFormat::kLINEAR
            && inOut[position].desc.type == nvinfer1::DataType::kBF16;
    }

    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::DynamicPluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        std::array<nvinfer1::PluginTensorDesc, 3> inputDescriptions{};
        std::array<nvinfer1::PluginTensorDesc, 2> outputDescriptions{};
        for (int32_t index = 0; index < nbInputs && index < 3; ++index)
        {
            inputDescriptions[index] = inputs[index].desc;
        }
        for (int32_t index = 0; index < nbOutputs && index < 2; ++index)
        {
            outputDescriptions[index] = outputs[index].desc;
        }
        return validateDescriptors(inputDescriptions.data(), nbInputs,
                   outputDescriptions.data(), nbOutputs, mChannels) ? 0 : 1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return mFast
            ? static_cast<size_t>(kMaxBlocks) * sizeof(WelfordState)
            : static_cast<size_t>(kMaxBlocks + 2) * sizeof(float);
    }

    int32_t getNbTactics() noexcept override { return 1; }
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override
    {
        if (tactics == nullptr || nbTactics != 1)
        {
            return 1;
        }
        tactics[0] = kTactic;
        return 0;
    }
    char const* getTimingCacheID() noexcept override
    {
        return mCooperative
            ? "bf16_cooperative_two_pass_image_layer_norm_v1"
            : (mFast
            ? "bf16_welford_image_layer_norm_v1"
            : "bf16_two_pass_image_layer_norm_v1");
    }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override
    {
        return mCooperative
            ? "{\"algorithm\":\"cooperative_two_pass_centered_variance\",\"accumulation\":\"fp32\"}"
            : (mFast
            ? "{\"algorithm\":\"two_kernel_welford\",\"accumulation\":\"fp32\"}"
            : "{\"algorithm\":\"two_pass_centered_variance\",\"accumulation\":\"fp32\"}");
    }
    int32_t setTactic(int32_t tactic) noexcept override
    {
        if (tactic != kTactic)
        {
            return 1;
        }
        mTactic = tactic;
        return 0;
    }

    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* inputs,
        int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
        int32_t nbOutputs) noexcept override
    {
        return validateDescriptors(inputs, nbInputs, outputs, nbOutputs, mChannels)
            ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override
    {
        if (mTactic != kTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        int32_t const batch = inputDescriptions[0].dims.d[0];
        int64_t const tokens = inputDescriptions[0].dims.d[1];
        int32_t const channels = inputDescriptions[0].dims.d[2];
        cudaError_t const error = mCooperative
            ? launchImageLayerNormCooperative(
                inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], workspace,
                batch, tokens, channels, mEpsilon, stream)
            : (mFast ? launchImageLayerNormWelford(
                inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], workspace,
                batch, tokens, channels, mEpsilon, stream)
            : launchImageLayerNorm(
                inputs[0], inputs[1], inputs[2], outputs[0], outputs[1], workspace,
                batch, tokens, channels, mEpsilon, stream));
        return error == cudaSuccess ? 0 : 1;
    }

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext*) noexcept override
    {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override
    {
        refreshSerializationFields();
        return &mSerializationCollection;
    }

private:
    void refreshSerializationFields() noexcept
    {
        mSerializationFields = {
            nvinfer1::PluginField{"channels", &mChannels,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"epsilon", &mEpsilon,
                nvinfer1::PluginFieldType::kFLOAT32, 1},
        };
        mSerializationCollection.nbFields
            = static_cast<int32_t>(mSerializationFields.size());
        mSerializationCollection.fields = mSerializationFields.data();
    }

    int32_t mChannels{0};
    float mEpsilon{1.0e-4F};
    int32_t mTactic{kTactic};
    bool mFast{false};
    bool mCooperative{false};
    std::array<nvinfer1::PluginField, 2> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftImageLayerNormPluginCreator final : public nvinfer1::IPluginCreatorV3One
{
public:
    explicit DrftImageLayerNormPluginCreator(
        bool fast = false, bool cooperative = false)
        : mFast(fast), mCooperative(cooperative)
    {
        mFields = {
            nvinfer1::PluginField{"channels", nullptr,
                nvinfer1::PluginFieldType::kINT32, 1},
            nvinfer1::PluginField{"epsilon", nullptr,
                nvinfer1::PluginFieldType::kFLOAT32, 1},
        };
        mCollection.nbFields = static_cast<int32_t>(mFields.size());
        mCollection.fields = mFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase) noexcept override
    {
        int32_t channels = 0;
        float epsilon = 0.0F;
        if (fields == nullptr)
        {
            return nullptr;
        }
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name == nullptr || field.data == nullptr || field.length != 1)
            {
                continue;
            }
            if (std::strcmp(field.name, "channels") == 0
                && field.type == nvinfer1::PluginFieldType::kINT32)
            {
                channels = *static_cast<int32_t const*>(field.data);
            }
            else if (std::strcmp(field.name, "epsilon") == 0
                && field.type == nvinfer1::PluginFieldType::kFLOAT32)
            {
                epsilon = *static_cast<float const*>(field.data);
            }
        }
        if (channels <= 0 || epsilon <= 0.0F || !std::isfinite(epsilon))
        {
            return nullptr;
        }
        return new (std::nothrow) DrftImageLayerNormPlugin(
            channels, epsilon, mFast, mCooperative);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    char const* getPluginName() const noexcept override
    {
        return mCooperative
            ? kCooperativePluginName : (mFast ? kFastPluginName : kPluginName);
    }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 2> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
    bool mFast{false};
    bool mCooperative{false};
};
} // namespace

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormPluginCreator()
{
    static DrftImageLayerNormPluginCreator creator(false);
    return &creator;
}

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormFastPluginCreator()
{
    static DrftImageLayerNormPluginCreator creator(true);
    return &creator;
}

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetImageLayerNormCooperativePluginCreator()
{
    static DrftImageLayerNormPluginCreator creator(false, true);
    return &creator;
}
