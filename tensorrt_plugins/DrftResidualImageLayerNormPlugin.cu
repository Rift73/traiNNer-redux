#include <NvInfer.h>
#include <NvInferPlugin.h>
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
constexpr char kPluginName[] = "DrftResidualImageLayerNorm_TRT";
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

__device__ __forceinline__ void finishBlockReduction(
    float value, float* partial)
{
    value = warpReduceSum(value);
    __shared__ float warpSums[kThreads / 32];
    int32_t const lane = static_cast<int32_t>(threadIdx.x) & 31;
    int32_t const warp = static_cast<int32_t>(threadIdx.x) >> 5;
    if (lane == 0)
    {
        warpSums[warp] = value;
    }
    __syncthreads();
    if (warp == 0)
    {
        value = lane < (kThreads / 32) ? warpSums[lane] : 0.0F;
        value = warpReduceSum(value);
        if (lane == 0)
        {
            partial[blockIdx.x] = value;
        }
    }
}

__global__ void addAndReduceImage(__nv_bfloat16 const* left,
    __nv_bfloat16 const* right, __nv_bfloat16* residual,
    float* partial, int64_t elements)
{
    float sum = 0.0F;
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        __nv_bfloat16 const value = __float2bfloat16_rn(
            __bfloat162float(left[index]) + __bfloat162float(right[index]));
        residual[index] = value;
        sum += __bfloat162float(value);
    }
    finishBlockReduction(sum, partial);
}

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
            statistics[0] = __bfloat162float(__float2bfloat16_rn(
                sum / static_cast<float>(elements)));
        }
    }
}

__global__ void reduceCenteredSquare(__nv_bfloat16 const* input,
    float const* statistics, float* partial, int64_t elements)
{
    float sum = 0.0F;
    float const mean = statistics[0];
    int64_t const stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        float const centered = __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(input[index]) - mean));
        sum += __bfloat162float(__float2bfloat16_rn(centered * centered));
    }
    finishBlockReduction(sum, partial);
}

__global__ void finalizeStd(float const* partial, float* statistics,
    __nv_bfloat16* stdOutput, int32_t blocks, int64_t elements,
    float epsilon)
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
            __nv_bfloat16 const stdValue
                = __float2bfloat16_rn(sqrtf(rootInput));
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
    for (int64_t index
             = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements; index += stride)
    {
        int32_t const channel = static_cast<int32_t>(index % channels);
        float const centered = __bfloat162float(__float2bfloat16_rn(
            __bfloat162float(input[index]) - mean));
        float const normalized = __bfloat162float(__float2bfloat16_rn(
            centered / stdValue));
        float const scaled = __bfloat162float(__float2bfloat16_rn(
            normalized * __bfloat162float(weight[channel])));
        output[index] = __float2bfloat16_rn(
            scaled + __bfloat162float(bias[channel]));
    }
}

cudaError_t launchResidualImageLayerNorm(void const* left, void const* right,
    void const* weight, void const* bias, void* residualOutput,
    void* normalizedOutput, void* stdOutput, void* workspace,
    int32_t batch, int64_t tokens, int32_t channels, float epsilon,
    cudaStream_t stream) noexcept
{
    if (left == nullptr || right == nullptr || weight == nullptr
        || bias == nullptr || residualOutput == nullptr
        || normalizedOutput == nullptr || stdOutput == nullptr
        || workspace == nullptr || batch != 1 || tokens <= 0 || channels <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int64_t const elements = tokens * channels;
    int32_t const blocks = static_cast<int32_t>(
        (elements + kThreads - 1) / kThreads < kMaxBlocks
            ? (elements + kThreads - 1) / kThreads : kMaxBlocks);
    auto* partial = static_cast<float*>(workspace);
    auto* statistics = partial + kMaxBlocks;
    auto* residual = static_cast<__nv_bfloat16*>(residualOutput);

    addAndReduceImage<<<blocks, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(left),
        static_cast<__nv_bfloat16 const*>(right), residual, partial, elements);
    finalizeMean<<<1, kThreads, 0, stream>>>(
        partial, statistics, blocks, elements);
    reduceCenteredSquare<<<blocks, kThreads, 0, stream>>>(
        residual, statistics, partial, elements);
    finalizeStd<<<1, kThreads, 0, stream>>>(partial, statistics,
        static_cast<__nv_bfloat16*>(stdOutput), blocks, elements, epsilon);
    normalizeImage<<<blocks, kThreads, 0, stream>>>(residual,
        static_cast<__nv_bfloat16 const*>(weight),
        static_cast<__nv_bfloat16 const*>(bias), statistics,
        static_cast<__nv_bfloat16*>(normalizedOutput), elements, channels);
    return cudaPeekAtLastError();
}

bool sameOrWildcard(int32_t left, int32_t right) noexcept
{
    return left == -1 || right == -1 || left == right;
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs, int32_t channels) noexcept
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 4
        || nbOutputs != 3 || channels <= 0)
    {
        return false;
    }
    auto const& left = inputs[0].dims;
    auto const& right = inputs[1].dims;
    auto const& weight = inputs[2].dims;
    auto const& bias = inputs[3].dims;
    auto const& residual = outputs[0].dims;
    auto const& normalized = outputs[1].dims;
    auto const& stdOutput = outputs[2].dims;
    if (left.nbDims != 3 || right.nbDims != 3 || weight.nbDims != 1
        || bias.nbDims != 1 || residual.nbDims != 3
        || normalized.nbDims != 3 || stdOutput.nbDims != 3)
    {
        return false;
    }
    return (left.d[0] == -1 || left.d[0] == 1)
        && (left.d[2] == -1 || left.d[2] == channels)
        && sameOrWildcard(right.d[0], left.d[0])
        && sameOrWildcard(right.d[1], left.d[1])
        && sameOrWildcard(right.d[2], left.d[2])
        && (weight.d[0] == -1 || weight.d[0] == channels)
        && (bias.d[0] == -1 || bias.d[0] == channels)
        && sameOrWildcard(residual.d[0], left.d[0])
        && sameOrWildcard(residual.d[1], left.d[1])
        && sameOrWildcard(residual.d[2], left.d[2])
        && sameOrWildcard(normalized.d[0], left.d[0])
        && sameOrWildcard(normalized.d[1], left.d[1])
        && sameOrWildcard(normalized.d[2], left.d[2])
        && sameOrWildcard(stdOutput.d[0], left.d[0])
        && (stdOutput.d[1] == -1 || stdOutput.d[1] == 1)
        && (stdOutput.d[2] == -1 || stdOutput.d[2] == 1);
}

class DrftResidualImageLayerNormPlugin final : public nvinfer1::IPluginV3,
    public nvinfer1::IPluginV3OneCore, public nvinfer1::IPluginV3OneBuild,
    public nvinfer1::IPluginV3OneRuntime
{
public:
    DrftResidualImageLayerNormPlugin(int32_t channels, float epsilon)
        : mChannels(channels), mEpsilon(epsilon)
    {
        refreshSerializationFields();
    }

    DrftResidualImageLayerNormPlugin(
        DrftResidualImageLayerNormPlugin const& other)
        : mChannels(other.mChannels), mEpsilon(other.mEpsilon),
          mTactic(other.mTactic)
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
        return new (std::nothrow) DrftResidualImageLayerNormPlugin(*this);
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 3; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
        int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr
            || nbOutputs != 3 || nbInputs != 4)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        outputTypes[1] = inputTypes[0];
        outputTypes[2] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& builder) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 4
            || nbShapeInputs != 0 || nbOutputs != 3)
        {
            return 1;
        }
        outputs[0] = inputs[0];
        outputs[1] = inputs[0];
        outputs[2].nbDims = 3;
        outputs[2].d[0] = inputs[0].d[0];
        outputs[2].d[1] = builder.constant(1);
        outputs[2].d[2] = builder.constant(1);
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 4 || nbOutputs != 3
            || position < 0 || position >= 7)
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
        std::array<nvinfer1::PluginTensorDesc, 4> inputDescriptions{};
        std::array<nvinfer1::PluginTensorDesc, 3> outputDescriptions{};
        for (int32_t index = 0; index < nbInputs && index < 4; ++index)
        {
            inputDescriptions[index] = inputs[index].desc;
        }
        for (int32_t index = 0; index < nbOutputs && index < 3; ++index)
        {
            outputDescriptions[index] = outputs[index].desc;
        }
        return validateDescriptors(inputDescriptions.data(), nbInputs,
                   outputDescriptions.data(), nbOutputs, mChannels) ? 0 : 1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return static_cast<size_t>(kMaxBlocks + 2) * sizeof(float);
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
        return "bf16_residual_two_pass_image_layer_norm_v1";
    }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override
    {
        return "{\"algorithm\":\"bf16_residual_plus_two_pass_centered_variance\",\"accumulation\":\"fp32\"}";
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
        return validateDescriptors(inputs, nbInputs, outputs, nbOutputs,
                   mChannels) ? 0 : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override
    {
        if (mTactic != kTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        int32_t const batch = inputDescriptions[0].dims.d[0];
        int64_t const tokens = inputDescriptions[0].dims.d[1];
        int32_t const channels = inputDescriptions[0].dims.d[2];
        cudaError_t const error = launchResidualImageLayerNorm(inputs[0],
            inputs[1], inputs[2], inputs[3], outputs[0], outputs[1],
            outputs[2], workspace, batch, tokens, channels, mEpsilon, stream);
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
    std::array<nvinfer1::PluginField, 2> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftResidualImageLayerNormPluginCreator final
    : public nvinfer1::IPluginCreatorV3One
{
public:
    DrftResidualImageLayerNormPluginCreator()
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
            if (field.name == nullptr || field.data == nullptr
                || field.length != 1)
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
        return new (std::nothrow)
            DrftResidualImageLayerNormPlugin(channels, epsilon);
    }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 2> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
};
} // namespace

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetResidualImageLayerNormPluginCreator()
{
    static DrftResidualImageLayerNormPluginCreator creator;
    return &creator;
}
