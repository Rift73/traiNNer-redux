#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <array>
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
constexpr char kPluginName[] = "DrftFFNSwiGLUDepthwise_TRT";
constexpr char kPluginVersion[] = "1";
constexpr int32_t kTactic = 1;
constexpr int32_t kSpatialTile = 16;
constexpr int32_t kChannelTile = 8;
constexpr int32_t kHaloTile = kSpatialTile + 2;
constexpr int32_t kThreads = kSpatialTile * kSpatialTile;

__device__ __forceinline__ float bf16Round(float value)
{
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void fusedSwiGLUDepthwise(__nv_bfloat16 const* gate,
    __nv_bfloat16 const* value, __nv_bfloat16 const* weight,
    __nv_bfloat16 const* bias, __nv_bfloat16* output, int32_t batch,
    int32_t height, int32_t width, int32_t channels)
{
    __shared__ __half activated[kHaloTile * kHaloTile * kChannelTile];

    int32_t const thread = static_cast<int32_t>(threadIdx.x);
    int32_t const localY = thread / kSpatialTile;
    int32_t const localX = thread - localY * kSpatialTile;
    int32_t const channelTiles = (channels + kChannelTile - 1) / kChannelTile;
    int32_t const batchChannelTile = static_cast<int32_t>(blockIdx.z);
    int32_t const batchIndex = batchChannelTile / channelTiles;
    int32_t const channelBase
        = (batchChannelTile - batchIndex * channelTiles) * kChannelTile;
    int32_t const imageYBase = static_cast<int32_t>(blockIdx.y) * kSpatialTile;
    int32_t const imageXBase = static_cast<int32_t>(blockIdx.x) * kSpatialTile;
    constexpr int32_t sharedElements
        = kHaloTile * kHaloTile * kChannelTile;

    for (int32_t index = thread; index < sharedElements; index += kThreads)
    {
        int32_t const channelOffset = index % kChannelTile;
        int32_t const spatial = index / kChannelTile;
        int32_t const haloY = spatial / kHaloTile;
        int32_t const haloX = spatial - haloY * kHaloTile;
        int32_t const y = imageYBase + haloY - 1;
        int32_t const x = imageXBase + haloX - 1;
        int32_t const channel = channelBase + channelOffset;
        float activatedValue = 0.0F;
        if (batchIndex < batch && channel < channels
            && static_cast<uint32_t>(y) < static_cast<uint32_t>(height)
            && static_cast<uint32_t>(x) < static_cast<uint32_t>(width))
        {
            int64_t const tensorIndex
                = ((static_cast<int64_t>(batchIndex) * height + y) * width + x)
                * channels + channel;
            float const gateValue = __bfloat162float(gate[tensorIndex]);
            float const valueValue = __bfloat162float(value[tensorIndex]);
            float const sigmoidValue
                = bf16Round(1.0F / (1.0F + expf(-gateValue)));
            float const siluValue = bf16Round(gateValue * sigmoidValue);
            activatedValue = bf16Round(siluValue * valueValue);
        }
        activated[index] = __float2half_rn(activatedValue);
    }
    __syncthreads();

    int32_t const y = imageYBase + localY;
    int32_t const x = imageXBase + localX;
    if (batchIndex >= batch
        || static_cast<uint32_t>(y) >= static_cast<uint32_t>(height)
        || static_cast<uint32_t>(x) >= static_cast<uint32_t>(width))
    {
        return;
    }

    #pragma unroll
    for (int32_t channelOffset = 0; channelOffset < kChannelTile;
         ++channelOffset)
    {
        int32_t const channel = channelBase + channelOffset;
        if (channel >= channels)
        {
            break;
        }
        float sum = 0.0F;
        #pragma unroll
        for (int32_t kernelY = 0; kernelY < 3; ++kernelY)
        {
            #pragma unroll
            for (int32_t kernelX = 0; kernelX < 3; ++kernelX)
            {
                int32_t const sharedIndex
                    = (((localY + kernelY) * kHaloTile
                           + localX + kernelX)
                          * kChannelTile)
                    + channelOffset;
                int32_t const weightIndex
                    = (channel * 3 + kernelY) * 3 + kernelX;
                __half const weightHalf = __float2half_rn(
                    __bfloat162float(weight[weightIndex]));
                sum += __half2float(activated[sharedIndex])
                    * __half2float(weightHalf);
            }
        }
        __half const convolutionHalf = __float2half_rn(sum);
        float const convolutionBf16
            = bf16Round(__half2float(convolutionHalf));
        float const biased = bf16Round(
            convolutionBf16 + __bfloat162float(bias[channel]));
        int64_t const outputIndex
            = ((static_cast<int64_t>(batchIndex) * height + y) * width + x)
            * channels + channel;
        output[outputIndex] = __float2bfloat16_rn(biased);
    }
}

cudaError_t launchFusedFFN(void const* gate, void const* value,
    void const* weight, void const* bias, void* output, int32_t batch,
    int32_t height, int32_t width, int32_t channels,
    cudaStream_t stream) noexcept
{
    if (gate == nullptr || value == nullptr || weight == nullptr
        || bias == nullptr || output == nullptr || batch <= 0 || height <= 0
        || width <= 0 || channels <= 0)
    {
        return cudaErrorInvalidValue;
    }
    int32_t const channelTiles = (channels + kChannelTile - 1) / kChannelTile;
    dim3 const grid(static_cast<uint32_t>(
                        (width + kSpatialTile - 1) / kSpatialTile),
        static_cast<uint32_t>((height + kSpatialTile - 1) / kSpatialTile),
        static_cast<uint32_t>(batch * channelTiles));
    fusedSwiGLUDepthwise<<<grid, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16 const*>(gate),
        static_cast<__nv_bfloat16 const*>(value),
        static_cast<__nv_bfloat16 const*>(weight),
        static_cast<__nv_bfloat16 const*>(bias),
        static_cast<__nv_bfloat16*>(output), batch, height, width, channels);
    return cudaGetLastError();
}

bool concreteOrWildcard(int32_t actual, int32_t expected)
{
    return actual == -1 || actual == expected;
}

bool validateDescriptors(nvinfer1::PluginTensorDesc const* inputs,
    int32_t nbInputs, nvinfer1::PluginTensorDesc const* outputs,
    int32_t nbOutputs, int32_t channels)
{
    if (inputs == nullptr || outputs == nullptr || nbInputs != 4
        || nbOutputs != 1 || channels <= 0)
    {
        return false;
    }
    auto const& gate = inputs[0].dims;
    auto const& value = inputs[1].dims;
    auto const& weight = inputs[2].dims;
    auto const& bias = inputs[3].dims;
    auto const& output = outputs[0].dims;
    if (gate.nbDims != 4 || value.nbDims != 4 || weight.nbDims != 4
        || bias.nbDims != 1 || output.nbDims != 4)
    {
        return false;
    }
    return concreteOrWildcard(gate.d[3], channels)
        && concreteOrWildcard(value.d[0], gate.d[0])
        && concreteOrWildcard(value.d[1], gate.d[1])
        && concreteOrWildcard(value.d[2], gate.d[2])
        && concreteOrWildcard(value.d[3], channels)
        && concreteOrWildcard(weight.d[0], channels)
        && concreteOrWildcard(weight.d[1], 1)
        && concreteOrWildcard(weight.d[2], 3)
        && concreteOrWildcard(weight.d[3], 3)
        && concreteOrWildcard(bias.d[0], channels)
        && concreteOrWildcard(output.d[0], gate.d[0])
        && concreteOrWildcard(output.d[1], gate.d[1])
        && concreteOrWildcard(output.d[2], gate.d[2])
        && concreteOrWildcard(output.d[3], channels);
}

class DrftFFNSwiGLUDepthwisePlugin final : public nvinfer1::IPluginV3,
                                           public nvinfer1::IPluginV3OneCore,
                                           public nvinfer1::IPluginV3OneBuild,
                                           public nvinfer1::IPluginV3OneRuntime
{
public:
    explicit DrftFFNSwiGLUDepthwisePlugin(int32_t channels)
        : mChannels(channels)
    {
        refreshSerializationFields();
    }

    DrftFFNSwiGLUDepthwisePlugin(
        DrftFFNSwiGLUDepthwisePlugin const& other)
        : mChannels(other.mChannels), mTactic(other.mTactic)
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
        return new (std::nothrow) DrftFFNSwiGLUDepthwisePlugin(*this);
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes,
        int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override
    {
        if (outputTypes == nullptr || inputTypes == nullptr || nbOutputs != 1
            || nbInputs != 4)
        {
            return 1;
        }
        outputTypes[0] = inputTypes[0];
        return 0;
    }

    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs, nvinfer1::DimsExprs const*, int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder&) noexcept override
    {
        if (inputs == nullptr || outputs == nullptr || nbInputs != 4
            || nbShapeInputs != 0 || nbOutputs != 1)
        {
            return 1;
        }
        outputs[0] = inputs[0];
        return 0;
    }

    bool supportsFormatCombination(int32_t position,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs, int32_t nbOutputs) noexcept override
    {
        if (inOut == nullptr || nbInputs != 4 || nbOutputs != 1
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
        std::array<nvinfer1::PluginTensorDesc, 4> inputDescriptions{};
        std::array<nvinfer1::PluginTensorDesc, 1> outputDescriptions{};
        for (int32_t index = 0; index < nbInputs && index < 4; ++index)
        {
            inputDescriptions[index] = inputs[index].desc;
        }
        if (nbOutputs == 1)
        {
            outputDescriptions[0] = outputs[0].desc;
        }
        return validateDescriptors(inputDescriptions.data(), nbInputs,
                   outputDescriptions.data(), nbOutputs, mChannels)
            ? 0
            : 1;
    }

    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) const noexcept override
    {
        return 0;
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
        return "bf16_swiglu_fp16_depthwise_bf16_output_v1";
    }
    int32_t getFormatCombinationLimit() noexcept override { return 1; }
    char const* getMetadataString() noexcept override
    {
        return "{\"algorithm\":\"tiled_swiglu_depthwise\",\"kernel\":\"3x3\"}";
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
        return validateDescriptors(
                   inputs, nbInputs, outputs, nbOutputs, mChannels)
            ? 0
            : 1;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDescriptions,
        nvinfer1::PluginTensorDesc const*, void const* const* inputs,
        void* const* outputs, void*, cudaStream_t stream) noexcept override
    {
        if (mTactic != kTactic || inputDescriptions == nullptr
            || inputs == nullptr || outputs == nullptr)
        {
            return 1;
        }
        auto const& dims = inputDescriptions[0].dims;
        if (dims.nbDims != 4)
        {
            return 1;
        }
        cudaError_t const error = launchFusedFFN(inputs[0], inputs[1],
            inputs[2], inputs[3], outputs[0], dims.d[0], dims.d[1], dims.d[2],
            mChannels, stream);
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
        mSerializationFields = {nvinfer1::PluginField{"channels", &mChannels,
            nvinfer1::PluginFieldType::kINT32, 1}};
        mSerializationCollection.nbFields = 1;
        mSerializationCollection.fields = mSerializationFields.data();
    }

    int32_t mChannels{0};
    int32_t mTactic{kTactic};
    std::array<nvinfer1::PluginField, 1> mSerializationFields{};
    nvinfer1::PluginFieldCollection mSerializationCollection{};
};

class DrftFFNSwiGLUDepthwisePluginCreator final
    : public nvinfer1::IPluginCreatorV3One
{
public:
    DrftFFNSwiGLUDepthwisePluginCreator()
    {
        mFields = {nvinfer1::PluginField{"channels", nullptr,
            nvinfer1::PluginFieldType::kINT32, 1}};
        mCollection.nbFields = 1;
        mCollection.fields = mFields.data();
    }

    nvinfer1::IPluginV3* createPlugin(char const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase) noexcept override
    {
        int32_t channels = 0;
        if (fields == nullptr)
        {
            return nullptr;
        }
        for (int32_t index = 0; index < fields->nbFields; ++index)
        {
            auto const& field = fields->fields[index];
            if (field.name != nullptr && field.data != nullptr
                && field.length == 1
                && field.type == nvinfer1::PluginFieldType::kINT32
                && std::strcmp(field.name, "channels") == 0)
            {
                channels = *static_cast<int32_t const*>(field.data);
            }
        }
        return channels > 0
            ? new (std::nothrow) DrftFFNSwiGLUDepthwisePlugin(channels)
            : nullptr;
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override
    {
        return &mCollection;
    }
    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    char const* getPluginNamespace() const noexcept override { return ""; }

private:
    std::array<nvinfer1::PluginField, 1> mFields{};
    nvinfer1::PluginFieldCollection mCollection{};
};
} // namespace

DRFT_PLUGIN_EXPORT nvinfer1::IPluginCreatorInterface*
drftGetFFNSwiGLUDepthwisePluginCreator()
{
    static DrftFFNSwiGLUDepthwisePluginCreator creator;
    return &creator;
}
