// metal_wrapper.h

#ifdef METAL_WRAPPER_H
#ifdef __APPLE__
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>

template<typename T>
class MetalBuffer {
public:
    MetalBuffer(id<MTLDevice> device, std::vector<T> &data);
    // ... (methods to read and write, etc.)

private:
    id<MTLBuffer> buffer_;
};

template<typename KernelFunction>
class MetalPipeline {
public:
    MetalPipeline(id<MTLDevice> device, const std::string &functionName);
    // ... (methods to set arguments, etc.)

private:
    id<MTLComputePipelineState> pipeline_;
};

template<typename T, typename KernelFunction>
class MetalComputation {
public:
    MetalComputation(id<MTLDevice> device, const std::string &functionName);

    void compute(MetalBuffer<T> &inputBuffer, MetalBuffer<T> &outputBuffer);
};

// Define implementation in metal_wrapper.cpp
#endif// __APPLE__
#endif// METAL_WRAPPER_H