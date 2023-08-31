// metal_wrapper.cpp

#include <freyr/algorithms/metal_wrapper.h>

template<typename T>
MetalBuffer<T>::MetalBuffer(id<MTLDevice> device, std::vector<T> &data) {
    buffer_ = [device newBufferWithBytes:data.data() length:data.size() * sizeof(T) options:MTLResourceStorageModeShared];
}