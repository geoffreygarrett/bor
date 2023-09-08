#ifndef FREYR_DEVICE_BUFFER_H
#define FREYR_DEVICE_BUFFER_H

#ifdef __APPLE__
#    include "metal/buffer_impl.h"
#endif

#ifdef __CUDA_ARCH__
#    include "cuda/buffer_impl.h"
#endif

#ifdef __TBB_H
#    include "tbb/buffer_impl.h"
#endif

#endif// FREYR_DEVICE_BUFFER_H