// File: device_manager.h
#ifndef FREYR_DEVICE_MANAGER_H
#define FREYR_DEVICE_MANAGER_H

#ifdef __APPLE__
#    include "metal/manager_impl.h"
#endif

#ifdef __CUDA_ARCH__
#    include "cuda/manager_impl.h"
#endif

#endif// FREYR_DEVICE_MANAGER_H
