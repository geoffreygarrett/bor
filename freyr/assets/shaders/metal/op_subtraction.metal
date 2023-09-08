

#include <metal_stdlib>
using namespace metal;

kernel void op_subtraction(device const float *inA,
                        device const float *inB,
                        device float       *result,
                        uint                index [[thread_position_in_grid]]) {
    result[index] = inA[index] - inB[index];
}