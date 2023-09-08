//
//  add.metal
//
//  Created by Srimukh Sripada on 03.12.21.
//     ^^^^^ Must pay credits later ^^^^^
//

#include <metal_stdlib>
using namespace metal;

kernel void op_addition(device const float *inA,
                        device const float *inB,
                        device float       *result,
                        uint                index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}