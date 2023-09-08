#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float3 fragNormal;
    float3 fragPos;
};

vertex VertexOut vertex_main(const VertexIn vertex_in [[stage_in]],
                             constant float4x4& model [[buffer(0)]],
                             constant float4x4& view [[buffer(1)]],
                             constant float4x4& projection [[buffer(2)]]) {
    VertexOut vertex_out;

    float4 worldPos = model * float4(vertex_in.position, 1.0);
    vertex_out.fragPos = worldPos.xyz;
    vertex_out.fragNormal = (model * float4(vertex_in.normal, 0.0)).xyz;
    vertex_out.position = projection * view * worldPos;

    return vertex_out;
}
