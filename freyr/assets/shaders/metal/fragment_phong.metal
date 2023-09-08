#include <metal_stdlib>
using namespace metal;

fragment float4 fragment_main(const VertexOut  vertex_out [[stage_in]],
                              constant float3 &lightPos [[buffer(0)]],
                              constant float3 &lightColor [[buffer(1)]],
                              constant float3 &objectColor [[buffer(2)]],
                              constant float  &ambientStrength [[buffer(3)]],
                              constant float  &specularStrength [[buffer(4)]],
                              constant float  &shininess [[buffer(5)]]) {

    float3 normal   = normalize(vertex_out.fragNormal);
    float3 fragPos  = vertex_out.fragPos;
    float3 lightDir = normalize(lightPos - fragPos);
    float3 viewDir  = normalize(-fragPos);

    float3 ambient    = calcAmbient(lightColor, ambientStrength);
    float3 diffuse    = calcDiffuse(normal, lightDir, lightColor);
    float3 reflectDir = reflect(-lightDir, normal);
    float3 specular   = calcSpecular(viewDir, reflectDir, lightColor, specularStrength, shininess);

    float3 result = (ambient + diffuse + specular) * objectColor;
    return float4(result, 1.0);
}
