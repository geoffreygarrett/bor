float3 calcDiffuse(float3 normal, float3 lightDir, float3 lightColor) {
    float diff = max(dot(normal, lightDir), 0.0);
    return diff * lightColor;
}
