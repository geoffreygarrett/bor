float3 calcSpecular(float3 viewDir, float3 reflectDir, float3 lightColor, float specularStrength, float shininess) {
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return specularStrength * spec * lightColor;
}
