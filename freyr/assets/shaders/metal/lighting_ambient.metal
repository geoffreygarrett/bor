float3 calcAmbient(float3 lightColor, float ambientStrength) {
    return ambientStrength * lightColor;
}
