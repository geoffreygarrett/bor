vec3 CalcSpecular(vec3 viewDir, vec3 reflectDir, vec3 lightColor, float specularStrength, float shininess) {
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    return specularStrength * spec * lightColor;
}
