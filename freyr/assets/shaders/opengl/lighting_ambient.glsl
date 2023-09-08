vec3 CalcAmbient(vec3 lightColor, float ambientStrength) {
    return ambientStrength * lightColor;
}
