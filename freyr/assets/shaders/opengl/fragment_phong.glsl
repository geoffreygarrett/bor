#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 FragPos;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

uniform float ambientStrength;
uniform float specularStrength;
uniform float shininess;

// Include the modular shader components here, or use your preferred method to inject them.

void main()
{
    vec3 ambient = CalcAmbient(lightColor, ambientStrength);
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    vec3 diffuse = CalcDiffuse(norm, lightDir, lightColor);
    vec3 viewDir = normalize(-FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    vec3 specular = CalcSpecular(viewDir, reflectDir, lightColor, specularStrength, shininess);

    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
}
