#include <cmath>
#include <iostream>
#include <tuple>

std::tuple<float, float> inverse_phong(float ip, float ca, float cd, float lx, float lz) {
    // Normalize the light direction vector
    float magnitude = std::sqrt(lx * lx + lz * lz);
    lx /= magnitude;
    lz /= magnitude;

    // Rewrite the Phong equation to solve for the dot product L.N
    float dot_product = (ip - ca) / cd;

    // Given L = [lx, lz] and L.N = dot_product,
    // solve for N using the equation 1 = sqrt(Nx^2 + Nz^2)

    // Solving the system:
    // dot_product = 0.71 * Nx + 0.71 * Nz
    // 1 = sqrt(Nx^2 + Nz^2)

    float nx = std::sqrt(1 / (lx * lx + lz * lz) - (dot_product * dot_product) / (lx * lx));
    float nz = (dot_product - lx * nx) / lz;

    return std::make_tuple(nx, nz);
}

int main() {
    float ip = 0.8;// Intensity at point x
    float ca = 0.1;// Ambient light constant
    float cd = 0.9;// Diffusion reflection constant

    // Light direction (45 degrees above the object)
    float lx = 0.71;
    float lz = 0.71;

    auto [nx, nz] = inverse_phong(ip, ca, cd, lx, lz);

    std::cout << "Calculated Normal: Nx = " << nx << ", Nz = " << nz << std::endl;

    return 0;
}