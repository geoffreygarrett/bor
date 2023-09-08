kernel void gamma_correction(texture2d<float, access::write> outTexture [[texture(0)]],
                             texture2d<float, access::read> inTexture [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
  // Read original color
  float4 color = inTexture.read(gid);

  // Apply gamma correction
  const float gamma = 2.2;
  color.rgb = pow(color.rgb, float3(1.0 / gamma));

  // Write back
  outTexture.write(color, gid);
}
