@group(0) @binding(0) var srcTexture: texture_storage_2d<rgba8unorm, read>;

@fragment
fn frag(@location(0) uv: vec2f) -> @location(0) vec4f {
    let size = textureDimensions(srcTexture);
    let coord = vec2u(uv * vec2f(size));
    return textureLoad(srcTexture, coord);
}