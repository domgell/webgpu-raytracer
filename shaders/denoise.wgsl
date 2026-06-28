@group(0) @binding(0) var<uniform> state: State;
@group(0) @binding(1) var<storage, read> sampleInfos: array<SampleInfo>;

@group(1) @binding(0) var<uniform> stepIndex: u32;
@group(1) @binding(1) var inputTexture: texture_storage_2d<rgba8unorm, read>;
@group(1) @binding(2) var outputTexture: texture_storage_2d<rgba8unorm, write>;

// --------------------------------------- State ---------------------------------------

struct State {
    sigmaColor: f32,
    sigmaNormal: f32,
    sigmaDepth: f32,
    sigmaAlbedo: f32,
    sigmaReflectPosition: f32,
    sigmaReflectDistance: f32,
}

// ------------------------------------ Sample Info ------------------------------------

struct SampleInfo {
    normal: vec3f,
    depth: f32,
    color: vec3f,
    roughness: f32,
    reflectHitPosition: vec3f,
    reflectHitDistance: f32,
    reflectHitAlbedo: vec3f,
}

// -------------------------------------- Texture --------------------------------------

fn readTexture(coord: vec2u) -> vec3f {
    return textureLoad(inputTexture, coord).rgb;
}

fn writeTexture(coord: vec2u, color: vec3f) {
    textureStore(outputTexture, coord, vec4f(color, 1.0));
}

// -------------------------------------- Random ---------------------------------------

fn initRandom(x: u32, y: u32, z: u32) -> u32 {
    var seed = x * 1973u + y * 9277u + z * 26699u;
    seed ^= seed >> 16;
    seed *= 0x85ebca6bu;
    seed ^= seed >> 13;
    seed *= 0xc2b2ae35u;
    return seed;
}

fn nextRandom(rng: ptr<function, u32>) -> f32 {
    *rng = *rng * 1664525u + 1013904223u;
    return f32(*rng & 0x00ffffffu) / 16777216.0;
}

// --------------------------------------- Main ----------------------------------------

fn bilateralWeight(dist2: f32, sigma: f32) -> f32 {
    return 1.0 / (1.0 + dist2 / (sigma * sigma) + 1e-6);
}

const kernel = array<f32, 5>(0.0625, 0.25, 0.375, 0.25, 0.0625);

override workgroupSize: u32 = 8;

@compute @workgroup_size(workgroupSize, workgroupSize, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2u(gid.xy);
    let size = textureDimensions(outputTexture);
    if any(coord >= size) {
        return;
    }

    let centerColor = readTexture(coord);
    let centerIndex = coord.y * size.x + coord.x;
    let centerInfo = sampleInfos[centerIndex];

    let step = 1u << stepIndex;
    var rng = initRandom(coord.x, coord.y, step);

    var colorSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var y: i32 = -2; y <= 2; y++) {
        for (var x: i32 = -2; x <= 2; x++) {
            let jitter = vec2f(nextRandom(&rng), nextRandom(&rng)) - vec2f(0.5);
            let jitterOffset = vec2i(jitter * f32(step));
            let offset = vec2i(x, y) * i32(step) + jitterOffset;
            let clamped = clamp(vec2i(coord) + offset, vec2i(0), vec2i(size) - vec2i(1));
            let sampleCoord = vec2u(clamped);

            let sampleColor = readTexture(sampleCoord);
            let sampleIndex = sampleCoord.y * size.x + sampleCoord.x;
            let sampleInfo = sampleInfos[sampleIndex];

            var weight = kernel[u32(x + 2)] * kernel[u32(y + 2)];

            // Handle reflective surface
            if centerInfo.roughness < 1 {
                let mirrorness = 1.0 - centerInfo.roughness;
                var rWeight = 1.0;

                // Reflection position weight
                let reflectPosDiff = centerInfo.reflectHitPosition - sampleInfo.reflectHitPosition;
                let adaptiveSigmaPos = state.sigmaReflectPosition * (1.0 + centerInfo.roughness * 8.0);
                rWeight *= bilateralWeight(dot(reflectPosDiff, reflectPosDiff), adaptiveSigmaPos);

                // Reflection distance weight
                let reflectDistDiff = centerInfo.reflectHitDistance - sampleInfo.reflectHitDistance;
                let adaptiveSigmaDist = state.sigmaReflectDistance * (1.0 + sampleInfo.roughness);
                rWeight *= bilateralWeight(reflectDistDiff * reflectDistDiff, adaptiveSigmaDist);

                // Reflection albedo weight
                let reflectAlbedoDiff = centerInfo.reflectHitAlbedo - sampleInfo.reflectHitAlbedo;
                let sigmaReflectAlbedo = mix(0.02, 0.15, centerInfo.roughness);
                rWeight *= bilateralWeight(dot(reflectAlbedoDiff, reflectAlbedoDiff), sigmaReflectAlbedo);

                weight *= mix(1.0, rWeight, mirrorness);
            }

            // Color weight
            let colorDiff = centerColor - sampleColor;
            weight *= bilateralWeight(dot(colorDiff, colorDiff), state.sigmaColor);

            // Normal weight
            let normalDot = clamp(dot(centerInfo.normal, sampleInfo.normal), 0.0, 1.0);
            weight *= pow(normalDot, state.sigmaNormal * 128.0);

            // Depth weight
            let depthDiff = centerInfo.depth - sampleInfo.depth;
            weight *= bilateralWeight(depthDiff * depthDiff, state.sigmaDepth);

            // Albedo weight
            let albedoDiff = centerInfo.color - sampleInfo.color;
            weight *= bilateralWeight(dot(albedoDiff, albedoDiff), state.sigmaAlbedo);

            colorSum += sampleColor * weight;
            weightSum += weight;
        }
    }

    if weightSum > 0.001 {
        let filtered = colorSum / weightSum;
        writeTexture(coord, filtered);
    }
}