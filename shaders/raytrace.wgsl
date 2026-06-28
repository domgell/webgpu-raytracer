// Scene bind group
@group(0) @binding(0) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(1) var<storage, read> meshBases: array<MeshBase>;
@group(0) @binding(2) var<storage, read> meshInstances: array<MeshInstance>;
@group(0) @binding(3) var<storage, read> materials: array<Material>;
@group(0) @binding(4) var<storage, read> lightMeshIndices: array<u32>;
@group(0) @binding(5) var<storage, read> meshNodes: array<Node>;
@group(0) @binding(6) var<storage, read> sceneNodes: array<Node>;

// Screen bind group
@group(1) @binding(0) var outputTexture: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(1) var<storage, read_write> accumulationBuffer: array<vec3f>;
@group(1) @binding(2) var<storage, read_write> sampleInfos: array<SampleInfo>;

// State bind group
@group(2) @binding(0) var<uniform> state: State;
@group(2) @binding(1) var<uniform> camera: Camera;

// Texture atlas bind group
@group(3) @binding(0) var textureAtlas: texture_2d<f32>;
@group(3) @binding(1) var textureAtlasSampler: sampler;
@group(3) @binding(2) var<storage, read> textureAtlasMinMax: array<vec4f>;

// -------------------------------------- Camera ---------------------------------------

struct Camera {
    viewProjection: mat4x4f,
    invViewProjection: mat4x4f,
}

// ---------------------------------------- Ray ----------------------------------------

struct Ray {
    origin: vec3f,
    direction: vec3f,
}

fn createCameraRay(coord: vec2u, offset: vec2f) -> Ray {
    let size = vec2f(textureDimensions(outputTexture));
    let ndc = (vec2f(coord) + offset) / size;

    let clipXY = vec2f(ndc.x * 2.0 - 1.0, 1.0 - ndc.y * 2.0);

    let clipNear = vec4f(clipXY, 0.0, 1.0);
    let clipFar  = vec4f(clipXY, 1.0, 1.0);

    var nearWorld = camera.invViewProjection * clipNear;
    nearWorld /= nearWorld.w;

    var farWorld = camera.invViewProjection * clipFar;
    farWorld /= farWorld.w;

    let origin = nearWorld.xyz;
    let direction = normalize(farWorld.xyz - nearWorld.xyz);

    return Ray(origin, direction);
}

// ------------------------------------- Triangle --------------------------------------

struct Triangle {
    a: vec3f,
    b: vec3f,
    c: vec3f,
    normal: vec3f,
    auv: vec2f,
    buv: vec2f,
    cuv: vec2f,
}

struct TriangleHit {
    t: f32,
    u: f32,
    v: f32,
    hit: bool,
}

fn intersectTriangle(ray: Ray, tri: Triangle) -> TriangleHit {
    let edge1 = tri.b - tri.a;
    let edge2 = tri.c - tri.a;
    let h = cross(ray.direction, edge2);
    let a = dot(edge1, h);

    var hit: TriangleHit;

    if (a < minDistance) {
        return hit;
    }

    let f = 1.0 / a;
    let s = ray.origin - tri.a;
    let u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit;
    }

    let q = cross(s, edge1);
    let v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0) {
        return hit;
    }

    let t = f * dot(edge2, q);

    if (t < minDistance || t > maxDistance) {
        return hit;
    }

    hit.hit = true;
    hit.t = t;
    hit.u = u;
    hit.v = v;
    return hit;
}

// ---------------------------------------- Hit ----------------------------------------

struct Hit {
    position: vec3f,
    distance: f32,
    normal: vec3f,
    materialIndex: u32,
    uv: vec2f,
    hit: bool,
}

fn transformPoint(p: vec3f, transform: mat4x4f) -> vec3f {
    return (transform * vec4f(p, 1)).xyz;
}

fn transformNormal(n: vec3f, transform: mat4x4f) -> vec3f {
    return (transform * vec4f(n, 0)).xyz;
}

const meshNodeStackSize = 32;
const sceneNodeStackSize = 32;

fn hitScene(ray: Ray) -> Hit {
    var hit: Hit;
    hit.distance = maxDistance;

    var stack = array<u32, sceneNodeStackSize>();
    var stackIndex = 0;

    stack[stackIndex] = 0;
    stackIndex++;

    while (stackIndex > 0) {
        let nodeIndex = stack[stackIndex - 1];
        stackIndex--;
        let node = sceneNodes[nodeIndex];

        let boundsDist = intersectBounds(ray, node.min, node.max);
        if (boundsDist < minDistance || boundsDist > hit.distance) {
            continue;
        }

        // Leaf
        if node.countOrLeaf != 0 {
            for (var i = 0u; i < node.countOrLeaf; i++) {
                let meshInstance = meshInstances[node.leftOrOffset + i];
                let meshHit = hitMesh(ray, meshInstance);
                if (meshHit.hit && meshHit.distance < hit.distance) {
                    hit = meshHit;
                }
            }
        }
        // Inner
        else {
            stack[stackIndex] = node.leftOrOffset; // Push left
            stack[stackIndex + 1] = node.leftOrOffset + 1; // Push right
            stackIndex += 2;
        }
    }

    return hit;
}

fn hitMesh(worldRay: Ray, meshInstance: MeshInstance) -> Hit {
    var localRay: Ray;
    localRay.origin = transformPoint(worldRay.origin, meshInstance.invTransform);
    localRay.direction = transformNormal(worldRay.direction, meshInstance.invTransform);

    let directionScale = length(localRay.direction);
    localRay.direction /= directionScale;

    var hit: Hit;
    hit.distance = maxDistance;

    let meshBase = meshBases[meshInstance.meshBaseIndex];

    let t = intersectBounds(localRay, meshBase.min, meshBase.max);
    if t < minDistance || t > hit.distance {
        return hit;
    }

    var stack = array<u32, meshNodeStackSize>();
    var stackIndex = 0;

    stack[stackIndex] = 0;
    stackIndex++;

    while (stackIndex > 0) {
        let nodeIndex = stack[stackIndex - 1];
        stackIndex--;
        let node = meshNodes[meshBase.nodeStart + nodeIndex];

        let boundsDist = intersectBounds(localRay, node.min, node.max);
        if (boundsDist < minDistance || boundsDist > hit.distance) {
            continue;
        }

        // Leaf
        if node.countOrLeaf != 0 {
            // Try intersect each triangle
            for (var i = 0u; i < node.countOrLeaf; i++) {
                var tri = triangles[meshBase.triangleStart + node.leftOrOffset + i];
                let triHit = intersectTriangle(localRay, tri);
                if (triHit.hit && triHit.t < hit.distance) {
                    hit.hit = true;
                    hit.distance = triHit.t;
                    hit.normal = tri.normal;
                    hit.uv = (1 - triHit.u - triHit.v) * tri.auv + triHit.u * tri.buv + triHit.v * tri.cuv;
                }
            }
        }
        // Inner
        else {
            stack[stackIndex] = node.leftOrOffset; // Push left
            stack[stackIndex + 1] = node.leftOrOffset + 1; // Push right
            stackIndex += 2;
        }
    }

    if !hit.hit {
        return hit;
    }

    hit.distance /= directionScale;
    hit.position = worldRay.origin + hit.distance * worldRay.direction;
    hit.normal = normalize(transformNormal(hit.normal, meshInstance.invTransposeTransform));
    hit.materialIndex = meshInstance.materialIndex;
    return hit;
}

fn intersectBounds(ray: Ray, boundsMin: vec3f, boundsMax: vec3f) -> f32 {
    let invDir = 1.0 / ray.direction;
    let t0s = (boundsMin - ray.origin) * invDir;
    let t1s = (boundsMax - ray.origin) * invDir;

    let tsmaller = min(t0s, t1s);
    let tbigger = max(t0s, t1s);

    let tmin = max(max(tsmaller.x, tsmaller.y), max(tsmaller.z, minDistance));
    let tmax = min(min(tbigger.x, tbigger.y), min(tbigger.z, maxDistance));

    if (tmax < tmin) {
        return -1;
    }

    return tmin;
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

fn randomUnit(rng: ptr<function, u32>) -> vec3f {
    let z = nextRandom(rng) * 2.0 - 1.0;
    let r = sqrt(1.0 - z * z);
    let phi = nextRandom(rng) * 2.0 * pi;
    let x = r * cos(phi);
    let y = r * sin(phi);
    return vec3f(x, y, z);
}

fn cosineSampleHemisphere(n: vec3f, rng: ptr<function, u32>) -> vec3f {
    let r1 = 2.0 * pi * nextRandom(rng);
    let r2 = nextRandom(rng);
    let r2s = sqrt(r2);

    let s = sin(r1);
    let c = cos(r1);
    let x = c * r2s;
    let y = s * r2s;
    let z = sqrt(1.0 - r2);

    // Duff's method
    let sgn = sign(n.z + 1e-8);
    let a = -1.0 / (sgn + n.z);
    let b = n.x * n.y * a;

    let tangent = vec3f(1.0 + sgn * n.x * n.x * a, sgn * b, -sgn * n.x);
    let bitangent = vec3f(b, sgn + n.y * n.y * a, -sgn * n.y);
    return normalize(x * tangent + y * bitangent + z * n);
}

// --------------------------------------- Trace ---------------------------------------

fn traceScene(cameraRay: Ray, sampleInfo: ptr<function, SampleInfo>, rng: ptr<function, u32>) -> vec3f {
    var radiance = vec3f(0);
    var throughput = vec3f(1);
    var ray = cameraRay;

    var reflected = true;
    let useNee = state.nee > 0;

    for (var bounceCount = 0u; bounceCount < state.maxRayBounces; bounceCount += 1) {
        let hit = hitScene(ray);

        // Hit sky
        if !hit.hit {
            let t = 0.5 * (ray.direction.y + 1.0);
            let skyColor = (1.0 - t) * vec3f(1.0) + t * vec3f(0.5, 0.7, 1.0);
            radiance += throughput * skyColor * skyLight;
            break;
        }

        // Russian roulette termination
        if bounceCount >= state.maxRayBounces / 2 {
            let p = clamp(max(throughput.x, max(throughput.y, throughput.z)), 0.05, 0.95);
            if nextRandom(rng) >= p {
                break;
            }
            throughput /= p;
        }

        let material = materials[hit.materialIndex];
        let materialColor = sampleMaterialColor(material, hit.uv);
        let surfaceNormal = faceForward(hit.normal, ray.direction, hit.normal);
        let surfacePosition = hit.position - surfaceNormal * minDistance;

        // Write sample info
        if bounceCount == 0 {
            sampleInfo.normal = surfaceNormal;
            sampleInfo.depth = hit.distance;
            sampleInfo.color = materialColor;
            sampleInfo.roughness = material.roughness;
        }
        if bounceCount == 1 {
            sampleInfo.reflectHitDistance = hit.distance;
            sampleInfo.reflectHitPosition = hit.position;
            sampleInfo.reflectHitAlbedo = materialColor;
        }

        // Hit light directly
        if material.light > 0 {
            if !useNee || reflected {
                radiance += throughput * materialColor * material.light;
            }
            break;
        }

        reflected = nextRandom(rng) > material.roughness;

        // NEE
        if useNee && !reflected {
            let nee = sampleLightContribution(hit, surfacePosition, materialColor, rng);
            radiance += throughput * nee;
        }

        // Next ray direction
        if reflected {
            ray.direction = normalize(reflect(ray.direction, surfaceNormal) + material.roughness * randomUnit(rng));
        } else {
            ray.direction = cosineSampleHemisphere(surfaceNormal, rng);
        }

        // Next ray position
        ray.origin = surfacePosition;

        throughput *= materialColor;
    }

    return radiance;
}

fn sampleLightContribution(surfaceHit: Hit, surfacePosition: vec3f, surfaceColor: vec3f, rng: ptr<function, u32>) -> vec3f {
    if state.lightMeshInstanceCount <= 0 {
        return vec3f(0);
    }

    // Random light mesh instance
    let randomLightIndex = u32(nextRandom(rng) * f32(state.lightMeshInstanceCount));
    let lightMeshInstanceIndex = lightMeshIndices[randomLightIndex];
    let lightMeshInstance = meshInstances[lightMeshInstanceIndex];
    let lightMeshBase = meshBases[lightMeshInstance.meshBaseIndex];

    // Random triangle on light
    let triangleIndex = u32(nextRandom(rng) * f32(lightMeshBase.triangleCount));
    var tri = triangles[lightMeshBase.triangleStart + triangleIndex];
    tri.a = transformPoint(tri.a, lightMeshInstance.transform);
    tri.b = transformPoint(tri.b, lightMeshInstance.transform);
    tri.c = transformPoint(tri.c, lightMeshInstance.transform);
    tri.normal = normalize(transformNormal(tri.normal, lightMeshInstance.invTransposeTransform));

    // Uniform point on triangle
    let r1 = sqrt(nextRandom(rng));
    let r2 = nextRandom(rng);
    let samplePoint = (1 - r1) * tri.a + r1 * (1 - r2) * tri.b + r1 * r2 * tri.c;

    // Shoot light ray
    let toLight = samplePoint - surfaceHit.position;
    let lightDist = length(toLight);
    let lightDir = toLight / lightDist;
    let lightRay = Ray(surfacePosition, lightDir);
    let lightHit = hitScene(lightRay);

    // Hit something else before the light
    if (lightHit.hit && lightHit.distance < lightDist - minDistance) {
        return vec3f(0);
    }

    // Cosines
    let cosSurface = max(0.0, dot(surfaceHit.normal, lightDir));
    let cosLight = max(0.0, dot(tri.normal, -lightDir));
    if (cosSurface <= 0.0 || cosLight <= 0.0) {
        return vec3f(0);
    }

    let area = length(cross(tri.b - tri.a, tri.c - tri.a)) * 0.5;
    let pdf = 1.0 / (area * f32(state.lightMeshInstanceCount) * f32(lightMeshBase.triangleCount));
    let geom = (cosSurface * cosLight) / (lightDist * lightDist);
    let lightMaterial = materials[lightMeshInstance.materialIndex];
    let lightMaterialColor = sampleMaterialColor(lightMaterial, lightHit.uv);
    let emission = lightMaterialColor * lightMaterial.light;
    let contribution = emission * geom / pdf;

    return contribution * (surfaceColor / pi);
}

// --------------------------------------- State ---------------------------------------

struct State {
    frameIndex: u32,
    // Settings
    raysPerPixel: u32,
    maxRayBounces: u32,
    nee: u32,
    // Scene state
    meshInstanceCount: u32,
    lightMeshInstanceCount: u32,
}

// --------------------------------------- Mesh ----------------------------------------

struct MeshBase {
    min: vec3f,
    nodeStart: u32,
    max: vec3f,
    triangleStart: u32,
    triangleCount: u32,
}

struct MeshInstance {
    transform: mat4x4f,
    invTransform: mat4x4f,
    invTransposeTransform: mat4x4f,
    meshBaseIndex: u32,
    materialIndex: u32,
}

// --------------------------------------- Node ----------------------------------------

struct Node {
    min: vec3f,
    leftOrOffset: u32,
    max: vec3f,
    countOrLeaf: u32,
}

// ------------------------------------- Material --------------------------------------

struct Material {
    color: u32,
    roughness: f32,
    light: f32,
    textureId: u32,
}

fn sampleMaterialColor(material: Material, uv: vec2f) -> vec3f {
    let baseColor = unpack4x8unorm(material.color).rgb;
    let textureColor = sampleTextureAtlas(material.textureId, uv);
    return baseColor * textureColor;
}

// --------------------------------------- Atlas ---------------------------------------

fn sampleTextureAtlas(id: u32, uv: vec2f) -> vec3f {
    if id >= arrayLength(&textureAtlasMinMax) {
        return vec3f(1);
    }

    let minMaxCoord = textureAtlasMinMax[id];
    let atlasSize = textureDimensions(textureAtlas);
    let atlasUV = mix(minMaxCoord.xy, minMaxCoord.zw, uv) / vec2f(atlasSize);
    return textureSampleLevel(textureAtlas, textureAtlasSampler, atlasUV, 0).rgb;
}

// ----------------------------------- Accumulation ------------------------------------

fn accumulateColor(coord: vec2u, color: vec3f) {
    let index = coord.y * textureDimensions(outputTexture).x + coord.x;
    //let previous = textureLoad(accumulationTexture, coord).rgb;
    let previous = accumulationBuffer[index];
    let weight = 1.0 / (f32(state.frameIndex) + 1.0);
    let accumulated = mix(previous, color, weight);
    //textureStore(accumulationTexture, coord, vec4f(accumulated, 1));
    accumulationBuffer[index] = accumulated;
    textureStore(outputTexture, coord, vec4f(accumulated, 1));
}

// ------------------------------------ SampleInfo -------------------------------------

struct SampleInfo {
    normal: vec3f,
    depth: f32,
    color: vec3f,
    roughness: f32,
    reflectHitPosition: vec3f,
    reflectHitDistance: f32,
    reflectHitAlbedo: vec3f,
}

// --------------------------------------- Main ----------------------------------------

const minDistance: f32 = 0.001;
const maxDistance: f32 = 10000;
const skyLight: f32 = 0.5;
const pi: f32 = 3.14159265;

override workgroupSize: u32 = 8;

@compute @workgroup_size(workgroupSize, workgroupSize, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2u(gid.xy);
    let size = textureDimensions(outputTexture);
    if any(coord >= size) {
        return;
    }

    var rng = initRandom(coord.x, coord.y, state.frameIndex);

    var colorSum: vec3f;
    var sampleInfo: SampleInfo;
    for (var i = 0u; i < state.raysPerPixel; i++) {
        let offset = vec2f(nextRandom(&rng) - 0.5, nextRandom(&rng) - 0.5);
        let ray = createCameraRay(coord, offset);
        let colorSample = traceScene(ray, &sampleInfo, &rng);
        colorSum += colorSample;
    }

    let colorAvg = colorSum / f32(state.raysPerPixel);
    accumulateColor(coord, colorAvg);

    let sampleInfoIndex = coord.y * size.x + coord.x;
    sampleInfos[sampleInfoIndex] = sampleInfo;
}