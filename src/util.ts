import {assert} from "@domgell/ts-util";
import {Scene} from "./scene.ts";
import {mat4, Matrix4, quat, vec2, vec3, vec4, Vector2, Vector3} from "dom-game-math";
import * as Game from "@domgell/game-util";
import MeshInstance = Scene.MeshInstance;
import {DebugRenderer} from "@domgell/webgpu-samples";
import {BVH, transformBounds} from "./bvh.ts";
import {buildBuffer} from "@domgell/webgpu-builder";
import * as GPU from "../../webgpu-util";

// --------------------------------------- Scene ---------------------------------------

/**
 * Build the demo scene
 */
export async function createRoomScene() {
    const scene = Scene.create();
    const builder = Scene.builder(scene);
    const cubeMeshBase = builder.meshBase(Game.cubeGeometry);
    const quadMeshBase = builder.meshBase(Game.quadGeometry);

    const sphereMesh = await Game.importMesh("res/sphere.glb");
    const sphereMeshBase = builder.meshBase(sphereMesh.geometry);

    const wallSize = 4;

    // Floor
    builder.meshInstance({
        transform: mat4.compose({
            rotation: quat.fromEuler({pitch: 90}),
            scale: vec3.new(wallSize),
        }),
        materialIndex: builder.material({textureId: 0}),
        meshBaseIndex: quadMeshBase,
    });
    // Ceiling
    builder.meshInstance({
        transform: mat4.compose({
            rotation: quat.fromEuler({pitch: -90}),
            scale: vec3.new(wallSize),
            translation: vec3.new(0, wallSize * 2, 0),
        }),
        materialIndex: builder.material({color: vec3.new(0.95)}),
        meshBaseIndex: quadMeshBase,
    });
    // Light
    builder.meshInstance({
        transform: mat4.compose({
            rotation: quat.fromEuler({pitch: -90}),
            scale: vec3.new(wallSize / 3),
            translation: vec3.new(0, wallSize * 2 - 0.001, 0),
        }),
        materialIndex: builder.material({light: 10}),
        meshBaseIndex: quadMeshBase,
    });
    // Back wall
    builder.meshInstance({
        transform: mat4.compose({
            scale: vec3.new(wallSize),
            translation: vec3.new(0, wallSize, wallSize),
        }),
        materialIndex: builder.material({color: vec3.new(0.95)}),
        meshBaseIndex: quadMeshBase,
    });
    // Front wall
    builder.meshInstance({
        transform: mat4.compose({
            scale: vec3.new(wallSize),
            translation: vec3.new(0, wallSize, -wallSize),
            rotation: quat.fromEuler({yaw: 180}),
        }),
        materialIndex: builder.material({color: vec3.new(0.95)}),
        meshBaseIndex: quadMeshBase,
    });
    // Left wall
    builder.meshInstance({
        transform: mat4.compose({
            scale: vec3.new(wallSize),
            translation: vec3.new(wallSize, wallSize, 0),
            rotation: quat.fromEuler({yaw: 90}),
        }),
        materialIndex: builder.material({color: vec3.new(0.95, 0.15, 0.15)}),
        meshBaseIndex: quadMeshBase,
    });
    // Right wall
    builder.meshInstance({
        transform: mat4.compose({
            scale: vec3.new(wallSize),
            translation: vec3.new(-wallSize, wallSize, 0),
            rotation: quat.fromEuler({yaw: -90}),
        }),
        materialIndex: builder.material({color: vec3.new(0.15, 0.95, 0.15)}),
        meshBaseIndex: quadMeshBase,
    });
    // Cube
    builder.meshInstance({
        transform: mat4.compose({
            translation: vec3.new(-wallSize * 0.4, 1, -1),
            rotation: quat.fromEuler({yaw: -15}),
            scale: vec3.new(1, 1, 1)
        }),
        materialIndex: builder.material({color: vec3.new(0.0, 0.95, 0.65)}),
        meshBaseIndex: cubeMeshBase,
    });
    // Cube
    builder.meshInstance({
        transform: mat4.compose({
            translation: vec3.new(wallSize * 0.4, 2, wallSize * 0.4),
            rotation: quat.fromEuler({yaw: 15}),
            scale: vec3.new(1, 2, 1)
        }),
        materialIndex: builder.material({color: vec3.new(1, 0, 1)}),
        meshBaseIndex: cubeMeshBase,
    });
    // Sphere
    builder.meshInstance({
        transform: mat4.compose({
            translation: vec3.new(-wallSize * 0.4, 3.2, -1),
            rotation: quat.fromEuler({roll: 90, yaw: 90}),
            scale: vec3.new(1.2)
        }),
        materialIndex: builder.material({color: vec3.new(1), roughness: 0.3}),
        meshBaseIndex: sphereMeshBase,
    });

    scene.meshBases.forEach(m => BVH.buildMeshNodes(m, scene));
    BVH.buildSceneNodes(scene);

    return scene;
}

// ------------------------------------- Triangle --------------------------------------

export interface Triangle {
    a: Vector3,
    b: Vector3,
    c: Vector3,
    normal: Vector3,
    auv: Vector2,
    buv: Vector2,
    cuv: Vector2
}

function intersectRayTriangle(ray: Ray, triangle: { a: Vector3, b: Vector3, c: Vector3 }) {
    const edge1 = vec3.sub(triangle.b, triangle.a);
    const edge2 = vec3.sub(triangle.c, triangle.a);
    const h = vec3.cross(ray.direction, edge2);
    const a = vec3.dot(edge1, h);

    if (a < minDistance) return -1;

    const f = 1 / a;
    const s = vec3.sub(ray.origin, triangle.a);
    const u = f * vec3.dot(s, h);

    if (u < 0 || u > 1) return -1;

    const q = vec3.cross(s, edge1);
    const v = f * vec3.dot(ray.direction, q);

    if (v < 0 || u + v > 1) return -1;

    const t = f * vec3.dot(edge2, q);

    if (t < minDistance || t > maxDistance) return -1;

    return t;
}

export function getTriangles(geom: Game.Geometry) {
    assert(geom.indices !== undefined && geom.vertices.uv !== undefined);

    const triangles: Triangle[] = [];

    for (let i = 0; i < geom.indices.length; i += 3) {
        const ia = geom.indices[i];
        const ib = geom.indices[i + 1];
        const ic = geom.indices[i + 2];

        const a = vec3.new(
            geom.vertices.positions[ia * 3],
            geom.vertices.positions[ia * 3 + 1],
            geom.vertices.positions[ia * 3 + 2],
        );

        const b = vec3.new(
            geom.vertices.positions[ib * 3],
            geom.vertices.positions[ib * 3 + 1],
            geom.vertices.positions[ib * 3 + 2],
        );

        const c = vec3.new(
            geom.vertices.positions[ic * 3],
            geom.vertices.positions[ic * 3 + 1],
            geom.vertices.positions[ic * 3 + 2],
        );

        const ab = vec3.sub(b, a);
        const ac = vec3.sub(c, a);
        const normal = vec3.cross(ab, ac);
        vec3.normalize(normal, normal);

        const auv = vec2.new(
            geom.vertices.uv[ia * 2],
            geom.vertices.uv[ia * 2 + 1],
        );

        const buv = vec2.new(
            geom.vertices.uv[ib * 2],
            geom.vertices.uv[ib * 2 + 1],
        );

        const cuv = vec2.new(
            geom.vertices.uv[ic * 2],
            geom.vertices.uv[ic * 2 + 1],
        );

        triangles.push({a, b, c, normal, auv, buv, cuv});
    }

    return triangles;
}

// ----------------------------------- Mouse Picking -----------------------------------

interface Ray {
    origin: Vector3,
    direction: Vector3,
}

export function createMouseRay(
    invViewProjection: Matrix4,
    mousePosition: Vector2,
    width: number,
    height: number
) {
    const ndcX = (2 * mousePosition.x) / width - 1;
    const ndcY = 1 - (2 * mousePosition.y) / height;

    const near = vec4.transform(vec4.new(ndcX, ndcY, 0, 1), invViewProjection);
    vec4.div(near, near.w, near);

    const far = vec4.transform(vec4.new(ndcX, ndcY, 1, 1), invViewProjection);
    vec4.div(far, far.w, far);

    const direction = vec3.sub(far, near);
    vec3.normalize(direction, direction);

    return {origin: near, direction};
}

const minDistance = 0.001;
const maxDistance = 10000;

export function pickMeshInstance(ray: Ray, scene: Scene) {
    let closest = maxDistance;
    let pick: MeshInstance | undefined;

    for (let i = 0; i < scene.meshInstances.length; i++) {
        const meshInstance = scene.meshInstances[i];
        const meshBase = scene.meshBases[meshInstance.meshBaseIndex];

        for (let j = 0; j < meshBase.triangleCount; j++) {
            const triangle = scene.triangles[meshBase.triangleStart + j];
            const a = vec3.transform(triangle.a, meshInstance.transform);
            const b = vec3.transform(triangle.b, meshInstance.transform);
            const c = vec3.transform(triangle.c, meshInstance.transform);
            const t = intersectRayTriangle(ray, {a, b, c});

            if (t > minDistance && t < closest) {
                closest = t;
                pick = meshInstance;
            }
        }
    }

    return pick;
}

// ----------------------------------- Debug Drawing -----------------------------------

export function drawMeshInstanceOutline(
    meshInstance: Scene.MeshInstance,
    scene: Scene,
    debugRenderer: DebugRenderer
) {
    if (meshInstance === undefined) return;
    const meshBase = scene.meshBases[meshInstance.meshBaseIndex];

    for (let i = 0; i < meshBase.triangleCount; i++) {
        const triangle = scene.triangles[meshBase.triangleStart + i];
        const a = vec3.transform(triangle.a, meshInstance.transform);
        const b = vec3.transform(triangle.b, meshInstance.transform);
        const c = vec3.transform(triangle.c, meshInstance.transform);
        debugRenderer.line(a, b, vec3.new(1, 0, 0));
        debugRenderer.line(b, c, vec3.new(1, 0, 0));
        debugRenderer.line(c, a, vec3.new(1, 0, 0));
    }
}

export function drawSceneNodes(scene: Scene, debugRenderer: DebugRenderer) {
    for (let sceneNode of scene.sceneNodes) {
        debugRenderer.aabb(sceneNode.bounds.min, sceneNode.bounds.max);
    }
}

export function drawMeshNodes(scene: Scene, debugRenderer: DebugRenderer) {
    for (let meshInstance of scene.meshInstances) {
        const meshBase = scene.meshBases[meshInstance.meshBaseIndex];
        for (let i = 0; i < meshBase.nodeCount; i++) {
            const node = scene.meshNodes[meshBase.nodeStart + i];
            const bounds = transformBounds(node.bounds, meshInstance.transform);
            debugRenderer.aabb(bounds.min, bounds.max, vec3.new(0.0, 0.65, 1));
        }
        const bounds = transformBounds(meshBase.bounds, meshInstance.transform);
        debugRenderer.aabb(bounds.min, bounds.max, vec3.new(0.65, 0, 1));
    }
}

// ------------------------------------ Performance ------------------------------------

export function rollingAverage(maxSamples = 32) {
    const samples = new Float64Array(maxSamples);
    let head = 0;
    let size = 0;
    let sum = 0;

    return {
        get(): number {
            return size === 0 ? 0 : sum / size;
        },
        add(sample: number) {
            if (size < maxSamples) {
                samples[head] = sample;
                sum += sample;
                size++;
                head = (head + 1) % maxSamples;
                return;
            }

            // Overwrite oldest
            const old = samples[head];
            sum -= old;

            samples[head] = sample;
            sum += sample;

            head = (head + 1) % maxSamples;
        },
        clear() {
            head = 0;
            size = 0;
            sum = 0;
        }
    };
}

export function createTimestampState(device: GPUDevice, timestampCount: number) {
    const timestampQuerySet = device.createQuerySet({
        type: "timestamp",
        count: timestampCount * 2,
    });

    const timestampQueryBuffer = buildBuffer(device)
        .size(timestampCount * 4 * GPU.Size.u32)
        .usage("query-resolve", "copy-src")
        .build("TimestampQueryBuffer");

    const timestampResultBuffer = buildBuffer(device)
        .size(timestampCount * 4 * GPU.Size.u32)
        .usage("copy-dst", "map-read")
        .build("TimestampResultBuffer");

    return {
        resolve(cmd: GPUCommandEncoder) {
            cmd.resolveQuerySet(timestampQuerySet, 0, timestampCount * 2, timestampQueryBuffer, 0);
            if (timestampResultBuffer.mapState === "unmapped") {
                cmd.copyBufferToBuffer(timestampQueryBuffer, timestampResultBuffer);
            }
        },
        update(...timers: ReturnType<typeof rollingAverage>[]) {
            assert(timers.length === timestampCount);

            // Collect GPU performance timings
            if (timestampResultBuffer.mapState === "unmapped") {
                timestampResultBuffer.mapAsync(GPUMapMode.READ).then(() => {
                    const timestamps = new BigInt64Array(timestampResultBuffer.getMappedRange());

                    for (let i = 0; i < timers.length; i++) {
                        const timer = timers[i];
                        const gpuTimeNs = Number(timestamps[i * 2 + 1] - timestamps[i * 2]);
                        timer.add(gpuTimeNs / 1e6);
                    }

                    timestampResultBuffer.unmap();
                });
            }
        },
        getTimestampWrite(index: number) {
            return {
                querySet: timestampQuerySet,
                beginningOfPassWriteIndex: index * 2,
                endOfPassWriteIndex: index * 2 + 1,
            }
        }
    }
}