import {mat4, Matrix4, vec3, Vector3} from "dom-game-math";
import * as Game from "@domgell/game-util";
import {getTriangles, Triangle} from "./util.ts";
import {BVH} from "./bvh.ts";

export interface Scene {
    meshInstances: Scene.MeshInstance[],
    meshBases: Scene.MeshBase[],
    materials: Scene.Material[],
    triangles: Triangle[],
    sceneNodes: BVH.Node[],
    meshNodes: BVH.Node[],
}

export namespace Scene {
    export interface Material {
        color: Vector3,
        roughness: number,
        light: number,
        textureId: number,
    }

    export interface MeshInstance {
        transform: Matrix4,
        invTransform: Matrix4,
        invTransposeTransform: Matrix4,
        meshBaseIndex: number,
        materialIndex: number,
    }

    export interface MeshBase {
        bounds: BVH.Bounds,
        triangleStart: number,
        triangleCount: number,
        nodeStart: number,
        nodeCount: number,
    }

    export type Builder = Scene & {
        meshInstance(instance: Partial<MeshInstance> & { materialIndex: number, meshBaseIndex: number }): number,
        meshBase(geom: Game.Geometry): number,
        material(material: Partial<Material>): number,
    }

    export function create(scene: Partial<Scene> = {}): Scene {
        return {
            meshInstances: scene.meshInstances ?? [],
            meshBases: scene.meshBases ?? [],
            materials: scene.materials ?? [],
            triangles: scene.triangles ?? [],
            sceneNodes: scene.sceneNodes ?? [],
            meshNodes: scene.meshNodes ?? [],
        };
    }

    export function builder(scene: Scene) {
        return {
            meshInstance(instance: Partial<MeshInstance> & {
                materialIndex: number,
                meshBaseIndex: number
            }): number {
                instance.transform ??= mat4.idt;
                instance.invTransform = mat4.invert(instance.transform);
                instance.invTransposeTransform = mat4.transpose(instance.invTransform);
                return scene.meshInstances.push(instance as MeshInstance) - 1;
            },
            meshBase(geom: Game.Geometry): number {
                const triangles = getTriangles(geom);
                const triangleStart = scene.triangles.length;
                const triangleCount = triangles.length;
                scene.triangles.push(...triangles);

                const meshBase: MeshBase = {
                    triangleStart,
                    triangleCount,
                    nodeCount: 0,
                    nodeStart: 0,
                    bounds: {} as BVH.Bounds,
                };

                return scene.meshBases.push(meshBase) - 1;
            },
            material(material: Partial<Material>): number {
                material.color ??= vec3.one;
                material.roughness ??= 1;
                material.light ??= 0;
                material.textureId ??= -1;
                return scene.materials.push(material as Material) - 1;
            },
        };
    }
}