import * as GPU from "@domgell/webgpu-util";
import {buildBindGroup, buildBuffer, buildTexture} from "@domgell/webgpu-builder";
import {getSamplerFilterNearestClampToEdge, TextureAtlas} from "@domgell/webgpu-samples";
import {ArrayBufferWriter} from "@domgell/webgpu-arraybuffer-writer";
import {packNormalizedRgb1} from "@domgell/webgpu-util";
import {Scene} from "./scene.ts";

export interface Raytracer {
    run(encoder: GPUComputePassEncoder): void,
    updateSettings(state: Raytracer.Settings): void,
    updateScene(scene: Scene): void,
    resetAccumulation(): void,
    outputTexture: GPUTexture,
    sampleInfoBuffer: GPUBuffer,
}

export namespace Raytracer {
    interface CreateArgs {
        device: GPUDevice,
        outputTexture: GPUTexture,
        cameraBuffer: GPUBuffer,
        textureAtlas: TextureAtlas,
        shader: string,
        workgroupSize?: number,
    }

    export interface Settings {
        raysPerPixel: number,
        maxRayBounces: number,
        nee: boolean,
    }

    const maxTriangles = 1024;
    const maxMeshBases = 64;
    const maxMeshInstances = 128;
    const maxMaterials = 128;
    const maxMeshNodes = 1024;
    const maxSceneNodes = 256;

    export function create({
        device,
        outputTexture,
        cameraBuffer,
        textureAtlas,
        shader,
        workgroupSize = 16
    }: CreateArgs): Raytracer {

        // --------------------------------- Pipeline ----------------------------------

        const pl = GPU.createComputePipeline(device, {
            shader,
            constants: {workgroupSize},
            label: "Raytracer.Pipeline",
        });

        const structByteSizes = GPU.shaderStructByteSizes(shader);

        // ------------------------------- Bind Group 0 --------------------------------

        const triangleBuffer = buildBuffer(device)
            .size(structByteSizes.Triangle * maxTriangles)
            .usage("storage", "copy-dst")
            .build("Raytracer.TriangleBuffer");

        const meshBaseBuffer = buildBuffer(device)
            .size(structByteSizes.MeshBase * maxMeshBases)
            .usage("storage", "copy-dst")
            .build("Raytracer.MeshBaseBuffer");

        const meshInstanceBuffer = buildBuffer(device)
            .size(structByteSizes.MeshInstance * maxMeshInstances)
            .usage("storage", "copy-dst")
            .build("Raytracer.MeshInstanceBuffer");

        const materialBuffer = buildBuffer(device)
            .size(structByteSizes.Material * maxMaterials)
            .usage("storage", "copy-dst")
            .build("Raytracer.MaterialBuffer");

        const lightMeshIndicesBuffer = buildBuffer(device)
            .size(GPU.Size.u32 * maxMeshInstances)
            .usage("storage", "copy-dst")
            .build("Raytracer.LightMeshIndicesBuffer");

        const meshNodesBuffer = buildBuffer(device)
            .size(structByteSizes.Node * maxMeshNodes)
            .usage("storage", "copy-dst")
            .build("Raytracer.MeshNodesBuffer");

        const sceneNodesBuffer = buildBuffer(device)
            .size(structByteSizes.Node * maxSceneNodes)
            .usage("storage", "copy-dst")
            .build("Raytracer.SceneNodesBuffer");

        const bg0 = buildBindGroup(device)
            .layout(pl.getBindGroupLayout(0))
            .entries(triangleBuffer, meshBaseBuffer, meshInstanceBuffer, materialBuffer, lightMeshIndicesBuffer, meshNodesBuffer, sceneNodesBuffer)
            .build("Raytracer.BindGroup0");

        // ------------------------------- Bind Group 1 --------------------------------

        const accumulationTexture = buildTexture(device)
            .size(outputTexture.width, outputTexture.height)
            .format("rgba32float")
            .usage("storage-binding")
            .build("Raytracer.AccumulationTexture");

        const sampleInfoBuffer = buildBuffer(device)
            .size(structByteSizes.SampleInfo * outputTexture.width * outputTexture.height)
            .usage("storage")
            .build("Raytracer.SampleInfoBuffer");

        const bg1 = buildBindGroup(device)
            .layout(pl.getBindGroupLayout(1))
            .entries(outputTexture, accumulationTexture, sampleInfoBuffer)
            .build("Raytracer.BindGroup2");

        // ------------------------------- Bind Group 2 --------------------------------

        const stateBuffer = buildBuffer(device)
            .size(structByteSizes.State)
            .usage("uniform", "copy-dst")
            .build("Raytracer.StateBuffer");

        const bg2 = buildBindGroup(device)
            .layout(pl.getBindGroupLayout(2))
            .entries(stateBuffer, cameraBuffer)
            .build("Raytracer.BindGroup2");

        // ------------------------------- Bind Group 3 --------------------------------

        const textureAtlasSampler = getSamplerFilterNearestClampToEdge(device);

        const bg3 = buildBindGroup(device)
            .layout(pl.getBindGroupLayout(3))
            .entries(textureAtlas.texture, textureAtlasSampler, textureAtlas.minMaxBuffer)
            .build("Raytracer.BindGroup3");

        // -----------------------------------------------------------------------------

        const dispatchX = Math.ceil(outputTexture.width / workgroupSize);
        const dispatchY = Math.ceil(outputTexture.height / workgroupSize);
        const bufferWriter = ArrayBufferWriter(Math.max(
            meshInstanceBuffer.size, materialBuffer.size, triangleBuffer.size, meshNodesBuffer.size, sceneNodesBuffer.size
        ));

        let frameIndex = 0;

        const raytracer: Raytracer = {
            outputTexture,
            sampleInfoBuffer,
            run(encoder) {
                GPU.writeBuffer(device, stateBuffer, new Uint32Array([frameIndex++]));

                encoder.pushDebugGroup("Raytracer.ComputePass");
                encoder.setPipeline(pl);
                encoder.setBindGroup(0, bg0);
                encoder.setBindGroup(1, bg1);
                encoder.setBindGroup(2, bg2);
                encoder.setBindGroup(3, bg3);
                encoder.dispatchWorkgroups(dispatchX, dispatchY);
                encoder.popDebugGroup();
            },
            resetAccumulation() {
                frameIndex = 0;
            },
            updateSettings(state) {
                this.resetAccumulation();

                GPU.writeBuffer(device, stateBuffer, new Uint32Array([
                    state.raysPerPixel,
                    state.maxRayBounces,
                    state.nee ? 1 : 0,
                ]), {writeOffset: GPU.Size.u32});
            },
            updateScene(scene) {
                this.resetAccumulation();

                bufferWriter.reset();
                for (const triangle of scene.triangles) {
                    bufferWriter
                        .vec3f(triangle.a)
                        .vec3f(triangle.b)
                        .vec3f(triangle.c)
                        .vec3f(triangle.normal)
                        .vec2f(triangle.auv)
                        .vec2f(triangle.buv)
                        .vec2f(triangle.cuv);
                }
                GPU.writeBuffer(device, triangleBuffer, bufferWriter.arrayBuffer, {writeSize: triangleBuffer.size});

                bufferWriter.reset();
                for (const meshInstance of scene.meshInstances) {
                    bufferWriter
                        .mat4x4f(meshInstance.transform)
                        .mat4x4f(meshInstance.invTransform)
                        .mat4x4f(meshInstance.invTransposeTransform)
                        .u32(meshInstance.meshBaseIndex)
                        .u32(meshInstance.materialIndex);
                }
                GPU.writeBuffer(device, meshInstanceBuffer, bufferWriter.arrayBuffer, {writeSize: meshInstanceBuffer.size});

                bufferWriter.reset();
                for (const meshBase of scene.meshBases) {
                    bufferWriter
                        .vec3f(meshBase.bounds.min)
                        .u32(meshBase.nodeStart)
                        .vec3f(meshBase.bounds.max)
                        .u32(meshBase.triangleStart)
                        .u32(meshBase.triangleCount);
                }
                GPU.writeBuffer(device, meshBaseBuffer, bufferWriter.arrayBuffer, {writeSize: meshBaseBuffer.size});

                bufferWriter.reset();
                for (const material of scene.materials) {
                    bufferWriter
                        .u32(packNormalizedRgb1(material.color))
                        .f32(material.roughness)
                        .f32(material.light)
                        .u32(material.textureId);
                }
                GPU.writeBuffer(device, materialBuffer, bufferWriter.arrayBuffer, {writeSize: materialBuffer.size});

                bufferWriter.reset();
                for (const meshNode of scene.meshNodes) {
                    bufferWriter
                        .vec3f(meshNode.bounds.min)
                        .u32(meshNode.leftOrOffset)
                        .vec3f(meshNode.bounds.max)
                        .u32(meshNode.countOrLeaf);
                }
                GPU.writeBuffer(device, meshNodesBuffer, bufferWriter.arrayBuffer, {writeSize: meshNodesBuffer.size});

                bufferWriter.reset();
                for (const sceneNode of scene.sceneNodes) {
                    bufferWriter
                        .vec3f(sceneNode.bounds.min)
                        .u32(sceneNode.leftOrOffset)
                        .vec3f(sceneNode.bounds.max)
                        .u32(sceneNode.countOrLeaf);
                }
                GPU.writeBuffer(device, sceneNodesBuffer, bufferWriter.arrayBuffer, {writeSize: sceneNodesBuffer.size});

                const lightMeshIndices: number[] = [];
                for (let i = 0; i < scene.meshInstances.length; i++) {
                    const material = scene.materials[scene.meshInstances[i].materialIndex];
                    if (material.light > 0) lightMeshIndices.push(i);
                }
                GPU.writeBuffer(device, lightMeshIndicesBuffer, new Uint32Array(lightMeshIndices));

                GPU.writeBuffer(device, stateBuffer, new Uint32Array([
                    scene.meshInstances.length,
                    lightMeshIndices.length,
                ]), {writeOffset: 4 * GPU.Size.u32});
            },
        };

        raytracer.updateSettings({raysPerPixel: 2, nee: true, maxRayBounces: 4});
        return raytracer;
    }
}