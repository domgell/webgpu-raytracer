import * as GPU from "@domgell/webgpu-util";
import {buildBindGroup, buildBuffer, buildTexture} from "@domgell/webgpu-builder";
import {getSamplerFilterNearestClampToEdge, TextureAtlas} from "@domgell/webgpu-samples";
import {ArrayBufferWriter} from "@domgell/webgpu-arraybuffer-writer";
import {packNormalizedRgb1} from "@domgell/webgpu-util";
import {Scene} from "./scene.ts";

export interface Raytracer {
    device: GPUDevice,
    stateBuffer: GPUBuffer,
    pipeline: GPUComputePipeline,
    screenWidth: number,
    screenHeight: number,
    workgroupSize: number,
    frameIndex: number,
    sampleInfoBuffer: GPUBuffer,
    accumulationBuffer: GPUBuffer,
    sampleInfoByteSize: number,
    // Bind groups
    sceneBindGroup: GPUBindGroup,
    screenBindGroup: GPUBindGroup,
    stateBindGroup: GPUBindGroup,
    textureAtlasBindGroup: GPUBindGroup,
    // Scene
    sceneBufferWriter: ArrayBufferWriter,
    triangleBuffer: GPUBuffer,
    meshBaseBuffer: GPUBuffer,
    meshInstanceBuffer: GPUBuffer,
    materialBuffer: GPUBuffer,
    lightMeshIndicesBuffer: GPUBuffer,
    meshNodesBuffer: GPUBuffer,
    sceneNodesBuffer: GPUBuffer,
}

const maxTriangles = 1024;
const maxMeshBases = 64;
const maxMeshInstances = 128;
const maxMaterials = 128;
const maxMeshNodes = 1024;
const maxSceneNodes = 256;

export function createRaytracer({device, outputTexture, cameraBuffer, textureAtlas, shader, workgroupSize = 16}: {
    device: GPUDevice,
    outputTexture: GPUTexture,
    cameraBuffer: GPUBuffer,
    textureAtlas: TextureAtlas,
    shader: string,
    workgroupSize?: number,
}): Raytracer {
    // --------------------------------- Pipeline ----------------------------------

    const pipeline = GPU.createComputePipeline(device, {
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

    const sceneBindGroup = buildBindGroup(device)
        .layout(pipeline.getBindGroupLayout(0))
        .entries(triangleBuffer, meshBaseBuffer, meshInstanceBuffer, materialBuffer, lightMeshIndicesBuffer, meshNodesBuffer, sceneNodesBuffer)
        .build("Raytracer.BindGroup0");

    // ------------------------------- Bind Group 1 --------------------------------

    const accumulationBuffer = buildBuffer(device)
        .size(GPU.Size.vec4f * outputTexture.width * outputTexture.height)
        .usage("storage")
        .build("Raytracer.AccumulationBuffer");

    const sampleInfoBuffer = buildBuffer(device)
        .size(structByteSizes.SampleInfo * outputTexture.width * outputTexture.height)
        .usage("storage")
        .build("Raytracer.SampleInfoBuffer");

    const screenBindGroup = buildBindGroup(device)
        .layout(pipeline.getBindGroupLayout(1))
        .entries(outputTexture, accumulationBuffer, sampleInfoBuffer)
        .build("Raytracer.BindGroup1");

    // ------------------------------- Bind Group 2 --------------------------------

    const stateBuffer = buildBuffer(device)
        .size(structByteSizes.State)
        .usage("uniform", "copy-dst")
        .build("Raytracer.StateBuffer");

    const stateBindGroup = buildBindGroup(device)
        .layout(pipeline.getBindGroupLayout(2))
        .entries(stateBuffer, cameraBuffer)
        .build("Raytracer.BindGroup2");

    // ------------------------------- Bind Group 3 --------------------------------

    const textureAtlasSampler = getSamplerFilterNearestClampToEdge(device);

    const textureAtlasBindGroup = buildBindGroup(device)
        .layout(pipeline.getBindGroupLayout(3))
        .entries(textureAtlas.texture, textureAtlasSampler, textureAtlas.minMaxBuffer)
        .build("Raytracer.BindGroup3");

    // ---------------------------------------------------------------------------------

    const sceneBufferWriter = ArrayBufferWriter(Math.max(
        meshInstanceBuffer.size, materialBuffer.size, triangleBuffer.size, meshNodesBuffer.size, sceneNodesBuffer.size
    ));

    return {
        device,
        stateBuffer,
        pipeline,
        screenWidth: outputTexture.width,
        screenHeight: outputTexture.height,
        workgroupSize,
        frameIndex: 0,
        sampleInfoBuffer,
        accumulationBuffer,
        sampleInfoByteSize: structByteSizes.SampleInfo,
        sceneBindGroup,
        screenBindGroup,
        stateBindGroup,
        textureAtlasBindGroup,
        sceneBufferWriter,
        triangleBuffer,
        meshBaseBuffer,
        meshInstanceBuffer,
        materialBuffer,
        lightMeshIndicesBuffer,
        meshNodesBuffer,
        sceneNodesBuffer,
    };
}

export function resizeRaytracer(rt: Raytracer, newOutputTexture: GPUTexture) {
    rt.screenWidth = newOutputTexture.width;
    rt.screenHeight = newOutputTexture.height;

    rt.accumulationBuffer.destroy();
    rt.accumulationBuffer = buildBuffer(rt.device)
        .size(GPU.Size.vec4f * rt.screenWidth * rt.screenHeight)
        .usage(rt.accumulationBuffer.usage)
        .build(rt.accumulationBuffer.label);

    rt.sampleInfoBuffer.destroy();
    rt.sampleInfoBuffer = buildBuffer(rt.device)
        .size(rt.sampleInfoByteSize * rt.screenWidth * rt.screenHeight)
        .usage(rt.sampleInfoBuffer.usage)
        .build(rt.sampleInfoBuffer.label);

    rt.screenBindGroup = buildBindGroup(rt.device)
        .layout(rt.pipeline.getBindGroupLayout(1))
        .entries(newOutputTexture, rt.accumulationBuffer, rt.sampleInfoBuffer)
        .build(rt.screenBindGroup.label);
}

export function runRaytracer(rt: Raytracer, pass: GPUComputePassEncoder) {
    GPU.writeBuffer(rt.device, rt.stateBuffer, new Uint32Array([rt.frameIndex++]));

    pass.pushDebugGroup("Raytracer.ComputePass");

    pass.setPipeline(rt.pipeline);
    pass.setBindGroup(0, rt.sceneBindGroup);
    pass.setBindGroup(1, rt.screenBindGroup);
    pass.setBindGroup(2, rt.stateBindGroup);
    pass.setBindGroup(3, rt.textureAtlasBindGroup);

    const dispatchX = Math.ceil(rt.screenWidth / rt.workgroupSize);
    const dispatchY = Math.ceil(rt.screenHeight / rt.workgroupSize);
    pass.dispatchWorkgroups(dispatchX, dispatchY);

    pass.popDebugGroup();
}

export function updateRaytracerSettings(rt: Raytracer, settings: {
    raysPerPixel: number,
    maxRayBounces: number,
    nee: boolean
}) {
    rt.frameIndex = 0; // Reset accumulation

    const data = new Uint32Array([settings.raysPerPixel, settings.maxRayBounces, settings.nee ? 1 : 0]);
    GPU.writeBuffer(rt.device, rt.stateBuffer, data, {writeOffset: GPU.Size.u32});
}

export function updateRaytracerScene(rt: Raytracer, scene: Scene) {
    rt.frameIndex = 0; // Reset accumulation

    for (const triangle of scene.triangles) {
        rt.sceneBufferWriter
            .vec3f(triangle.a)
            .vec3f(triangle.b)
            .vec3f(triangle.c)
            .vec3f(triangle.normal)
            .vec2f(triangle.auv)
            .vec2f(triangle.buv)
            .vec2f(triangle.cuv);
    }
    GPU.writeBuffer(rt.device, rt.triangleBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.triangleBuffer.size});

    rt.sceneBufferWriter.reset();
    for (const meshInstance of scene.meshInstances) {
        rt.sceneBufferWriter
            .mat4x4f(meshInstance.transform)
            .mat4x4f(meshInstance.invTransform)
            .mat4x4f(meshInstance.invTransposeTransform)
            .u32(meshInstance.meshBaseIndex)
            .u32(meshInstance.materialIndex);
    }
    GPU.writeBuffer(rt.device, rt.meshInstanceBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.meshInstanceBuffer.size});

    rt.sceneBufferWriter.reset();
    for (const meshBase of scene.meshBases) {
        rt.sceneBufferWriter
            .vec3f(meshBase.bounds.min)
            .u32(meshBase.nodeStart)
            .vec3f(meshBase.bounds.max)
            .u32(meshBase.triangleStart)
            .u32(meshBase.triangleCount);
    }
    GPU.writeBuffer(rt.device, rt.meshBaseBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.meshBaseBuffer.size});

    rt.sceneBufferWriter.reset();
    for (const material of scene.materials) {
        rt.sceneBufferWriter
            .u32(packNormalizedRgb1(material.color))
            .f32(material.roughness)
            .f32(material.light)
            .u32(material.textureId);
    }
    GPU.writeBuffer(rt.device, rt.materialBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.materialBuffer.size});

    rt.sceneBufferWriter.reset();
    for (const meshNode of scene.meshNodes) {
        rt.sceneBufferWriter
            .vec3f(meshNode.bounds.min)
            .u32(meshNode.leftOrOffset)
            .vec3f(meshNode.bounds.max)
            .u32(meshNode.countOrLeaf);
    }
    GPU.writeBuffer(rt.device, rt.meshNodesBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.meshNodesBuffer.size});

    rt.sceneBufferWriter.reset();
    for (const sceneNode of scene.sceneNodes) {
        rt.sceneBufferWriter
            .vec3f(sceneNode.bounds.min)
            .u32(sceneNode.leftOrOffset)
            .vec3f(sceneNode.bounds.max)
            .u32(sceneNode.countOrLeaf);
    }
    GPU.writeBuffer(rt.device, rt.sceneNodesBuffer, rt.sceneBufferWriter.arrayBuffer, {writeSize: rt.sceneNodesBuffer.size});

    const lightMeshIndices: number[] = [];
    for (let i = 0; i < scene.meshInstances.length; i++) {
        const material = scene.materials[scene.meshInstances[i].materialIndex];
        if (material.light > 0) lightMeshIndices.push(i);
    }
    GPU.writeBuffer(rt.device, rt.lightMeshIndicesBuffer, new Uint32Array(lightMeshIndices));

    GPU.writeBuffer(rt.device, rt.stateBuffer, new Uint32Array([
        scene.meshInstances.length,
        lightMeshIndices.length,
    ]), {writeOffset: 4 * GPU.Size.u32});
}