import * as GPU from "@domgell/webgpu-util";
import {buildBindGroup, buildBindGroupLayout, buildBuffer, buildTexture} from "@domgell/webgpu-builder";
import {TextureAtlas} from "@domgell/webgpu-samples";
import * as Game from "@domgell/game-util";
import {Raytracer} from "./raytracer.ts";

export interface Denoiser {
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    outputTexture: GPUTexture,
    inputTexture: GPUTexture,
    sampleInfoBuffer: GPUBuffer,
    stateBuffer: GPUBuffer,
    workgroupSize: number,
    // Bind groups
    stateBindGroup: GPUBindGroup,
    pingTextureBindGroup: GPUBindGroup,
    pongTextureBindGroup: GPUBindGroup,
    sampleInfoBindGroup: GPUBindGroup,
}

const denoiserSteps = 5;
const minDynamicBufferOffset = 256;

export function createDenoiser({device, sampleInfoBuffer, outputTexture, shader, workgroupSize = 16}: {
    device: GPUDevice,
    sampleInfoBuffer: GPUBuffer,
    outputTexture: GPUTexture,
    shader: string,
    workgroupSize?: number,
}): Denoiser {
    // ------------------------------------ Layout -------------------------------------

    // State bind group layout needs to be provided explicitly,
    // because it uses dynamic buffer offset
    const stateBindGroupLayout = buildBindGroupLayout(device)
        .uniform({hasDynamicOffset: true})
        .uniform()
        .build("Denoiser.StateBindGroupLayout");

    const bindGroupLayouts = GPU.generateBindGroupLayoutEntries(shader)
        .map(entries => device.createBindGroupLayout({entries}));
    bindGroupLayouts[0] = stateBindGroupLayout;

    const layout = device.createPipelineLayout({
        bindGroupLayouts,
        label: "Denoiser.PipelineLayout"
    });

    // --------------------------------- Pipeline ----------------------------------

    const pipeline = GPU.createComputePipeline(device, {
        shader,
        layout,
        constants: {workgroupSize},
        label: "Denoiser.Pipeline",
    });

    const structByteSizes = GPU.shaderStructByteSizes(shader);

    // ------------------------------- State Bind Group --------------------------------

    const stateBuffer = buildBuffer(device)
        .size(structByteSizes.State)
        .usage("uniform", "copy-dst")
        .build("Denoiser.StateBuffer");

    const stepIndexBuffer = buildBuffer(device)
        .size(denoiserSteps * minDynamicBufferOffset)
        .usage("uniform")
        .mapped()
        .build("Denoiser.StepBuffer");

    // Step value update needs to happen *during* compute pass encoding.
    // Ideally immediates/pushconstants would be used for this when they are fully supported in WebGPU.
    // Instead, each step index is stored and dynamic buffer offsets are used when encoding.
    const range = new Uint32Array(stepIndexBuffer.getMappedRange());
    for (let i = 0; i < denoiserSteps; i++) range[i * (minDynamicBufferOffset / 4)] = i;
    stepIndexBuffer.unmap();

    const stateBindGroup = buildBindGroup(device)
        .entries({buffer: stepIndexBuffer, size: 4}, stateBuffer)
        .layout(stateBindGroupLayout)
        .build("Denoiser.StateBindGroup");

    // ---------------------------------------------------------------------------------

    const inputTexture = buildTexture(device)
        .size(outputTexture.width, outputTexture.height)
        .format(outputTexture.format)
        .usage("storage-binding", "copy-dst")
        .build("Denoiser.InputTexture");

    const pingTextureBindGroup = buildBindGroup(device)
        .entries(outputTexture, inputTexture)
        .layout(pipeline.getBindGroupLayout(2))
        .build("Denoiser.PingTextureBindGroup");

    const pongTextureBindGroup = buildBindGroup(device)
        .entries(inputTexture, outputTexture)
        .layout(pipeline.getBindGroupLayout(2))
        .build("Denoiser.PongTextureBindGroup");

    const sampleInfoBindGroup = buildBindGroup(device)
        .entries(sampleInfoBuffer)
        .layout(pipeline.getBindGroupLayout(1))
        .build("Denoiser.SampleInfoBindGroup");

    // ---------------------------------------------------------------------------------

    return {
        device,
        pipeline,
        outputTexture,
        inputTexture,
        sampleInfoBuffer,
        stateBuffer,
        workgroupSize,
        stateBindGroup,
        pingTextureBindGroup,
        pongTextureBindGroup,
        sampleInfoBindGroup,
    };
}

export function resizeDenoiser(denoiser: Denoiser, newOutputTexture: GPUTexture, newSampleInfoBuffer: GPUBuffer) {
    denoiser.outputTexture = newOutputTexture;

    denoiser.inputTexture.destroy();
    denoiser.inputTexture = buildTexture(denoiser.device)
        .size(newOutputTexture.width, newOutputTexture.height)
        .format(newOutputTexture.format)
        .usage(denoiser.inputTexture.usage)
        .build(denoiser.inputTexture.label);

    denoiser.sampleInfoBuffer = newSampleInfoBuffer;
    denoiser.sampleInfoBindGroup = buildBindGroup(denoiser.device)
        .entries(denoiser.sampleInfoBuffer)
        .layout(denoiser.pipeline.getBindGroupLayout(1))
        .build(denoiser.sampleInfoBindGroup.label);

    denoiser.pingTextureBindGroup = buildBindGroup(denoiser.device)
        .entries(denoiser.outputTexture, denoiser.inputTexture)
        .layout(denoiser.pipeline.getBindGroupLayout(2))
        .build(denoiser.pingTextureBindGroup.label);

    denoiser.pongTextureBindGroup = buildBindGroup(denoiser.device)
        .entries(denoiser.inputTexture, denoiser.outputTexture)
        .layout(denoiser.pipeline.getBindGroupLayout(2))
        .build(denoiser.pongTextureBindGroup.label);
}

export function runDenoiser(denoiser: Denoiser, pass: GPUComputePassEncoder) {
    pass.pushDebugGroup("Denoiser.ComputePass");

    for (let i = 0; i < denoiserSteps; i++) {
        pass.pushDebugGroup(`Denoiser.Dispatch[${i}]`);
        pass.setPipeline(denoiser.pipeline);

        pass.setBindGroup(0, denoiser.stateBindGroup, [i * minDynamicBufferOffset]);
        pass.setBindGroup(1, denoiser.sampleInfoBindGroup);
        pass.setBindGroup(2, i % 2 === 0 ? denoiser.pingTextureBindGroup : denoiser.pongTextureBindGroup);

        const dispatchX = Math.ceil(denoiser.outputTexture.width / denoiser.workgroupSize);
        const dispatchY = Math.ceil(denoiser.outputTexture.height / denoiser.workgroupSize);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.popDebugGroup();
    }

    pass.popDebugGroup();
}

export function updateDenoiserSettings(denoiser: Denoiser, settings: {
    sigmaColor: number,
    sigmaNormal: number,
    sigmaDepth: number,
    sigmaAlbedo: number,
    sigmaReflectPosition: number,
    sigmaReflectDistance: number,
}) {
    GPU.writeBuffer(denoiser.device, denoiser.stateBuffer, [
        settings.sigmaColor,
        settings.sigmaNormal,
        settings.sigmaDepth,
        settings.sigmaAlbedo,
        settings.sigmaReflectPosition,
        settings.sigmaReflectDistance,
    ]);
}