import * as GPU from "@domgell/webgpu-util";
import {buildBindGroup, buildBuffer, buildTexture} from "@domgell/webgpu-builder";

export interface Denoiser {
    run(encoder: GPUComputePassEncoder): void,
    updateSettings(settings: Denoiser.Settings): void,
}

export namespace Denoiser {
    interface CreateArgs {
        device: GPUDevice,
        outputTexture: GPUTexture,
        sampleInfoBuffer: GPUBuffer,
        shader: string,
        workgroupSize?: number,
    }

    export interface Settings {
        sigmaColor: number,
        sigmaNormal: number,
        sigmaDepth: number,
        sigmaAlbedo: number,
        sigmaReflectPosition: number,
        sigmaReflectDistance: number,
    }

    export function create({
        device,
        outputTexture,
        sampleInfoBuffer,
        shader,
        workgroupSize = 8,
    }: CreateArgs): Denoiser {

        // --------------------------------- Pipeline ----------------------------------

        const pl = GPU.createComputePipeline(device, {
            shader,
            constants: {workgroupSize},
            label: "Denoiser.Pipeline",
        });

        const structByteSizes = GPU.shaderStructByteSizes(shader);

        // ------------------------------- Bind Group 0 --------------------------------

        const stateBuffer = buildBuffer(device)
            .size(structByteSizes.State)
            .usage("uniform", "copy-dst")
            .build("Denoiser.StateBuffer");

        const bg0 = buildBindGroup(device)
            .entries(stateBuffer, sampleInfoBuffer)
            .layout(pl.getBindGroupLayout(0))
            .build("Denoiser.BindGroup0");

        // ------------------------------- Bind Group 1 --------------------------------

        const inputTexture = buildTexture(device)
            .size(outputTexture.width, outputTexture.height)
            .format(outputTexture.format)
            .usage("storage-binding", "copy-dst")
            .build("Denoiser.InputTexture");

        // -------------------------------------------------------------------------------------

        const dispatchX = Math.ceil(outputTexture.width / workgroupSize);
        const dispatchY = Math.ceil(outputTexture.height / workgroupSize);

        const denoiser: Denoiser = {
            run(encoder) {
                encoder.pushDebugGroup("Denoiser.ComputePass");
                for (let i = 0; i < 5; i++) {
                    encoder.pushDebugGroup(`Denoiser.Dispatch[${i}]`);
                    encoder.setPipeline(pl);

                    // Buffer update needs to happen *during* compute pass encoding,
                    // ideally immediates/pushconstants would be used for this when they are supported in WebGPU
                    const stepBuffer = buildBuffer(device)
                        .size(GPU.Size.minimumUniformSize)
                        .usage("uniform")
                        .data(new Uint32Array([i]))
                        .build(`Denoiser.StepBuffer[${i}]`);

                    const pingTexture = i % 2 === 0 ? inputTexture : outputTexture;
                    const pongTexture = i % 2 === 0 ? outputTexture : inputTexture;

                    const bg1 = buildBindGroup(device)
                        .entries(stepBuffer, pongTexture, pingTexture)
                        .layout(pl.getBindGroupLayout(1))
                        .build(`Denoiser.BindGroup1[${i}]`);

                    encoder.setBindGroup(0, bg0);
                    encoder.setBindGroup(1, bg1);
                    encoder.dispatchWorkgroups(dispatchX, dispatchY);
                    encoder.popDebugGroup();
                }
                encoder.popDebugGroup();
            },
            updateSettings(settings) {
                GPU.writeBuffer(device, stateBuffer, [
                    settings.sigmaColor,
                    settings.sigmaNormal,
                    settings.sigmaDepth,
                    settings.sigmaAlbedo,
                    settings.sigmaReflectPosition,
                    settings.sigmaReflectDistance,
                ]);
            }
        };

        denoiser.updateSettings({
            sigmaColor: 3,
            sigmaNormal: 0.3,
            sigmaDepth: 0.5,
            sigmaAlbedo: 0.01,
            sigmaReflectPosition: 0.5,
            sigmaReflectDistance: 0.25,
        });
        return denoiser;
    }
}