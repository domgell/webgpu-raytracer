import {ImGui} from "@mori2003/jsimgui";
import {mat4, quat, vec3} from "dom-game-math";
import {Scene} from "./scene.ts";
import {TextureAtlas} from "@domgell/webgpu-samples";

export interface Settings {
    // Raytracing
    raysPerPixel: number,
    maxRayBounces: number,
    nee: boolean,
    accumulation: boolean,
    // Denoising
    denoising: boolean,
    sigmaColor: number,
    sigmaNormal: number,
    sigmaDepth: number,
    sigmaAlbedo: number,
    sigmaReflectPosition: number,
    sigmaReflectDistance: number,
    // BVH
    showSceneBVH: boolean,
    showMeshBVH: boolean,
    // Timing
    cpuTime: number,
    timestampQuerySupport: boolean,
    gpuRenderTime: number,
    gpuDenoiserComputeTime: number,
    gpuRaytracingComputeTime: number,
    fps: number,
}

export function drawSettingsUI(settings: Settings) {
    let changed = false;

    if (ImGui.CollapsingHeader("Raytracing")) {
        const raysPerPixel = [settings.raysPerPixel] as [number];
        if (ImGui.SliderInt("Rays Per Pixel", raysPerPixel, 1, 64)) {
            settings.raysPerPixel = raysPerPixel[0];
            changed = true;
        }

        const maxRayBounces = [settings.maxRayBounces] as [number];
        if (ImGui.SliderInt("Max Ray Bounces", maxRayBounces, 1, 16)) {
            settings.maxRayBounces = maxRayBounces[0];
            changed = true;
        }

        const nee = [settings.nee] as [boolean];
        if (ImGui.Checkbox("Next Event Estimation", nee)) {
            settings.nee = nee[0];
            changed = true;
        }

        const accumulation = [settings.accumulation] as [boolean];
        if (ImGui.Checkbox("Accumulation", accumulation)) {
            settings.accumulation = accumulation[0];
            changed = true;
        }

        const showSceneBVH = [settings.showSceneBVH] as [boolean];
        if (ImGui.Checkbox("Show Scene BVH", showSceneBVH)) {
            settings.showSceneBVH = showSceneBVH[0];
        }

        const showMeshBVH = [settings.showMeshBVH] as [boolean];
        if (ImGui.Checkbox("Show Mesh BVH", showMeshBVH)) {
            settings.showMeshBVH = showMeshBVH[0];
        }
    }

    if (ImGui.CollapsingHeader("Denoising")) {
        const denoising = [settings.denoising] as [boolean];
        if (ImGui.Checkbox("Enable denoising", denoising)) {
            settings.denoising = denoising[0];
            changed = true;
        }

        ImGui.SeparatorText("Weights");

        const sigmaColor = [settings.sigmaColor] as [number];
        if (ImGui.SliderFloat("Color", sigmaColor, 0.01, 5)) {
            settings.sigmaColor = sigmaColor[0];
            changed = true;
        }

        const sigmaNormal = [settings.sigmaNormal] as [number];
        if (ImGui.SliderFloat("Normal", sigmaNormal, 0.01, 5)) {
            settings.sigmaNormal = sigmaNormal[0];
            changed = true;
        }

        const sigmaDepth = [settings.sigmaDepth] as [number];
        if (ImGui.SliderFloat("Depth", sigmaDepth, 0.01, 5)) {
            settings.sigmaDepth = sigmaDepth[0];
            changed = true;
        }

        const sigmaAlbedo = [settings.sigmaAlbedo] as [number];
        if (ImGui.SliderFloat("Albedo", sigmaAlbedo, 0.01, 5)) {
            settings.sigmaAlbedo = sigmaAlbedo[0];
            changed = true;
        }

        const sigmaReflectPosition = [settings.sigmaReflectPosition] as [number];
        if (ImGui.SliderFloat("Reflect Position", sigmaReflectPosition, 0.01, 5)) {
            settings.sigmaReflectPosition = sigmaReflectPosition[0];
            changed = true;
        }

        const sigmaReflectDistance = [settings.sigmaReflectDistance] as [number];
        if (ImGui.SliderFloat("Reflect Distance", sigmaReflectDistance, 0.01, 5)) {
            settings.sigmaReflectDistance = sigmaReflectDistance[0];
            changed = true;
        }
    }

    if (ImGui.CollapsingHeader("Performance", ImGui.TreeNodeFlags.DefaultOpen)) {
        ImGui.Text(`FPS (VSync): ${settings.fps.toFixed()}`);

        if (settings.timestampQuerySupport) {
            const sum = settings.gpuRaytracingComputeTime + settings.gpuDenoiserComputeTime + settings.gpuRenderTime;
            ImGui.Text(`FPS (GPU): ${(1000 / sum).toFixed()}`);
            ImGui.Text(`GPU Raytrace Time: ${settings.gpuRaytracingComputeTime.toFixed(2)} ms`);
            ImGui.Text(`GPU Denoise Time: ${settings.gpuDenoiserComputeTime.toFixed(3)} ms`);
            ImGui.Text(`GPU Render Time: ${settings.gpuRenderTime.toFixed(3)} ms`);
        }

        ImGui.Text(`CPU Time: ${settings.cpuTime.toFixed(4)} ms`);

        if (!settings.timestampQuerySupport) {
            ImGui.Spacing()
            ImGui.Text("GPU Performance unavailable.\nWebGPU `timestamp-query` is not supported");
        }
    }

    return changed;
}

export function drawMeshUI(meshInstance: Scene.MeshInstance | undefined, scene: Scene, atlas: TextureAtlas) {
    if (meshInstance === undefined) return false;
    if (!ImGui.CollapsingHeader(`Mesh`, ImGui.TreeNodeFlags.DefaultOpen)) return false;

    const transform = mat4.decompose(meshInstance.transform);
    const pos = vec3.toArray(transform.translation);
    const scale = vec3.toArray(transform.scale);
    const euler = quat.toEuler(transform.rotation);
    const rot = [euler.pitch, euler.yaw, euler.roll] as [number, number, number];

    let transformChanged = false;
    ImGui.SeparatorText("Transform");
    if (ImGui.DragFloat3("Translation", pos, 0.1)) transformChanged = true;
    if (ImGui.DragFloat3("Rotation", rot, 1)) transformChanged = true;
    if (ImGui.DragFloat3("Scale", scale, 0.1, 0.001, 1000)) transformChanged = true;

    if (transformChanged) {
        mat4.compose({
            translation: vec3.fromArray(pos),
            rotation: quat.fromEuler({pitch: rot[0], yaw: rot[1], roll: rot[2]}),
            scale: vec3.fromArray(scale)
        }, meshInstance.transform);

        mat4.invert(meshInstance.transform, meshInstance.invTransform);
        mat4.transpose(meshInstance.invTransform, meshInstance.invTransposeTransform);
    }

    const material = scene.materials[meshInstance.materialIndex];
    const color = vec3.toArray(material.color);
    const roughness = [material.roughness] as [number];
    const light = [material.light] as [number];
    const textureId = [material.textureId] as [number];

    // Combine texture names into special string to be used with ImGui
    const textureNames = atlas.entries.entries().toArray()
        .sort(([_, entry]) => entry.id)
        .map(([name, _]) => name);
    textureNames.push("none");
    const textureNamesStr = textureNames.join("\0") + "\0";

    let materialChanged = false;
    ImGui.SeparatorText("Material");
    if (ImGui.ColorEdit3("Color", color)) materialChanged = true;
    if (ImGui.SliderFloat("Roughness", roughness, 0, 1)) materialChanged = true;
    if (ImGui.SliderFloat("Light", light, 0, 30)) materialChanged = true;
    if (ImGui.Combo("Texture", textureId, textureNamesStr)) materialChanged = true;

    if (materialChanged) {
        material.color = vec3.fromArray(color);
        material.roughness = roughness[0];
        material.light = light[0];
        material.textureId = textureId[0];
    }

    return transformChanged || materialChanged;
}