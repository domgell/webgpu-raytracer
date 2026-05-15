import * as GPU from "@domgell/webgpu-util";
import * as Game from "@domgell/game-util";
import {ImGui, ImGuiImplWeb} from "@mori2003/jsimgui";
import {Raytracer} from "./raytracer.ts";
import {TextureAtlas, DebugRenderer, TextureCopyRenderer, FullscreenRenderer} from "@domgell/webgpu-samples";
import {mat4, vec2, vec3} from "dom-game-math";
import {Input} from "@domgell/game-input";
import {defaultColorTargetState} from "@domgell/webgpu-util";
import {buildBindGroup, buildBuffer, buildRenderPass, buildTexture} from "@domgell/webgpu-builder";
import {
    createMouseRay,
    createRoomScene,
    drawMeshInstanceOutline,
    drawMeshNodes,
    drawSceneNodes,
    pickMeshInstance
} from "./util.ts";
import {drawMeshUI, drawSettingsUI, Settings} from "./ui.ts";
import {Scene} from "./scene.ts";
import {Denoiser} from "./denoiser.ts";
import {BVH} from "./bvh.ts";

// --------------------------------------- Init ----------------------------------------

// Canvas
const canvas = document.querySelector("canvas")!;
canvas.width = window.innerWidth - 20;
canvas.height = window.innerHeight - 20;

const input = Input.create(canvas);
const saved = Game.load("saved");

// WebGPU
const {device, context, adapter} = await GPU.initWebGPU(canvas, {
    device: {
        requiredFeatures: ["texture-formats-tier2"],
        requiredLimits: {maxStorageBuffersPerShaderStage: 9}
    },
    context: {usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT}
});

// ImGui
await ImGuiImplWeb.Init({device, canvas, backend: "webgpu"});
if (saved?.imgui) ImGui.LoadIniSettingsFromMemory(saved.imgui);

// ---------------------------------- Init Renderers -----------------------------------

const outputTexture = buildTexture(device)
    .size(canvas.width, canvas.height)
    .format("rgba8unorm")
    .usage("storage-binding", "copy-src", "texture-binding")
    .build("OutputTexture");

const textureAtlas = TextureAtlas.create(device, {tile: await Game.importImage("res/tile.png")});

const cameraBuffer = buildBuffer(device)
    .size(GPU.Size.mat4x4f * 2)
    .usage("uniform", "copy-dst")
    .build("CameraBuffer");

const raytracer = Raytracer.create({
    device,
    outputTexture,
    cameraBuffer,
    textureAtlas,
    shader: await Game.importText("shaders/raytrace.wgsl"),
});

const denoiser = Denoiser.create({
    device,
    outputTexture,
    sampleInfoBuffer: raytracer.sampleInfoBuffer,
    shader: await Game.importText("shaders/denoise.wgsl"),
});

const colorState = defaultColorTargetState({format: "bgra8unorm"});
const debugRenderer = DebugRenderer.create({device, camera: cameraBuffer, color: colorState});
const outputRenderer = TextureCopyRenderer.create({device, texture: outputTexture, color: colorState});

// ------------------------------------ Init Scene -------------------------------------

const camera = Game.Camera3d.create(saved?.camera ?? {position: vec3.new(0, 4, -11)});

const scene = await createRoomScene();
raytracer.updateScene(scene);

// ------------------------------------- Settings --------------------------------------

const settings: Settings = {
    raysPerPixel: 1,
    maxRayBounces: 4,
    nee: true,
    accumulation: true,
    showMeshBVH: false,
    showSceneBVH: false,
    denoising: false,
    sigmaColor: 3,
    sigmaNormal: 0.3,
    sigmaDepth: 0.5,
    sigmaAlbedo: 0.01,
    sigmaReflectPosition: 0.5,
    sigmaReflectDistance: 0.25,
};

raytracer.updateSettings(settings);
denoiser.updateSettings(settings);

// -------------------------------------- Update ---------------------------------------

let pickMesh: Scene.MeshInstance | undefined;

Game.runUpdate((dt, t) => {

    // ------------------------------------- Input -------------------------------------

    // Update camera input
    const uiFocus = ImGui.GetIO().WantCaptureMouse || ImGui.GetIO().WantCaptureKeyboard;
    const capture = input.isDown("RightMouse") && !uiFocus;
    input.setMouseCapture(capture);
    if (capture) {
        camera.updateFirstPerson({
            move: vec2.mul(input.wasd(), dt * 15),
            look: vec2.mul(input.mouseDelta(), 0.1),
        });
        raytracer.resetAccumulation();
    }

    const viewProjection = camera.viewProjection(canvas.width / canvas.height);
    const invViewProjection = mat4.invert(viewProjection);

    // Pick mesh
    if (input.isJustPressed("LeftMouse") && !uiFocus) {
        const mouseRay = createMouseRay(invViewProjection, input.mousePosition(), canvas.width, canvas.height);
        const newPickMesh = pickMeshInstance(mouseRay, scene);

        if (newPickMesh === undefined || newPickMesh === pickMesh) pickMesh = undefined; // Unselect
        else pickMesh = newPickMesh; // Select
    }

    // -------------------------------------- UI ---------------------------------------

    ImGuiImplWeb.BeginRender();
    ImGui.SetNextWindowPos({x: 0, y: 0}, ImGui.Cond.FirstUseEver);
    ImGui.Begin("Settings", null, ImGui.WindowFlags.AlwaysAutoResize);

    if (drawSettingsUI(settings)) raytracer.updateSettings(settings);

    // Rebuild scene BVH and write scene data on mesh update
    if (drawMeshUI(pickMesh, scene, textureAtlas)) {
        scene.sceneNodes.length = 0;
        BVH.buildSceneNodes(scene);
        raytracer.updateScene(scene);
    }

    ImGui.End();

    // Draw picked mesh
    if (pickMesh !== undefined) drawMeshInstanceOutline(pickMesh, scene, debugRenderer);

    // Draw BVH outlines
    if (settings.showSceneBVH) drawSceneNodes(scene, debugRenderer);
    if (settings.showMeshBVH) drawMeshNodes(scene, debugRenderer);

    // Clear accumulation every frame when disabled
    if (!settings.accumulation) raytracer.resetAccumulation();

    // ------------------------------------ Render -------------------------------------

    // Write camera buffer
    GPU.writeBuffer(device, cameraBuffer, [...viewProjection, ...invViewProjection]);

    const cmd = device.createCommandEncoder();

    // Run raytracing compute shader
    const computePass = cmd.beginComputePass();
    raytracer.run(computePass);
    if (settings.denoising) denoiser.run(computePass);
    computePass.end();

    // Render raytracing result to screen texture, and render UI and debug lines
    const currentTexture = context.getCurrentTexture();
    const renderPass = buildRenderPass(cmd).color(currentTexture).build();
    outputRenderer.render(renderPass);
    debugRenderer.render(renderPass);
    ImGuiImplWeb.EndRender(renderPass);
    renderPass.end();

    // ---------------------------------------------------------------------------------

    GPU.submitCommandEncoder(device, cmd);

    input.flush();

    // Save camera and UI state to browser local storage to reuse between runs
    Game.save({camera, imgui: ImGui.SaveIniSettingsToMemory()}, "saved");
});