import * as GPU from "@domgell/webgpu-util";
import * as Game from "@domgell/game-util";
import {ImGui, ImGuiImplWeb} from "@mori2003/jsimgui";
import {
    createRaytracer,
    Raytracer,
    resizeRaytracer,
    runRaytracer,
    updateRaytracerScene,
    updateRaytracerSettings
} from "./raytracer.ts";
import {
    TextureAtlas,
    DebugRenderer,
    TextureCopyRenderer,
    FullscreenRenderer,
    getSamplerFilterNearestClampToEdge
} from "@domgell/webgpu-samples";
import {mat4, vec2, vec3} from "dom-game-math";
import {Input} from "@domgell/game-input";
import {defaultColorTargetState, generatePipelineLayout} from "@domgell/webgpu-util";
import {buildBindGroup, buildBuffer, buildRenderPass, buildTexture} from "@domgell/webgpu-builder";
import {
    createMouseRay,
    createRoomScene, createTimestampState,
    drawMeshInstanceOutline,
    drawMeshNodes,
    drawSceneNodes,
    pickMeshInstance, rollingAverage
} from "./util.ts";
import {drawMeshUI, drawSettingsUI, Settings} from "./ui.ts";
import {Scene} from "./scene.ts";
import {createDenoiser, Denoiser, resizeDenoiser, runDenoiser, updateDenoiserSettings} from "./denoiser.ts";
import {BVH} from "./bvh.ts";
import {assert, fail} from "@domgell/ts-util";

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
        requestedFeatures: ["timestamp-query"],
        requiredLimits: {maxStorageBuffersPerShaderStage: 10}
    },
    context: {
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    }
});

const timestampQuerySupport = adapter.features.has("timestamp-query");

// ImGui
await ImGuiImplWeb.Init({device, canvas, backend: "webgpu"});
if (saved?.imgui) ImGui.LoadIniSettingsFromMemory(saved.imgui);

// ------------------------------------ Performance ------------------------------------

const cpuTime = rollingAverage();
const gpuRaytracingComputeTime = rollingAverage();
const gpuDenoiserComputeTime = rollingAverage();
const gpuRenderTime = rollingAverage();

const timestampState = timestampQuerySupport ? createTimestampState(device, 3) : undefined;

// ---------------------------------- Init Renderers -----------------------------------

let outputTexture = buildTexture(device)
    .size(canvas.width, canvas.height)
    .format("rgba8unorm")
    .usage("storage-binding", "copy-src", "texture-binding")
    .build("OutputTexture");

const textureAtlas = TextureAtlas.create(device, {tile: await Game.importImage("res/tile.png")});

const cameraBuffer = buildBuffer(device)
    .size(GPU.Size.mat4x4f * 2)
    .usage("uniform", "copy-dst")
    .build("CameraBuffer");

const raytracer = createRaytracer({
    device,
    outputTexture,
    cameraBuffer,
    textureAtlas,
    shader: await Game.importText("shaders/raytrace.wgsl"),
});

const denoiser = createDenoiser({
    device,
    outputTexture,
    sampleInfoBuffer: raytracer.sampleInfoBuffer,
    shader: await Game.importText("shaders/denoise.wgsl"),
});

const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
const colorState = defaultColorTargetState({format: canvasFormat});
const debugRenderer = DebugRenderer.create({device, camera: cameraBuffer, color: colorState});

const outputRenderer = FullscreenRenderer.create({
    device,
    color: colorState,
    shader: await Game.importText("shaders/output.wgsl")
});

let outputRendererBG = buildBindGroup(device)
    .layout(outputRenderer.getBindGroupLayout(0))
    .entries(outputTexture)
    .build("OutputRenderer.BindGroup");

// ------------------------------------ Init Scene -------------------------------------

const camera = Game.Camera3d.create(saved?.camera ?? {position: vec3.new(0, 4, -11)});

const scene = await createRoomScene();
updateRaytracerScene(raytracer, scene);

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
    timestampQuerySupport,
    cpuTime: 0,
    gpuRenderTime: 0,
    gpuDenoiserComputeTime: 0,
    gpuRaytracingComputeTime: 0,
    fps: 0,
};

updateRaytracerSettings(raytracer, settings);
updateDenoiserSettings(denoiser, settings);

// -------------------------------------- Update ---------------------------------------

let pickMesh: Scene.MeshInstance | undefined;

Game.runUpdate((dt, t) => {

    // ------------------------------------ Resize -------------------------------------

    canvas.width = window.innerWidth - 20;
    canvas.height = window.innerHeight - 20;

    if (canvas.width !== outputTexture.width || canvas.height !== outputTexture.height) {
        context.configure(context.getConfiguration()!);

        outputTexture.destroy();
        outputTexture = buildTexture(device)
            .size(canvas.width, canvas.height)
            .format(outputTexture.format)
            .usage(outputTexture.usage)
            .build(outputTexture.label);

        outputRendererBG = buildBindGroup(device)
            .layout(outputRenderer.getBindGroupLayout(0))
            .entries(outputTexture)
            .build("OutputRenderer.BindGroup");

        resizeRaytracer(raytracer, outputTexture);
        resizeDenoiser(denoiser, outputTexture, raytracer.sampleInfoBuffer);
        raytracer.frameIndex = 0;
    }

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
        raytracer.frameIndex = 0; // Reset accumulation on camera move
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

    if (drawSettingsUI(settings)) updateRaytracerSettings(raytracer, settings);

    // Rebuild scene BVH and write scene data on mesh update
    if (drawMeshUI(pickMesh, scene, textureAtlas)) {
        scene.sceneNodes.length = 0;
        BVH.buildSceneNodes(scene);
        updateRaytracerScene(raytracer, scene);
    }

    ImGui.End();

    // Draw picked mesh
    if (pickMesh !== undefined) drawMeshInstanceOutline(pickMesh, scene, debugRenderer);

    // Draw BVH outlines
    if (settings.showSceneBVH) drawSceneNodes(scene, debugRenderer);
    if (settings.showMeshBVH) drawMeshNodes(scene, debugRenderer);

    // Disable accumulation
    if (!settings.accumulation) raytracer.frameIndex = 0;

    // ------------------------------------ Render -------------------------------------

    // Write camera buffer
    GPU.writeBuffer(device, cameraBuffer, [...viewProjection, ...invViewProjection]);

    const cmd = device.createCommandEncoder();

    // Run raytracing compute shader
    const raytracePass = cmd.beginComputePass({timestampWrites: timestampState?.getTimestampWrite(0)});
    runRaytracer(raytracer, raytracePass);
    raytracePass.end();

    // Run denoising compute shader
    if (settings.denoising) {
        const denoisePass = cmd.beginComputePass({timestampWrites: timestampState?.getTimestampWrite(1)});
        raytracer.frameIndex = 0; // Disable accumulation when using denoising
        runDenoiser(denoiser, denoisePass);
        denoisePass.end();
    } else {
        gpuDenoiserComputeTime.clear();
    }

    // Render raytracing result to screen texture, and render UI and debug lines
    const currentTexture = context.getCurrentTexture();
    const renderPass = cmd.beginRenderPass({
        colorAttachments: [GPU.createColorAttachment(currentTexture, [0, 0, 0, 1])],
        timestampWrites: timestampState?.getTimestampWrite(2),
    });
    outputRenderer.render(renderPass, [outputRendererBG]);
    debugRenderer.render(renderPass);
    ImGuiImplWeb.EndRender(renderPass);
    renderPass.end();

    timestampState?.resolve(cmd);
    GPU.submitCommandEncoder(device, cmd);

    // ---------------------------------- Performance ----------------------------------

    timestampState?.update(gpuRaytracingComputeTime, gpuDenoiserComputeTime, gpuRenderTime);

    // CPU time this frame
    const now = performance.now() / 1000;
    cpuTime.add(now - t);

    settings.cpuTime = cpuTime.get();
    settings.fps = 1 / dt;
    settings.gpuRaytracingComputeTime = gpuRaytracingComputeTime.get();
    settings.gpuDenoiserComputeTime = gpuDenoiserComputeTime.get();
    settings.gpuRenderTime = gpuRenderTime.get();

    // ---------------------------------------------------------------------------------

    input.flush();

    // Save camera and UI state to browser local storage to reuse between runs
    Game.save({camera, imgui: ImGui.SaveIniSettingsToMemory()}, "saved");
});