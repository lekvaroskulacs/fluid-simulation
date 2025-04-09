import shader from "./shaders/water.wgsl"
import { TriangleMesh } from "./triangle_mesh";
import { mat4, vec3 } from "gl-matrix";
import { Plane } from "./plane_mesh";
import { Skybox } from "./skybox";

export class Renderer {

    canvas: HTMLCanvasElement

    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;

    uniformBuffer: GPUBuffer;
    time_uniformBuffer: GPUBuffer;
    waveOptions_uniformBuffer: GPUBuffer;
    sceneOptions_uniformBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
    pipeline: GPURenderPipeline;

    mesh: Plane;

    skybox: Skybox;
    cameraForward: vec3 = vec3.fromValues(0, 0, 0);
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        this.cameraInputs();
        await this.setupDevice();
        this.setupAssets();
        await this.setupPipeline();
        const inputs = document.getElementsByClassName("listened-for-input");
        for (let i = 0; i < inputs.length; i++) {
            inputs.item(i)?.addEventListener("input", () => {
                const item = <HTMLInputElement> inputs.item(i);
                localStorage.setItem(item.id, item.value);
                this.writeOptionBuffer();
            });
        }
        this.skybox = new Skybox(this.device, this.format, this.context, this.canvas);
        await this.skybox.init(this.cameraForward);
        this.render();
    }

    cameraInputs() {
        let yaw = 1.5; // Rotation around the Z axis (left/right)
        let pitch = -0.5; // Rotation around the X axis (up/down)
        const sensitivity = 0.002; // Controls how fast the camera rotates

        pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));

        this.cameraForward = vec3.fromValues(
            Math.cos(pitch) * Math.sin(yaw),
            Math.sin(pitch),
            Math.cos(pitch) * Math.cos(yaw),
        );  

        document.addEventListener('mousemove', (event) => {
            if ( (event.buttons & 1) !== 1 ) //Check if primary mouse button is pressed
                return;

            const dx = -event.movementX; // Change in mouse X position
            const dy = -event.movementY; // Change in mouse Y position
        
            // Update yaw and pitch based on mouse movement
            yaw -= dx * sensitivity;
            pitch -= dy * sensitivity;
        
            // Clamp pitch to avoid gimbal lock
            pitch = Math.max(-Math.PI / 2 + 0.00001, Math.min(Math.PI / 2 - 0.00001, pitch));

            this.cameraForward = vec3.fromValues(
                Math.cos(pitch) * Math.sin(yaw),
                Math.sin(pitch),
                Math.cos(pitch) * Math.cos(yaw),
            );
            
            this.skybox.cameraForward = this.cameraForward;
            
        });
    }

    async setupDevice() {
        this.adapter = <GPUAdapter> await navigator.gpu?.requestAdapter();
        this.device = await this.adapter.requestDevice();
        this.context = <GPUCanvasContext> this.canvas.getContext("webgpu");
        this.format = "bgra8unorm";
        this.context.configure({
            device: this.device,
            format: this.format
        });
    }

    async setupPipeline() {
        this.uniformBuffer = this.device.createBuffer({
            size: 64 * 3,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.time_uniformBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.waveOptions_uniformBuffer = this.device.createBuffer({
            size: 7 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.sceneOptions_uniformBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX,
                    buffer: {
                        type: "uniform"
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform"
                    }
                },

            ]
        });

        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.uniformBuffer
                    }
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.time_uniformBuffer
                    }
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.waveOptions_uniformBuffer
                    }
                },
                {
                    binding: 3,
                    resource: {
                        buffer: this.sceneOptions_uniformBuffer
                    }
                },
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        });

        this.pipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: shader
                }),
                entryPoint: "vs_main",
                buffers: [this.mesh.bufferLayout]
            },

            fragment: {
                module: this.device.createShaderModule({
                    code: shader
                }),
                entryPoint: "fs_main",
                targets: [{
                    format: this.format
                }]
            },

            primitive: {
                topology: "triangle-strip",
                stripIndexFormat: "uint16"
            },

            layout: pipelineLayout
        });
    }

    setupAssets() {
        this.mesh = new Plane(3, 254, this.device);
    }

    render() {
        if (!this)
            console.log("this is null");
        this.writeBuffers();

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const textureView: GPUTextureView = this.context.getCurrentTexture().createView();
        const renderpass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: {r: 133.0/255.0, g: 211.0/255.0, b: 241.0/255.0, a: 0},
                loadOp: "load",
                storeOp: "store"
            }]
        });

        renderpass.setPipeline(this.pipeline);
        renderpass.setVertexBuffer(0, this.mesh.vertexBuffer);
        renderpass.setIndexBuffer(this.mesh.indexBuffer, "uint16");
        renderpass.setBindGroup(0, this.bindGroup);
        renderpass.drawIndexed(this.mesh.indexBuffer.size / Uint16Array.BYTES_PER_ELEMENT);
        renderpass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(() => this.render());
    }


    writeBuffers() {
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 10);

        const view = mat4.create();
        const cameraPos = vec3.fromValues(0, 1, 0);
        const forwardWorld = vec3.create();
        vec3.add(forwardWorld, this.cameraForward, cameraPos);
        mat4.lookAt(view, cameraPos, forwardWorld, [0, 1, 0]);
        
        const model = mat4.create();
        mat4.scale(model, model, [1, 1, 1]);

        const time = performance.now() / 1000;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, <ArrayBuffer>model);
        this.device.queue.writeBuffer(this.uniformBuffer, 64, <ArrayBuffer>view);
        this.device.queue.writeBuffer(this.uniformBuffer, 128, <ArrayBuffer>projection);
        this.device.queue.writeBuffer(this.time_uniformBuffer, 0, new Float32Array([time]));

        this.writeOptionBuffer();
    }

    writeOptionBuffer() {
        const _amplitude: number = +(localStorage.getItem("amplitude") ?? 1);
        const _frequency: number = +(localStorage.getItem("frequency") ?? 1);
        const _amplitudeMult: number = +(localStorage.getItem("amplitudeMultiplier") ?? 1);
        const _frequencyMult: number = +(localStorage.getItem("frequencyMultiplier") ?? 1);
        const _basePhase: number = +(localStorage.getItem("basePhase") ?? 0);
        const _baseSpeed: number = +(localStorage.getItem("baseSpeed") ?? 1);
        const _maxWaves: number = +(localStorage.getItem("maxWaves") ?? 1);

        const _sunPosition: number = +(localStorage.getItem("sunPosition") ?? 1);

        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 0, new Float32Array([_amplitude]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 4, new Float32Array([_frequency]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 8, new Float32Array([_amplitudeMult]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 12, new Float32Array([_frequencyMult]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 16, new Float32Array([_basePhase]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 20, new Float32Array([_baseSpeed]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 24, new Float32Array([_maxWaves]));

        this.device.queue.writeBuffer(this.sceneOptions_uniformBuffer, 0, new Float32Array([_sunPosition]));
    }

    
}