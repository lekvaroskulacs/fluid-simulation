import shader from "./shaders/water.wgsl"
import { TriangleMesh } from "./triangle_mesh";
import { mat4 } from "gl-matrix";
import { Plane } from "./plane_mesh";

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
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        await this.setupDevice();
        this.setupAssets();
        await this.setupPipeline();
        //requestAnimationFrame(this.render);
        this.render();
        const inputs = document.getElementsByClassName("listened-for-input");
        for (let i = 0; i < inputs.length; i++) {
            inputs.item(i)?.addEventListener("input", () => {
                const item = <HTMLInputElement> inputs.item(i);
                localStorage.setItem(item.id, item.value);
                this.writeOptionBuffer();
            });
        }
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

        const texture = await this.createNoiseTexture();
        const sampler = this.device.createSampler({
            magFilter: "linear",
            minFilter: "linear"
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
                clearValue: {r: 133.0/255.0, g: 211.0/255.0, b: 241.0/255.0, a: 1},
                loadOp: "clear",
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
        mat4.lookAt(view, [-3, 0, 3], [0, 0, 0], [0, 0, 1]);

        const model = mat4.create();
        mat4.scale(model, model, [1, 1, 1]);

        const time = performance.now() / 1000;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, <ArrayBuffer>model);
        this.device.queue.writeBuffer(this.uniformBuffer, 64, <ArrayBuffer>view);
        this.device.queue.writeBuffer(this.uniformBuffer, 128, <ArrayBuffer>projection);
        this.device.queue.writeBuffer(this.time_uniformBuffer, 0, new Float32Array([time]));

        this.writeOptionBuffer();
    }

    async createNoiseTexture(): Promise<GPUTexture> {
        const response = await fetch("./src/assets/noiseTexture.png");
        const imageBitmap = await createImageBitmap(await response.blob());

        const texture = this.device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture },
            [imageBitmap.width, imageBitmap.height]
        );

        return texture;
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

    renderSkybox() {
        // Cube vertices (positions)
        const cubeVertexBuffer: Float32Array = new Float32Array([
            -1.0, -1.0,  1.0, // Front-bottom-left
             1.0, -1.0,  1.0, // Front-bottom-right
             1.0,  1.0,  1.0, // Front-top-right
            -1.0,  1.0,  1.0, // Front-top-left
            -1.0, -1.0, -1.0, // Back-bottom-left
             1.0, -1.0, -1.0, // Back-bottom-right
             1.0,  1.0, -1.0, // Back-top-right
            -1.0,  1.0, -1.0  // Back-top-left
        ]);

        // Cube indices (to form triangles)
        const cubeIndexBuffer: Float32Array = new Float32Array([
            // Front face
            0, 1, 2, 2, 3, 0,
            // Right face
            1, 5, 6, 6, 2, 1,
            // Back face
            5, 4, 7, 7, 6, 5,
            // Left face
            4, 0, 3, 3, 7, 4,
            // Bottom face
            4, 5, 1, 1, 0, 4,
            // Top face
            3, 2, 6, 6, 7, 3
        ]);


    }
}