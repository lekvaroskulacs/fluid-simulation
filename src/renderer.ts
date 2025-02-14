import shader from "./shaders/shaders.wgsl"
import { TriangleMesh } from "./triangle_mesh";
import { mat4 } from "gl-matrix";

export class Renderer {

    canvas: HTMLCanvasElement

    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;

    uniformBuffer: GPUBuffer;
    bindGroup: GPUBindGroup;
    pipeline: GPURenderPipeline;

    mesh: TriangleMesh;
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        await this.setupDevice();
        this.setupAssets();
        await this.setupPipeline();
        this.render();
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
        })

        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {}
            }]
        });

        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: this.uniformBuffer
                }
            }]
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
                topology: "triangle-list"
            },

            layout: pipelineLayout
        });
    }

    setupAssets() {
        this.mesh = new TriangleMesh(this.device);
    }

    render() {
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4, this.canvas.width/this.canvas.height, 0.1, 10);

        const view = mat4.create();
        mat4.lookAt(view, [-2, 0, 2], [0, 0, 0], [0, 0, 1]);

        const model = mat4.create();
        mat4.rotate(model, model, 0.0, [0, 0, 1]);

        this.device.queue.writeBuffer(this.uniformBuffer, 0, <ArrayBuffer>model);
        this.device.queue.writeBuffer(this.uniformBuffer, 64, <ArrayBuffer>view);
        this.device.queue.writeBuffer(this.uniformBuffer, 128, <ArrayBuffer>projection);

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const textureView: GPUTextureView = this.context.getCurrentTexture().createView();
        const renderpass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: {r: 0, g: 0, b: 0, a: 1},
                loadOp: "clear",
                storeOp: "store"
            }]
        });

        renderpass.setPipeline(this.pipeline);
        renderpass.setVertexBuffer(0, this.mesh.buffer);
        renderpass.setBindGroup(0, this.bindGroup);
        renderpass.draw(3, 1, 0, 0);
        renderpass.end();
        
        this.device.queue.submit([commandEncoder.finish()]);
    }

}