import skybox from "./shaders/skybox.wgsl"
import { mat4, vec3 } from "gl-matrix";

export class Skybox {

    canvas: HTMLCanvasElement;

    device: GPUDevice;
    format: GPUTextureFormat;
    context: GPUCanvasContext;

    depthTextureView: GPUTextureView;
    pipeline: GPURenderPipeline;
    bindGroup: GPUBindGroup;

    viewDirInverseBuffer: GPUBuffer;

    cameraForward: vec3;


    constructor(device: GPUDevice, format: GPUTextureFormat, context: GPUCanvasContext, canvas: HTMLCanvasElement) {
        this.device = device;
        this.format = format;
        this.context = context;
        this.canvas = canvas;
    }

    async init(cameraForward: vec3) {
        await this.setupSkybox();
        this.cameraForward = cameraForward
        this.renderSkybox();
    }

    async createSkyboxTexture(): Promise<GPUTexture> {
        const bitmaps: ImageBitmap[] = [];
        for (let i = 1; i <= 6; i++) {
            const response = <Response> await fetch("./src/assets/skybox" + i + ".jpg");
            bitmaps[i-1] = <ImageBitmap> await createImageBitmap(await response.blob());
        }
        

        const texture = this.device.createTexture({
            size: [bitmaps[0].width, bitmaps[0].height, 6],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        });

        for (let i = 0; i < 6; i++) {
            this.device.queue.copyExternalImageToTexture(
                { source: bitmaps[i] },
                { texture , origin: [0, 0, i]},
                [bitmaps[0].width, bitmaps[0].height]
            );
        }

        return texture;
    }

    async setupSkybox() {
        
        const bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    texture: {
                        viewDimension: "cube"
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "uniform"
                    }
                }
            ]
        })


        this.viewDirInverseBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
        });
        
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: (await this.createSkyboxTexture()).createView({ dimension: 'cube' }) },
                { binding: 1, resource: this.device.createSampler() },
                { binding: 2, resource: { buffer: this.viewDirInverseBuffer } }

            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        })

        this.pipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: skybox
                }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: skybox
                }),
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            depthStencil: {
                depthWriteEnabled: true, 
                depthCompare: 'less-equal', // Render the skybox behind everything else
                format: 'depth24plus'
            },
            primitive: {
                topology: 'triangle-list'
            },
            layout: pipelineLayout
        });

        const depthTexture = this.device.createTexture({
            size: [this.context.canvas.width, this.context.canvas.height, 1],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });
        
        this.depthTextureView = depthTexture.createView();

    }

    renderSkybox() {
        const projection = mat4.create();
        mat4.perspective(projection, 60 * Math.PI / 180, this.canvas.width / this.canvas.height, 0.1, 10);

        const skyboxView = mat4.create();
        mat4.lookAt(skyboxView, [0, 0, 0], this.cameraForward, [0, 1, 0]);

        skyboxView[12] = skyboxView[13] = skyboxView[14] = 0; 

        const viewProj = mat4.create();
        mat4.mul(viewProj, projection, skyboxView);
        const inverse = mat4.create();
        mat4.invert(inverse, viewProj);
        
        this.device.queue.writeBuffer(this.viewDirInverseBuffer, 0, new Float32Array(inverse).buffer);

        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const textureView: GPUTextureView = this.context.getCurrentTexture().createView();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 133.0 / 255.0, g: 211.0 / 255.0, b: 241.0 / 255.0, a: 1 },
                loadOp: "clear",
                storeOp: "store"
            }],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(() => this.renderSkybox());
    }

}