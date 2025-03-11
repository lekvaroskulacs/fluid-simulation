import skybox from "./shaders/skybox.wgsl"
import { mat4, vec3 } from "gl-matrix";

export class Skybox {

    device: GPUDevice;
    format: GPUTextureFormat;
    context: GPUCanvasContext;

    depthTextureView: GPUTextureView;
    pipeline: GPURenderPipeline;
    bindGroup: GPUBindGroup;
    vertexBuffer: GPUBuffer;
    indexBuffer: GPUBuffer;
    viewBuffer: GPUBuffer;

    cameraForward: vec3;


    constructor(device: GPUDevice, format: GPUTextureFormat, context: GPUCanvasContext) {
        this.device = device;
        this.format = format;
        this.context = context;
    }

    async init(cameraForward: vec3) {
        await this.setupSkybox();
        this.cameraForward = cameraForward
        this.renderSkybox();
    }

    async createSkyboxTexture(): Promise<GPUTexture> {
        const bitmaps: ImageBitmap[] = [];
        for (let i = 1; i <= 6; i++) {
            const response = <Response> await fetch("./src/assets/skybox" + i + ".png");
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

        this.vertexBuffer = this.device.createBuffer({
            size: cubeVertexBuffer.BYTES_PER_ELEMENT * cubeVertexBuffer.length,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
            mappedAtCreation: true
        });

        new Float32Array(this.vertexBuffer.getMappedRange()).set(cubeVertexBuffer);
        this.vertexBuffer.unmap();

        this.indexBuffer = this.device.createBuffer({
            size: cubeIndexBuffer.BYTES_PER_ELEMENT * cubeIndexBuffer.length,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDEX,
            mappedAtCreation: true
        });

        new Uint16Array(this.indexBuffer.getMappedRange()).set(cubeIndexBuffer);
        this.indexBuffer.unmap();

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


        this.viewBuffer = this.device.createBuffer({
            size: 64,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM
        });
        

        // Bind the cube map texture and sampler
        this.bindGroup = this.device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: (await this.createSkyboxTexture()).createView({ dimension: 'cube' }) },
                { binding: 1, resource: this.device.createSampler() },
                { binding: 2, resource: { buffer: this.viewBuffer } }

            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout]
        })

        // Example: Setting up the render pipeline for the skybox
        this.pipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: skybox
                }),
                entryPoint: 'vs_main',
                buffers: [{
                    arrayStride: 3 * 4, // 3 floats (x, y, z)
                    attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
                }]
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: skybox
                }),
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
            depthStencil: {
                depthWriteEnabled: false, // Disable depth writing for the skybox
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
        const skyboxView = mat4.create();
        mat4.lookAt(skyboxView, [0, 0, 0], this.cameraForward, [0, 1, 0]);
        this.device.queue.writeBuffer(this.viewBuffer, 0, new Float32Array(skyboxView).buffer);

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
        renderPass.setVertexBuffer(0, this.vertexBuffer);
        renderPass.setIndexBuffer(this.indexBuffer, 'uint16');
        renderPass.drawIndexed(36); // Draw the cube (36 indices)
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(() => this.renderSkybox());
    }

}