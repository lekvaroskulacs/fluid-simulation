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
    translationMatrices: GPUBuffer;

    bindGroup: GPUBindGroup;
    pipeline: GPURenderPipeline;
    depthTextureView: GPUTextureView;

    meshes: Plane[];

    skybox: Skybox;
    cameraForward: vec3 = vec3.fromValues(0, 0, 0);
    cameraPos: vec3 = vec3.fromValues(-6, 3, 0);
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.meshes = [];
    }

    async init() {
        this.cameraInputs();
        await this.setupDevice();
        this.skybox = new Skybox(this.device, this.format, this.context, this.canvas);
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
            
            if (this.cameraForward) this.skybox.cameraForward = this.cameraForward;
            
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === "w")
                vec3.add(this.cameraPos, this.cameraPos, vec3.scale(vec3.create(), this.cameraForward, 0.1));
            if (event.key === "s")
                vec3.add(this.cameraPos, this.cameraPos, vec3.scale(vec3.create(), this.cameraForward, -0.1));
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
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.translationMatrices = this.device.createBuffer({
            size: this.meshes.length * 16 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false
        })

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
                {
                    binding: 4,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    texture: {
                        viewDimension: "cube"
                    }
                },
                {
                    binding: 5,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    sampler: {}
                },
                {
                    binding: 6,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only'
                    }
                },
                {
                    binding: 7,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "read-only-storage"
                    }
                }
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
                { 
                    binding: 4, 
                    resource: (await this.skybox.createSkyboxTexture()).createView({ dimension: 'cube' }) 
                },
                { 
                    binding: 5, 
                    resource: this.device.createSampler() 
                },
                {
                    binding: 6,
                    resource: (await this.createGaussianTexture()).createView()
                },
                {
                    binding: 7,
                    resource: {
                        buffer: this.translationMatrices
                    }
                }
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
                buffers: [this.meshes[0].bufferLayout]
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
                stripIndexFormat: "uint32"
            },

            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },

            layout: pipelineLayout
        });

        const depthTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.depthTextureView = depthTexture.createView();

    }

    setupAssets() {
        const gridSize = 4;
        for (let i = 0; i < gridSize * gridSize; i++) {
            let x = Math.floor(i / gridSize);
            let z = i % gridSize;
            x = x - gridSize / 2.0;
            z = z - gridSize / 2.0;
            this.meshes.push(new Plane(1, 200, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), this.device));
        }
        
    }

    render() {
        if (!this)
            console.log("this is null");
        

        
        const commandEncoder: GPUCommandEncoder = this.device.createCommandEncoder();
        const textureView: GPUTextureView = this.context.getCurrentTexture().createView();
        const renderpass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: {r: 133.0/255.0, g: 211.0/255.0, b: 241.0/255.0, a: 0},
                loadOp: "load",
                storeOp: "store"
            }],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });

        renderpass.setPipeline(this.pipeline);
        renderpass.setBindGroup(0, this.bindGroup);
        this.writeBuffers();
        

        renderpass.setVertexBuffer(0, this.meshes[0].vertexBuffer);
        renderpass.setIndexBuffer(this.meshes[0].indexBuffer, "uint32");
        renderpass.drawIndexed(this.meshes[0].indexBuffer.size / Uint32Array.BYTES_PER_ELEMENT, this.meshes.length);
            
        
        renderpass.end();
        this.device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(() => this.render());
    }


    writeBuffers() {
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 10);

        const view = mat4.create();
        const cameraPos = this.cameraPos;
        const forwardWorld = vec3.create();
        vec3.add(forwardWorld, this.cameraForward, cameraPos);
        mat4.lookAt(view, cameraPos, forwardWorld, [0, 1, 0]);

        const time = performance.now() / 1000;

        const matrixData = new Float32Array(16 * this.meshes.length);
        this.meshes.forEach( (plane, idx) => {
            matrixData.set(plane.translationMatrix, idx * 16);
        });

        this.device.queue.writeBuffer(this.translationMatrices, 0, matrixData.buffer);
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
        const _horizontalDisplacement: number = +(localStorage.getItem("horizontalDisplacement") ?? 1);
        const _maxWaves: number = +(localStorage.getItem("maxWaves") ?? 1);

        const _sunPosition: number = +(localStorage.getItem("sunPosition") ?? 1);

        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 0, new Float32Array([_amplitude]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 4, new Float32Array([_frequency]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 8, new Float32Array([_amplitudeMult]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 12, new Float32Array([_frequencyMult]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 16, new Float32Array([_basePhase]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 20, new Float32Array([_horizontalDisplacement]));
        this.device.queue.writeBuffer(this.waveOptions_uniformBuffer, 24, new Float32Array([_maxWaves]));

        this.device.queue.writeBuffer(this.sceneOptions_uniformBuffer, 16, new Float32Array([_sunPosition]));
        const cam = [this.cameraPos[0], this.cameraPos[1], this.cameraPos[2], 1];
        //console.log(`camera buffer written: ${cam}`);
        this.device.queue.writeBuffer(this.sceneOptions_uniformBuffer, 0, new Float32Array(cam));
    }

    async createGaussianTexture() {
        const data = new Float32Array(256 * 256 * 2);
        for (let i = 0; i <= data.length; i++) {
            let idx = i * 2;

            let theta  = 2 * Math.PI * Math.random();
            let R   = Math.sqrt(-2 * Math.log(Math.random()));
            let x   = R * Math.cos(theta);
            let y   = R * Math.sin(theta);
            
            // Sigmoid
            // x = 1 / (1 + Math.exp(-x * 0.5));
            // y = 1 / (1 + Math.exp(-y * 0.5));

            data[idx] = x;
            data[idx + 1] = y;
        }

        const texture = this.device.createTexture({
            size : [256, 256],
            format: 'rg32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
        });

        this.device.queue.writeTexture(
            {texture}, 
            data, 
            {bytesPerRow: 256 * 4 * 2}, 
            {width: 256, height: 256}
        );
        return texture;
    }

    
}