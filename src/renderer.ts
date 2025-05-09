import shader from "./shaders/water.wgsl"
import { TriangleMesh } from "./triangle_mesh";
import { mat4, vec3 } from "gl-matrix";
import { Plane } from "./plane_mesh";
import { Skybox } from "./skybox";
import Rand, { PRNG } from 'rand-seed';
import { AdaptiveWaterTessellation } from "./plane_generation";

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
    cameraPos: vec3 = vec3.fromValues(4, 0.5, -5);

    paused: boolean;
    previousFrameTime: number;
    deltaTime: number;
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.meshes = [];
        this.previousFrameTime = performance.now() / 1000000;
        this.deltaTime = 0;

        this.paused = true;

        document.addEventListener("keydown", e => {
            if (e.key == "p") {
                this.paused = !this.paused;
                this.skybox.pauseSwitch();
                if (!this.paused) {
                    this.skybox.renderSkybox();
                    this.render();
                }
            }
        })
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
        let yaw = -Math.PI / 4; // Rotation around the Z axis (left/right)
        let pitch = 0; // Rotation around the X axis (up/down)
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
            if (event.key === "e")
                vec3.add(this.cameraPos, this.cameraPos, vec3.scale(vec3.create(), this.cameraForward, 0.01));
        });
    }

    async setupDevice() {
        this.adapter = <GPUAdapter> await navigator.gpu?.requestAdapter();
        this.device = await this.adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: 2147483644
            }
        });
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
/*
        const debugLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {format: 'rgba32float'}
                }
            ]
        })
*/
        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout,]
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
                topology: 'triangle-strip',
                stripIndexFormat: 'uint32'
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
        const gridSize = 6;
        for (let i = 0; i < gridSize * gridSize; i++) {
            let x = Math.floor(i / gridSize);
            let z = i % gridSize;
            x = x - gridSize / 2.0;
            z = z - gridSize / 2.0;
            this.meshes.push(new Plane(1, 256, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(2 * x, 0, 2 * z)), this.device));
        }
        
    }

    async render() {
        this.deltaTime = performance.now() / 1000000 - this.previousFrameTime;
        
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
/*
        const debugTex = this.device.createTexture({
            format: 'rgba32float',
            size: [256, 256],
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });

        const debugGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: debugTex.createView() }
            ]
        });
        renderpass.setBindGroup(1, debugGroup);
        */
        this.writeBuffers();
        

        renderpass.setVertexBuffer(0, this.meshes[0].vertexBuffer);
        renderpass.setIndexBuffer(this.meshes[0].indexBuffer, "uint32");
        renderpass.drawIndexed(this.meshes[0].indexBuffer.size / Uint32Array.BYTES_PER_ELEMENT, this.meshes.length);
            
        
        renderpass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        if (this.paused) return;

        requestAnimationFrame(() => this.render());

    }


    writeBuffers() {
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 100);

        //vec3.add(this.cameraPos, this.cameraPos, vec3.scale(vec3.create(), this.cameraForward, this.deltaTime));
        //vec3.rotateY(this.cameraForward, this.cameraForward, [0, 0, 0], this.deltaTime * 0.1);
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
        const rand = new Rand("93b5facc");
        for (let i = 0; i <= data.length; i++) {
            let idx = i * 2;

            let theta  = 2 * Math.PI * rand.next();
            let R   = Math.sqrt(-2 * Math.log(rand.next()));
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
/*
    async render() {
        const renderPipeline = this.device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: this.device.createShaderModule({
            code: `
        struct VertexIn {
        @location(0) position: vec3<f32>,
        @location(1) normal: vec3<f32>,
        @location(2) uv: vec2<f32>,
        };

        struct VertexOut {
        @builtin(position) position: vec4<f32>,
        @location(0) normal: vec3<f32>,
        @location(1) worldPos: vec3<f32>,
        };

        @group(0) @binding(0) var<uniform> viewProj: mat4x4<f32>;

        @vertex
        fn main(input: VertexIn) -> VertexOut {
        var output: VertexOut;
        output.position = viewProj * vec4f(input.position, 1.0);
        output.normal = input.normal;
        output.worldPos = input.position;
        return output;
        }
            `
            }),
            entryPoint: 'main',
            buffers: [
            {
                arrayStride: 8 * 4,
                attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x3' }, // position
                { shaderLocation: 1, offset: 12, format: 'float32x3' }, // normal
                { shaderLocation: 2, offset: 24, format: 'float32x2' }, // uv
                ]
            }
            ]
        },
        fragment: {
            module: this.device.createShaderModule({
            code: `
        @fragment
        fn main(@location(0) normal: vec3<f32>, @location(1) worldPos: vec3<f32>) -> @location(0) vec4<f32> {
        let lightDir = normalize(vec3<f32>(-0.4, 0.5, 0.3));
        let lighting = max(dot(normal, lightDir), 0.0);
        return vec4<f32>(0.0, 0.4, 0.8, 1.0);
        }
            `
            }),
            entryPoint: 'main',
            targets: [{ format: 'bgra8unorm' }]
        },
        primitive: {
            topology: 'triangle-strip',
            stripIndexFormat: 'uint32',
        },
        depthStencil: {
            format: 'depth24plus',
            depthWriteEnabled: true,
            depthCompare: 'less'
        }
        });
        
        const viewMatrix = mat4.create();
        const projectionMatrix = mat4.create();
        const viewProjMatrix = mat4.create();

        // Set up perspective projection (field of view, aspect ratio, near, far)
        mat4.perspective(projectionMatrix, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 500.0);

        // Set up the view (camera) matrix
        const cameraPosition = vec3.fromValues(30, 20, 30);
        const target = vec3.fromValues(8, 0, 8);
        const up = vec3.fromValues(0, 1, 0);
        mat4.lookAt(viewMatrix, cameraPosition, target, up);

        // Multiply projection * view
        mat4.multiply(viewProjMatrix, projectionMatrix, viewMatrix);

        // Create GPU buffer
        const viewProjBuffer = this.device.createBuffer({
        size: 64, // 4x4 matrix = 16 floats = 64 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Upload data
        this.device.queue.writeBuffer(viewProjBuffer, 0, viewProjMatrix as Float32Array);

        const bindGroupLayout = renderPipeline.getBindGroupLayout(0);
        const viewProjBindGroup = this.device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            {
            binding: 0,
            resource: {
                buffer: viewProjBuffer
            }
            }
        ]
        });

        const tessellation = new AdaptiveWaterTessellation(this.device, this.cameraPos)

        const patchData = tessellation.getPatchInfo();
        const vertexBuffer = tessellation.getVertexBuffer();
        const indexBuffers = tessellation.getIndexBuffers();

        const commandEncoder = this.device.createCommandEncoder();
        const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: this.context.getCurrentTexture().createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
        }],
        depthStencilAttachment: {
            view: this.depthTextureView,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
            depthClearValue: 1.0,
        }
        });

        pass.setPipeline(renderPipeline);
        pass.setVertexBuffer(0, vertexBuffer);
        pass.setBindGroup(0, viewProjBindGroup); // assuming you created a uniform bind group for view-projection

        let vertexBaseOffset = 0;
        for (let i = 0; i < patchData.length; i++) {
        const lod = patchData[i].lod;
        const res = [32, 16, 8][lod];
        const numIndices = (res + 1) * 2 * res + (res - 1); // include primitive restarts
        pass.setIndexBuffer(indexBuffers[lod], 'uint32');
        pass.drawIndexed(numIndices, 1, 0, vertexBaseOffset / 32, 0);
        vertexBaseOffset += (res + 1) * (res + 1) * 32; // 32 bytes per vertex
        }

        await vertexBuffer.mapAsync(GPUMapMode.READ);
        console.log(vertexBuffer.getMappedRange());
        pass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }*/
}