import shader from "./shaders/water.wgsl"
import { TriangleMesh } from "./triangle_mesh";
import { mat4, vec3 } from "gl-matrix";
import { Plane } from "./plane_mesh";
import { Skybox } from "./skybox";
import Rand, { PRNG } from 'rand-seed';
import { AdaptiveWaterTessellation } from "./plane_generation";

interface LODData {
    lod0: Plane[];
    lod1: Plane[];
    lod2: Plane[];
}

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
    translationMatrices: {
        lod0: GPUBuffer,
        lod1: GPUBuffer,
        lod2: GPUBuffer
    }
    translationGroup: {
        lod0: GPUBindGroup,
        lod1: GPUBindGroup,
        lod2: GPUBindGroup
    }
    bindGroup: GPUBindGroup;
    pipeline: GPURenderPipeline;
    depthTextureView: GPUTextureView;

    meshes: {
        lod0: Plane[],
        lod1: Plane[],
        lod2: Plane[],
        totalLength: () => number
    }
    lod0Dist = 10;
    lod1Dist = 20;
    gridSize = 30;

    skybox: Skybox;
    cameraForward: vec3 = vec3.fromValues(0, 0, 0);
    cameraPos: vec3 = vec3.fromValues(0, 0.5, 0);

    paused: boolean;
    previousFrameTime: number;
    deltaTime: number;
    
    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.meshes = { lod0: [], lod1: [], lod2: [],
            totalLength: () => { return this.meshes.lod0.length + this.meshes.lod1.length + this.meshes.lod2.length }
        };
        this.previousFrameTime = performance.now() / 1000;
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

    lodsReadyForProcessing = true;
    async recalculateLODs() {
        this.lodsReadyForProcessing = false;
        const chunkAmount = 3;
        let currentChunk = 0;

        const lod0: Plane[] = [];
        const lod1: Plane[] = [];
        const lod2: Plane[] = [];

        const processChunk = () => {

            for (let i = currentChunk; i < Math.min(currentChunk + chunkAmount, this.gridSize * this.gridSize); i++) {
                let x = Math.floor(i / this.gridSize);
                let y = i % this.gridSize;
                const worldX = (x - this.gridSize / 2.0) * 2;
                const worldZ = (y - this.gridSize / 2.0) * 2;

                const distance = vec3.distance(vec3.fromValues(worldX, 0, worldZ), this.cameraPos);

                let detail = 10;
                if (distance < this.lod0Dist) {
                    detail = 256;
                    lod0.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(worldX, 0, worldZ)), this.device));
                } else if (distance < this.lod1Dist) {
                    detail = 60;
                    lod1.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(worldX, 0, worldZ)), this.device));
                } else {
                    detail = 10;
                    lod2.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(worldX, 0, worldZ)), this.device));
                }
            }

            currentChunk += chunkAmount;

            if (currentChunk < this.gridSize * this.gridSize) {
                requestAnimationFrame(processChunk);
            } else {
                // Update LOD meshes
                this.meshes.lod0 = lod0;
                this.meshes.lod1 = lod1;
                this.meshes.lod2 = lod2;
                this.lodsReadyForProcessing = true;
            }
        };

        processChunk();
    }

    async setupPipeline() {
        const maxLodCount = 1000;
        const lodBufferSize = maxLodCount * 16 * 4;
        const lod2BufferSize = maxLodCount * 100 * 16 * 4; // lod2 is the most frequent

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

        const lod0 = this.device.createBuffer({
            size: lodBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: 'lod0'
        })

        const lod1 = this.device.createBuffer({
            size: lodBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: 'lod1'
        })
        
        const lod2 = this.device.createBuffer({
            size: lod2BufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: false,
            label: 'lod2'
        })

        this.translationMatrices = { lod0: lod0, lod1: lod1, lod2: lod2}

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
            ]
        });

        const translationsLayout = this.device.createBindGroupLayout({
            entries: [
            {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: {
                        type: "read-only-storage"
                    }
            }]
        })

        const bindgourpsLod0 = this.device.createBindGroup({
            layout: translationsLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.translationMatrices.lod0
                    }
                }
            ]
        })

        const bindgourpsLod1 = this.device.createBindGroup({
            layout: translationsLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.translationMatrices.lod1
                    }
                }
            ]
        })

        const bindgourpsLod2 = this.device.createBindGroup({
            layout: translationsLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.translationMatrices.lod2
                    }
                }
            ]
        })

        this.translationGroup = { lod0: bindgourpsLod0, lod1: bindgourpsLod1, lod2: bindgourpsLod2}

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [bindGroupLayout, translationsLayout]
        });

        this.pipeline = this.device.createRenderPipeline({
            vertex: {
                module: this.device.createShaderModule({
                    code: shader
                }),
                entryPoint: "vs_main",
                buffers: [this.meshes.lod0[0].bufferLayout]
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
        for (let i = 0; i < this.gridSize * this.gridSize; i++) {
            let x = Math.floor(i / this.gridSize);
            let z = i % this.gridSize;
            x = (x - this.gridSize / 2.0) * 2; 
            z = (z - this.gridSize / 2.0) * 2;
            // Simple LOD
            var detail = 10;
            if (vec3.distance(vec3.fromValues(x, 0, z), this.cameraPos) < this.lod0Dist) {
                detail = 256;
                this.meshes.lod0.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), this.device));
            }
            else if (vec3.distance(vec3.fromValues(x, 0, z), this.cameraPos) < this.lod1Dist) {
                detail = 60;
                this.meshes.lod1.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), this.device));
            }
            else {
                detail = 10;
                this.meshes.lod2.push(new Plane(1, detail, mat4.translate(mat4.create(), mat4.create(), vec3.fromValues(x, 0, z)), this.device));
            }     
        }
        
    }

    cnt: number = 0;
    async render() {
        this.deltaTime = performance.now() / 1000 - this.previousFrameTime;
        this.previousFrameTime = performance.now() / 1000;

        this.updateCamera();
        
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

        this.writeBuffers();
        
        renderpass.setPipeline(this.pipeline);
        renderpass.setBindGroup(0, this.bindGroup);
        
        renderpass.setBindGroup(1, this.translationGroup.lod0);
        renderpass.setVertexBuffer(0, this.meshes.lod0[0].vertexBuffer);
        renderpass.setIndexBuffer(this.meshes.lod0[0].indexBuffer, "uint32");
        renderpass.drawIndexed(this.meshes.lod0[0].indexBuffer.size / Uint32Array.BYTES_PER_ELEMENT, this.meshes.lod0.length);

        renderpass.setBindGroup(1, this.translationGroup.lod1);
        renderpass.setVertexBuffer(0, this.meshes.lod1[0].vertexBuffer);
        renderpass.setIndexBuffer(this.meshes.lod1[0].indexBuffer, "uint32");
        renderpass.drawIndexed(this.meshes.lod1[0].indexBuffer.size / Uint32Array.BYTES_PER_ELEMENT, this.meshes.lod1.length);

        renderpass.setBindGroup(1, this.translationGroup.lod2);
        renderpass.setVertexBuffer(0, this.meshes.lod2[0].vertexBuffer);
        renderpass.setIndexBuffer(this.meshes.lod2[0].indexBuffer, "uint32");
        renderpass.drawIndexed(this.meshes.lod2[0].indexBuffer.size / Uint32Array.BYTES_PER_ELEMENT, this.meshes.lod2.length);
            
        
        renderpass.end();
        this.device.queue.submit([commandEncoder.finish()]);

        if (this.lodsReadyForProcessing) {
            this.recalculateLODs();
        }

        if (this.paused) return;

        requestAnimationFrame(() => this.render());

    }


    updateCamera() {
        const speed = 0.1;
        const time = performance.now() / 1000 * speed;
        const bobSpeed = 4.0;

        const a = 10; 
        const b = 10; 

        const currentX = a * Math.cos(time);
        const currentZ = b * Math.sin(2 * time) / 2;
        const currentY = 1.5 + 0.2 * Math.sin(time * bobSpeed); 

        // Calculate next position (slightly ahead in time)
        const nextTime = time + 0.01; // Small time step for forward direction
        const nextX = a * Math.cos(nextTime);
        const nextZ = b * Math.sin(2 * nextTime) / 2;
        const nextY =  1.5 + 0.2 * Math.sin(nextTime * bobSpeed);

        
        // Update camera position
        this.cameraPos[0] = currentX;
        this.cameraPos[1] = currentY;
        this.cameraPos[2] = currentZ;

        // Calculate cameraForward vector (direction of motion)
        const forward = vec3.fromValues(nextX - currentX, nextY - currentY, nextZ - currentZ);
        vec3.normalize(this.cameraForward, forward);
    
    }

    writeBuffers() {
        const time = performance.now() / 1000;
        
        const projection = mat4.create();
        mat4.perspective(projection, Math.PI / 4, this.canvas.width / this.canvas.height, 0.1, 30);
   
        const view = mat4.create();
        const cameraPos = this.cameraPos;
        const forwardWorld = vec3.create();
        vec3.add(forwardWorld, this.cameraForward, cameraPos);
        mat4.lookAt(view, cameraPos, forwardWorld, [0, 1, 0]);

        const matrixData0 = new Float32Array(16 * this.meshes.lod0.length);
        const matrixData1 = new Float32Array(16 * this.meshes.lod1.length);
        const matrixData2 = new Float32Array(16 * this.meshes.lod2.length);
        this.meshes.lod0.forEach( (plane, idx) => {
            matrixData0.set(plane.translationMatrix, idx * 16);
        });
        this.meshes.lod1.forEach( (plane, idx) => {
            matrixData1.set(plane.translationMatrix, idx * 16);
        });
        this.meshes.lod2.forEach( (plane, idx) => {
            matrixData2.set(plane.translationMatrix, idx * 16);
        });


        this.device.queue.writeBuffer(this.translationMatrices.lod0, 0, matrixData0.buffer);
        this.device.queue.writeBuffer(this.translationMatrices.lod1, 0, matrixData1.buffer);
        this.device.queue.writeBuffer(this.translationMatrices.lod2, 0, matrixData2.buffer);
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
}