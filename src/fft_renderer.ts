import jonswap from './shaders/generate_jonswap.wgsl'
import fft from './shaders/fft_water.wgsl'
import spectrum from './shaders/time_spectrum.wgsl'
import { create } from 'domain';
import { buffer } from 'stream/consumers';

export class FFTRenderer {

    canvas: HTMLCanvasElement

    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;

    jonswapBindGroup: GPUBindGroup;
    jonswapPipeline: GPUComputePipeline;
    spectrumBindGroup: GPUBindGroup;
    spectrumPipeline: GPUComputePipeline;
    testBindGroup: GPUBindGroup;
    testPipeline: GPURenderPipeline;

    timeBuffer: GPUBuffer;

    spectrumTexture: GPUTexture;
    timeSpectrumTexture: GPUTexture;

    gridSize = 256; // Grid size

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        await this.setupDevice();
        await this.setupPipeline();
        await this.computeSpectrum();
        await this.render();
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
        
        this.spectrumTexture = this.device.createTexture({
            size: [this.gridSize, this.gridSize],
            format: "rg32float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.timeSpectrumTexture = this.device.createTexture({
            size: [this.gridSize, this.gridSize],
            format: 'rg32float',
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        })

        const jonswapLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'write-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d',
                    }
                },
            ]
        });

        const renderLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d',
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d',
                    }
                },
            ]
        });

        const spectrumLayout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d'
                    }
                },

                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'write-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: 'uniform'
                    }
                }
            ]
        })

        
        this.jonswapPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({bindGroupLayouts: [jonswapLayout]}),
            compute: {
                module: this.device.createShaderModule({
                    code: jonswap
                }),
                entryPoint: "cs_main",
            },
        });
        
        this.testPipeline = this.device.createRenderPipeline({
            layout: this.device.createPipelineLayout({bindGroupLayouts: [renderLayout]}),
            vertex: {
                module: this.device.createShaderModule({
                    code: fft
                }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.device.createShaderModule({
                    code: fft
                }),
                entryPoint: 'fs_main',
                targets: [{ format: this.format }]
            },
        });
        
        this.spectrumPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({bindGroupLayouts: [spectrumLayout]}),
            compute: {
                module: this.device.createShaderModule({
                    code: spectrum
                }),
                entryPoint: "cs_main"
            }
        });

        const textureView = this.spectrumTexture.createView();
        const gaussianTexture = await this.createGaussianTexture();
        const spectrumView = this.timeSpectrumTexture.createView();

        this.jonswapBindGroup = this.device.createBindGroup({
            layout: this.jonswapPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: textureView,
                },
                {
                    binding: 1,
                    resource: gaussianTexture.createView()
                },
            ],
        });
        this.testBindGroup = this.device.createBindGroup({
            layout: this.testPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: textureView,
                },
                {
                    binding: 1,
                    resource: gaussianTexture.createView()
                },
                {
                    binding: 2,
                    resource: spectrumView
                }
            ]
        })

        this.timeBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.spectrumBindGroup = this.device.createBindGroup({
            layout: this.spectrumPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: textureView
                },
                {
                    binding: 1,
                    resource: spectrumView
                }, 
                {
                    binding: 2,
                    resource: {
                        buffer: this.timeBuffer
                    }
                }
            ]
        })
    }

    async computeSpectrum() {

        const commandEncoder = this.device.createCommandEncoder();
        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.jonswapPipeline);
        computePass.setBindGroup(0, this.jonswapBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.gridSize / 8), Math.ceil(this.gridSize / 8));
        computePass.end();


        this.device.queue.submit([commandEncoder.finish()]);
        
    }

    async createGaussianTexture() {
        const data = new Float32Array(this.gridSize * this.gridSize * 2);
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
            size : [this.gridSize, this.gridSize],
            format: 'rg32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
        });

        this.device.queue.writeTexture(
            {texture}, 
            data, 
            {bytesPerRow: this.gridSize * 4 * 2}, 
            {width: this.gridSize, height: this.gridSize}
        );
        return texture;
    }

    async computeSpectrumEvolution() {
        const time = performance.now() / 1000;
        this.device.queue.writeBuffer(this.timeBuffer, 0, new Float32Array([time]).buffer);

        const commandEncoder = this.device.createCommandEncoder();

        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.spectrumPipeline);
        computePass.setBindGroup(0, this.spectrumBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.gridSize / 8), Math.ceil(this.gridSize / 8));
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }

    async render() {
        this.computeSpectrumEvolution();
        this.test();

        //requestAnimationFrame(() => this.render());
    }

    async test() {
        /*
        const buffer = this.device.createBuffer({
            size: this.gridSize * this.gridSize * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture: this.spectrumTexture },
            { buffer, bytesPerRow: this.gridSize * 8 },
            [this.gridSize, this.gridSize]
        );
        this.device.queue.submit([commandEncoder.finish()]);
        
        await buffer.mapAsync(GPUMapMode.READ);
        const spectrumData = new Float32Array(buffer.getMappedRange());
        console.log(spectrumData); // Inspect real/imaginary values
        */

        var commandEncoder = this.device.createCommandEncoder();
        const textureView: GPUTextureView = this.context.getCurrentTexture().createView();
        const renderPass: GPURenderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 133.0 / 255.0, g: 211.0 / 255.0, b: 241.0 / 255.0, a: 1 },
                loadOp: "clear",
                storeOp: "store"
            }],
        });

        renderPass.setPipeline(this.testPipeline);
        renderPass.setBindGroup(0, this.testBindGroup);
        renderPass.draw(3);
        renderPass.end();

        this.device.queue.submit([commandEncoder.finish()]);

    }
}