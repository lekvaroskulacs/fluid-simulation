import jonswap from './shaders/generate_jonswap.wgsl'
import fft from './shaders/fft_water.wgsl'
import spectrum from './shaders/time_spectrum.wgsl'
import fft_horizontal from './shaders/fft_horizontal.wgsl'
import fft_vertical from './shaders/fft_vertical.wgsl'
import fft_scale from './shaders/fft_scale.wgsl'
import { isBuffer } from 'node:util'
import { IFFT2D } from './ifft_2d'
import { Tester } from './test'

export class FFTRenderer {

    canvas: HTMLCanvasElement

    adapter: GPUAdapter;
    device: GPUDevice;
    context: GPUCanvasContext;
    format: GPUTextureFormat;

    jonswapBindGroup: GPUBindGroup;
    jonswapPipeline: GPUComputePipeline;
    conjugatePipeline: GPUComputePipeline;
    spectrumBindGroup: GPUBindGroup;
    spectrumPipeline: GPUComputePipeline;
    testBindGroup: GPUBindGroup;
    testPipeline: GPURenderPipeline;

    timeBuffer: GPUBuffer;

    spectrumTexture: GPUTexture;
    spectrumConjugateTexture: GPUTexture;
    timeSpectrumTexture: GPUTexture;
    waveData: GPUTexture;
    noise: GPUTexture;

    gridSize = 256; // Grid size
    
    testViews: GPUTextureView[];
    testCallBack: () => any;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        await this.setupDevice();
        await this.setupPipeline();
        await this.computeSpectrum();
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

    async createBuffersAndTextures() {
        this.timeBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.spectrumTexture = this.device.createTexture({
            size: [this.gridSize, this.gridSize],
            format: "rg32float",
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.spectrumConjugateTexture = this.device.createTexture({
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
                GPUTextureUsage.COPY_SRC        |
                GPUTextureUsage.COPY_DST,
        });

        this.waveData = this.device.createTexture({
            size: [this.gridSize, this.gridSize],
            format: 'rgba32float',
            usage:
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC,
        });

        this.noise = await this.createGaussianTexture();
        this.tempTextures = IFFT2D.createTexturePair(this.device, this.gridSize);
    }

    async setupPipeline() {
        await this.createBuffersAndTextures();

        // Bindgroup layouts
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
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rgba32float',
                        access: 'write-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'write-only',
                        viewDimension: '2d'
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
                },
                {
                    binding: 3,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rgba32float',
                        access: 'read-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.COMPUTE,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
                        viewDimension: '2d'
                    }
                },
            ]
        })

        // Pipelines
        this.jonswapPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({bindGroupLayouts: [jonswapLayout]}),
            compute: {
                module: this.device.createShaderModule({
                    code: jonswap
                }),
                entryPoint: "initial_spectrum",
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

        // Texture views for testing
        const textureView = this.spectrumTexture.createView();
        const spectrumConjugateView = this.spectrumConjugateTexture.createView();
        const gaussianTexture = this.noise.createView();
        const spectrumView = this.timeSpectrumTexture.createView();
        const waveDataView = this.waveData.createView();

        // Bind groups
        this.jonswapBindGroup = this.device.createBindGroup({
            layout: this.jonswapPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: textureView,
                },
                {
                    binding: 1,
                    resource: gaussianTexture
                },
                {
                    binding: 2,
                    resource: waveDataView
                },
                {
                    binding: 3,
                    resource: spectrumConjugateView
                }
            ],
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
                },
                {
                    binding: 3,
                    resource: waveDataView
                },
                {
                    binding: 4,
                    resource: spectrumConjugateView
                }
            ]
        })

    }

    async createGaussianTexture() {
        const data = new Float32Array(this.gridSize * this.gridSize * 2);
        for (let i = 0; i <= data.length; i++) {
            let idx = i * 2;
            
            let theta  = 2 * Math.PI * Math.random();
            let R   = Math.sqrt(-2 * Math.log(Math.random()));
            let x   = R * Math.cos(theta);
            let y   = R * Math.sin(theta);

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
    
    async computeSpectrum() {
        const commandEncoder = this.device.createCommandEncoder();
        
        const computePass = commandEncoder.beginComputePass();
        computePass.setPipeline(this.jonswapPipeline);
        computePass.setBindGroup(0, this.jonswapBindGroup);
        computePass.dispatchWorkgroups(Math.ceil(this.gridSize / 8), Math.ceil(this.gridSize / 8));
        computePass.end();

        this.device.queue.submit([commandEncoder.finish()]);
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

    tempTextures: { readable: GPUTexture, writable: GPUTexture };
    async render() {
        this.computeSpectrumEvolution();
        
        const ifft = new IFFT2D(this.device, this.gridSize);
        const inputTextures = IFFT2D.createTexturePair(this.device, this.gridSize);
        //this.tempTextures = IFFT2D.createTexturePair(this.device, this.gridSize);

        const cmd = this.device.createCommandEncoder();
        cmd.copyTextureToTexture(
            { texture: this.spectrumTexture },
            { texture: inputTextures.readable },
            [this.gridSize, this.gridSize]
        );
        cmd.copyTextureToTexture(
            { texture: this.spectrumTexture },
            { texture: inputTextures.writable },
            [this.gridSize, this.gridSize]
        );
        this.device.queue.submit([cmd.finish()]);

        const commandEncoder = await ifft.runIFFT2D(inputTextures.readable, inputTextures.writable, this.tempTextures.readable, this.tempTextures.writable);
        this.device.queue.submit([commandEncoder.finish()]);
        
        this.testCallBack();
        //requestAnimationFrame(() => this.render());
    }


}