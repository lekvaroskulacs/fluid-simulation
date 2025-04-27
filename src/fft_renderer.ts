import jonswap from './shaders/generate_jonswap.wgsl'
import fft from './shaders/fft_water.wgsl'
import spectrum from './shaders/time_spectrum.wgsl'
import fft_horizontal from './shaders/fft_horizontal.wgsl'
import fft_vertical from './shaders/fft_vertical.wgsl'
import fft_scale from './shaders/fft_scale.wgsl'
import { isBuffer } from 'node:util'

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

    gridSize = 256; // Grid size

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
    }

    async init() {
        await this.setupDevice();
        await this.setupPipeline();
        await this.computeSpectrum();
        this.fftSetup();
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
                GPUTextureUsage.COPY_SRC,
        })

        this.waveData = this.device.createTexture({
            size: [this.gridSize, this.gridSize],
            format: 'rgba32float',
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
                    visibility: GPUShaderStage.VERTEX,
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
                {
                    binding: 3,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rgba32float',
                        access: 'read-only',
                        viewDimension: '2d'
                    }
                },
                {
                    binding: 4,
                    visibility: GPUShaderStage.FRAGMENT,
                    storageTexture: {
                        format: 'rg32float',
                        access: 'read-only',
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

        
        this.jonswapPipeline = this.device.createComputePipeline({
            layout: this.device.createPipelineLayout({bindGroupLayouts: [jonswapLayout]}),
            compute: {
                module: this.device.createShaderModule({
                    code: jonswap
                }),
                entryPoint: "initial_spectrum",
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
        const spectrumConjugateView = this.spectrumConjugateTexture.createView();
        const gaussianTexture = await this.createGaussianTexture();
        const spectrumView = this.timeSpectrumTexture.createView();
        const waveDataView = this.waveData.createView();

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
        this.fftCompute();
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
            { texture: this.timeSpectrumTexture },
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

    fftPipelines: GPUComputePipeline[];
    fftBindGroups: GPUBindGroup[];
    propertiesBuffer: GPUBuffer;

    fftSetup() {
        this.fftPipelines = [];
        this.fftBindGroups = [];
        const N = 256;
        const ifftInputTexture = this.device.createTexture({
            size: [N, N],
            format: "rg32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        const ifftTempTexture1 = this.device.createTexture({
            size: [N, N],
            format: "rg32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, 
        });

        const ifftTempTexture2 = this.device.createTexture({
            size: [N, N],
            format: "rg32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING, 
        });

        const ifftOutputTexture = this.device.createTexture({
            size: [N, N],
            format: "r32float",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
        });

        const twiddleBuffer = this.device.createBuffer({
            size: N * 2 * 4, // N complex numbers (real, imag)
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
          });
          
        const twiddleData = new Float32Array(twiddleBuffer.getMappedRange());
        for (let k = 0; k < N; k++) {
          const angle = (2 * Math.PI * k) / N;
          twiddleData[2 * k] = Math.cos(angle);
          twiddleData[2 * k + 1] = Math.sin(angle);
        }
        twiddleBuffer.unmap();

        this.propertiesBuffer = this.device.createBuffer({
            size: 4 * 2,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const horizontalPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: fft_horizontal
                }),
                entryPoint: 'cs_main',
            },
        });
          
        const verticalPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: fft_horizontal
                }),
                entryPoint: 'cs_main',
            },
        });
          
        const scalingPipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: this.device.createShaderModule({
                    code: fft_scale
                }),
                entryPoint: 'cs_main',
            },
        });

        const horizontalBindGroup = this.device.createBindGroup({
            layout: horizontalPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: ifftInputTexture.createView() }, 
                { binding: 1, resource: ifftTempTexture1.createView() }, 
                { binding: 2, resource: { buffer: twiddleBuffer } }, 
                { binding: 3, resource: { buffer: this.propertiesBuffer } }
            ],
        });
        
        const verticalBindGroup = this.device.createBindGroup({
            layout: verticalPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: ifftTempTexture1.createView() }, 
                { binding: 1, resource: ifftTempTexture2.createView() }, 
                { binding: 2, resource: { buffer: twiddleBuffer } },
                { binding: 3, resource: { buffer: this.propertiesBuffer } }
            ],
        });
        
        const scalingBindGroup = this.device.createBindGroup({
            layout: scalingPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: ifftTempTexture2.createView() }, 
                { binding: 1, resource: ifftOutputTexture.createView() },
            ],
        });

        this.fftPipelines.push(horizontalPipeline, verticalPipeline, scalingPipeline);
        this.fftBindGroups.push(horizontalBindGroup, verticalBindGroup, scalingBindGroup);
    }

    fftCompute() {
        let N = 256;
        let pingPong = false;
        
        //submitting a commandbuffer 17 times in a frame is crazy, but i dont know what other choice i have
        //because the properties uniform needs to be updated for every iteration of the fft
        for (let i = 0; i < Math.log2(N); i++) {
            const commandEncoder = this.device.createCommandEncoder();
            this.device.queue.writeBuffer(this.propertiesBuffer, 0, new Float32Array([i]).buffer);

            const horizontalPass = commandEncoder.beginComputePass();
            horizontalPass.setPipeline(this.fftPipelines[0]);
            horizontalPass.setBindGroup(0, this.fftBindGroups[0]);
            horizontalPass.dispatchWorkgroups(Math.ceil(N / 64), N); // 64 threads per row
            horizontalPass.end();
            this.device.queue.submit([commandEncoder.finish()]);

        }

        for (let i = 0; i < Math.log2(N); i++) {
            const commandEncoder = this.device.createCommandEncoder();
            this.device.queue.writeBuffer(this.propertiesBuffer, 0, new Float32Array([i]).buffer);

            const verticalPass = commandEncoder.beginComputePass();
            verticalPass.setPipeline(this.fftPipelines[1]);
            verticalPass.setBindGroup(0, this.fftBindGroups[1]);
            verticalPass.dispatchWorkgroups(N, Math.ceil(N / 64)); // 64 threads per column
            verticalPass.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }
        const commandEncoder = this.device.createCommandEncoder();
        // Scaling pass (extract real part)
        const scalingPass = commandEncoder.beginComputePass();
        scalingPass.setPipeline(this.fftPipelines[2]);
        scalingPass.setBindGroup(0, this.fftBindGroups[2]);
        scalingPass.dispatchWorkgroups(Math.ceil(N / 8), Math.ceil(N / 8)); // 8x8 workgroup
        scalingPass.end();

        this.device.queue.submit([commandEncoder.finish()]);
    }
}