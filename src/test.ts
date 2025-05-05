import fft from './shaders/fft_water.wgsl'
import { FFTRenderer } from './fft_renderer';

export class Tester {

    renderer: FFTRenderer;

    testPipeline: GPURenderPipeline;
    testBindGroup: GPUBindGroup;

    constructor(renderer: FFTRenderer) {
        this.renderer = renderer;
        
    }

    async init() {
        await this.renderer.init();
        this.renderer.testCallBack = () => this.test();
        await this.testSetup();

        this.renderer.render();
    }

    testSetup() {
        const renderLayout = this.renderer.device.createBindGroupLayout({
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

        this.testPipeline = this.renderer.device.createRenderPipeline({
            layout: this.renderer.device.createPipelineLayout({bindGroupLayouts: [renderLayout]}),
            vertex: {
                module: this.renderer.device.createShaderModule({
                    code: fft
                }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: this.renderer.device.createShaderModule({
                    code: fft
                }),
                entryPoint: 'fs_main',
                targets: [{ format: 'bgra8unorm' }]
            },
        });

        this.testBindGroup = this.renderer.device.createBindGroup({
            layout: this.testPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: this.renderer.spectrumTexture.createView(),
                },
                {
                    binding: 1,
                    resource: this.renderer.noise.createView()
                },
                {
                    binding: 2,
                    resource: this.renderer.timeSpectrumTexture.createView()
                },
                {
                    binding: 3,
                    resource: this.renderer.waveData.createView()
                },
                {
                    binding: 4,
                    resource: this.renderer.tempTextures.writable.createView()
                }
            ]
        });

    }

    async test() {
        
        const buffer = this.renderer.device.createBuffer({
            size: this.renderer.gridSize * this.renderer.gridSize * 8,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        var commandEncoder = this.renderer.device.createCommandEncoder();
        commandEncoder.copyTextureToBuffer(
            { texture: this.renderer.tempTextures.writable },
            { buffer, bytesPerRow: this.renderer.gridSize * 8},
            [this.renderer.gridSize, this.renderer.gridSize]
        );
        this.renderer.device.queue.submit([commandEncoder.finish()]);
        
        await buffer.mapAsync(GPUMapMode.READ);
        const spectrumData = new Float32Array(buffer.getMappedRange());
        console.log(spectrumData); // Inspect real/imaginary values
        spectrumData.forEach((s,i) => {
            if (isNaN(s))
                console.log(i);
        })

        

        var commandEncoder = this.renderer.device.createCommandEncoder();
        const textureView: GPUTextureView = this.renderer.context.getCurrentTexture().createView();
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

        this.renderer.device.queue.submit([commandEncoder.finish()]);

    }

}