import { off } from 'node:process';
import ifftShaderCode from './shaders/ifft.wgsl'
import { util } from 'webpack';

interface TwiddleFactor {
    twiddleX: number;
    twiddleY: number;
    index1: number;
    index2: number;
}

export class IFFT2D {
    private device: GPUDevice;
    private pipelines: {
        horizontal: GPUComputePipeline;
        vertical: GPUComputePipeline;
        scale: GPUComputePipeline;
    };
    private precomputedData: GPUBuffer;
    private uniforms: GPUBuffer;
    private bindGroupLayout: GPUBindGroupLayout;
    private size: number;
    private localWorkGroupsX: number = 8;
    private localWorkGroupsY: number = 8;

    constructor(device: GPUDevice, size: number) {
        this.device = device;
        this.size = size;
        
        // Create resources
        this.precomputeData();
        //this.createUniformBuffer();
        //this.precomputeTwiddleFactors(size);
        this.createPipelines();
    }
    
    private complexExp(a: [number, number]): [number, number] {
      return [Math.cos(a[1]) * Math.exp(a[0]), Math.sin(a[1]) * Math.exp(a[0])];
    }
/*
    private precomputeTwiddleFactors(size: number) {
        const buffer = new Float32Array(size * 2 * 4);
        
        // We'll simulate the [1, 8, 1] thread dispatch with id.x always 0
        const idX = 0;
        
        for (let idY = 0; idY < size; idY++) {
          const b = size >> (idX + 1);  // For size=8, b=4
          const mult: [number, number] = [0, 2 * Math.PI / size];
          
          // Calculate input indices
          const i = (2 * b * Math.floor(idY / b) + (idY % b)) % size;
          
          // Calculate twiddle factor
          const exponent: [number, number] = [
            -mult[0] * (Math.floor(idY / b) * b),
            -mult[1] * (Math.floor(idY / b) * b)
          ];
          const [twiddleX, twiddleY] = this.complexExp(exponent);
          
          // First write at original position
          buffer[idY * 4] = twiddleX;
          buffer[idY * 4 + 1] = twiddleY;
          buffer[idY * 4 + 2] = i;
          buffer[idY * 4 + 3] = i + b;
          
          // Second write at offset position (with negated twiddle)
          buffer[(idY + size) * 4] = -twiddleX;
          buffer[(idY + size) * 4 + 1] = -twiddleY;
          buffer[(idY + size) * 4 + 2] = i;
          buffer[(idY + size) * 4 + 3] = i + b;
        }
        console.log(buffer);
    
        this.precomputedData = this.device.createBuffer({
            size: buffer.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.precomputedData.getMappedRange()).set(buffer);
        this.precomputedData.unmap();
      }
        */
    
    // twiddle calculation looks wrong
    private precomputeData() {
        const logSize = Math.log2(this.size);
        const data = new Float32Array(this.size * logSize * 4);
        
        for (let step = 0; step < logSize; step++) {
            const b = this.size >> (step + 1);
            const mult = 2 * Math.PI / this.size;
            
            for (let y = 0; y < this.size; y++) {
                const i = (2 * b * Math.floor(y / b) + y % b) % this.size;
                const angle = mult * (Math.floor(y / b) * b);
                const twiddleX = Math.cos(angle);
                const twiddleY = -Math.sin(angle); // Negative for IFFT
                
                const idx = (step * this.size + y) * 4;
                data[idx] = twiddleX;
                data[idx + 1] = twiddleY;
                data[idx + 2] = i;       // First index
                data[idx + 3] = i + b;    // Second index
            }
        }
        //console.log(data);
        this.precomputedData = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.precomputedData.getMappedRange()).set(data);
        this.precomputedData.unmap();
    }
/*
    private createUniformBuffer() {
        const logSize = Math.log2(this.size);
        this.uniforms = this.device.createBuffer({
            size: 256 * logSize * 4, // step (u32) + pingPong (u32) + size (u32) + mode (u32)
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        const uniformData = new Uint32Array(256 * logSize); // 256 bytes per block, 4 bytes per element
        for (let i = 0; i < logSize; i++) {
            const baseIndex = 256 * i; // Offset in 32-bit words
            uniformData[baseIndex + 0] = i; // step
            uniformData[baseIndex + 1] = i % 2 == 0 ? 1 : 0;
            uniformData[baseIndex + 2] = this.size;
            uniformData[baseIndex + 3] = 0;
        }
        
        for(let i = 0; i < uniformData.length; ++i) {
            console.log(i + " " + uniformData[i]);
        }

        this.device.queue.writeBuffer(
            this.uniforms,
            0,
            uniformData.buffer,
            uniformData.byteOffset,
            uniformData.byteLength
        );
    }
        */
    private createUniformBuffer(step: number, pingPong: boolean, mode: number): GPUBuffer {
        const uniformData = new Uint32Array(4);
        uniformData[0] = step;
        uniformData[1] = pingPong ? 1 : 0;
        uniformData[2] = this.size;
        uniformData[3] = mode;
    
        const buffer = this.device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
    
        new Uint32Array(buffer.getMappedRange()).set(uniformData);
        buffer.unmap();
    
        return buffer;
    }

    private createPipelines() {
        // Create bind group layout
        this.bindGroupLayout = this.device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rg32float', access: 'read-only' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rg32float', access: 'read-only' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rg32float', access: 'write-only' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { format: 'rg32float', access: 'write-only' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
            ]
        });

        const shaderModule = this.device.createShaderModule({
            code: ifftShaderCode // The WGSL code from previous examples
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [this.bindGroupLayout]
        });

        this.pipelines = {
            horizontal: this.device.createComputePipeline({
                layout: pipelineLayout,
                compute: { module: shaderModule, entryPoint: 'horizontalStepIFFT' }
            }),
            vertical: this.device.createComputePipeline({
                layout: pipelineLayout,
                compute: { module: shaderModule, entryPoint: 'verticalStepIFFT' }
            }),
            scale: this.device.createComputePipeline({
                layout: pipelineLayout,
                compute: { module: shaderModule, entryPoint: 'scale' }
            })
        };
    }

    async runIFFT2D(
        inputRead: GPUTexture,  // rgba32float, readable
        inputWrite: GPUTexture, // rg32float, writable
        bufferRead: GPUTexture, // rgba32float, readable
        bufferWrite: GPUTexture,// rg32float, writable
        outputToInput: boolean = false,
        scale: boolean = true,
        encoder?: GPUCommandEncoder
    ): Promise<GPUCommandEncoder> {
        const commandEncoder = encoder || this.device.createCommandEncoder();
        const logSize = Math.log2(this.size);
        let pingPong = false;


        // Horizontal pass
        for (let i = 0; i < logSize; i++) {
            pingPong = !pingPong;
            const uniformBuffer = this.createUniformBuffer(i, pingPong, 0);
            const bindGroup = this.createBindGroup(
                inputRead, bufferRead, 
                inputWrite, bufferWrite, 
                pingPong, uniformBuffer
            );
            
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipelines.horizontal);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(
                this.size / this.localWorkGroupsX,
                this.size / this.localWorkGroupsY
            );
            passEncoder.end();

            // Copy write -> read if needed
            if (i < logSize - 1) {
                this.copyBetweenPasses(
                    commandEncoder,
                    pingPong ? bufferWrite : inputWrite,
                    pingPong ? bufferRead : inputRead
                );
            }
        }

        // Vertical pass
        for (let i = 0; i < logSize; i++) {
            pingPong = !pingPong;
            const uniformBuffer = this.createUniformBuffer(i, pingPong, 0);

            const bindGroup = this.createBindGroup(
                inputRead, bufferRead, 
                inputWrite, bufferWrite, 
                pingPong, uniformBuffer
            );
            
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipelines.vertical);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(
                this.size / this.localWorkGroupsX,
                this.size / this.localWorkGroupsY
            );
            passEncoder.end();

            if (i < logSize - 1) {
                this.copyBetweenPasses(
                    commandEncoder,
                    pingPong ? bufferWrite : inputWrite,
                    pingPong ? bufferRead : inputRead
                );
            }
        }

        // Final copy if needed
        if (pingPong && outputToInput) {
            commandEncoder.copyTextureToTexture(
                { texture: bufferWrite },
                { texture: inputRead },
                [this.size, this.size, 1]
            );
        }

        if (!pingPong && !outputToInput) {
            commandEncoder.copyTextureToTexture(
                { texture: inputWrite },
                { texture: bufferRead },
                [this.size, this.size, 1]
            );
        }

        // Scale pass
        if (scale) {
            
            const uniformBuffer = this.createUniformBuffer(0, false, 0);
            const bindGroup = this.createBindGroup(
                inputRead, bufferRead, // Dummy read texture
                inputWrite, bufferWrite,
                false, uniformBuffer
            );
            
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipelines.scale);
            passEncoder.setBindGroup(0, bindGroup);
            passEncoder.dispatchWorkgroups(
                this.size / this.localWorkGroupsX,
                this.size / this.localWorkGroupsY
            );
            passEncoder.end();
            
        }

        return commandEncoder;
    }

    id: number = 0;
    private createBindGroup(
        inputRead: GPUTexture,
        bufferRead: GPUTexture,
        inputWrite: GPUTexture,
        bufferWrite: GPUTexture,
        pingPong: boolean,
        uniformBuffer: GPUBuffer
    ): GPUBindGroup {
        return this.device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: this.precomputedData } },
                { binding: 1, resource: inputRead.createView() },
                { binding: 2, resource: bufferRead.createView() },
                { binding: 3, resource: inputWrite.createView() },
                { binding: 4, resource: bufferWrite.createView() },
                { binding: 5, resource: { buffer: uniformBuffer} }
            ],
            label: `IFFT2D Bind Group ${this.id++}`
        });
    }

    private copyBetweenPasses(
        encoder: GPUCommandEncoder,
        source: GPUTexture,
        destination: GPUTexture
    ) {
        encoder.copyTextureToTexture(
            { texture: source },
            { texture: destination },
            [this.size, this.size, 1]
        );
    }

    // Helper to create texture pairs
    static createTexturePair(device: GPUDevice, size: number) {
        const readable = device.createTexture({
            size: [size, size],
            format: 'rg32float',
            usage: GPUTextureUsage.STORAGE_BINDING | 
                   GPUTextureUsage.COPY_DST | 
                   GPUTextureUsage.COPY_SRC
        });
        
        const writable = device.createTexture({
            size: [size, size],
            format: 'rg32float',
            usage: GPUTextureUsage.STORAGE_BINDING | 
                   GPUTextureUsage.COPY_SRC        |
                   GPUTextureUsage.COPY_DST
        });
        
        return { readable, writable };
    }
}