import ifftShaderCode from './shaders/ifft.wgsl'

class IFFT2D {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private precomputedData: GPUBuffer;
    private uniforms: GPUBuffer;
    private bindGroups: GPUBindGroup[] = [];
    private size: number;
    private localWorkGroupsX: number = 8;
    private localWorkGroupsY: number = 8;

    constructor(device: GPUDevice, size: number) {
        this.device = device;
        this.size = size;
        
        // Precompute twiddle factors and input indices
        this.precomputeData();
        
        // Create shader module and pipeline
        const shaderModule = device.createShaderModule({
            code: ifftShaderCode
        });
        
        this.pipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: 'horizontalStepIFFT' // Main entry point, others will be used via constants
            }
        });
        
        // Create uniform buffer
        this.uniforms = device.createBuffer({
            size: 16, // step (u32) + pingPong (u32) + size (u32) + padding
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
    }

    private precomputeData() {
        const logSize = Math.log2(this.size);
        const data = new Float32Array(this.size * logSize * 4); // RGBA for each step and index
        
        for (let step = 0; step < logSize; step++) {
            const b = this.size >> (step + 1);
            const mult = 2 * Math.PI / this.size;
            
            for (let y = 0; y < this.size; y++) {
                const i = (2 * b * Math.floor(y / b) + y % b) % this.size;
                const angle = mult * Math.floor(y / b) * b;
                const twiddleX = Math.cos(angle);
                const twiddleY = -Math.sin(angle); // Negative for IFFT
                
                const idx = (step * this.size + y) * 4;
                data[idx] = twiddleX;
                data[idx + 1] = twiddleY;
                data[idx + 2] = i;
                data[idx + 3] = i + b;
            }
        }
        
        this.precomputedData = this.device.createBuffer({
            size: data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        
        new Float32Array(this.precomputedData.getMappedRange()).set(data);
        this.precomputedData.unmap();
    }

    async runIFFT2D(
        input: GPUTexture,
        buffer: GPUTexture,
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
            this.updateBindGroup(input, buffer, pingPong, i);
            
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, this.bindGroups[pingPong ? 1 : 0]);
            passEncoder.dispatchWorkgroups(
                this.size / this.localWorkGroupsX,
                this.size / this.localWorkGroupsY
            );
            passEncoder.end();
        }

        // Vertical pass
        for (let i = 0; i < logSize; i++) {
            pingPong = !pingPong;
            this.updateBindGroup(input, buffer, pingPong, i);
            
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.pipeline);
            passEncoder.setBindGroup(0, this.bindGroups[pingPong ? 1 : 0]);
            passEncoder.dispatchWorkgroups(
                this.size / this.localWorkGroupsX,
                this.size / this.localWorkGroupsY
            );
            passEncoder.end();
        }

        // Copy results if needed
        if (pingPong && outputToInput) {
            commandEncoder.copyTextureToTexture(
                { texture: buffer },
                { texture: input },
                [this.size, this.size, 1]
            );
        }

        if (!pingPong && !outputToInput) {
            commandEncoder.copyTextureToTexture(
                { texture: input },
                { texture: buffer },
                [this.size, this.size, 1]
            );
        }

        // Scaling if needed
        if (scale) {
            // Would need another compute pass with scale shader
        }

        return commandEncoder;
    }

    /// Ha az uniformok nem jól updatelődnek akkor lehetne minden külön stephez egy külön uniformot létrehozni
    private updateBindGroup(input: GPUTexture, buffer: GPUTexture, pingPong: boolean, step: number) {
        // Update uniform buffer
        const uniformData = new Uint32Array(4);
        uniformData[0] = step;
        uniformData[1] = pingPong ? 1 : 0;
        uniformData[2] = this.size;
        
        this.device.queue.writeBuffer(
            this.uniforms,
            0,
            uniformData.buffer
        );

        // Create or update bind group
        if (!this.bindGroups[pingPong ? 1 : 0]) {
            this.bindGroups[pingPong ? 1 : 0] = this.device.createBindGroup({
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.precomputedData } },
                    { binding: 1, resource: input.createView() },
                    { binding: 2, resource: buffer.createView() },
                    { binding: 3, resource: { buffer: this.uniforms } }
                ]
            });
        }
    }
}