export class Plane {

    vertexBuffer: GPUBuffer;
    bufferLayout: GPUVertexBufferLayout;
    indexBuffer: GPUBuffer;

    constructor(size: number, detail: number, device: GPUDevice) {
        const _vert_data = [];
        for(let y = 0; y <= detail; y++) {
            for(let x = 0; x <= detail; x++) {
                const u = x / detail;
                const v = y / detail;
                const px = u * 2 * size - size;
                const pz = v * 2 * size - size;
                const py = 0;
                _vert_data.push(px, py, pz);
            }
        }
        const vertices = new Float32Array(_vert_data);

        const _index_data = [];
        
        for(let y = 0; y < detail; y++) {
            var bottom = 0;
            var top = 0;
            for(let x = 0; x <= detail; x++) {
                bottom = y * (detail + 1) + x;
                top = (y + 1) * (detail + 1) + x;
                _index_data.push(bottom, top);
            }
            if (y < detail - 1)
                _index_data.push(top, bottom + 1);
        }
        const indices = new Uint16Array(_index_data);

        this.vertexBuffer = device.createBuffer({
            size: vertices.length * Float32Array.BYTES_PER_ELEMENT, 
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        
        new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
        this.vertexBuffer.unmap();

        this.indexBuffer = device.createBuffer({
            size: indices.length * Uint16Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        })

        new Uint16Array(this.indexBuffer.getMappedRange()).set(indices);
        this.indexBuffer.unmap();

        this.bufferLayout = {
            arrayStride: 12,
            attributes: [
                {
                    format: "float32x3",
                    offset: 0,
                    shaderLocation: 0
                }
            ]
        }
        
    }

} 