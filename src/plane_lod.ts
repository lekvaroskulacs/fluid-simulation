import { mat4, vec3 } from 'gl-matrix';

// Configuration
const CHUNK_SIZE = 100; // World units per chunk
const LOD_LEVELS = [32, 16, 8, 4]; // Subdivisions per LOD
const LOD_DISTANCES = [50, 100, 200, 500]; // Distance thresholds

class PlaneLOD {
    private device: GPUDevice;
    private pipeline: GPURenderPipeline;
    private lodMeshes: {
        vertices: GPUBuffer,
        indices: GPUBuffer,
        indexCount: number
    }[];
    private chunks: Chunk[] = [];
    private instanceBuffers: GPUBuffer[] = [];

    constructor(device: GPUDevice, pipeline: GPURenderPipeline) {
        this.device = device;
        this.pipeline = pipeline;
        this.lodMeshes = LOD_LEVELS.map(subdivisions => 
            this.createPlaneMesh(subdivisions)
        );
        this.createChunks();
        this.createInstanceBuffers();
    }

    private createPlaneMesh(subdivisions: number) {
        const vertices: number[] = [];
        const indices: number[] = [];
        
        // Generate vertices (position + UV)
        for (let y = 0; y <= subdivisions; y++) {
            for (let x = 0; x <= subdivisions; x++) {
                const u = x / subdivisions;
                const v = y / subdivisions;
                vertices.push(
                    2 * u - 1, 0, 2 * v - 1, // Position (XZ plane -1 to 1)
                    u, v                      // UV
                );
            }
        }

        // Generate indices
        for (let y = 0; y < subdivisions; y++) {
            for (let x = 0; x < subdivisions; x++) {
                const tl = y * (subdivisions + 1) + x;
                const tr = tl + 1;
                const bl = (y + 1) * (subdivisions + 1) + x;
                const br = bl + 1;
                
                indices.push(tl, bl, tr);
                indices.push(tr, bl, br);
            }
        }

        // Create GPU buffers
        const vertexBuffer = this.device.createBuffer({
            size: vertices.length * 4,
            usage: GPUBufferUsage.VERTEX,
            mappedAtCreation: true,
        });
        new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
        vertexBuffer.unmap();

        const indexBuffer = this.device.createBuffer({
            size: indices.length * 4,
            usage: GPUBufferUsage.INDEX,
            mappedAtCreation: true,
        });
        new Uint32Array(indexBuffer.getMappedRange()).set(indices);
        indexBuffer.unmap();

        return {
            vertices: vertexBuffer,
            indices: indexBuffer,
            indexCount: indices.length
        };
    }

    private createChunks() {
        // Create grid of chunks (example 5x5 grid)
        for (let z = -2; z <= 2; z++) {
            for (let x = -2; x <= 2; x++) {
                this.chunks.push({
                    position: [x * CHUNK_SIZE, 0, z * CHUNK_SIZE],
                    lod: 0
                });
            }
        }
    }

    private createInstanceBuffers() {
        // Create one instance buffer per LOD level
        this.instanceBuffers = LOD_LEVELS.map(() => 
            this.device.createBuffer({
                size: 16 * 4 * this.chunks.length, // mat4 per instance
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
            })
        );
    }

    updateLOD(cameraPosition: vec3) {
        this.chunks.forEach(chunk => {
            const dist = vec3.distance(chunk.position, cameraPosition);
            chunk.lod = LOD_DISTANCES.findIndex(d => dist < d);
            chunk.lod = Math.min(Math.max(chunk.lod, 0), LOD_LEVELS.length - 1);
        });
    }

    render(pass: GPURenderPassEncoder, cameraMatrix: GPUBuffer) {
        const lodGroups = this.groupChunksByLOD();

        lodGroups.forEach((chunks, lod) => {
            if (chunks.length === 0) return;

            const mesh = this.lodMeshes[lod];
            const matrices = chunks.map(chunk => 
                this.getChunkMatrix(chunk.position)
            );

            // Update instance buffer
            this.device.queue.writeBuffer(
                this.instanceBuffers[lod],
                0,
                new Float32Array(matrices.flat()),
                chunks.length * 16 * 4
            );

            pass.setPipeline(this.pipeline);
            pass.setVertexBuffer(0, mesh.vertices);
            pass.setIndexBuffer(mesh.indices, 'uint32');
            pass.setVertexBuffer(1, this.instanceBuffers[lod]);
            pass.drawIndexed(mesh.indexCount, chunks.length);
        });
    }

    private groupChunksByLOD() {
        const groups = new Map<number, Chunk[]>();
        LOD_LEVELS.forEach((_, i) => groups.set(i, []));
        this.chunks.forEach(chunk => groups.get(chunk.lod)!.push(chunk));
        return groups;
    }

    private getChunkMatrix(position: vec3) {
        const matrix = mat4.create();
        mat4.translate(matrix, matrix, position);
        mat4.scale(matrix, matrix, [CHUNK_SIZE/2, 1, CHUNK_SIZE/2]);
        return Array.from(matrix);
    }
}

type Chunk = {
    position: vec3;
    lod: number;
};