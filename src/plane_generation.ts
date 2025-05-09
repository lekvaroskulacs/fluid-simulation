import { mat4, vec3 } from 'gl-matrix';

interface PatchInfo {
  worldPos: vec3;
  lod: number;
}

export class AdaptiveWaterTessellation {
  private device: GPUDevice;
  private patchInfoBuffer: GPUBuffer;
  private vertexBuffer: GPUBuffer;
  private indexBuffers: GPUBuffer[] = [];
  private computePipeline: GPUComputePipeline;
  private bindGroup: GPUBindGroup;

  private PATCHES_X = 10;
  private PATCHES_Z = 10;
  private PATCH_SIZE = 1.0;
  private lodResolutions = [32, 16, 8];
  private patchInfoData: PatchInfo[] = [];

  constructor(device: GPUDevice, cameraPosition: vec3) {
    this.device = device;

    // Compute patch LODs
    for (let x = 0; x < this.PATCHES_X; x++) {
      for (let z = 0; z < this.PATCHES_Z; z++) {
        const pos = vec3.fromValues(x * this.PATCH_SIZE, 0, z * this.PATCH_SIZE);
        const lod = this.computeLOD(cameraPosition, pos);
        this.patchInfoData.push({ worldPos: pos, lod });
      }
    }

    this.patchInfoBuffer = device.createBuffer({
      size: this.patchInfoData.length * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.patchInfoBuffer, 0, new Float32Array(this.patchInfoData.flatMap(p => [...p.worldPos, p.lod])));

    const maxVertices = 256 * 256 * this.PATCHES_X * this.PATCHES_Z;
    this.vertexBuffer = device.createBuffer({
      size: maxVertices * 4 * 8,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    });

    for (const res of this.lodResolutions) {
      const data = this.generateTriangleStripIndices(res);
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(buffer, 0, data);
      this.indexBuffers.push(buffer);
    }

    const computeShaderModule = device.createShaderModule({
      code: `
struct Vertex {
  position: vec3<f32>,
  normal: vec3<f32>,
  uv: vec2<f32>,
};

struct PatchInfo {
  worldPos: vec3<f32>,
  lod: u32,
};

@group(0) @binding(0) var<storage, read_write> vertexBuffer: array<Vertex>;
@group(0) @binding(1) var<storage, read> patchInfo: array<PatchInfo>;

fn computeHeight(x: f32, z: f32) -> f32 {
  return sin(x * 0.3 + z * 0.2) * 0.5;
}

@compute @workgroup_size(1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let patchId = id.x;
  let info = patchInfo[patchId];
  var resolution: u32;
  switch(info.lod) {
    case 0: { resolution = 32u; }
    case 1: { resolution = 16u; }
    default: { resolution = 8u; }
  }

  let baseIndex = patchId * 256u * 256u;
  let origin = info.worldPos;

  for (var x = 0u; x <= resolution; x += 1u) {
    for (var z = 0u; z <= resolution; z += 1u) {
      let fx = f32(x) / f32(resolution);
      let fz = f32(z) / f32(resolution);
      let worldX = origin.x + fx;
      let worldZ = origin.z + fz;
      let height = computeHeight(worldX, worldZ);
      let index = baseIndex + x * (resolution + 1u) + z;
      vertexBuffer[index] = Vertex(vec3f(worldX, height, worldZ), vec3f(0, 1, 0), vec2f(fx, fz));
    }
  }
}`
    });

    this.computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: computeShaderModule,
        entryPoint: 'cs_main'
      }
    });

    this.bindGroup = device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.vertexBuffer } },
        { binding: 1, resource: { buffer: this.patchInfoBuffer } },
      ]
    });
  }

  private computeLOD(camera: vec3, patchPos: vec3): number {
    const dist = vec3.distance(camera, patchPos);
    if (dist < 5) return 0;
    else if (dist < 10) return 1;
    return 2;
  }

  private generateTriangleStripIndices(res: number): Uint32Array {
    const indices: number[] = [];
    for (let y = 0; y < res; y++) {
      if (y > 0) indices.push(0xFFFFFFFF); // primitive restart
      for (let x = 0; x <= res; x++) {
        indices.push(y * (res + 1) + x);
        indices.push((y + 1) * (res + 1) + x);
      }
    }
    return new Uint32Array(indices);
  }

  public dispatch() {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(this.PATCHES_X * this.PATCHES_Z);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  public getVertexBuffer(): GPUBuffer {
    return this.vertexBuffer;
  }

  public getIndexBuffers(): GPUBuffer[] {
    return this.indexBuffers;
  }

  public getPatchInfo(): PatchInfo[] {
    return this.patchInfoData;
  }
}