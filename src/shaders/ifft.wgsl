// IFFT Shader Code
struct Uniforms {
    step: u32,
    pingPong: u32,
    size: u32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> precomputedData: array<vec4<f32>>;
@group(0) @binding(1) var inputTexture: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(2) var bufferTexture: texture_storage_2d<rgba32float, read_write>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.r * b.r - a.g * b.g,
        a.r * b.g + a.g * b.r
    );
}

@compute @workgroup_size(8, 8, 1)
fn horizontalStepIFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let data = precomputedData[uniforms.step * uniforms.size + id.x];
    let inputsIndices = vec2<u32>(data.b, data.a); // ba contains the indices
    
    var result: vec2<f32>;
    if (uniforms.pingPong == 1) {
        let val1 = textureLoad(inputTexture, vec2<u32>(inputsIndices.x, id.y), 0).rg;
        let val2 = textureLoad(inputTexture, vec2<u32>(inputsIndices.y, id.y), 0).rg;
        result = val1 + complexMult(vec2<f32>(data.r, -data.g), val2);
        textureStore(bufferTexture, vec2<u32>(id.x, id.y), vec4<f32>(result, 0.0, 1.0));
    } else {
        let val1 = textureLoad(bufferTexture, vec2<u32>(inputsIndices.x, id.y), 0).rg;
        let val2 = textureLoad(bufferTexture, vec2<u32>(inputsIndices.y, id.y), 0).rg;
        result = val1 + complexMult(vec2<f32>(data.r, -data.g), val2);
        textureStore(inputTexture, vec2<u32>(id.x, id.y), vec4<f32>(result, 0.0, 1.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn verticalStepIFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    let data = precomputedData[uniforms.step * uniforms.size + id.y];
    let inputsIndices = vec2<u32>(data.b, data.a); // ba contains the indices
    
    var result: vec2<f32>;
    if (uniforms.pingPong == 1) {
        let val1 = textureLoad(inputTexture, vec2<u32>(id.x, inputsIndices.x), 0).rg;
        let val2 = textureLoad(inputTexture, vec2<u32>(id.x, inputsIndices.y), 0).rg;
        result = val1 + complexMult(vec2<f32>(data.r, -data.g), val2);
        textureStore(bufferTexture, vec2<u32>(id.x, id.y), vec4<f32>(result, 0.0, 1.0));
    } else {
        let val1 = textureLoad(bufferTexture, vec2<u32>(id.x, inputsIndices.x), 0).rg;
        let val2 = textureLoad(bufferTexture, vec2<u32>(id.x, inputsIndices.y), 0).rg;
        result = val1 + complexMult(vec2<f32>(data.r, -data.g), val2);
        textureStore(inputTexture, vec2<u32>(id.x, id.y), vec4<f32>(result, 0.0, 1.0));
    }
}

@compute @workgroup_size(8, 8, 1)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = f32(uniforms.size);
    let value = textureLoad(inputTexture, vec2<u32>(id.x, id.y), 0).rg;
    textureStore(inputTexture, vec2<u32>(id.x, id.y), vec4<f32>(value / (size * size), 0.0, 1.0));
}