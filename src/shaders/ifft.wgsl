// IFFT Shader Code
struct Uniforms {
    step: u32,
    pingPong: u32,
    size: u32,
    padding: u32,
};

@group(0) @binding(0) var<storage, read> precomputedData: array<vec4<f32>>;

@group(0) @binding(1) var inputTextureRead: texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var bufferTextureRead: texture_storage_2d<rg32float, read>;

@group(0) @binding(3) var inputTextureWrite: texture_storage_2d<rg32float, write>;
@group(0) @binding(4) var bufferTextureWrite: texture_storage_2d<rg32float, write>;

@group(0) @binding(5) var<uniform> uniforms: Uniforms;

fn complexMult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.r * b.r - a.g * b.g,
        a.r * b.g + a.g * b.r
    );
}

@compute @workgroup_size(8, 8, 1)
fn horizontalStepIFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    
    let data = precomputedData[uniforms.step * uniforms.size + id.x];
    let indices = vec2<u32>(u32(data.b), u32(data.a));
    let twiddle = vec2<f32>(data.r, -data.g); // Conjugate for IFFT
    
    if (uniforms.pingPong == 1) {
        // Read from input (rgba), write to buffer (rg)
        let val1 = textureLoad(inputTextureRead, vec2<i32>(i32(indices.x), i32(id.y))).rg;
        let val2 = textureLoad(inputTextureRead, vec2<i32>(i32(indices.y), i32(id.y))).rg;
        let result = val1 + complexMult(twiddle, val2);
        textureStore(bufferTextureWrite, vec2<u32>(id.x, id.y), vec4f(result, 0.0, 1.0));
    } else {
        // Read from buffer (rgba), write to input (rg)
        let val1 = textureLoad(bufferTextureRead, vec2<i32>(i32(indices.x), i32(id.y))).rg;
        let val2 = textureLoad(bufferTextureRead, vec2<i32>(i32(indices.y), i32(id.y))).rg;
        let result = val1 + complexMult(twiddle, val2);
        textureStore(inputTextureWrite, vec2<u32>(id.x, id.y), vec4f(result, 0.0, 1.0));
    }
    
}

@compute @workgroup_size(8, 8, 1)
fn verticalStepIFFT(@builtin(global_invocation_id) id: vec3<u32>) {
    
    let data = precomputedData[uniforms.step * uniforms.size + id.y];
    let indices = vec2<u32>(u32(data.b), u32(data.a));
    let twiddle = vec2<f32>(data.r, -data.g); // Conjugate for IFFT
    
    if (uniforms.pingPong == 1) {
        // Read from input (rgba), write to buffer (rg)
        let val1 = textureLoad(inputTextureRead, vec2<i32>(i32(indices.x), i32(id.x))).rg;
        let val2 = textureLoad(inputTextureRead, vec2<i32>(i32(indices.x), i32(id.y))).rg;
        let result = val1 + complexMult(twiddle, val2);
        textureStore(bufferTextureWrite, vec2<u32>(id.x, id.y), vec4f(result, 0.0, 1.0));
    } else {
        // Read from buffer (rgba), write to input (rg)
        let val1 = textureLoad(bufferTextureRead, vec2<i32>(i32(indices.x), i32(id.x))).rg;
        let val2 = textureLoad(bufferTextureRead, vec2<i32>(i32(indices.x), i32(id.y))).rg;
        let result = val1 + complexMult(twiddle, val2);
        textureStore(inputTextureWrite, vec2<u32>(id.x, id.y), vec4f(result, 0.0, 1.0));
    }

}

@compute @workgroup_size(8, 8, 1)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = f32(uniforms.size);
    let value = textureLoad(inputTextureRead, vec2<u32>(id.x, id.y)).rg;
    textureStore(inputTextureWrite, vec2<u32>(id.x, id.y), vec4<f32>(value / (size * size), 0.0, 1.0));
}