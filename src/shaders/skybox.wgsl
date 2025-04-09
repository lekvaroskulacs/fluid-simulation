struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) FragPosition: vec4<f32>,
};

struct TransformData {
    viewDirInverse: mat4x4<f32>
}

@group(0) @binding(2) var<uniform> transform: TransformData;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let pos = array(
        vec2f(-1, 3),
        vec2f(-1,-1),
        vec2f( 3,-1),
    );
    var output: VertexOutput;
    output.Position = vec4<f32>(pos[idx], 1, 1);  
    output.FragPosition = output.Position;
    return output;
}

@group(0) @binding(0) var cubeMap: texture_cube<f32>;
@group(0) @binding(1) var cubeSampler: sampler;

@fragment
fn fs_main(@location(0) FragPosition: vec4<f32>) -> @location(0) vec4<f32> {
    let t =  transform.viewDirInverse * FragPosition;
    return textureSample(cubeMap, cubeSampler, normalize(t.xyz / t.w) * vec3f(1, 1, -1));
    //return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}