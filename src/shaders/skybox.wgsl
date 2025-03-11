struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) TexCoord: vec3<f32>,
};

@group(0) @binding(2) var<uniform> view: mat4x4<f32>;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VertexOutput {
    var output: VertexOutput;
    output.Position = view * vec4<f32>(position, 1.0); // Use the vertex position directly
    output.TexCoord = normalize(position); // Use the vertex position as the texture coordinate
    return output;
}

@group(0) @binding(0) var cubeMap: texture_cube<f32>;
@group(0) @binding(1) var cubeSampler: sampler;

@fragment
fn fs_main(@location(0) TexCoord: vec3<f32>) -> @location(0) vec4<f32> {
    // Sample the cube map using the texture coordinate
    return textureSample(cubeMap, cubeSampler, normalize(TexCoord));
    //return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}