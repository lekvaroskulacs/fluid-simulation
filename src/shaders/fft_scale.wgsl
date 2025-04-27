@group(0) @binding(0) var<storage> input: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var<storage> output: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let N = 256;
    let complexVal = textureLoad(input, id.xy);
    let scale = 1.0 / (f32(N) * f32(N));
    textureStore(output, id.xy, vec4f(complexVal.x * scale, 0, 0, 0));
}