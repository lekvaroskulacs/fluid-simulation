@group(0) @binding(0) var stationary_spectrum: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var evolved_spectrum: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> time: f32;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    var sum = 0.0;
    for (var kx: u32 = 0; kx < 256; kx++) {
        for (var ky: u32 = 0; ky < 256; ky++) {
            let h0k = textureLoad(stationary_spectrum, vec2u(kx, ky)).x;
            sum += h0k * exp(f32(dot(vec2u(kx, ky), id.xy)));
        }
    }

    textureStore(evolved_spectrum, id.xy, vec4f( textureLoad(stationary_spectrum, vec2u(id.x, id.y)).x, 0.0, 0.0, 0.0));
}