@group(0) @binding(0) var stationary_spectrum: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var evolved_spectrum: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var waveData: texture_storage_2d<rgba32float, read>;
@group(0) @binding(4) var conjugateSpectrum: texture_storage_2d<rg32float, read>;

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let wave = textureLoad(waveData, id.xy);
    let phase = wave.w * time;
    let exponent = vec2f(cos(phase), sin(phase));
    let H0K = textureLoad(stationary_spectrum, id.xy);
    let H0minusK = textureLoad(conjugateSpectrum, id.xy);


    let h = complex_mult(H0K.xy, exponent) + complex_mult(H0minusK.xy, vec2f(exponent.x, -exponent.y));


    textureStore(evolved_spectrum, id.xy, vec4f(h.x, 0.0, 0.0, 1.0));
}