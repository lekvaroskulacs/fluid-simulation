@group(0) @binding(0) var stationary_spectrum: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var evolved_spectrum: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<uniform> time: f32;
@group(0) @binding(3) var waveData: texture_storage_2d<rgba32float, read>;
@group(0) @binding(4) var conjugateSpectrum: texture_storage_2d<rg32float, read>;

@group(1) @binding(0) var dy_dxz: texture_storage_2d<rg32float, write>;
@group(1) @binding(1) var dx_dz: texture_storage_2d<rg32float, write>;

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Simulating ocean water, Jerry Tessendorf 2004., formula (43)
// Calculating h = h0 * exp(i * phase) for every frequency (phase = frequency * time)
@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let wave = textureLoad(waveData, id.xy);
    let phase = wave.w * time;
    let exponent = vec2f(cos(phase), sin(phase));
    
    let H0K = textureLoad(stationary_spectrum, id.xy);
    let H0minusK = textureLoad(conjugateSpectrum, id.xy);

    // h0 * exp(i * phase) = h0 * (cos(phase) + i * sin(phase)) using Euler's formula
    // h0 is a complex number so we can just use complex multiplication to get h
    // the conjugate of h0 is used so that the inverse fft guarantees a real number
    let h = complex_mult(H0K.xy, exponent) + complex_mult(H0minusK.xy, vec2f(exponent.x, -exponent.y));
    let ih = vec2f(-h.y, h.x);

    let displacementX = ih * wave.x * wave.y;
	let displacementY = h;
	let displacementZ = ih * wave.z * wave.y;
	
	let displacementX_dx = -h * wave.x * wave.x * wave.y;
	let displacementY_dx = ih * wave.x;
	let displacementZ_dx = -h * wave.x * wave.z * wave.y;

    let dx_dz_val = vec2f(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
    let dy_dxz_val = vec2f(displacementY.x - displacementZ_dx.y, displacementY.y + displacementZ_dx.x);

    textureStore(evolved_spectrum, id.xy, vec4f(h.x, h.y, 0.0, 1.0));
    textureStore(dx_dz, id.xy, vec4f(dx_dz_val.x, dx_dz_val.y, 0.0, 1.0));
    textureStore(dy_dxz, id.xy, vec4f(dy_dxz_val.x, dy_dxz_val.y, 0.0, 1.0));

}