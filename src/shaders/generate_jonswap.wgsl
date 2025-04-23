
const PI = 3.141592653589793;
const GRAVITY = 9.81;
const L = 1000.0; // Physical ocean patch size (meters)
const WIND_SPEED = 15; // m/s
const PEAK_ENHANCEMENT = 3.3;
const DIRECTION = 0.0; // Main wave direction (radians)
const FETCH = 10.0; // Distance that wind can blow unobstructed


@group(0) @binding(0) var spectrum: texture_storage_2d<rg32float, write>;
@group(0) @binding(1) var random: texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var waveData: texture_storage_2d<rgba32float, write>;
@group(0) @binding(3) var conjugateSpectrum: texture_storage_2d<rg32float, write>;

fn rand(uv: vec2u) -> f32 {
    var seed = f32(uv.x) * 1973.0 + f32(uv.y) * 9277.0;
    seed = sin(f32(seed)) * 43758.5453;
    return fract(seed) * 2.0 * PI;
}

//JONSWAP function based on: https://wikiwaves.org/Ocean-Wave_Spectra#JONSWAP_Spectrum
fn jonswap(omega: f32, theta: f32) -> f32 {
    // Pierson-Moskowitz spectrum
    const peak_frequency_mod = 1.0;
    let alpha = 0.076 * pow(WIND_SPEED / (GRAVITY * 1.0), 0.22);
    let omega_p = peak_frequency_mod * pow(GRAVITY / (WIND_SPEED * FETCH), 1.0/3.0);
    let S_pm = alpha * pow(GRAVITY, 2.0) / pow(omega, 5.0) * exp(-1.25 * pow(omega_p / omega, 4.0));

    // Peak enhancement
    var sigma = 0.09;
    if (omega <= omega_p) {
        sigma = 0.07;
    }
    let gamma_factor = pow(PEAK_ENHANCEMENT, exp(-pow((omega - omega_p), 2.0) / (2.0 * sigma * sigma * omega_p * omega_p)));

    // Directional spreading (cos^2)
    let dir_spread = pow(cos(theta - DIRECTION), 2.0);  
    return S_pm * gamma_factor * dir_spread;
}

//Formulas taken from https://github.com/gasgiant/FFT-Ocean
@compute @workgroup_size(8, 8)
fn initial_spectrum(@builtin(global_invocation_id) id: vec3u) {

    let N = f32(textureDimensions(spectrum).x);
    let deltaK = 2 * PI / L;
    let nx = f32(id.x) - f32(N) / 2.0;
    let ny = f32(id.y) - f32(N) / 2.0;
    let k = vec2f(nx, ny) * deltaK;
    let kLength = sqrt(k.x * k.x + k.y * k.y);

    let kAngle = atan2(k.y, k.x);
    let omega = sqrt(GRAVITY * kLength);
    let dOmegadk = 1 / (2 * sqrt(GRAVITY * kLength));

    let S = jonswap(omega, kAngle);

    let H0K = vec2f(textureLoad(random, id.xy).xy) * sqrt(2 * S * abs(dOmegadk) / kLength * deltaK * deltaK);
    let minusID = vec2u(u32(N) - id.x, u32(N) - id.y);

    textureStore(spectrum, id.xy, vec4f(H0K.x, H0K.y, 0.0, 1.0));
    textureStore(conjugateSpectrum, minusID.xy, vec4f(H0K.x, -H0K.y, 0.0, 1.0));
    textureStore(waveData, id.xy, vec4f(k.x, 1 / kLength, k.y, omega));
}
