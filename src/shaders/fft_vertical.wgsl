@group(0) @binding(0) var<storage> input: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var<storage> output: texture_storage_2d<rg32float, write>;
@group(0) @binding(2) var<storage> twiddles: array<vec2f>;

@compute @workgroup_size(1, 64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
    let col = id.x;
    let n = id.y;
    let N = 256;
    let stage = u32(log2(f32(N))); // Total stages = log2(N)
    let halfN = N / 2;

    // Decimation-in-time butterfly operation
    let stride = 1 << (stage - 1);
    let twiddleIndex = (n % halfN) * (N / (2 * halfN));
    let twiddle = twiddles[twiddleIndex];

    // Load two complex numbers from the input row
    let a = textureLoad(input, vec2u(n, row));
    let b = textureLoad(input, vec2u(n + stride, row));

    // Butterfly operation
    let t = vec2f(
        twiddle.x * b.x - twiddle.y * b.y,
        twiddle.x * b.y + twiddle.y * b.x
    );
    let out0 = a + t;
    let out1 = a - t;

    // Write results
    textureStore(output, vec2u(n, row), out0);
    textureStore(output, vec2u(n + stride, row), out1);
}