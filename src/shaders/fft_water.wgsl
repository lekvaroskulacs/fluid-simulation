struct VertexOutput {
    @builtin(position) Position: vec4<f32>,
    @location(0) TexCoord: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let pos = array(
        vec2f(-1, 3),
        vec2f(-1,-1),
        vec2f( 3,-1)
    );
    var output: VertexOutput;
    output.Position = vec4f(pos[idx], 0.0, 1.0);
    output.TexCoord = vec4f(pos[idx] * 0.5 + 0.5, 0.0, 1.0);
    return output;
}

@group(0) @binding(0) var tex: texture_storage_2d<rg32float, read>;
@group(0) @binding(1) var random: texture_storage_2d<rg32float, read>;
@group(0) @binding(2) var timeSpectrum: texture_storage_2d<rg32float, read>;
@group(0) @binding(3) var waveData: texture_storage_2d<rgba32float, read>;
@group(0) @binding(4) var conjugateSpectrum: texture_storage_2d<rg32float, read>;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let texCoord = vec2u(input.Position.xy);
    var pixel = textureLoad(tex, texCoord);
    //var noise = textureLoad(random, texCoord);
    var spectrum = textureLoad(timeSpectrum, texCoord);
    var wave = textureLoad(waveData, texCoord);
    var con = textureLoad(conjugateSpectrum, texCoord);
    
    return vec4f(spectrum.x, 0.0, 0.0, 0.1);  
    
}