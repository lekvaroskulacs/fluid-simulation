struct TransformData {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>
};

struct Fragment {
    @builtin(position) Position : vec4<f32>,
    @location(0) Normal : vec4<f32>,
    @location(1) WorldPosition : vec4<f32>
};

struct WaveOptions {
    _amplitude: f32,
    _frequency: f32,
    _amplitudeMult: f32,
    _frequencyMult: f32,
    _basePhase: f32,
    _baseSpeed: f32,
    _maxWaves: f32
};

@binding(0) @group(0) var<uniform> transformUBO: TransformData;

@binding(1) @group(0) var<uniform> time: f32;

@binding(2) @group(0) var<uniform> waveOptions: WaveOptions;

@binding(3) @group(0) var noiseTexture: texture_2d<f32>;

@binding(4) @group(0) var noiseSampler: sampler;

const PI: f32 = 3.141592653589793;

@vertex 
fn vs_main(@location(0) vertexPosition: vec3<f32>, @builtin(vertex_index) v_id: u32) -> Fragment {
    var output : Fragment;
    var p = vec4<f32>(vertexPosition, 1.0);

    var dz_dx: f32 = 0.0; // Partial derivative with respect to x
    var dz_dy: f32 = 0.0; // Partial derivative with respect to y

    var amplitude: f32 = waveOptions._amplitude;
    var frequency: f32 = waveOptions._frequency;
    let amplitudeMult: f32 = waveOptions._amplitudeMult;
    let frequencyMult: f32 = waveOptions._frequencyMult;
    let basePhase: f32 = waveOptions._basePhase;
    let baseSpeed: f32 = waveOptions._baseSpeed;
    var amplitudeSum = 0.0;

    let maxWaves: i32 = i32(waveOptions._maxWaves);

    let directions: array<vec2<f32>, 4> = array(
        vec2<f32>(1.0, 0.0),   
        vec2<f32>(0.0, 1.0),   
        vec2<f32>(1.0, 1.0),   
        vec2<f32>(-1.0, 1.0)   
    );

    var seed = 0.0;
    let seedIter = 500.0;

    for (var wave: i32 = 0; wave < maxWaves; wave += 1) {
        
        let phase = basePhase; //+ f32(wave) * PI / 2.0; 

        //let sample1 = textureSampleLevel(noiseTexture, noiseSampler, vec2<f32>(f32(wave) / f32(maxWaves), 0.0), 0.0);
        //let sample2 = textureSampleLevel(noiseTexture, noiseSampler, vec2<f32>(1.0 - f32(wave) / f32(maxWaves), 1.0), 0.0);

        let direction = vec2<f32>(sin(seed), cos(seed));
        seed += seedIter;

        let angle = dot(normalize(direction), vec2<f32>(p.x, p.y)) * frequency + time * baseSpeed + phase;
        let sinVal = sin(angle);
        let cosVal = cos(angle);

        p.z += amplitude * exp(sinVal - 1);
        dz_dx += amplitude * cosVal * direction.x * frequency * exp(sinVal - 1);
        dz_dy += amplitude * cosVal * direction.y * frequency * exp(sinVal - 1);

        p.x += direction.x * dz_dx * amplitude;
        p.y += direction.y * dz_dy * amplitude;

        amplitude = amplitude * amplitudeMult;
        frequency = frequency * frequencyMult;
        amplitudeSum += amplitude;
    }

    p.z = p.z / amplitudeSum; 
    dz_dx /= amplitudeSum;
    dz_dy /= amplitudeSum;

    var tangent: vec3<f32> = normalize(vec3<f32>(1, 0, dz_dx + 0.00001));
    var binormal: vec3<f32> = normalize(vec3<f32>(0, 1, dz_dy + 0.00001));

    var obj_space_normal = normalize(cross(tangent, binormal));
    var world_space_normal = normalize(transformUBO.model * vec4<f32>(obj_space_normal, 0)).xyz;

    output.Normal = vec4<f32>(world_space_normal, 1);
    output.WorldPosition = transformUBO.view * transformUBO.model * p;
    output.Position = transformUBO.projection * output.WorldPosition;
    return output;
}

@fragment
fn fs_main(@location(0) Normal: vec4<f32>, @location(1) WorldPosition: vec4<f32>) -> @location(0) vec4<f32> {
    var ambient = vec4<f32>(37, 150, 190, 1) / 255;

    var normal = normalize(Normal.xyz);
    var sun = normalize(vec3<f32>(1, 1, 3));
    var lambert = max(dot(sun, normal), 0.0);

    var camera = vec3<f32>(-3, 0, 3);
    var cameraDir = normalize(camera - WorldPosition.xyz);
    var halfway = normalize(cameraDir + sun);
    var specular = pow(max(dot(halfway, normal), 0.0), 32.0);

    var color = ambient * (lambert + specular);
    return color;
}

