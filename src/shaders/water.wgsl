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

struct SceneOptions {
    _cameraPosition: vec4<f32>,
    _sunPosition: f32,
}

@binding(0) @group(0) var<uniform> transformUBO: TransformData;

@binding(1) @group(0) var<uniform> time: f32;

@binding(2) @group(0) var<uniform> waveOptions: WaveOptions;

@binding(3) @group(0) var<uniform> sceneOptions: SceneOptions;

@binding(4) @group(0) var cubeMap: texture_cube<f32>;
@binding(5) @group(0) var cubeSampler: sampler;

const PI: f32 = 3.141592653589793;
const e: f32 = 2.718281828459045;

@vertex 
fn vs_main(@location(0) vertexPosition: vec3<f32>, @builtin(vertex_index) v_id: u32) -> Fragment {
    var output : Fragment;
    var p = vec4<f32>(vertexPosition, 1.0);
    p = transformUBO.model * p;

    var dy_dx: f32 = 0.0; // Partial derivative with respect to x
    var dy_dz: f32 = 0.0; // Partial derivative with respect to y

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
    let seedIter = 1.0;

    for (var wave: i32 = 0; wave < maxWaves; wave += 1) {
        
        let phase = seed; //+ f32(wave) * PI / 2.0; 

        //let sample1 = textureSampleLevel(noiseTexture, noiseSampler, vec2<f32>(f32(wave) / f32(maxWaves), 0.0), 0.0);
        //let sample2 = textureSampleLevel(noiseTexture, noiseSampler, vec2<f32>(1.0 - f32(wave) / f32(maxWaves), 1.0), 0.0);

        let direction = vec2<f32>(sin(seed), cos(seed));
        seed += seedIter;

        let angle = dot(normalize(direction), vec2<f32>(p.x, p.z)) * frequency + time * baseSpeed + phase;
        let sinVal = sin(angle);
        let cosVal = cos(angle);

        p.y += amplitude * exp(sinVal - 1) / e;
        dy_dx += amplitude * cosVal * direction.x * frequency * exp(sinVal - 1) / e;
        dy_dz += amplitude * cosVal * direction.y * frequency * exp(sinVal - 1) / e;

        p.x += direction.x * dy_dx * 0.1;
        p.z += direction.y * dy_dz * 0.1;

        amplitude = amplitude * amplitudeMult;
        frequency = frequency * frequencyMult;
        amplitudeSum += amplitude;
    }

    p.y = p.y / amplitudeSum; 
    dy_dx /= amplitudeSum;
    dy_dz /= amplitudeSum;

    var tangent: vec3<f32> = normalize(vec3<f32>(0, dy_dx, 1));
    var binormal: vec3<f32> = normalize(vec3<f32>(1, dy_dz, 0));

    //var tangent: vec3<f32> = normalize(vec3<f32>(1, 0, dy_dx + 0.0001));
    //var binormal: vec3<f32> = normalize(vec3<f32>(0, 1, dy_dz + 0.0001));

    var obj_space_normal = normalize(cross(tangent, binormal));
    var world_space_normal = normalize(transformUBO.model * vec4<f32>(obj_space_normal, 0)).xyz;

    output.Normal = vec4<f32>(world_space_normal, 1);
    output.WorldPosition = p;
    output.Position = transformUBO.projection * transformUBO.view * output.WorldPosition;
    return output;
}

@fragment
fn fs_main(@location(0) Normal: vec4<f32>, @location(1) WorldPosition: vec4<f32>) -> @location(0) vec4<f32> {
    
    const SPECULAR_SHININESS = 300.0;
    const SPECULAR_STRENGTH = 0.7;
    const FRESNEL_SHININESS = 5.0;
    const FRESNEL_STRENGTH = 1.0;
    const REFLECTION_STRENGTH = 1.0;
    const DIFFUSE_REFLECTANCE = 1.0;
    const SUN_DIRECTION = vec3f(-0.5, 1, 0.5);
    const AMBIENT_RGB = vec3f(153, 179, 216);
    
    var ambient = vec4<f32>(AMBIENT_RGB, 1) / 255;

    var normal = normalize(Normal.xyz);
    let sunPos = sceneOptions._sunPosition;
    var sun = normalize(SUN_DIRECTION);
    var lambert = max(dot(sun, normal), 0.0) * DIFFUSE_REFLECTANCE;

    var camera = sceneOptions._cameraPosition.xyz;
    var viewDir = normalize(camera - WorldPosition.xyz);
    var halfway = normalize(viewDir + sun);
    var specular = pow(max(dot(halfway, normal), 0.0), SPECULAR_SHININESS);

    var reflectedDir = reflect(-viewDir, normal);
    //reversed z coord, because skybox is rendered the same way 
    var reflected = textureSample(cubeMap, cubeSampler, reflectedDir * vec3f(1, 1, -1));

    //Schlick fresnel
    //r0 = ((1 - 1.33) / (1 + 1.33)) ^ 2
    var r0 = 0.02;
    var fresnel = r0 + (1 - r0) * pow(1 - dot(viewDir, normal), FRESNEL_SHININESS);
    fresnel *= FRESNEL_STRENGTH;

    //var color = ambient * (lambert + fresnel * specular + reflected * fresnel);
    //var color = ambient * (lambert + specular * fresnel);
    var color = ambient * (lambert + specular * fresnel * reflected * SPECULAR_STRENGTH + reflected * fresnel * REFLECTION_STRENGTH);
    return color;
}

