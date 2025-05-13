struct TransformData {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>
};

struct Fragment {
    @builtin(position) Position : vec4<f32>,
    @location(0) Normal : vec4<f32>,
    @location(1) WorldPosition : vec4<f32>,
};

struct WaveOptions {
    _amplitude: f32,
    _frequency: f32,
    _amplitudeMult: f32,
    _frequencyMult: f32,
    _basePhase: f32,
    _horizontalDisplacement: f32,
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

@binding(6) @group(0) var noise: texture_storage_2d<rg32float, read>;

@binding(0) @group(1) var<storage> translations: array<mat4x4f>;

//@binding(0) @group(1) var normals: texture_storage_2d<rgba32float, write>;

const PI: f32 = 3.141592653589793;
const e: f32 = 2.718281828459045;

fn evaluateWave(pos: vec2<f32>) -> vec3<f32> {
    var p = vec4f(pos.x, 0, pos.y, 0);

    var dy_dx: f32 = 0.0; // Partial derivative with respect to x
    var dy_dz: f32 = 0.0; // Partial derivative with respect to y

    var amplitude: f32 = waveOptions._amplitude;
    var frequency: f32 = waveOptions._frequency;
    let amplitudeMult: f32 = waveOptions._amplitudeMult;
    let frequencyMult: f32 = waveOptions._frequencyMult;
    let basePhase: f32 = waveOptions._basePhase;
    let horizontalDisplacement: f32 = waveOptions._horizontalDisplacement;
    var amplitudeSum = 0.0;

    let maxWaves: u32 = u32(waveOptions._maxWaves);

    let directions: array<vec2<f32>, 4> = array(
        vec2<f32>(1.0, 0.0),   
        vec2<f32>(0.0, 1.0),   
        vec2<f32>(1.0, 1.0),   
        vec2<f32>(-1.0, 1.0)   
    );

    var seed = 0.0;
    let seedIter = 1.0;

    var smoothed_dy_dx = 0.0;
    var smoothed_dy_dz = 0.0;
    let smoothing_factor = 0.1;

    for (var wave: u32 = 0; wave < maxWaves; wave += 1) {
        
        let phase = seed; //+ f32(wave) * PI / 2.0; 

        let noiseId = vec2u(wave / 256, wave % 256);
        //let direction = normalize(vec2f(textureLoad(noise, noiseId).xy));
        let direction = normalize(vec2f(sin(seed), cos(seed)));
        seed += seedIter;

        let angle = dot(normalize(direction), vec2<f32>(p.x, p.z)) * frequency + time + phase;
        let sinVal = sin(angle);
        let cosVal = cos(angle);

        p.y += amplitude * exp(sinVal - 1) / e;
        dy_dx += amplitude * cosVal * direction.x * frequency * exp(sinVal - 1) / e;
        dy_dz += amplitude * cosVal * direction.y * frequency * exp(sinVal - 1) / e;

        smoothed_dy_dx = mix(smoothed_dy_dx, dy_dx, smoothing_factor);
        smoothed_dy_dz = mix(smoothed_dy_dz, dy_dz, smoothing_factor);

        p.x += direction.x * dy_dx * horizontalDisplacement;
        p.z += direction.y * dy_dz * horizontalDisplacement;

        amplitude = amplitude * amplitudeMult;
        frequency = frequency * frequencyMult;
        amplitudeSum += amplitude;
    }
    //dy_dx = smoothed_dy_dx;
    //dy_dz = smoothed_dy_dz;

    p.y = p.y / amplitudeSum; 
    dy_dx /= amplitudeSum;
    dy_dz /= amplitudeSum;

    return vec3f(p.y, dy_dx, dy_dz);
}

@vertex 
fn vs_main(@location(0) vertexPosition: vec3<f32>, @builtin(vertex_index) v_id: u32, @builtin(instance_index) i_id: u32) -> Fragment {
    var output : Fragment;
    var p = vec4<f32>(vertexPosition, 1.0);
    p = translations[i_id] * p;

    

    //p = vec4f(p.x, evaluateWave(p.xz), p.z, 1.0);
    
    let center = vec2f(p.x, p.z);
    let values = evaluateWave(center);


    var tangent: vec3<f32> = normalize(vec3<f32>(0, values.y, 1));
    var binormal: vec3<f32> = normalize(vec3<f32>(1, values.z, 0));

    var obj_space_normal = normalize(cross(tangent, binormal));
    var world_space_normal = normalize(translations[i_id] * vec4<f32>(obj_space_normal, 0)).xyz;

    //output.Normal = vec4<f32>(world_space_normal, 1);
    output.Normal = vec4<f32>(obj_space_normal, 1.0);
    output.WorldPosition = vec4f(p.x, values.x, p.z, 1.0);
    output.Position = transformUBO.projection * transformUBO.view * output.WorldPosition;

    return output;
}

@fragment
fn fs_main(@location(0) Normal: vec4<f32>, @location(1) WorldPosition: vec4<f32>, @builtin(sample_index) id: u32) -> @location(0) vec4<f32> {
    const SPECULAR_SHININESS = 400.0;
    const SPECULAR_STRENGTH = 2.0;
    const FRESNEL_SHININESS = 5.0;
    const FRESNEL_STRENGTH = 1.0;
    const REFLECTION_STRENGTH = 0.5;
    const DIFFUSE_REFLECTANCE = 1;
    const SUN_DIRECTION = vec3f(-0.4, 0.2, 0.5);
    const SUN_INTENSITY = 1.0;
    const AMBIENT_RGB = vec3f(17, 64, 77);
    const AMBIENT_STRENGTH = 0.2;
    const SPECULAR_RGB = vec3f(255, 255, 255);


    var ambient = vec4<f32>(AMBIENT_RGB, 1) / 255;
    var specularColor = vec4<f32>(SPECULAR_RGB, 1) / 255;

    var camera = sceneOptions._cameraPosition.xyz;
    var viewDir = normalize(camera - WorldPosition.xyz);
    var normal = normalize(Normal.xyz);

    if (dot(normal, vec3f(0, 1, 0)) < 0.2) {
        normal = vec3f(0, 1, 0);
    }
    
    //Schlick fresnel
    //r0 = ((1 - 1.33) / (1 + 1.33)) ^ 2
    var r0 = 0.02;
    var fresnel = r0 + (1 - r0) * pow(1 - dot(viewDir, normal), FRESNEL_SHININESS);
    fresnel *= FRESNEL_STRENGTH;

    let sunPos = sceneOptions._sunPosition;
    var sun = normalize(SUN_DIRECTION);
    var diffuse = max(dot(sun, normal), 0.0) * DIFFUSE_REFLECTANCE * ambient;

    let wrap = 0.8; // tweak for softness — 0.4 to 0.8 is typical
    let dotNL = dot(normal, sun);
    let wrappedDiffuse = ((dotNL + wrap) / (1.0 + wrap)) * SUN_INTENSITY;
    let sssColor = ambient; // reflected color
    let sss = wrappedDiffuse * sssColor;

    var halfway = normalize(viewDir + sun);
    var specular = pow(max(dot(halfway, normal), 0.0), SPECULAR_SHININESS) * SPECULAR_STRENGTH;

    var reflectedDir = reflect(-viewDir, normal);
    //reversed z coord, because skybox is rendered the same way 
    var reflected = textureSample(cubeMap, cubeSampler, reflectedDir * vec3f(1, 1, -1)) * REFLECTION_STRENGTH;

    //var color = ambient * AMBIENT_STRENGTH + lambert + specular * fresnel * specularColor * SPECULAR_STRENGTH + reflected * fresnel * REFLECTION_STRENGTH;
    //var color = sss + mix(diffuse, reflected, fresnel) + specular * fresnel;
    //var color = sss;
    var color =  sss + specular * fresnel + reflected * fresnel;
    return vec4f(acesToneMapper(color.xyz), 1.0);

    //return Normal;
}

fn acesToneMapper(x: vec3<f32>) -> vec3<f32> {
    const a = 2.51;
    const b = 0.03;
    const c = 2.43;
    const d = 0.59;
    const e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d ) + e), vec3f(0.0), vec3f(1.0));
}
