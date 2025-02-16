struct TransformData {
    model: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>
}
@binding(0) @group(0) var<uniform> transformUBO: TransformData;

@binding(1) @group(0) var<uniform> time: f32;

struct Fragment {
    @builtin(position) Position : vec4<f32>,
    @location(0) Normal : vec4<f32>,
    @location(1) WorldPosition : vec4<f32>
};

const PI: f32 = 3.141592653589793;

@vertex 
fn vs_main(@location(0) vertexPosition: vec3<f32>, @builtin(vertex_index) v_id: u32) -> Fragment {
    var output : Fragment;
    var p = vec4<f32>(vertexPosition, 1.0);

    var dz_dx: f32 = 0.0; // Partial derivative with respect to x
    var dz_dy: f32 = 0.0; // Partial derivative with respect to y

    let baseAmplitude: f32 = 1.0;
    let baseFrequency: f32 = 1.0;
    let basePhase: f32 = 1.0;

    let directions = array<vec2<f32>, 4>(
        vec2<f32>(1.0, 0.0),   
        vec2<f32>(0.0, 1.0),   
        vec2<f32>(1.0, 1.0),   
        vec2<f32>(-1.0, 1.0)   
    );

    let maxWaves: i32 = 12;

    for (var wave: i32 = 0; wave < maxWaves; wave += 1) {
        let amplitude = baseAmplitude / (f32(wave) + 1.0);
        let frequency = baseFrequency * (f32(wave) + 1.0);
        let phase = basePhase + f32(wave) * PI / 2.0; 

        let direction = directions[wave % 4];

        let angle = dot(direction, vec2<f32>(p.x, p.y)) * frequency + 0.5 * time * phase;
        let sinVal = sin(angle);
        let cosVal = cos(angle);

        p.z += amplitude * sinVal;
        dz_dx += amplitude * cosVal * direction.x * frequency;
        dz_dy += amplitude * cosVal * direction.y * frequency;
    }

    p.z = p.z / f32(maxWaves); 

    var tangent: vec3<f32> = normalize(vec3<f32>(1, 0, dz_dx));
    var binormal: vec3<f32> = normalize(vec3<f32>(0, 1, dz_dy));

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
    var sun = normalize(vec3<f32>(1, 1, 5));
    var lambert = max(dot(sun, normal), 0.0);

    var camera = vec3<f32>(-3, 0, 3);
    var cameraDir = normalize(camera - WorldPosition.xyz);
    var halfway = normalize(cameraDir + sun);
    var specular = pow(max(dot(halfway, normal), 0.0), 32.0);

    var color = ambient * (lambert + specular);
    return color;
}

