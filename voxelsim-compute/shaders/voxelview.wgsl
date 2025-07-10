// The camera uniform that matches the CameraUniform struct in Rust.
struct CameraUniform {
    view_proj: mat4x4<f32>,
};
// This is the bind group that will contain our camera buffer.
@group(0) @binding(0)
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(1) position: vec3<f32>,
    @location(2) value: u32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) value: u32,
};

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    // Use the camera's view-projection matrix to transform the vertex position.
    let world_position = vec4<f32>(model.position + instance.position, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.value = instance.value;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let brightness = f32(in.metadata % 10u) * 0.05;
    return vec4<f32>(0.3 + brightness, 0.4, 0.5 + brightness, 1.0);
}
