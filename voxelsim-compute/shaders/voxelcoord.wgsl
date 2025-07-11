struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) coord: vec3<i32>,
    @location(1) value: u32,
}

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    // The calculation is the same, but 'camera' is now sourced from push constants.
    let instance_pos = vec3<f32>(instance.coord);
    let world_position = vec4<f32>(model.position + instance_pos, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.coord = instance.coord;
    out.value = instance.value;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec3<i32> {
    return in.coord;
}
