// The camera data structure, matching the Rust struct.
struct CameraUniform {
    view_proj: mat4x4<f32>,
}

// Use a push constant for the camera data. This removes the need for a bind group.
var<push_constant> camera: CameraUniform;

struct OutputData {
    coord: vec3<i32>,
};

// The output buffer is a struct containing an atomic counter and the data array
struct OutputBuffer {
    count: atomic<u32>,
    // Use a runtime-sized array for the data
    data: array<OutputData>,
};

// A separate buffer to hold one atomic flag per instance
struct AtomicFlags {
    flags: array<atomic<u32>>,
};

// BINDINGS
@group(0) @binding(0) var external_depth_texture: texture_depth_2d;
@group(0) @binding(1) var external_depth_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output_buffer: OutputBuffer;
@group(0) @binding(3) var<storage, read_write> instance_flags: AtomicFlags;

const EPSILON: f32 = 0.000001;

struct InstanceInput {
    @builtin(instance_index) index: u32,
    @location(1) coord: vec3<i32>,
    @location(2) value: u32,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) coord: vec3<i32>,
    @location(1) value: u32,
    @location(2) instance_index: u32,

}

// VERTEX SHADER
// Pass through vertex data and instance index
@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput
) -> VertexOutput {
    var out: VertexOutput;
    
    let instance_pos = vec3<f32>(instance.coord);
    let world_position = vec4<f32>(model.position + instance_pos, 1.0);
    out.clip_position = camera.view_proj * world_position;
    out.coord = instance.coord;
    out.value = instance.value;
    out.instance_index = instance.index;
    return out;
}


// FRAGMENT SHADER
@fragment
fn fs_main(in: VertexOutput) -> @location(0) u32 {
    // 1. Sample the external depth texture at the fragment's screen coordinate
    let frag_coords: vec2<f32> = in.clip_position.xy;
    //let sampled_depth = textureSample(external_depth_texture, external_depth_sampler, frag_coords);
    let coord = vec2<i32>(floor(frag_coords));
    let sampled_depth = textureLoad(external_depth_texture, coord, 0);
    // 2. Perform the depth check
    if (in.clip_position.z + EPSILON < sampled_depth) {
        // 3. Try to claim the write for this instance.
        // atomicExchange sets the flag to 1 and returns the *old* value.
        let previous_flag = atomicExchange(&instance_flags.flags[in.instance_index], 1u);

        // 4. If the old value was 0, we are the first! Write to the output buffer.
        if (previous_flag == 0u) {
            // Atomically increment the output counter and get the index to write to
            let output_index = atomicAdd(&output_buffer.count, 1u);
            
            // Write our instance data to the list
            output_buffer.data[output_index].coord = in.coord;
            
        }
    }

    // 5. Discard the fragment. We don't want to write to any color/depth attachment.
    return 0;
}
