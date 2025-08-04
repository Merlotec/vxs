// Clean Frustum Culling Compute Shader
// Properly handles Vector3<i32> coordinates

// Match the CameraUniform structure from voxelcoord.wgsl exactly
struct CameraUniform {
    view_proj: mat4x4<f32>,
    pos: vec3<f32>,
}

struct CellInstance {
    position: vec3<i32>,  // Matches Rust Vector3<i32> exactly
    value: u32,
}

struct CullParams {
    instance_count: u32,
    visible_count: atomic<u32>,
    _padding: array<u32, 2>,
}

@group(0) @binding(0) var<storage, read> input_instances: array<CellInstance>;
@group(0) @binding(1) var<storage, read_write> output_instances: array<CellInstance>;
@group(0) @binding(2) var<storage, read_write> cull_params: CullParams;
@group(0) @binding(3) var<uniform> camera: CameraUniform;

// Test if a unit cube at center is visible in the frustum
// Use the same transformation approach as voxelcoord.wgsl
fn is_cube_visible(center: vec3<f32>) -> bool {
    // Transform center to clip space (same as voxelcoord.wgsl)
    let world_position = vec4<f32>(center, 1.0);
    let clip_position = camera.view_proj * world_position;
    
    // Perform perspective divide to get NDC coordinates
    let ndc = clip_position.xyz / clip_position.w;
    
    // Test if the point is within the NDC cube [-1, 1] in all axes
    // Add some tolerance for the cube size
    let tolerance = 1.0; // Be generous with culling bounds
    
    return ndc.x >= -tolerance && ndc.x <= tolerance &&
           ndc.y >= -tolerance && ndc.y <= tolerance &&
           ndc.z >= -tolerance && ndc.z <= tolerance;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if index >= cull_params.instance_count {
        return;
    }
    
    let instance = input_instances[index];
    
    // Convert integer coordinates to float world position
    let world_pos = vec3<f32>(
        f32(instance.position.x),
        f32(instance.position.y),
        f32(instance.position.z)
    );
    
    // Test visibility
    if is_cube_visible(world_pos) {
        // Get next output slot atomically
        let output_index = atomicAdd(&cull_params.visible_count, 1u);
        
        // Write visible instance (with bounds check)
        if output_index < arrayLength(&output_instances) {
            output_instances[output_index] = instance;
        }
    }
}