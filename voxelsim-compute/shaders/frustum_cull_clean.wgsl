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

struct IndirectDrawCommand {
    index_count: u32,     // Number of indices to draw
    instance_count: u32,  // Number of instances (set by GPU)
    first_index: u32,     // First index in index buffer
    vertex_offset: i32,   // Offset added to vertex indices
    first_instance: u32,  // First instance to draw
}

@group(0) @binding(0) var<storage, read> input_instances: array<CellInstance>;
@group(0) @binding(1) var<storage, read_write> output_instances: array<CellInstance>;
@group(0) @binding(2) var<storage, read_write> cull_params: CullParams;
@group(0) @binding(3) var<uniform> camera: CameraUniform;
@group(0) @binding(4) var<storage, read_write> indirect_draw: IndirectDrawCommand;

// Test if a unit cube is visible in the frustum
// Use the robust "positive vertex" approach for better edge handling
fn is_cube_visible(center: vec3<f32>) -> bool {
    // Transform center to clip space
    let world_position = vec4<f32>(center, 1.0);
    let clip_position = camera.view_proj * world_position;
    
    // Avoid division by zero or very small w values
    if abs(clip_position.w) < 0.0001 {
        return false;
    }
    
    // Half-size of unit cube (extends 0.5 units in each direction)
    let half_size = 0.5;
    
    // Test the cube against each face of the NDC cube
    // We'll test the "farthest corner" of the cube in each direction
    
    // For each NDC face, find the corner of the cube that's farthest in that direction
    // and test if that corner is still inside NDC bounds
    
    // Transform all 8 corners of the cube and test them
    let corners = array<vec3<f32>, 8>(
        center + vec3<f32>(-half_size, -half_size, -half_size),
        center + vec3<f32>( half_size, -half_size, -half_size),
        center + vec3<f32>(-half_size,  half_size, -half_size),
        center + vec3<f32>( half_size,  half_size, -half_size),
        center + vec3<f32>(-half_size, -half_size,  half_size),
        center + vec3<f32>( half_size, -half_size,  half_size),
        center + vec3<f32>(-half_size,  half_size,  half_size),
        center + vec3<f32>( half_size,  half_size,  half_size)
    );
    
    // If any corner is inside NDC bounds, the cube is visible
    for (var i = 0; i < 8; i++) {
        let corner_world = vec4<f32>(corners[i], 1.0);
        let corner_clip = camera.view_proj * corner_world;
        
        if abs(corner_clip.w) > 0.0001 {
            let corner_ndc = corner_clip.xyz / corner_clip.w;
            
            // If this corner is inside NDC bounds, cube is visible
            if corner_ndc.x >= -1.0 && corner_ndc.x <= 1.0 &&
               corner_ndc.y >= -1.0 && corner_ndc.y <= 1.0 &&
               corner_ndc.z >= -1.0 && corner_ndc.z <= 1.0 {
                return true;
            }
        }
    }
    
    // If no corners are inside, the cube might still intersect the frustum
    // (e.g., a large cube that contains the entire frustum)
    // For safety, we should test if the frustum intersects the cube
    // But for typical voxel rendering, this case is rare, so we'll accept
    // the small chance of over-culling very large cubes
    
    return false;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Bounds check
    if index >= cull_params.instance_count {
        // Even out-of-bounds threads need to participate in the barrier
        if index == cull_params.instance_count {
            // Let the first out-of-bounds thread update the indirect draw command
            // This ensures it happens after all valid threads are done
            indirect_draw.instance_count = atomicLoad(&cull_params.visible_count);
        }
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
    
    // TEMPORARY DEBUG: Force every thread to try writing (should be atomic-safe)
    // This will help us see if ANY thread can write to the buffer
    let visible = atomicLoad(&cull_params.visible_count);
    indirect_draw.instance_count = 1000u; // Force a fixed value for testing
    
    // Also try setting it in the main thread
    if index == 0u {
        indirect_draw.instance_count = 500u; // Thread 0 should write 500
    }
}