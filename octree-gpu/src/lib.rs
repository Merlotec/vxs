use std::ffi::{c_int, c_uint, c_ulonglong};

// bindings.rs

/// Mirrors the C struct exactly.
#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub struct VulkanExtBindingsInfo {
    pub rp_mem_fd: c_int,
    pub rp_width: u32,
    pub rp_height: u32,
    pub rp_size: u64,
    pub rp_sem_fd: c_int,

    pub rm_mem_fd: c_int,
    pub rm_size: u64,
    pub rm_sem_fd: c_int,
}

/// Opaque handle to a C-allocated RenderBindings. Size is unknown on the Rust side.
/// Use only behind raw pointers.
#[repr(C)]
pub struct RenderBindings {
    _private: [u8; 0],
}

#[link(name = "CudaOctree", kind = "static")]
unsafe extern "C" {
    pub fn init_cuda_thread();
    pub fn test_vk_texture(
        memFD: c_int,
        semFD: c_int,
        width: c_int,
        height: c_int,
        vk_allocation_size: u64,
        vk_bind_offset: u64,
        is_dedicated_allocation: c_int,
    ) -> c_int;

    pub fn probe_cuda_import(
        mem_fd: c_int,
        vk_allocation_size: u64,
        vk_bind_offset: u64,
        width: c_uint,
        height: c_uint,
        color_attachment: c_int,
        dedicated_allocation: c_int,
        cu_fmt: c_int,
        num_channels: c_uint,
    ) -> c_int;

    /// int bind_vk(RenderBindings* out, VulkanExtBindingsInfo* vk_bnd);
    pub fn bind_vk(out: *mut RenderBindings, vk_bnd: *mut VulkanExtBindingsInfo) -> c_int;

    /// RenderBindings* init_bindings(VulkanExtBindingsInfo* vk_bindings, uint32_t count);
    pub fn init_bindings(
        vk_bindings: *const VulkanExtBindingsInfo,
        count: u32,
    ) -> *mut RenderBindings;
}
