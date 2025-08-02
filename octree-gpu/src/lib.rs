use std::ffi::{c_int, c_uint, c_ulonglong};

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
}
