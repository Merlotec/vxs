use std::ffi::c_int;

#[link(name = "CudaOctree", kind = "static")]
unsafe extern "C" {
    pub fn init_cuda_thread();
    pub fn test_vk_texture(
        memFD: c_int,
        semFD: c_int,
        size: c_int,
        width: c_int,
        height: c_int,
    ) -> c_int;
}
