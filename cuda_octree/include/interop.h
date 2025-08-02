#include <cuda.h>
#include <stdint.h>

#define CHECK(call) do {                                 \
    CUresult _e = (call);                                \
    if (_e != CUDA_SUCCESS) {                            \
        const char *n = 0, *d = 0;                       \
        cuGetErrorName(_e, &n);                          \
        cuGetErrorString(_e, &d);                        \
        fprintf(stderr, "%s failed: %d %s - %s\n",       \
                #call, _e, n?n:"", d?d:"");              \
        return -1;                                       \
    }                                                    \
} while (0)

/// One set of bindings for a pipeline.
struct RenderBindings {

  /// The render target of the main pass.
  CUmipmappedArray rp_target;
  uint32_t rp_width; uint32_t rp_height;
  uint64_t rp_size;
  CUexternalSemaphore rp_sem;

  // Removal buffer.
  CUarray rm_buffer;
  uint64_t rm_size;
  CUexternalSemaphore rm_sem;
};

struct VulkanExtBindingsInfo {
  int rp_mem_fd;
  uint32_t rp_width;
  uint32_t rp_height;
  uint64_t rp_size;
  int rp_sem_fd;

  int rm_mem_fd;
  uint64_t rm_size;
  int rm_sem_fd; 
};

extern "C" {
  int bind_vk(RenderBindings* out, VulkanExtBindingsInfo* vk_bnd);
  
  RenderBindings* init_bindings(VulkanExtBindingsInfo* vk_bindings, uint32_t count);
}

