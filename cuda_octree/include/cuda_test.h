#pragma once
#include <stdint.h>

extern "C" {
  int init_cuda_thread();
  int test_vk_texture(int memFD, int semFD, int width, int height, uint64_t vk_allocation_size, uint64_t vk_bind_offset, int is_dedicated_allocation);

  int probe_cuda_import(
      int mem_fd,
      unsigned long long vk_allocation_size,
      unsigned long long vk_bind_offset,   // 0 if dedicated allocation
      unsigned width, unsigned height,
      int color_attachment,                // 1 if image used as color target
      int dedicated_allocation,            // 1 if VkMemoryDedicatedAllocateInfo was used
      int cu_fmt, unsigned num_channels  // e.g., CU_AD_FORMAT_SIGNED_INT32, 4
  ); 
}
