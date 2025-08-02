#import "interop.h"
#include <cassert>
#include <iostream>

int cuda_vk_import_rp_tex(
    RenderBindings* out,
    int mem_fd,
    unsigned long long vk_allocation_size,
    unsigned long long vk_bind_offset,   // 0 if dedicated allocation
    unsigned width, unsigned height,
    int color_attachment,                // 1 if image used as color target
    int dedicated_allocation,            // 1 if VkMemoryDedicatedAllocateInfo was used
    int cu_fmt, unsigned num_channels  // e.g., CU_AD_FORMAT_SIGNED_INT32, 4
) {

    CUexternalMemory extMem = 0;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
    memDesc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memDesc.handle.fd = mem_fd;                    // CUDA takes ownership on success
    memDesc.size      = vk_allocation_size;        // <-- EXACT Vulkan allocation size
    memDesc.flags     = dedicated_allocation ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;
    CHECK(cuImportExternalMemory(&extMem, &memDesc));

    CUmipmappedArray mip = 0;
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC mm = {};
    mm.offset    = vk_bind_offset;   // 0 if dedicated
    mm.numLevels = 1;

    CUDA_ARRAY3D_DESCRIPTOR arr = {};
    arr.Width       = width;
    arr.Height      = height;
    arr.Depth       = 0;                       // 2D
    arr.Format      = (CUarray_format)cu_fmt;                  // e.g., CU_AD_FORMAT_SIGNED_INT32
    arr.NumChannels = num_channels;            // e.g., 4
    arr.Flags       = 0;
    if (color_attachment)
        arr.Flags |= CUDA_ARRAY3D_COLOR_ATTACHMENT; 

    mm.arrayDesc = arr;

    CUresult r = cuExternalMemoryGetMappedMipmappedArray(&mip, extMem, &mm);
    if (r != CUDA_SUCCESS) {
        const char *n=0,*d=0; cuGetErrorName(r,&n); cuGetErrorString(r,&d);
        fprintf(stderr, "cuExternalMemoryGetMappedMipmappedArray failed: %s - %s\n",
                n?n:"", d?d:"");
        // If buffer probe succeeded but array probe failed, it’s almost certainly format/flags.
        cuDestroyExternalMemory(extMem);
        return -2;
    }
    out->rp_target = mip;
    out->rp_size = vk_allocation_size;
    out->rp_width = width;
    out->rp_height = height;
    return 1;
}

int bind_vk(RenderBindings* out, VulkanExtBindingsInfo* vk_bnd) {
  int im_res = cuda_vk_import_rp_tex(out, vk_bnd->rp_mem_fd, vk_bnd->rp_size, 0, vk_bnd->rp_width, vk_bnd->rp_height, 1, 0, CU_AD_FORMAT_SIGNED_INT32, 4);
  if (!im_res)
    return im_res;

  return 1;
}

RenderBindings* init_bindings(VulkanExtBindingsInfo* vk_bindings, uint32_t count) {
  assert(count < 400);

  RenderBindings* bindings = (RenderBindings*) malloc(count * sizeof(RenderBindings));
  
  for (uint32_t i = 0; i < count; ++i) {
    RenderBindings* bnd = &bnd[i];
    int res = bind_vk(bnd, &vk_bindings[i]);
    if(!res) {
      free(bindings);
      return nullptr;
    }
  }

  return bindings;
}

  
