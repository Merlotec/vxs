#include "cuda_test.h"
#include <cuda.h>
#include <iostream>


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

int init_cuda_thread() {
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuCtxCreate(&ctx, 0, dev);
  return 0;
}

int test_vk_texture(int memFD, int semFD, int width, int height, uint64_t vk_allocation_size, uint64_t vk_bind_offset, int is_dedicated_allocation) {
  CUresult err;

  // 1) Import the Vulkan memory FD into CUDA
  CUexternalMemory extMem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc{};
  memDesc.type       = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  memDesc.handle.fd  = memFD;
  memDesc.size       = vk_allocation_size;
  memDesc.flags      = is_dedicated_allocation ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;
  err = cuImportExternalMemory(&extMem, &memDesc);
  if (err != CUDA_SUCCESS) {
    std::cerr << "cuImportExternalMemory failed: " << err << "\n";
    return -1;
  }

  // 2) Describe & map level-0 of the image as a 2D int4 array
  CUmipmappedArray mip = nullptr;
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrDesc{};
  arrDesc.numLevels               = 1;
  arrDesc.offset = vk_allocation_size;
  arrDesc.arrayDesc.Width         = width;
  arrDesc.arrayDesc.Height        = height;
  arrDesc.arrayDesc.Depth         = 0;                       // 2D
  arrDesc.arrayDesc.Format        = CU_AD_FORMAT_SIGNED_INT32;
  arrDesc.arrayDesc.NumChannels   = 4;                       // vec4<i32>
  arrDesc.arrayDesc.Flags         |= CUDA_ARRAY3D_COLOR_ATTACHMENT;
  err = cuExternalMemoryGetMappedMipmappedArray(&mip, extMem, &arrDesc);
  if (err != CUDA_SUCCESS) {
    const char* name   = nullptr;
    const char* desc   = nullptr;
    cuGetErrorName(err,  &name);
    cuGetErrorString(err, &desc);
    std::cerr << "cuExternalMemoryGetMappedMipmappedArray failed: " << err << name << desc << "\n";
    cuDestroyExternalMemory(extMem);
    return -1;
  }

  // 3) Pull out the single level
  CUarray cuArray = nullptr;
  err = cuMipmappedArrayGetLevel(&cuArray, mip, 0);
  if (err != CUDA_SUCCESS) {
    std::cerr << "cuMipmappedArrayGetLevel failed: " << err << "\n";
    cuDestroyExternalMemory(extMem);
    return -1;
  }

  // 4) Import & wait on the Vulkan-signaled semaphore
  CUexternalSemaphore extSem = nullptr;
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC semDesc{};
  semDesc.type      = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD;
  semDesc.handle.fd = semFD;
  err = cuImportExternalSemaphore(&extSem, &semDesc);
  if (err != CUDA_SUCCESS) {
    std::cerr << "cuImportExternalSemaphore failed: " << err << "\n";
    cuDestroyExternalMemory(extMem);
    return -1;
  }

  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS waitParams{};
  waitParams.flags = 0;
  // For a binary semaphore, fenceValue and other fields are ignored
  err = cuWaitExternalSemaphoresAsync(&extSem, &waitParams, 1, 0 /* default stream */);
  if (err != CUDA_SUCCESS) {
    std::cerr << "cuWaitExternalSemaphoresAsync failed: " << err << "\n";
    cuDestroyExternalSemaphore(extSem);
    cuDestroyExternalMemory(extMem);
    return -1;
  }

  // Optionally sync the stream to be 100% sure before host reading
  cuStreamSynchronize(0);

  // … now you can use `cuArray` in kernels or with cuTexObjectCreate …

  // 5) Cleanup
  cuDestroyExternalSemaphore(extSem);
  cuDestroyExternalMemory(extMem);
  return 0;
}

// Returns 0 on success; prints diagnostics on failures.
// Use exactly the Vulkan allocation size and bind offset you used on the Vulkan side.
int probe_cuda_import(
    int mem_fd,
    unsigned long long vk_allocation_size,
    unsigned long long vk_bind_offset,   // 0 if dedicated allocation
    unsigned width, unsigned height,
    int color_attachment,                // 1 if image used as color target
    int dedicated_allocation,            // 1 if VkMemoryDedicatedAllocateInfo was used
    int cu_fmt, unsigned num_channels  // e.g., CU_AD_FORMAT_SIGNED_INT32, 4
) {
    // CHECK(cuInit(0));
    // CUdevice dev; CHECK(cuDeviceGet(&dev, 0));
    // CUcontext ctx; CHECK(cuCtxCreate(&ctx, 0, dev));

    // 1) Import external memory
    CUexternalMemory extMem = 0;
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc = {};
    memDesc.type      = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
    memDesc.handle.fd = mem_fd;                    // CUDA takes ownership on success
    memDesc.size      = vk_allocation_size;        // <-- EXACT Vulkan allocation size
    memDesc.flags     = dedicated_allocation ? CUDA_EXTERNAL_MEMORY_DEDICATED : 0;
    CHECK(cuImportExternalMemory(&extMem, &memDesc));

    // 2) BUFFER probe (sanity check for FD/size/offset)
    CUdeviceptr devPtr = 0;
    CUDA_EXTERNAL_MEMORY_BUFFER_DESC b = {};
    b.offset = vk_bind_offset; // 0 for dedicated
    // map something modest, but aligned by construction (using the full tail is simplest)
    unsigned long long map_size = vk_allocation_size - vk_bind_offset;
    if (map_size > (1ull<<20)) map_size = (1ull<<20); // map up to 1 MiB for the probe
    b.size   = map_size;
    b.flags  = 0; // must be zero
    CHECK(cuExternalMemoryGetMappedBuffer(&devPtr, extMem, &b));  // may fail if size/offset wrong

    // Try reading a few bytes to host (proves the mapping is valid)
    unsigned char tmp[64];
    memset(tmp, 0xCD, sizeof(tmp));
    CHECK(cuMemcpyDtoH(tmp, devPtr, sizeof(tmp)));               // a small read should succeed

    // Free the temporary buffer mapping (required by CUDA; otherwise Destroy will complain)
    CHECK(cuMemFree(devPtr));  // driver API requires cuMemFree on mapped buffers
    // (Doc: buffers mapped from external memory must be freed with cuMemFree)  [oai_citation:2‡NVIDIA Docs](https://docs.nvidia.com/cuda/archive/11.4.4/pdf/CUDA_Driver_API.pdf)

    // 3) ARRAY probe (format/flags/extent correctness)
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
        arr.Flags |= CUDA_ARRAY3D_COLOR_ATTACHMENT; // REQUIRED for render targets  [oai_citation:3‡NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXTRES__INTEROP.html)
    // Optional: if you’ll use CUDA surface writes, also add CUDA_ARRAY3D_SURFACE_LDST.

    mm.arrayDesc = arr;

    CUresult r = cuExternalMemoryGetMappedMipmappedArray(&mip, extMem, &mm);
    if (r != CUDA_SUCCESS) {
        const char *n=0,*d=0; cuGetErrorName(r,&n); cuGetErrorString(r,&d);
        fprintf(stderr, "cuExternalMemoryGetMappedMipmappedArray failed: %s - %s\n",
                n?n:"", d?d:"");
        // If buffer probe succeeded but array probe failed, it’s almost certainly format/flags.
        cuDestroyExternalMemory(extMem);
        // cuCtxDestroy(ctx);
        return -2;
    }

    // Pull level-0 and sanity-copy a tiny tile to host
    CUarray level0 = 0; CHECK(cuMipmappedArrayGetLevel(&level0, mip, 0));

    // Query back CUDA’s view of the array (helps confirm your descriptor matched)
    CUDA_ARRAY_DESCRIPTOR arr_out = {};
    CHECK(cuArrayGetDescriptor(&arr_out, level0));
    fprintf(stderr, "CUDA array: %ux%u fmt=%u numCh=%u\n",
            (unsigned)arr_out.Width, (unsigned)arr_out.Height,
            (unsigned)arr_out.Format, (unsigned)arr_out.NumChannels);

    // Try a 16x16 read to host using cuMemcpy2D
    const unsigned tileW = (width  < 16 ? width  : 16);
    const unsigned tileH = (height < 16 ? height : 16);
    const size_t   elemB = 4 /*int32*/ * num_channels;  // vec4<i32> => 16 bytes/texel
    void* host = malloc(tileW * tileH * elemB);

    CUDA_MEMCPY2D c2d = {};
    c2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    c2d.srcArray      = level0;
    c2d.dstMemoryType = CU_MEMORYTYPE_HOST;
    c2d.dstHost       = host;
    c2d.dstPitch      = tileW * elemB;
    c2d.WidthInBytes  = tileW * elemB;
    c2d.Height        = tileH;
    CHECK(cuMemcpy2D(&c2d)); // if this works, the array is mapped and accessible

    // cleanup
    free(host);
    // No explicit cuArrayDestroy for external arrays; destroying the external memory drops it:
    CHECK(cuDestroyExternalMemory(extMem));
    // CHECK(cuCtxDestroy(ctx));
    return 0;
}
