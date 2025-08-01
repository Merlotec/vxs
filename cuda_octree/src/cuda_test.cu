#include "cuda_test.h"
#include <cuda.h>
#include <iostream>

int init_cuda_thread() {
  cuInit(0);
  CUdevice dev;
  cuDeviceGet(&dev, 0);
  CUcontext ctx;
  cuCtxCreate(&ctx, 0, dev);
  return 0;
}

int test_vk_texture(int memFD, int semFD, int width, int height) {
  CUresult err;

  int size = width * height * 4 * 4;

  // 1) Import the Vulkan memory FD into CUDA
  CUexternalMemory extMem = nullptr;
  CUDA_EXTERNAL_MEMORY_HANDLE_DESC memDesc{};
  memDesc.type       = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
  memDesc.handle.fd  = memFD;
  memDesc.size       = size;
  memDesc.flags      = 0;
  err = cuImportExternalMemory(&extMem, &memDesc);
  if (err != CUDA_SUCCESS) {
    std::cerr << "cuImportExternalMemory failed: " << err << "\n";
    return -1;
  }

  // 2) Describe & map level-0 of the image as a 2D int4 array
  CUmipmappedArray mip = nullptr;
  CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC arrDesc{};
  arrDesc.numLevels               = 1;
  arrDesc.offset = 0;
  arrDesc.arrayDesc.Width         = width;
  arrDesc.arrayDesc.Height        = height;
  arrDesc.arrayDesc.Depth         = 0;                       // 2D
  arrDesc.arrayDesc.Format        = CU_AD_FORMAT_SIGNED_INT32;
  arrDesc.arrayDesc.NumChannels   = 4;                       // vec4<i32>
  arrDesc.arrayDesc.Flags         = CUDA_ARRAY3D_COLOR_ATTACHMENT;
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
