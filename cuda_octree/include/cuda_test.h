#pragma once

extern "C" {
  int init_cuda_thread();
  int test_vk_texture(int memFD, int semFD, int width, int height);
}
