#import "interop.h"
#include <cassert>

RenderBindings bind_vk(VulkanExtBindingsInfo* vk_bnd) {
  
}

RenderBindings* init_bindings(VulkanExtBindingsInfo* vk_info, uint32_t count) {
  assert(count < 400);

  RenderBindings* bindings = (RenderBindings*) malloc(count * sizeof(RenderBindings));
  
  for (uint32_t i = 0; i < count; ++i) {
    RenderBindings bnd = bind_vk();
  }
}

  
