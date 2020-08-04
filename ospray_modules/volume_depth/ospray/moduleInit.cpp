// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ospray/version.h"
#include "render/VolumeDepth.hpp"

namespace ospray {

namespace volumedepth {

extern "C" OSPError OSPRAY_DLLEXPORT ospray_module_init_volume_depth(
    int16_t versionMajor, int16_t versionMinor, int16_t versionPatch) {
  auto status = moduleVersionCheck(versionMajor, versionMinor);

  if (status == OSP_NO_ERROR) {
    Renderer::registerType<VolumeDepth>("volume_depth");
  }

  return status;
}

} // namespace volumedepth
} // namespace ospray
