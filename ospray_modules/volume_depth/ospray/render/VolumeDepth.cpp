// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VolumeDepth.hpp"
// ispc exports
#include "VolumeDepth_ispc.h"

namespace ospray {

namespace volumedepth {

VolumeDepth::VolumeDepth() {
  ispcEquivalent = ispc::VolumeDepth_create(this);
}

std::string VolumeDepth::toString() const {
  return "ospray::render::VolumeDepth";
}

void VolumeDepth::commit() {
  Renderer::commit();

  ispc::VolumeDepth_set(
      getIE(), getParam<float>("volumeSamplingRate", 1.f), getParam<int>("depthMode", 0));
}

} // namespace volumedepth
} // namespace ospray
