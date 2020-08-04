// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VolumeDepth.hpp"
// ispc exports
#include "VolumeDepth_ispc.h"

namespace ospray {

namespace volumedepth {

VolumeDepth::VolumeDepth(int defaultNumSamples)
    : aoSamples(defaultNumSamples) {
  ispcEquivalent = ispc::VolumeDepth_create(this);
}

std::string VolumeDepth::toString() const {
  return "ospray::render::VolumeDepth";
}

void VolumeDepth::commit() {
  Renderer::commit();

  ispc::VolumeDepth_set(getIE(), getParam<int>("aoSamples", aoSamples),
      getParam<float>("aoRadius", 1e20f), getParam<float>("aoIntensity", 1.f),
      getParam<float>("volumeSamplingRate", 1.f));
}

} // namespace volumedepth
} // namespace ospray
