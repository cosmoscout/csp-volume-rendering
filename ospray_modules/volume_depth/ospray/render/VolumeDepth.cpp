// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "VolumeDepth.hpp"
#include "lights/AmbientLight.h"
#include "lights/HDRILight.h"
// ispc exports
#include "common/World_ispc.h"
#include "render/VolumeDepth_ispc.h"

#include <iostream>

namespace ospray {

namespace volumedepth {

VolumeDepth::VolumeDepth(int defaultNumSamples, bool defaultShadowsEnabled)
    : aoSamples(defaultNumSamples)
    , shadowsEnabled(defaultShadowsEnabled) {
  ispcEquivalent = ispc::VolumeDepth_create(this);
}

std::string VolumeDepth::toString() const {
  return "ospray::render::VolumeDepth";
}

void VolumeDepth::commit() {
  Renderer::commit();

  ispc::VolumeDepth_set(getIE(), getParam<int>("depthMode", 0),
      getParam<bool>("shadows", shadowsEnabled), getParam<int>("aoSamples", aoSamples),
      getParam<float>("aoDistance", getParam<float>("aoRadius", 1e20f)),
      getParam<float>("volumeSamplingRate", 1.f));
}

void* VolumeDepth::beginFrame(FrameBuffer*, World* world) {
  if (!world)
    return nullptr;

  if (world->scivisDataValid)
    return nullptr;

  std::vector<void*> lightArray;
  vec3f              aoColor = vec3f(0.f);

  if (world->lights) {
    for (auto&& light : *world->lights) {
      // extract color from ambient lights and remove them
      const AmbientLight* const ambient = dynamic_cast<const AmbientLight*>(light);
      if (ambient) {
        aoColor += ambient->radiance;
      } else {
        // also ignore HDRI lights TODO but put in background
        if (!dynamic_cast<const HDRILight*>(light)) {
          lightArray.push_back(light->getIE());
          if (light->getSecondIE().has_value()) {
            lightArray.push_back(light->getSecondIE().value());
					}
        }
      }
    }
  }

  void** lightPtr = lightArray.empty() ? nullptr : &lightArray[0];

  ispc::World_setSciVisData(world->getIE(), (ispc::vec3f&)aoColor, lightPtr, (uint32_t)lightArray.size());

  world->scivisDataValid = true;

  return nullptr;
}

} // namespace volumedepth
} // namespace ospray
