// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING

// ospray
#include "VolumeDepth.hpp"
#include "lights/AmbientLight.h"
#include "lights/HDRILight.h"
#include "lights/SunSkyLight.h"
// ispc exports
#include "common/World_ispc.h"
#include "render/VolumeDepth_ispc.h"

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

  rendererValid = false;

  filters       = std::vector<ScalarFilter>((ScalarFilter*)getParam<void*>("scalarFilters"),
      (ScalarFilter*)getParam<void*>("scalarFilters") + getParam<int>("numScalarFilters"));
  visibleLights = getParam<bool>("visibleLights", false);

  ispc::VolumeDepth_set(getIE(), getParam<int>("depthMode", 0), getParam<bool>("shadows", false),
      getParam<int>("aoSamples", 0),
      getParam<float>("aoDistance", getParam<float>("aoRadius", 1e20f)),
      getParam<float>("volumeSamplingRate", 1.f), filters.data(), (int32_t)filters.size(),
      getParam<float>("minDepth", 0), getParam<float>("maxDepth", inf));
}

void* VolumeDepth::beginFrame(FrameBuffer*, World* world) {
  if (!world)
    return nullptr;

  if (world->scivisDataValid && rendererValid)
    return nullptr;

  std::vector<void*> lightArray;
  std::vector<void*> lightVisibleArray;
  vec3f              aoColor = vec3f(0.f);

  if (world->lights) {
    for (auto&& light : *world->lights) {
      // extract color from ambient lights and remove them
      const auto ambient = dynamic_cast<const AmbientLight*>(light);
      if (ambient) {
        aoColor += ambient->radiance;
        if (visibleLights)
          lightVisibleArray.push_back(light->getIE());
        continue;
      }

      // hdri lights are only (potentially) visible, no illumination
      const auto hdri = dynamic_cast<const HDRILight*>(light);
      if (hdri) {
        if (visibleLights)
          lightVisibleArray.push_back(light->getIE());
        continue;
      }

      // sun-sky: only sun illuminates, sky only (potentially) in background
      const auto sunsky = dynamic_cast<const SunSkyLight*>(light);
      if (sunsky) {
        lightArray.push_back(light->getSecondIE().value()); // sun
        if (visibleLights) {
          lightVisibleArray.push_back(light->getIE());               // sky
          lightVisibleArray.push_back(light->getSecondIE().value()); // sun
        }
        continue;
      }

      // handle the remaining types of lights
      lightArray.push_back(light->getIE());
      if (visibleLights)
        lightVisibleArray.push_back(light->getIE());
    }
  }

  ispc::World_setSciVisData(world->getIE(), ispc::vec3f{aoColor[0], aoColor[1], aoColor[2]},
      lightArray.empty() ? nullptr : &lightArray[0], (uint32_t)lightArray.size(),
      lightVisibleArray.empty() ? nullptr : &lightVisibleArray[0],
      (uint32_t)lightVisibleArray.size());

  world->scivisDataValid = true;
  rendererValid          = true;

  return nullptr;
}

} // namespace volumedepth
} // namespace ospray
