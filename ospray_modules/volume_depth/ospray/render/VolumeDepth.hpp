// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ospray_module_volume_depth_export.h"

#include <vector>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4275)
#pragma warning(disable : 4996)
#endif
#include "render/Renderer.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace ospray {

namespace volumedepth {

struct ScalarFilter {
  int   attrIndex;
  float min;
  float max;
};

struct OSPRAY_MODULE_VOLUME_DEPTH_EXPORT VolumeDepth : public Renderer {
  VolumeDepth();
  std::string toString() const override;
  void        commit() override;
  void*       beginFrame(FrameBuffer*, World*) override;

 private:
  bool visibleLights{false};
  bool rendererValid{false};

  std::vector<ScalarFilter> filters;
};

} // namespace volumedepth
} // namespace ospray
