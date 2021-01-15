// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ospray_module_volume_depth_export.h"

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4275)
#endif
#include "render/Renderer.h"
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace ospray {

namespace volumedepth {

struct OSPRAY_MODULE_VOLUME_DEPTH_EXPORT VolumeDepth : public Renderer {
  VolumeDepth(int defaultAOSamples = 0, bool defaultShadowsEnabled = false);
  std::string toString() const override;
  void        commit() override;
  void*       beginFrame(FrameBuffer*, World*) override;

 private:
  int  aoSamples{0};
  bool shadowsEnabled{false};
};

} // namespace volumedepth
} // namespace ospray
