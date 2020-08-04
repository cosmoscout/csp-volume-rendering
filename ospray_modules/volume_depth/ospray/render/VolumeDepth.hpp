// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ospray_module_volume_depth_export.h"

#pragma warning(push)
#pragma warning(disable : 4275)
#include "render/Renderer.h"
#pragma warning(pop)

namespace ospray {

namespace volumedepth {

struct OSPRAY_MODULE_VOLUME_DEPTH_EXPORT VolumeDepth : public Renderer {
  VolumeDepth();
  std::string toString() const override;
  void        commit() override;
};

} // namespace volumedepth
} // namespace ospray
