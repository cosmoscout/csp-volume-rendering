////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
#define CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP

#include "OSPRayUtility.hpp"
#include "Renderer.hpp"

#include <ospray/ospray_cpp.h>
// Boost defines a function called likely so this macro from platform.h has to be undeffed
#undef likely

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

class OSPRayRenderer : public Renderer {
 public:
  OSPRayRenderer(std::shared_ptr<DataManager> dataManager, Renderer::VolumeStructure structure,
      Renderer::VolumeShape shape);
  ~OSPRayRenderer();

  OSPRayRenderer(const OSPRayRenderer& other) = delete;
  OSPRayRenderer& operator=(const OSPRayRenderer& other) = delete;

  void setTransferFunction(std::vector<glm::vec4> colors) override;

  std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> getFrame(int resolution,
      glm::mat4 cameraTransform, float samplingRate, DepthMode depthMode, bool denoiseColor,
      bool denoiseDepth, bool shading) override;

 private:
  std::shared_future<ospray::cpp::TransferFunction> mTransferFunction;
  std::optional<ospray::cpp::Volume>                mVolume;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
