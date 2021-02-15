////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
#define CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP

#include "Renderer.hpp"

#include "../../../../src/cs-core/GuiManager.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

class WebRTCRenderer : public Renderer {
 public:
  WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
      VolumeShape shape, std::shared_ptr<cs::core::GuiManager> guiManager);
  ~WebRTCRenderer();

  float getProgress() override;
  void  preloadData(DataManager::State state) override;
  void  cancelRendering() override;

 private:
  RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) override;

  glm::mat4 getOSPRayMVP(float volumeHeight, glm::mat4 observerTransform);

  std::shared_ptr<cs::core::GuiManager> mGuiManager;

  std::promise<RenderedImage> mResultPromise;
  glm::mat4                   mCurrentTransform;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
