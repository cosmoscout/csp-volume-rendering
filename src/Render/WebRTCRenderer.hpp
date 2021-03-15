////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
#define CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP

#include "Renderer.hpp"

#include "WebRTCUtility/Stream.hpp"

#include "../../../../src/cs-core/GuiManager.hpp"
#include "../../../../src/cs-utils/Signal.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

class WebRTCRenderer : public Renderer {
 public:
  WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
      VolumeShape shape, std::string signallingUrl);
  ~WebRTCRenderer();

  void update() override;

  float getProgress() override;
  void  preloadData(DataManager::State state) override;
  void  cancelRendering() override;

 private:
  RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) override;

  std::pair<std::string, glm::mat4> getOSPRayCamera(
      float volumeHeight, glm::mat4 observerTransform);

  const SampleType mType;
  webrtc::Stream   mStream;

  int                     mUncurrentRequiredSignal = -1;
  int                     mUncurrentReleaseSignal  = -1;
  std::mutex              mUncurrentRequiredMutex;
  std::mutex              mUncurrentReleaseMutex;
  std::mutex              mUncurrentDoneMutex;
  std::condition_variable mUncurrentRequiredCV;
  std::condition_variable mUncurrentReleaseCV;
  bool                    mContextCurrentIs       = true;
  bool                    mContextCurrentShouldBe = true;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
