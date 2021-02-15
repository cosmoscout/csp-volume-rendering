////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebRTCRenderer.hpp"

#include "../logger.hpp"
#include "OSPRayUtility.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCRenderer::WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
    VolumeShape shape, std::shared_ptr<cs::core::GuiManager> guiManager)
    : Renderer(dataManager, structure, shape)
    , mGuiManager(guiManager) {
  mGuiManager->getGui()->registerCallback("volumeRendering.captureColorImage",
      "Capture a single frame from the WebRTC stream.", std::function([this](std::string data) {
        // Decode base64
        using It = boost::archive::iterators::transform_width<
            boost::archive::iterators::binary_from_base64<std::string::const_iterator>, 8, 6>;
        auto binary = std::vector<unsigned char>(It(data.begin()), It(data.end()));

        // Remove padding.
        auto length = data.size();
        if (binary.size() > 2 && data[length - 1] == '=' && data[length - 2] == '=') {
          binary.erase(binary.end() - 2, binary.end());
        } else if (binary.size() > 1 && data[length - 1] == '=') {
          binary.erase(binary.end() - 1, binary.end());
        }

        // Load image
        int            w, h, c;
        int            channels = 4;
        unsigned char* image =
            stbi_load_from_memory(binary.data(), (int)binary.size(), &w, &h, &c, channels);

        RenderedImage result;
        result.mColorData = std::vector<unsigned char>(image, image + w * h * c);
        result.mDepthData = std::vector<float>(w * h);
        // result.mMVP = ;
        result.mValid = true;
        mResultPromise.set_value(result);
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCRenderer::~WebRTCRenderer() {
  mGuiManager->getGui()->unregisterCallback("volumeRendering.captureColorImage");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

float WebRTCRenderer::getProgress() {
  // TODO Implement
  return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCRenderer::preloadData(DataManager::State state) {
  // TODO Implement
  logger().warn("Preloading not implemented yet");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCRenderer::cancelRendering() {
  // TODO Implement
  logger().warn("Canceling not implemented yet");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::RenderedImage WebRTCRenderer::getFrameImpl(
    glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) {
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.volumeRendering.capture", parameters.mResolution);
  mResultPromise                    = std::promise<RenderedImage>();
  std::future<RenderedImage> future = mResultPromise.get_future();
  future.wait();
  return future.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
