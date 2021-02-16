////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebRTCRenderer.hpp"

#include "../logger.hpp"
#include "OSPRayUtility.hpp"

#include "../../../../src/cs-utils/convert.hpp"

#include <glm/gtc/matrix_transform.hpp>

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
        result.mMVP       = getOSPRayMVP(512., mCurrentTransform);
        result.mValid     = true;
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
  std::scoped_lock(mResultMutex);
  mCurrentTransform = cameraTransform;
  mGuiManager->getGui()->callJavascript(
      "CosmoScout.volumeRendering.capture", parameters.mResolution);
  mResultPromise                    = std::promise<RenderedImage>();
  std::future<RenderedImage> future = mResultPromise.get_future();
  future.wait();
  return future.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 WebRTCRenderer::getOSPRayMVP(float volumeHeight, glm::mat4 observerTransform) {
  // Scale observer transform according to the size of the volume
  observerTransform[3] =
      observerTransform[3] * glm::vec4(volumeHeight, volumeHeight, volumeHeight, 1);

  // Define vertical field of view for ospray camera
  float fov    = 90;
  float fovRad = cs::utils::convert::toRadians(fov);

  // Create camera transform looking along negative z
  glm::mat4 cameraTransform(1);
  cameraTransform[2][2] = -1;

  // Move camera to observer position relative to planet
  cameraTransform = observerTransform * cameraTransform;

  // Get base vectors of rotated coordinate system
  glm::vec3 camRight(cameraTransform[0]);
  camRight = glm::normalize(camRight);
  glm::vec3 camUp(cameraTransform[1]);
  camUp = glm::normalize(camUp);
  glm::vec3 camDir(cameraTransform[2]);
  camDir = glm::normalize(camDir);
  glm::vec3 camPos(cameraTransform[3]);

  // Get position of camera in rotated coordinate system
  float camXLen = glm::dot(camPos, camRight);
  float camYLen = glm::dot(camPos, camUp);
  float camZLen = glm::dot(camPos, camDir);

  // Get angle between camera position and forward vector
  float cameraAngleX = atan(camXLen / camZLen);
  float cameraAngleY = atan(camYLen / camZLen);

  // Get angle between ray towards center of volume and ray at edge of volume
  float modelAngleX = asin(volumeHeight / sqrt(camXLen * camXLen + camZLen * camZLen));
  float modelAngleY = asin(volumeHeight / sqrt(camYLen * camYLen + camZLen * camZLen));

  // Get angle between rays at edges of volume and forward vector
  float leftAngle, rightAngle, downAngle, upAngle;
  if (!isnan(modelAngleX) && !isnan(modelAngleY)) {
    leftAngle  = cameraAngleX - modelAngleX;
    rightAngle = cameraAngleX + modelAngleX;
    downAngle  = cameraAngleY - modelAngleY;
    upAngle    = cameraAngleY + modelAngleY;
  } else {
    // If the camera is inside the volume the model angles will be NaN,
    // so the angles are set to the edges of the field of view
    leftAngle  = -fovRad / 2;
    rightAngle = fovRad / 2;
    downAngle  = -fovRad / 2;
    upAngle    = fovRad / 2;
  }

  glm::mat4 view =
      glm::translate(glm::mat4(1.f), -glm::vec3(camXLen, camYLen, -camZLen) / volumeHeight);

  float nearClip = -camZLen / volumeHeight - 1;
  float farClip  = -camZLen / volumeHeight + 1;
  if (nearClip < 0) {
    nearClip = 0.00001f;
  }
  float     leftClip  = tan(leftAngle) * nearClip;
  float     rightClip = tan(rightAngle) * nearClip;
  float     downClip  = tan(downAngle) * nearClip;
  float     upClip    = tan(upAngle) * nearClip;
  glm::mat4 projection(0);
  projection[0][0] = 2 * nearClip / (rightClip - leftClip);
  projection[1][1] = 2 * nearClip / (upClip - downClip);
  projection[2][0] = (rightClip + leftClip) / (rightClip - leftClip);
  projection[2][1] = (upClip + downClip) / (upClip - downClip);
  projection[2][2] = -(farClip + nearClip) / (farClip - nearClip);
  projection[2][3] = -1;
  projection[3][2] = -2 * farClip * nearClip / (farClip - nearClip);

  return projection * view;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
