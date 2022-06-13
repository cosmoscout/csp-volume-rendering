////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Utility.hpp"
#include "logger.hpp"

#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::volumerendering::Utility {

////////////////////////////////////////////////////////////////////////////////////////////////////

CameraParams calculateCameraParams(
    float volumeHeight, glm::mat4 observerTransform, float fovY, float fovX) {
  CameraParams crop;

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
  if (!std::isnan(modelAngleX) && !std::isnan(modelAngleY) && modelAngleX < 3.141f / 4 && modelAngleY < 3.141f / 4) {
    leftAngle  = cameraAngleX - modelAngleX;
    rightAngle = cameraAngleX + modelAngleX;
    downAngle  = cameraAngleY - modelAngleY;
    upAngle    = cameraAngleY + modelAngleY;

    crop.mInside = false;
  } else {
    // If the camera is inside the volume the model angles will be NaN,
    // so the angles are set to the edges of the field of view
    leftAngle  = -fovX / 2;
    rightAngle = fovX / 2;
    downAngle  = -fovY / 2;
    upAngle    = fovY / 2;

    crop.mInside = true;
  }

  // Get model, view and projection matrices
  glm::mat4 model = glm::scale(glm::mat4(1), glm::vec3(volumeHeight));
  glm::mat4 view  = glm::translate(glm::mat4(1.f), -glm::vec3(camXLen, camYLen, -camZLen));

  float nearClip = -camZLen - volumeHeight;
  float farClip  = -camZLen + volumeHeight;
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

  // Get edges of volume in image space coordinates
  crop.mLeft   = 0.5f + tan(leftAngle) / (2 * tan(fovX / 2));
  crop.mRight  = 0.5f + tan(rightAngle) / (2 * tan(fovX / 2));
  crop.mBottom = 0.5f + tan(downAngle) / (2 * tan(fovY / 2));
  crop.mTop    = 0.5f + tan(upAngle) / (2 * tan(fovY / 2));

  crop.mPos     = camPos;
  crop.mUp      = camUp;
  crop.mForward = camDir;

  crop.mModelView  = view * model;
  crop.mProjection = projection;
  return crop;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::Utility
