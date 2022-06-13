////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_UTILITY_HPP
#define CSP_VOLUME_RENDERING_UTILITY_HPP

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering::Utility {

struct CameraParams {
  float mLeft;
  float mRight;
  float mTop;
  float mBottom;

  glm::vec3 mPos;
  glm::vec3 mUp;
  glm::vec3 mForward;

  glm::mat4 mModelView;
  glm::mat4 mProjection;

  bool mInside;
};

CameraParams calculateCameraParams(
    float volumeHeight, glm::mat4 observerTransform, float fovY = 3.141f, float fovX = 3.141f);

} // namespace csp::volumerendering::Utility

#endif // CSP_VOLUME_RENDERING_UTILITY_HPP
