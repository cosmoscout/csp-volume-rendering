////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP
#define CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP

#include "DisplayNode.hpp"

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering {

/// DisplayNode implementation, that displays pixels of a rendered volume image as discrete points.
/// The number of used points is equal to the number of pixels in the used depth image.
class PointsForwardWarped : public DisplayNode {
 public:
  PointsForwardWarped(
      VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor);

  PointsForwardWarped(PointsForwardWarped const& other) = delete;
  PointsForwardWarped(PointsForwardWarped&& other)      = default;

  PointsForwardWarped& operator=(PointsForwardWarped const& other) = delete;
  PointsForwardWarped& operator=(PointsForwardWarped&& other) = default;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;

 private:
  void createBuffers(std::vector<float> depthValues);

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaBufferObject      mIBO;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP
