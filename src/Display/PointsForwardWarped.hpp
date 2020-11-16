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

class PointsForwardWarped : public DisplayNode {
 public:
  PointsForwardWarped(VistaSceneGraph* sceneGraph, std::string const& centerName,
      std::string const& frameName, double startExistence, double endExistence, glm::dvec3 radii);

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
