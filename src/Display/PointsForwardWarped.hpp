////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP
#define CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

#include "../../../../src/cs-scene/CelestialObject.hpp"

namespace csp::volumerendering {

class PointsForwardWarped : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  PointsForwardWarped(std::string const& sCenterName, std::string const& sFrameName, double tStartExistence,
      double tEndExistence, glm::dvec3 radii);

  PointsForwardWarped(PointsForwardWarped const& other) = delete;
  PointsForwardWarped(PointsForwardWarped&& other)      = default;

  PointsForwardWarped& operator=(PointsForwardWarped const& other) = delete;
  PointsForwardWarped& operator=(PointsForwardWarped&& other) = default;

  void setTexture(std::vector<uint8_t>& texture, int width, int height);
  void setDepthTexture(std::vector<float>& texture, int width, int height);
  void setTransform(glm::mat4 transform);

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void createBuffers();

  glm::mat4              mTransform;
  VistaTexture           mTexture;
  VistaGLSLShader        mShader;
  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;

	std::vector<float> mDepthValues;
  uint32_t           mResolution;

  glm::dvec3 mRadii;

  bool mShaderDirty;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_POINTSFORWARDWARPED_HPP
