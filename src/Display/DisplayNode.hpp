////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
#define CSP_VOLUME_RENDERING_DISPLAYNODE_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

#include "../../../../src/cs-scene/CelestialObject.hpp"
#include "../../../../src/cs-utils/DefaultProperty.hpp"

namespace csp::volumerendering {

class DisplayNode : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  DisplayNode(VistaSceneGraph* sceneGraph, std::string const& centerName,
      std::string const& frameName, double startExistence, double endExistence, glm::dvec3 radii,
      int depthResolution);
  virtual ~DisplayNode();

  DisplayNode(DisplayNode const& other) = delete;
  DisplayNode(DisplayNode&& other)      = default;

  DisplayNode& operator=(DisplayNode const& other) = delete;
  DisplayNode& operator=(DisplayNode&& other) = default;

  void setEnabled(bool enabled);

  void setTexture(std::vector<uint8_t>& texture, int width, int height);
  void setDepthTexture(std::vector<float>& texture, int width, int height);
  void setTransform(glm::mat4 transform);
  void setMVPMatrix(glm::mat4 mvp);
  void setUseDepth(bool useDepth);
  void setDrawDepth(bool drawDepth);

  /// Interface implementation of IVistaOpenGLDraw.
  virtual bool Do() override = 0;
  bool         GetBoundingBox(VistaBoundingBox& bb) override;

 protected:
  VistaSceneGraph*                 mVistaSceneGraph;
  std::shared_ptr<VistaOpenGLNode> mVistaNode;

  bool mEnabled = false;

  glm::mat4       mTransform;
  VistaTexture    mTexture;
  VistaGLSLShader mShader;

  glm::mat4                               mRendererMVP;
  cs::utils::Property<std::vector<float>> pDepthValues;
  int                                     mDepthResolution;
  bool                                    mUseDepth  = true;
  bool                                    mDrawDepth = false;

  glm::dvec3 mRadii;

  bool mShaderDirty;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
