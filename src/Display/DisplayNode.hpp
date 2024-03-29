////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
#define CSP_VOLUME_RENDERING_DISPLAYNODE_HPP

#include "../Enums.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

#include "../../../../src/cs-core/Settings.hpp"
#include "../../../../src/cs-scene/CelestialBody.hpp"
#include "../../../../src/cs-utils/DefaultProperty.hpp"

namespace csp::volumerendering {

/// The abstract DisplayNode provides an interface for CelestialObjects that should display images
/// rendered with a Renderer.
class DisplayNode : public cs::scene::CelestialObject, public IVistaOpenGLDraw {
 public:
  /// Create a DisplayNode positioned at the given anchor using properties found in settings.
  /// It will automatically be added to the Vista scene graph on construction and removed on
  /// destruction. The depthResolution is used as an initial mesh resolution of the created objects.
  DisplayNode(VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor);
  virtual ~DisplayNode();

  DisplayNode(DisplayNode const& other) = delete;
  DisplayNode(DisplayNode&& other)      = default;

  DisplayNode& operator=(DisplayNode const& other) = delete;
  DisplayNode& operator=(DisplayNode&& other) = default;

  /// Enable rendering of the display node.
  void setEnabled(bool enabled);

  /// Set the color image that should be displayed.
  /// The color data should be given as an array of rgba values (8 bit per channel).
  void setTexture(uint8_t* texture, int width, int height);
  void setTexture(float* texture, int width, int height);
  /// Set the depth information for displaying the image.
  /// The depth data should be given as an array of z-positions in clip space per pixel.
  void setDepthTexture(float* texture, int width, int height);
  /// Sets the base transform of the display node.
  /// This should correspond to the perspective, from which the image was rendered.
  void setTransform(glm::mat4 transform);
  /// Sets the model view and projection matrices used in rendering the color image.
  void setRendererMatrices(glm::mat4 modelView, glm::mat4 projection);

  /// Enables using the depth information for image based rendering.
  void setUseDepth(bool useDepth);
  /// Enables drawing the depth data as grayscale instead of the color image.
  void setDrawDepth(bool drawDepth);

  virtual glm::dvec3 getRadii() const;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 protected:
  std::shared_ptr<VistaOpenGLNode> mVistaNode;

  VolumeShape mShape;

  bool mEnabled = false;

  glm::mat4       mTransform;
  VistaTexture    mTexture;
  VistaTexture    mDepthTexture;
  VistaGLSLShader mShader;

  glm::mat4 mRendererModelView;
  glm::mat4 mRendererProjection;
  glm::mat4 mRendererMVP;
  bool      mUseDepth  = true;
  bool      mDrawDepth = false;

  bool mShaderDirty;

  virtual bool DoImpl() = 0;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
