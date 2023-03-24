////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
#define CSP_VOLUME_RENDERING_DISPLAYNODE_HPP

#include "../Enums.hpp"
#include "../Render/Renderer.hpp"

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

  /// Sets the image that should be displayed.
  /// Includes color, depth, multiple layers, matrices etc.
  /// The color data should be given as an array of rgba values (8 bit per channel).
  /// The depth data should be given as an array of z-positions in clip space per pixel.
  virtual void setImage(Renderer::RenderedImage& image);

  /// Enables using the depth information for image based rendering.
  void setUseDepth(bool useDepth);
  /// Enables drawing the depth data as grayscale instead of the color image.
  void setDrawDepth(bool drawDepth);

  void setHoleFillingLevel(int value);

  virtual glm::dvec3 getRadii() const;

  glm::mat4 getVistaModelView() const;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 protected:
  std::shared_ptr<VistaOpenGLNode> mVistaNode;

  VolumeShape mShape;

  bool mEnabled = false;

  glm::mat4                                  mTransform;
  std::vector<std::unique_ptr<VistaTexture>> mTexture;
  std::vector<std::unique_ptr<VistaTexture>> mDepthTexture;
  VistaGLSLShader                            mShader;

  glm::mat4 mRendererModelView;
  glm::mat4 mRendererProjection;
  glm::mat4 mRendererMVP;
  bool      mInside;
  bool      mUseDepth  = true;
  bool      mDrawDepth = false;

  int mHoleFillingLevel = -1;

  bool mShaderDirty;

  virtual bool DoImpl() = 0;

 private:
  glm::mat4 mVistaModelView;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DISPLAYNODE_HPP
