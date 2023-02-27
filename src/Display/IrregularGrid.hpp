////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_IRREGULAR_GRID_HPP
#define CSP_VOLUME_RENDERING_IRREGULAR_GRID_HPP

#include "DisplayNode.hpp"
#include "SurfaceDetectionBuffer.hpp"

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaFramebufferObj.h>
#include <VistaOGLExt/VistaRenderbuffer.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering {

/// DisplayNode implementation, that displays rendered volume images on a irregular grid mesh.
class IrregularGrid : public DisplayNode {
 public:
  IrregularGrid(
      VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor);

  IrregularGrid(IrregularGrid const& other) = delete;
  IrregularGrid(IrregularGrid&& other)      = default;

  IrregularGrid& operator=(IrregularGrid const& other) = delete;
  IrregularGrid& operator=(IrregularGrid&& other) = default;

  void setDepthTexture(float* texture, int width, int height) override;

 protected:
  /// Interface implementation of IVistaOpenGLDraw.
  bool DoImpl() override;

 private:
  void createBuffers();
  void createFBOs(int width, int height);

  std::optional<SurfaceDetectionBuffer> mSurfaces;
  unsigned int                          mWidth;
  unsigned int                          mHeight;

  int mScreenWidth  = 0;
  int mScreenHeight = 0;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  size_t                 mVertexCount;

  VistaGLSLShader     mFullscreenQuadShader;
  VistaFramebufferObj mFBO;
  VistaTexture        mFBOColor;
  VistaTexture        mFBODepth;

  VistaGLSLShader                  mHoleFillingShader;
  const int                        mHoleFillingLevels = 6;
  VistaTexture                     mHoleFillingTexture;
  VistaTexture                     mHoleFillingDepth;
  std::vector<VistaFramebufferObj> mHoleFillingFBOs;
};

} // namespace csp::volumerendering

#endif
