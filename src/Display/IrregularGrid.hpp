////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_IRREGULAR_GRID_HPP
#define CSP_VOLUME_RENDERING_IRREGULAR_GRID_HPP

#include "DisplayNode.hpp"
#include "RegularGridBuffers.hpp"
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

  void setImage(Renderer::RenderedImage& image) override;

 protected:
  /// Interface implementation of IVistaOpenGLDraw.
  bool DoImpl() override;

 private:
  struct LayerBuffers {
    std::optional<SurfaceDetectionBuffer> mSurfaces;

    struct Grid {
      VistaVertexArrayObject mVAO;
      VistaBufferObject      mVBO;
      size_t                 mVertexCount = 0;
    } mGrid;

    struct HoleHilling {
      HoleHilling()
          : mTexture(GL_TEXTURE_2D)
          , mDepth(GL_TEXTURE_2D) {
      }

      std::vector<VistaFramebufferObj> mFBOs;
      VistaTexture                     mTexture;
      VistaTexture                     mDepth;
    } mHoleFilling;

    struct FullscreenQuad {
      FullscreenQuad()
          : mTexture(GL_TEXTURE_2D)
          , mDepth(GL_TEXTURE_2D) {
      }

      VistaFramebufferObj mFBO;
      VistaTexture        mTexture;
      VistaTexture        mDepth;
    } mFullscreenQuad;
  };

  void drawIrregularGrid(glm::mat4 matMV, glm::mat4 matP);
  void generateHoleFillingTex();
  void drawFullscreenQuad(glm::mat4 matMV, glm::mat4 matP);

  void createBuffers();
  void createFBOs(int width, int height);

  unsigned int mWidth;
  unsigned int mHeight;

  int mScreenWidth  = 0;
  int mScreenHeight = 0;

  bool mLayerCountChanged = false;

  // Using unique_ptrs here, because otherwise resizing the vector would release and break all
  // contained Vista objects.
  std::vector<std::unique_ptr<LayerBuffers>> mLayerBuffers;

  RegularGridBuffers mRegularGrid;

  const int mHoleFillingLevels = 4;

  VistaGLSLShader mFullscreenQuadShader;
  VistaGLSLShader mRegularGridShader;
  VistaGLSLShader mHoleFillingShader;
};

} // namespace csp::volumerendering

#endif
