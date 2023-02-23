////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IrregularGrid.hpp"

#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

#include <thrust/host_vector.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

IrregularGrid::IrregularGrid(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor)
    , mWidth(0)
    , mHeight(0)
    , mVertexCount(0)
    , mFBOColor(GL_TEXTURE_2D)
    , mFBODepth(GL_TEXTURE_2D)
    , mHoleFillingTexture(GL_TEXTURE_2D)
    , mHoleFillingFBOs(mHoleFillingLevels) {
  mFullscreenQuadShader.InitVertexShaderFromString(FULLSCREEN_QUAD_VERT);
  mFullscreenQuadShader.InitGeometryShaderFromString(FULLSCREEN_QUAD_GEOM);
  mFullscreenQuadShader.InitFragmentShaderFromString(FULLSCREEN_QUAD_FRAG);
  mFullscreenQuadShader.Link();
  mHoleFillingShader.InitVertexShaderFromString(FULLSCREEN_QUAD_VERT);
  mHoleFillingShader.InitGeometryShaderFromString(FULLSCREEN_QUAD_GEOM);
  mHoleFillingShader.InitFragmentShaderFromString(HOLE_FILLING_FRAG);
  mHoleFillingShader.Link();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setDepthTexture(float* texture, int width, int height) {
  mWidth  = width;
  mHeight = height;
  DisplayNode::setDepthTexture(texture, width, height);
  mSurfaces.emplace(texture, width, height);
  // mSurfaces->print();
  createBuffers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IrregularGrid::DoImpl() {
  if (!mSurfaces.has_value()) {
    return false;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();
    mShader.InitVertexShaderFromString(PASS_VERT);
    mShader.InitGeometryShaderFromString(IRREGULAR_GRID_GEOM);
    mShader.InitFragmentShaderFromString(IRREGULAR_GRID_FRAG);
    mShader.Link();

    mShaderDirty = false;
  }

  mShader.Bind();

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE, glm::value_ptr(mTransform));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererProjectionInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uDepthTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mShader.SetUniform(mShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uInside"), mInside);
  glUniform2ui(mShader.GetUniformLocation("uResolution"), mWidth, mHeight);

  int width, height;
  GetVistaSystem()
      ->GetDisplayManager()
      ->GetCurrentRenderInfo()
      ->m_pViewport->GetViewportProperties()
      ->GetSize(width, height);

  // TODO Only do this again, if width or height changed
  mFBO.Bind();
  mFBOColor.UploadTexture(width, height, 0, false, GL_RGBA, GL_UNSIGNED_BYTE);
  mFBOColor.SetMagFilter(GL_LINEAR);
  mFBOColor.SetMinFilter(GL_LINEAR);
  mFBO.Attach(&mFBOColor, GL_COLOR_ATTACHMENT0);
  mFBOColor.Unbind();
  mFBODepth.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL,
      GL_UNSIGNED_INT_24_8, 0);
  mFBODepth.SetMagFilter(GL_LINEAR);
  mFBODepth.SetMinFilter(GL_LINEAR);
  mFBO.Attach(&mFBODepth, GL_DEPTH_STENCIL_ATTACHMENT);
  mFBODepth.Unbind();
  glClearColor(.0f, .0f, .0f, .0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  // Draw first pass.
  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

  mVAO.Bind();
  glDrawArrays(GL_POINTS, 0, mVertexCount);
  mVAO.Release();

  mTexture.Unbind(GL_TEXTURE0);
  mDepthTexture.Unbind(GL_TEXTURE1);
  mShader.Release();

  glPopAttrib();

  mFBO.Release();

  // Hole filling.
  mHoleFillingTexture.UploadTexture(width / 2, height / 2, 0, false, GL_RGBA, GL_UNSIGNED_BYTE);
  mHoleFillingTexture.SetMagFilter(GL_LINEAR);
  mHoleFillingTexture.SetMinFilter(GL_LINEAR);
  mHoleFillingTexture.GenerateMipmaps();

  for (int i = 0; i < mHoleFillingLevels; ++i) {
    mHoleFillingFBOs[i].Bind();
    mHoleFillingFBOs[i].Attach(&mHoleFillingTexture, GL_COLOR_ATTACHMENT0, i);
    mHoleFillingFBOs[i].Release();
  }
  mHoleFillingTexture.Unbind();

  mHoleFillingShader.Bind();
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uColorBuffer"), 0);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uDepthBuffer"), 1);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uHoleFillingTexture"), 2);

  mFBOColor.Bind(GL_TEXTURE0);
  mFBODepth.Bind(GL_TEXTURE1);
  mHoleFillingTexture.Bind(GL_TEXTURE2);

  glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);

  int mipWidth  = width / 2;
  int mipHeight = height / 2;
  for (int i = 0; i < mHoleFillingLevels; ++i) {
    mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uCurrentLevel"), i);

    mHoleFillingFBOs[i].Bind();
    glViewport(0, 0, mipWidth, mipHeight);
    glClearColor(.0f, .0f, .0f, .0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    mHoleFillingFBOs[i].Release();

    mipWidth /= 2;
    mipHeight /= 2;
  }

  glPopAttrib();

  mHoleFillingShader.Release();

  // Draw second pass.
  mFullscreenQuadShader.Bind();
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexColor"), 0);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexDepth"), 1);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexHoleFilling"), 2);

  mFBOColor.Bind(GL_TEXTURE0);
  mFBODepth.Bind(GL_TEXTURE1);
  mHoleFillingTexture.Bind(GL_TEXTURE2);

  glPushAttrib(GL_ENABLE_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glDrawArrays(GL_POINTS, 0, 1);

  mFBOColor.Unbind(GL_TEXTURE0);
  mFBODepth.Unbind(GL_TEXTURE1);
  mHoleFillingTexture.Unbind(GL_TEXTURE2);
  mFullscreenQuadShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::createBuffers() {
  thrust::device_vector<SurfaceDetectionBuffer::Vertex> dVertices = mSurfaces->generateVertices();
  thrust::host_vector<SurfaceDetectionBuffer::Vertex>   hVertices = dVertices;
  mVertexCount                                                    = hVertices.size();

  mVAO.Bind();

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(
      hVertices.size() * sizeof(SurfaceDetectionBuffer::Vertex), hVertices.data(), GL_STATIC_DRAW);

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayInteger(0, 3, GL_UNSIGNED_INT, 3 * sizeof(unsigned int), 0, &mVBO);

  mVAO.Release();
  mVBO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
