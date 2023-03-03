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
    , mRegularGrid(128)
    , mHoleFillingTexture(GL_TEXTURE_2D)
    , mHoleFillingDepth(GL_TEXTURE_2D)
    , mHoleFillingFBOs(mHoleFillingLevels) {
  mFullscreenQuadShader.InitVertexShaderFromString(FULLSCREEN_QUAD_VERT);
  mFullscreenQuadShader.InitGeometryShaderFromString(FULLSCREEN_QUAD_GEOM);
  mFullscreenQuadShader.InitFragmentShaderFromString(FULLSCREEN_QUAD_FRAG);
  mFullscreenQuadShader.Link();

  mHoleFillingShader.InitVertexShaderFromString(FULLSCREEN_QUAD_VERT);
  mHoleFillingShader.InitGeometryShaderFromString(FULLSCREEN_QUAD_GEOM);
  mHoleFillingShader.InitFragmentShaderFromString(HOLE_FILLING_FRAG);
  mHoleFillingShader.Link();

  mRegularGridShader.InitVertexShaderFromString(BILLBOARD_VERT);
  mRegularGridShader.InitFragmentShaderFromString(REGULAR_GRID_FRAG);
  mRegularGridShader.Link();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setImage(Renderer::RenderedImage& image) {
  DisplayNode::setImage(image);
  mWidth  = image.getResolution();
  mHeight = image.getResolution();
  mSurfaces.emplace(image.getDepthData(), mWidth, mHeight);
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

  if (width != mScreenWidth || height != mScreenHeight) {
    createFBOs(width, height);
    mScreenWidth  = width;
    mScreenHeight = height;
  }

  mFBO.Bind();
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
  mHoleFillingShader.Bind();
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uColorBuffer"), 0);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uDepthBuffer"), 1);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uHoleFillingTexture"), 2);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uHoleFillingDepth"), 3);

  mFBOColor.Bind(GL_TEXTURE0);
  mFBODepth.Bind(GL_TEXTURE1);
  mHoleFillingTexture.Bind(GL_TEXTURE2);
  mHoleFillingDepth.Bind(GL_TEXTURE3);

  glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_ALWAYS);

  int mipWidth  = width / 2;
  int mipHeight = height / 2;
  for (int i = 0; i < mHoleFillingLevels; ++i) {
    mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uCurrentLevel"), i);

    mHoleFillingFBOs[i].Bind();
    glViewport(0, 0, mipWidth, mipHeight);
    glClearColor(.0f, .0f, .0f, .0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDrawArrays(GL_POINTS, 0, 1);
    mHoleFillingFBOs[i].Release();

    mipWidth /= 2;
    mipHeight /= 2;
  }

  glPopAttrib();

  mFBOColor.Unbind(GL_TEXTURE0);
  mFBODepth.Unbind(GL_TEXTURE1);
  mHoleFillingTexture.Unbind(GL_TEXTURE2);
  mHoleFillingDepth.Unbind(GL_TEXTURE3);
  mHoleFillingShader.Release();

  // Draw second pass.
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_STENCIL_TEST);
  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
  glStencilFunc(GL_ALWAYS, 1, 0xFF);
  glStencilMask(0xFF);
  glClear(GL_STENCIL_BUFFER_BIT);

  mRegularGridShader.Bind();
  glUniformMatrix4fv(
      mRegularGridShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
  glUniformMatrix4fv(
      mRegularGridShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE, glm::value_ptr(mTransform));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererProjectionInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(
      mRegularGridShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uTexture"), 0);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uDepthTexture"), 1);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mRegularGridShader.SetUniform(
      mRegularGridShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uInside"), mInside);
  glUniform2ui(mRegularGridShader.GetUniformLocation("uResolution"), mWidth, mHeight);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  mRegularGrid.Draw();
  mRegularGridShader.Release();

  glStencilFunc(mHoleFillingLevel == -3 ? GL_ALWAYS : GL_EQUAL, 1, 0xFF);
  glStencilMask(0x00);

  mFullscreenQuadShader.Bind();
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexColor"), 0);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexDepth"), 1);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexHoleFilling"), 2);
  mFullscreenQuadShader.SetUniform(
      mFullscreenQuadShader.GetUniformLocation("uMaxLevel"), mHoleFillingLevels - 1);
  mFullscreenQuadShader.SetUniform(
      mFullscreenQuadShader.GetUniformLocation("uHoleFillingLevel"), mHoleFillingLevel);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uResolution"),
      static_cast<float>(width), static_cast<float>(height));

  mFBOColor.Bind(GL_TEXTURE0);
  mFBODepth.Bind(GL_TEXTURE1);
  mHoleFillingTexture.Bind(GL_TEXTURE2);

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

void IrregularGrid::createFBOs(int width, int height) {
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
  mFBO.Release();

  mHoleFillingTexture.UploadTexture(width / 2, height / 2, 0, false, GL_RGBA, GL_UNSIGNED_BYTE);
  mHoleFillingTexture.SetMagFilter(GL_LINEAR);
  mHoleFillingTexture.SetMinFilter(GL_LINEAR_MIPMAP_LINEAR);
  mHoleFillingTexture.SetWrapS(GL_CLAMP_TO_BORDER);
  mHoleFillingTexture.SetWrapT(GL_CLAMP_TO_BORDER);
  mHoleFillingTexture.GenerateMipmaps();

  mHoleFillingDepth.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width / 2, height / 2, 0,
      GL_DEPTH_COMPONENT, GL_FLOAT, 0);
  VistaOGLUtils::CheckForOGLError(__FILE__, __LINE__);
  mHoleFillingDepth.SetMagFilter(GL_LINEAR);
  mHoleFillingDepth.SetMinFilter(GL_LINEAR_MIPMAP_LINEAR);
  mHoleFillingTexture.SetWrapS(GL_CLAMP_TO_BORDER);
  mHoleFillingTexture.SetWrapT(GL_CLAMP_TO_BORDER);
  mHoleFillingDepth.GenerateMipmaps();
  mHoleFillingDepth.Unbind();

  for (int i = 0; i < mHoleFillingLevels; ++i) {
    mHoleFillingFBOs[i].Bind();
    mHoleFillingFBOs[i].Attach(&mHoleFillingTexture, GL_COLOR_ATTACHMENT0, i);
    mHoleFillingFBOs[i].Attach(&mHoleFillingDepth, GL_DEPTH_ATTACHMENT, i);
    mHoleFillingFBOs[i].Release();
  }
  mHoleFillingTexture.Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
