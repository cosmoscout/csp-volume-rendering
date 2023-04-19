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
    , mRegularGrid(128) {
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

  mShader.InitVertexShaderFromString(PASS_VERT);
  mShader.InitGeometryShaderFromString(IRREGULAR_GRID_GEOM);
  mShader.InitFragmentShaderFromString(IRREGULAR_GRID_FRAG);
  mShader.Link();

  mShaderDirty = false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setImage(std::unique_ptr<Renderer::RenderedImage> image) {
  DisplayNode::setImage(std::move(image));
  mNewImage = 2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IrregularGrid::DoImpl() {
  if (--mNewImage == 0 && mImage) {
    {
      cs::utils::FrameTimings::ScopedTimer timer("Buffer");
      std::this_thread::sleep_for(std::chrono::milliseconds(10 * mImage->getLayerCount()));
    }
    cs::utils::FrameTimings::ScopedTimer timer("IrregularGrid::setImage");
    mWidth  = mImage->getResolution();
    mHeight = mImage->getResolution();

    if (mImage->getLayerCount() != mLayerBuffers.size()) {
      mLayerBuffers.resize(mImage->getLayerCount());
      mLayerCountChanged = true;
    }

    for (auto i = 0u; i < mImage->getLayerCount(); ++i) {
      if (!mLayerBuffers[i]) {
        mLayerBuffers[i] = std::make_unique<LayerBuffers>();
      }
      {
        cs::utils::FrameTimings::ScopedTimer timer("Surface detection");
        mLayerBuffers[i]->mSurfaces.emplace(mImage->getDepthData(i), mWidth, mHeight);
      }
      mLayerBuffers[i]->mHoleFilling.mFBOs.resize(mHoleFillingLevels);
    }

    createBuffers();
  }

  if (mLayerBuffers.size() == 0) {
    return false;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  // Get viewport size
  int width, height;
  GetVistaSystem()
      ->GetDisplayManager()
      ->GetCurrentRenderInfo()
      ->m_pViewport->GetViewportProperties()
      ->GetSize(width, height);

  if (width != mScreenWidth || height != mScreenHeight || mLayerCountChanged) {
    createFBOs(width, height);
    mScreenWidth       = width;
    mScreenHeight      = height;
    mLayerCountChanged = false;
  }

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  glm::mat4 matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());
  glm::mat4 matP  = glm::make_mat4x4(glMatP.data());

  // Do renderpasses
  drawIrregularGrid(matMV, matP);
  generateHoleFillingTex();
  drawFullscreenQuad(matMV, matP);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::drawIrregularGrid(glm::mat4 matMV, glm::mat4 matP) {
  cs::utils::FrameTimings::ScopedTimer timer("IrregularGrid::drawIrregularGrid");
  mShader.Bind();

  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glm::value_ptr(matP));
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

  int   layer              = 0;
  float halfLayerThickness = 1.f / mLayerBuffers.size();
  for (auto& layerBuffers : mLayerBuffers) {
    if (layer >= mTexture.size()) {
      continue;
    }
    glm::vec4 depthCenter(0, 0, 1.f - (halfLayerThickness * (layer * 2 + 1.f)), 1);
    glm::vec4 depthFar(0, 0, 1.f - (halfLayerThickness * (layer * 2 + 2.f)), 1);
    depthCenter = mRendererMVP * depthCenter;
    depthFar    = mRendererMVP * depthFar;
    mShader.SetUniform(
        mShader.GetUniformLocation("uDefaultDepthCenter"), depthCenter.z / depthCenter.w);
    mShader.SetUniform(mShader.GetUniformLocation("uDefaultDepthFar"), depthFar.z / depthFar.w);

    layerBuffers->mFullscreenQuad.mFBO.Bind();
    glClearColor(.0f, .0f, .0f, .0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    mTexture[layer]->Bind(GL_TEXTURE0);
    mDepthTexture[layer]->Bind(GL_TEXTURE1);

    // Draw first pass.
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ZERO);

    layerBuffers->mGrid.mVAO.Bind();
    glDrawArrays(GL_POINTS, 0, layerBuffers->mGrid.mVertexCount);
    layerBuffers->mGrid.mVAO.Release();

    glPopAttrib();

    mTexture[layer]->Unbind(GL_TEXTURE0);
    mDepthTexture[layer]->Unbind(GL_TEXTURE1);

    layerBuffers->mFullscreenQuad.mFBO.Release();
    layer++;
  }

  mShader.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::generateHoleFillingTex() {
  cs::utils::FrameTimings::ScopedTimer timer("IrregularGrid::generateHoleFillingTex");
  // Hole filling.
  mHoleFillingShader.Bind();
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uColorBuffer"), 0);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uDepthBuffer"), 1);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uHoleFillingTexture"), 2);
  mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uHoleFillingDepth"), 3);

  glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_ALWAYS);

  for (auto& layerBuffers : mLayerBuffers) {
    layerBuffers->mFullscreenQuad.mTexture.Bind(GL_TEXTURE0);
    layerBuffers->mFullscreenQuad.mDepth.Bind(GL_TEXTURE1);
    layerBuffers->mHoleFilling.mTexture.Bind(GL_TEXTURE2);
    layerBuffers->mHoleFilling.mDepth.Bind(GL_TEXTURE3);

    int mipWidth  = mScreenWidth / 2;
    int mipHeight = mScreenHeight / 2;
    for (int i = 0; i < mHoleFillingLevels; ++i) {
      mHoleFillingShader.SetUniform(mHoleFillingShader.GetUniformLocation("uCurrentLevel"), i);

      layerBuffers->mHoleFilling.mFBOs[i].Bind();
      glViewport(0, 0, mipWidth, mipHeight);
      glClearColor(.0f, .0f, .0f, .0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDrawArrays(GL_POINTS, 0, 1);
      layerBuffers->mHoleFilling.mFBOs[i].Release();

      mipWidth /= 2;
      mipHeight /= 2;
    }

    layerBuffers->mFullscreenQuad.mTexture.Unbind(GL_TEXTURE0);
    layerBuffers->mFullscreenQuad.mDepth.Unbind(GL_TEXTURE1);
    layerBuffers->mHoleFilling.mTexture.Unbind(GL_TEXTURE2);
    layerBuffers->mHoleFilling.mDepth.Unbind(GL_TEXTURE3);
  }

  glPopAttrib();

  mHoleFillingShader.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::drawFullscreenQuad(glm::mat4 matMV, glm::mat4 matP) {
  cs::utils::FrameTimings::ScopedTimer timer("IrregularGrid::drawFullscreenQuad");
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
  glUniformMatrix4fv(
      mRegularGridShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glm::value_ptr(matP));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE,
      glm::value_ptr(mTransform));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererProjectionInv"), 1,
      GL_FALSE, glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE,
      glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mRegularGridShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uTexture"), 0);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uDepthTexture"), 1);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uRadii"),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mRegularGridShader.SetUniform(
      mRegularGridShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mRegularGridShader.SetUniform(mRegularGridShader.GetUniformLocation("uInside"), mInside);
  glUniform2ui(mRegularGridShader.GetUniformLocation("uResolution"), mWidth, mHeight);

  for (auto i = 0u; i < mTexture.size(); ++i) {
    mTexture[i]->Bind(GL_TEXTURE0);
    mDepthTexture[i]->Bind(GL_TEXTURE1);

    mRegularGrid.Draw();
  }

  mRegularGridShader.Release();

  glStencilFunc(mHoleFillingLevel == -3 ? GL_ALWAYS : GL_EQUAL, 1, 0xFF);
  glStencilMask(0x00);

  mFullscreenQuadShader.Bind();
  glm::mat4 warp = mRendererMVP * glm::inverse(matP * matMV * mTransform);
  glUniformMatrix4fv(mFullscreenQuadShader.GetUniformLocation("uMatWarp"), 1, GL_FALSE, glm::value_ptr(warp));

  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexColor"), 0);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexDepth"), 1);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uTexHoleFilling"), 2);
  mFullscreenQuadShader.SetUniform(
      mFullscreenQuadShader.GetUniformLocation("uMaxLevel"), mHoleFillingLevels - 1);
  mFullscreenQuadShader.SetUniform(
      mFullscreenQuadShader.GetUniformLocation("uHoleFillingLevel"), mHoleFillingLevel);
  mFullscreenQuadShader.SetUniform(mFullscreenQuadShader.GetUniformLocation("uResolution"),
      static_cast<float>(mScreenWidth), static_cast<float>(mScreenHeight));

  for (auto layerBuffers = mLayerBuffers.rbegin(); layerBuffers != mLayerBuffers.rend();
       layerBuffers++) {
    int index = std::distance(layerBuffers, mLayerBuffers.rend()) - 1;

    (*layerBuffers)->mFullscreenQuad.mTexture.Bind(GL_TEXTURE0);
    (*layerBuffers)->mFullscreenQuad.mDepth.Bind(GL_TEXTURE1);
    (*layerBuffers)->mHoleFilling.mTexture.Bind(GL_TEXTURE2);

    if (mHoleFillingLevel == index || mHoleFillingLevel < 0)
      glDrawArrays(GL_POINTS, 0, 1);

    (*layerBuffers)->mFullscreenQuad.mTexture.Unbind(GL_TEXTURE0);
    (*layerBuffers)->mFullscreenQuad.mDepth.Unbind(GL_TEXTURE1);
    (*layerBuffers)->mHoleFilling.mTexture.Unbind(GL_TEXTURE2);
  }

  mFullscreenQuadShader.Release();

  glPopAttrib();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::createBuffers() {
  cs::utils::FrameTimings::ScopedTimer timer("Create buffers");
  for (auto& layerBuffers : mLayerBuffers) {
    thrust::device_vector<SurfaceDetectionBuffer::Vertex> dVertices =
        layerBuffers->mSurfaces->generateVertices();
    thrust::host_vector<SurfaceDetectionBuffer::Vertex> hVertices = dVertices;
    layerBuffers->mGrid.mVertexCount                              = hVertices.size();

    layerBuffers->mGrid.mVAO.Bind();

    layerBuffers->mGrid.mVBO.Bind(GL_ARRAY_BUFFER);
    layerBuffers->mGrid.mVBO.BufferData(hVertices.size() * sizeof(SurfaceDetectionBuffer::Vertex),
        hVertices.data(), GL_STATIC_DRAW);

    layerBuffers->mGrid.mVAO.EnableAttributeArray(0);
    layerBuffers->mGrid.mVAO.SpecifyAttributeArrayInteger(
        0, 3, GL_UNSIGNED_INT, 3 * sizeof(unsigned int), 0, &layerBuffers->mGrid.mVBO);

    layerBuffers->mGrid.mVAO.Release();
    layerBuffers->mGrid.mVBO.Release();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::createFBOs(int width, int height) {
  for (auto& layerBuffers : mLayerBuffers) {
    layerBuffers->mFullscreenQuad.mFBO.Bind();
    layerBuffers->mFullscreenQuad.mTexture.UploadTexture(
        width, height, 0, false, GL_RGBA, GL_UNSIGNED_BYTE);
    layerBuffers->mFullscreenQuad.mTexture.SetMagFilter(GL_LINEAR);
    layerBuffers->mFullscreenQuad.mTexture.SetMinFilter(GL_LINEAR);
    layerBuffers->mFullscreenQuad.mFBO.Attach(
        &layerBuffers->mFullscreenQuad.mTexture, GL_COLOR_ATTACHMENT0);
    layerBuffers->mFullscreenQuad.mTexture.Unbind();
    layerBuffers->mFullscreenQuad.mDepth.Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, width, height, 0, GL_DEPTH_STENCIL,
        GL_UNSIGNED_INT_24_8, 0);
    layerBuffers->mFullscreenQuad.mDepth.SetMagFilter(GL_LINEAR);
    layerBuffers->mFullscreenQuad.mDepth.SetMinFilter(GL_LINEAR);
    layerBuffers->mFullscreenQuad.mFBO.Attach(
        &layerBuffers->mFullscreenQuad.mDepth, GL_DEPTH_STENCIL_ATTACHMENT);
    layerBuffers->mFullscreenQuad.mDepth.Unbind();
    layerBuffers->mFullscreenQuad.mFBO.Release();

    layerBuffers->mHoleFilling.mTexture.UploadTexture(
        width / 2, height / 2, 0, false, GL_RGBA, GL_UNSIGNED_BYTE);
    layerBuffers->mHoleFilling.mTexture.SetMagFilter(GL_LINEAR);
    layerBuffers->mHoleFilling.mTexture.SetMinFilter(GL_LINEAR_MIPMAP_LINEAR);
    layerBuffers->mHoleFilling.mTexture.SetWrapS(GL_CLAMP_TO_BORDER);
    layerBuffers->mHoleFilling.mTexture.SetWrapT(GL_CLAMP_TO_BORDER);
    layerBuffers->mHoleFilling.mTexture.GenerateMipmaps();

    layerBuffers->mHoleFilling.mDepth.Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width / 2, height / 2, 0,
        GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    layerBuffers->mHoleFilling.mDepth.SetMagFilter(GL_LINEAR);
    layerBuffers->mHoleFilling.mDepth.SetMinFilter(GL_LINEAR_MIPMAP_LINEAR);
    layerBuffers->mHoleFilling.mDepth.SetWrapS(GL_CLAMP_TO_BORDER);
    layerBuffers->mHoleFilling.mDepth.SetWrapT(GL_CLAMP_TO_BORDER);
    layerBuffers->mHoleFilling.mDepth.GenerateMipmaps();
    layerBuffers->mHoleFilling.mDepth.Unbind();

    for (int i = 0; i < mHoleFillingLevels; ++i) {
      layerBuffers->mHoleFilling.mFBOs[i].Bind();
      layerBuffers->mHoleFilling.mFBOs[i].Attach(
          &layerBuffers->mHoleFilling.mTexture, GL_COLOR_ATTACHMENT0, i);
      layerBuffers->mHoleFilling.mFBOs[i].Attach(
          &layerBuffers->mHoleFilling.mDepth, GL_DEPTH_ATTACHMENT, i);
      layerBuffers->mHoleFilling.mFBOs[i].Release();
    }
    layerBuffers->mHoleFilling.mTexture.Unbind();
    layerBuffers->mHoleFilling.mDepth.Unbind();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
