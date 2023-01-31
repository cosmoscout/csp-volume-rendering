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

#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////


IrregularGrid::IrregularGrid(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor) {
  createBuffers(16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setDepthTexture(float* texture, int width, int height) {
  DisplayNode::setDepthTexture(texture, width, height);
  // TODO Generate quadtree
  // TODO Pass quadtree to createBuffers?
  createBuffers(width, height);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IrregularGrid::DoImpl() {
  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();
    mShader.InitVertexShaderFromString(BILLBOARD_VERT);
    mShader.InitFragmentShaderFromString(BILLBOARD_FRAG);
    mShader.Link();

    mVisShader = VistaGLSLShader();
    mVisShader.InitVertexShaderFromString(BILLBOARD_VERT);
    mVisShader.InitFragmentShaderFromString(VIS_FRAG);
    mVisShader.Link();

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

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, mIndexCount,
      GL_UNSIGNED_INT, nullptr);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mTexture.Unbind(GL_TEXTURE1);
  mShader.Release();

  // Visualization of grid
  mVisShader.Bind();

  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE, glm::value_ptr(mTransform));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererProjectionInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mVisShader.SetUniform(mVisShader.GetUniformLocation("uTexture"), 0);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uDepthTexture"), 1);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mVisShader.SetUniform(
      mVisShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uInside"), mInside);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  mVisVAO.Bind();
  glDrawElements(GL_LINES, mVisIndexCount, GL_UNSIGNED_INT, nullptr);
  mVisVAO.Release();
  mVisShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::createBuffers(uint32_t width, uint32_t height) {
  mIndexCount = (height - 1) * (2 + 2 * width);
  std::vector<float>    vertices(width * height * 3);
  std::vector<unsigned> indices(mIndexCount);

  for (uint32_t x = 0; x < width; ++x) {
    for (uint32_t y = 0; y < height; ++y) {
      vertices[(x * height + y) * 3 + 0] = 2.f / (width - 1) * x - 1.f;
      vertices[(x * height + y) * 3 + 1] = 2.f / (height - 1) * y - 1.f;
      vertices[(x * height + y) * 3 + 2] = 0.f;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < width - 1; ++x) {
    indices[index++] = x * height;
    for (uint32_t y = 0; y < height; ++y) {
      indices[index++] = x * height + y;
      indices[index++] = (x + 1) * height + y;
    }
    indices[index] = indices[index - 1];
    ++index;
  }

  mVAO.Bind();

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  mIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mIBO.BufferData(indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

  mVAO.Release();
  mIBO.Release();
  mVBO.Release();

  // Create visualization buffers
  mVisIndexCount = ((height - 1) * (width) + (height) * (width - 1)) * 2;
  std::vector<unsigned> visIndices(mVisIndexCount);
  index = 0;
  for (uint32_t x = 0; x < width; ++x) {
    for (uint32_t y = 0; y < height - 1; ++y) {
      visIndices[index++] = x * height + y;
      visIndices[index++] = x * height + y + 1;
    }
  }
  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width - 1; ++x) {
      visIndices[index++] = x * height + y;
      visIndices[index++] = (x + 1) * height + y;
    }
  }

  mVisVAO.Bind();

  mVBO.Bind(GL_ARRAY_BUFFER);

  mVisIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
  mVisIBO.BufferData(visIndices.size() * sizeof(unsigned), visIndices.data(), GL_STATIC_DRAW);

  mVisVAO.EnableAttributeArray(0);
  mVisVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

  mVisVAO.Release();
  mVisIBO.Release();
  mVBO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
