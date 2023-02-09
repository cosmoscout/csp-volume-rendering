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

#include <thrust/host_vector.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

IrregularGrid::IrregularGrid(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setDepthTexture(float* texture, int width, int height) {
  DisplayNode::setDepthTexture(texture, width, height);
  mSurfaces.emplace(texture, width, height);
  createBuffers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IrregularGrid::DoImpl() {
  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();
    mShader.InitVertexShaderFromString(IRREGULAR_GRID_VERT);
    mShader.InitFragmentShaderFromString(BILLBOARD_FRAG);
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

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawArrays(GL_POINTS, 0, mVertexCount);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mTexture.Unbind(GL_TEXTURE1);
  mShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::createBuffers() {
  thrust::device_vector<glm::uvec3> dVertices = mSurfaces->generateVertices();
  thrust::host_vector<glm::uvec3>   hVertices = dVertices;
  mVertexCount                                = hVertices.size();
  std::vector<unsigned int> stdVertices(
      (unsigned int*)hVertices.data(), (unsigned int*)hVertices.data() + hVertices.size() * 3);
  //logger().trace("{}", stdVertices);

  mVAO.Bind();

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(hVertices.size() * sizeof(glm::uvec3), hVertices.data(), GL_STATIC_DRAW);

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayInteger(0, 3, GL_UNSIGNED_INT, 3 * sizeof(unsigned int), 0, &mVBO);

  mVAO.Release();
  mVBO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
