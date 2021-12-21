////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PointsForwardWarped.hpp"

#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

PointsForwardWarped::PointsForwardWarped(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor)
    , mDepthResolution(256) {
  createBuffers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointsForwardWarped::DoImpl() {
  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    mShader.InitVertexShaderFromString(POINTS_FORWARD_VERT);
    mShader.InitFragmentShaderFromString(POINTS_FORWARD_FRAG);
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

  glm::mat4 matrix =
      glm::mat4(glMatP[0], glMatP[1], glMatP[2], glMatP[3], glMatP[4], glMatP[5], glMatP[6],
          glMatP[7], glMatP[8], glMatP[9], glMatP[10], glMatP[11], glMatP[12], glMatP[13],
          glMatP[14], glMatP[15]) *
      matMV * mTransform *
      glm::mat4(mRadii[0], 0, 0, 0, 0, mRadii[1], 0, 0, 0, 0, mRadii[2], 0, 0, 0, 0, 1) *
      inverse(mRendererMVP);

  std::array<GLint, 4> glViewport;
  glGetIntegerv(GL_VIEWPORT, glViewport.data());
  glm::vec4 volumeTop    = glm::vec4(0, 1, -mRendererMVP[3][2] / mRendererMVP[2][2], 1);
  volumeTop              = matrix * volumeTop;
  volumeTop              = glm::vec4(glm::vec3(volumeTop) * (1.f / volumeTop.w), 1);
  glm::vec4 volumeBottom = glm::vec4(0, -1, -mRendererMVP[3][2] / mRendererMVP[2][2], 1);
  volumeBottom           = matrix * volumeBottom;
  volumeBottom           = glm::vec4(glm::vec3(volumeBottom) * (1.f / volumeBottom.w), 1);
  glm::vec4 volumeRight  = glm::vec4(1, 0, -mRendererMVP[3][2] / mRendererMVP[2][2], 1);
  volumeRight            = matrix * volumeRight;
  volumeRight            = glm::vec4(glm::vec3(volumeRight) * (1.f / volumeRight.w), 1);
  glm::vec4 volumeLeft   = glm::vec4(-1, 0, -mRendererMVP[3][2] / mRendererMVP[2][2], 1);
  volumeLeft             = matrix * volumeLeft;
  volumeLeft             = glm::vec4(glm::vec3(volumeLeft) * (1.f / volumeLeft.w), 1);

  glm::vec4 volumeCenter = glm::vec4(0, 0, -mRendererMVP[3][2] / mRendererMVP[2][2], 1);
  volumeCenter           = inverse(mRendererMVP) * volumeCenter;
  volumeCenter           = glm::vec4(glm::vec3(volumeCenter) * (1.f / volumeCenter.w), 1);
  volumeCenter           = glm::vec4(mRadii, 1) * volumeCenter;
  volumeCenter           = glm::vec4((mTransform * glm::vec4(glm::vec3(volumeCenter), 1)).xyz(), 1);
  volumeCenter           = glm::vec4((matMV * glm::vec4(glm::vec3(volumeCenter), 1)).xyz(), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uBaseDepth"), volumeCenter[2] / volumeCenter[3]);

  int pointHeight = (int)ceil((volumeTop[1] / volumeTop[3] - volumeBottom[1] / volumeBottom[3]) /
                              2.f * (float)glViewport[3] / mDepthResolution);
  int pointWidth  = (int)ceil((volumeRight[0] / volumeRight[3] - volumeLeft[0] / volumeLeft[3]) /
                             2.f * (float)glViewport[2] / mDepthResolution);
  mShader.SetUniform(mShader.GetUniformLocation("uBasePointSize"),
      pointHeight > pointWidth ? pointHeight : pointWidth);

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uDepthTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mShader.SetUniform(mShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uDrawDepth"), mDrawDepth);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawArrays(GL_POINTS, 0, mDepthResolution * mDepthResolution);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mDepthTexture.Unbind(GL_TEXTURE1);
  mShader.Release();
  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointsForwardWarped::createBuffers() {
  std::vector<float>    vertices(mDepthResolution * mDepthResolution * 3);
  std::vector<unsigned> indices((mDepthResolution - 1) * (2 + 2 * mDepthResolution));

  for (int x = 0; x < mDepthResolution; ++x) {
    for (int y = 0; y < mDepthResolution; ++y) {
      vertices[(x * mDepthResolution + y) * 3 + 0] = 2.f / (mDepthResolution - 1) * x - 1.f;
      vertices[(x * mDepthResolution + y) * 3 + 1] = 2.f / (mDepthResolution - 1) * y - 1.f;
      vertices[(x * mDepthResolution + y) * 3 + 2] = 0.f;
    }
  }

  mVAO.Bind();

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

  mVAO.Release();
  mVBO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
