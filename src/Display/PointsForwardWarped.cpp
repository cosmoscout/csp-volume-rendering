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

PointsForwardWarped::PointsForwardWarped(std::string const& sCenterName,
    std::string const& sFrameName, double tStartExistence, double tEndExistence, glm::dvec3 radii)
    : cs::scene::CelestialObject(sCenterName, sFrameName, tStartExistence, tEndExistence)
    , mRadii(radii)
    , mTexture(GL_TEXTURE_2D)
    , mDepthValues(256 * 256)
    , mResolution(256) {
  pVisibleRadius = mRadii[0];

  createBuffers();

  mShaderDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointsForwardWarped::setTexture(std::vector<uint8_t>& texture, int width, int height) {
  mTexture.UploadTexture(width, height, texture.data(), false, GL_RGBA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointsForwardWarped::setDepthTexture(std::vector<float>& texture, int width, int height) {
  mDepthValues = texture;
  mResolution  = width;

  createBuffers();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointsForwardWarped::setTransform(glm::mat4 transform) {
  mTransform = transform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointsForwardWarped::Do() {
  if (!getIsInExistence() || !pVisible.get()) {
    return true;
  }

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

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());

  mTexture.Bind(GL_TEXTURE0);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawArrays(GL_POINTS, 0, mResolution * mResolution);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mShader.Release();
  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PointsForwardWarped::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PointsForwardWarped::createBuffers() {
  std::vector<float>    vertices(mResolution * mResolution * 3);
  std::vector<unsigned> indices((mResolution - 1) * (2 + 2 * mResolution));

  for (uint32_t x = 0; x < mResolution; ++x) {
    for (uint32_t y = 0; y < mResolution; ++y) {
      vertices[(x * mResolution + y) * 3 + 0] = 2.f / (mResolution - 1) * x - 1.f;
      vertices[(x * mResolution + y) * 3 + 1] = 2.f / (mResolution - 1) * y - 1.f;
      vertices[(x * mResolution + y) * 3 + 2] = mDepthValues[x * mResolution + y];
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
