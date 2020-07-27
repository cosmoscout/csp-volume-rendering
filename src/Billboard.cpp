////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Billboard.hpp"

#include "Shaders.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const uint32_t GRID_RESOLUTION_X = 100;
const uint32_t GRID_RESOLUTION_Y = 100;

////////////////////////////////////////////////////////////////////////////////////////////////////

Billboard::Billboard(std::string const& sCenterName, std::string const& sFrameName,
    double tStartExistence, double tEndExistence, glm::dvec3 radii)
    : cs::scene::CelestialObject(sCenterName, sFrameName, tStartExistence, tEndExistence)
    , mRadii(radii) {
  pVisibleRadius = mRadii[0];

  std::vector<float>    vertices(GRID_RESOLUTION_X * GRID_RESOLUTION_Y * 2);
  std::vector<unsigned> indices((GRID_RESOLUTION_X - 1) * (2 + 2 * GRID_RESOLUTION_Y));

  for (uint32_t x = 0; x < GRID_RESOLUTION_X; ++x) {
    for (uint32_t y = 0; y < GRID_RESOLUTION_Y; ++y) {
      vertices[(x * GRID_RESOLUTION_Y + y) * 2 + 0] = 2.f / (GRID_RESOLUTION_X - 1) * x - 1.f;
      vertices[(x * GRID_RESOLUTION_Y + y) * 2 + 1] = 2.f / (GRID_RESOLUTION_Y - 1) * y - 1.f;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < GRID_RESOLUTION_X - 1; ++x) {
    indices[index++] = x * GRID_RESOLUTION_Y;
    for (uint32_t y = 0; y < GRID_RESOLUTION_Y; ++y) {
      indices[index++] = x * GRID_RESOLUTION_Y + y;
      indices[index++] = (x + 1) * GRID_RESOLUTION_Y + y;
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
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0, &mVBO);

  mVAO.Release();
  mIBO.Release();
  mVBO.Release();

  mTexture = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
  std::vector<uint8_t> tempTexture(16);
  mTexture->UploadTexture(2, 2, tempTexture.data(), true, GL_RGBA);

  mShaderDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Billboard::setTexture(std::vector<uint8_t>& texture, int width, int height) {
  mTexture->UploadTexture(width, height, texture.data(), true, GL_RGBA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Billboard::setTransform(glm::mat4 transform) {
  mTransform = transform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Billboard::Do() {
  if (!getIsInExistence() || !pVisible.get()) {
    return true;
  }

  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();

    mShader.InitVertexShaderFromString(BILLBOARD_VERT);
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

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());

  mTexture->Bind(GL_TEXTURE0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, (GRID_RESOLUTION_X - 1) * (2 + 2 * GRID_RESOLUTION_Y),
      GL_UNSIGNED_INT, nullptr);
  mVAO.Release();

  // Clean up.
  glDisable(GL_BLEND);
  mTexture->Unbind(GL_TEXTURE0);
  mShader.Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Billboard::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
