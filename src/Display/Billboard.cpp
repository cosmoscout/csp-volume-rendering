////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Billboard.hpp"

#include "../Plugin.hpp"

#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

std::mutex              mUncurrentRequiredMutex;
std::mutex              mUncurrentReleaseMutex;
std::mutex              mUncurrentDoneMutex;
std::condition_variable mUncurrentRequiredCV;
std::condition_variable mUncurrentReleaseCV;
bool                    mContextCurrentIs       = true;
bool                    mContextCurrentShouldBe = true;

////////////////////////////////////////////////////////////////////////////////////////////////////

const uint32_t GRID_RESOLUTION = 256;

////////////////////////////////////////////////////////////////////////////////////////////////////

Billboard::Billboard(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor, GRID_RESOLUTION) {
  pDepthValues.connectAndTouch(
      [this](std::vector<float> depthValues) { createBuffers(depthValues); });

  Plugin::mOnUncurrentRequired.connect([]() {
    {
      std::lock_guard lock(mUncurrentRequiredMutex);
      mContextCurrentShouldBe = false;
    }
    {
      std::unique_lock<std::mutex> lock(mUncurrentDoneMutex);
      while (mContextCurrentIs) {
        mUncurrentRequiredCV.wait(lock);
      }
    }
  });
  Plugin::mOnUncurrentRelease.connect([]() {
    std::lock_guard lock(mUncurrentReleaseMutex);
    mContextCurrentShouldBe = true;
    mUncurrentReleaseCV.notify_all();
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Billboard::Do() {
  if (!mEnabled || !getIsInExistence() || !pVisible.get()) {
    return true;
  }

  bool doUncurrent = false;
  {
    std::lock_guard lock(mUncurrentRequiredMutex);
    doUncurrent = !mContextCurrentShouldBe;
  }
  if (doUncurrent) {
#ifdef _WIN32
    HDC   dc   = wglGetCurrentDC();
    HGLRC glrc = wglGetCurrentContext();

    // uncurrent
    wglMakeCurrent(NULL, NULL);

    mContextCurrentIs = false;
    mUncurrentRequiredCV.notify_all();
    std::unique_lock<std::mutex> lock(mUncurrentReleaseMutex);
    while (!mContextCurrentShouldBe) {
      mUncurrentReleaseCV.wait(lock);
    }

    // current
    wglMakeCurrent(dc, glrc);

    mContextCurrentIs = true;
#endif
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
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mShader.SetUniform(mShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uDrawDepth"), mDrawDepth);

  mTexture.Bind(GL_TEXTURE0);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, (GRID_RESOLUTION - 1) * (2 + 2 * GRID_RESOLUTION),
      GL_UNSIGNED_INT, nullptr);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mShader.Release();
  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Billboard::createBuffers(std::vector<float> depthValues) {
  std::vector<float>    vertices(GRID_RESOLUTION * GRID_RESOLUTION * 3);
  std::vector<unsigned> indices((GRID_RESOLUTION - 1) * (2 + 2 * GRID_RESOLUTION));

  for (uint32_t x = 0; x < GRID_RESOLUTION; ++x) {
    for (uint32_t y = 0; y < GRID_RESOLUTION; ++y) {
      vertices[(x * GRID_RESOLUTION + y) * 3 + 0] = 2.f / (GRID_RESOLUTION - 1) * x - 1.f;
      vertices[(x * GRID_RESOLUTION + y) * 3 + 1] = 2.f / (GRID_RESOLUTION - 1) * y - 1.f;
      vertices[(x * GRID_RESOLUTION + y) * 3 + 2] =
          depthValues[y * mDepthResolution / GRID_RESOLUTION * mDepthResolution +
                      x * mDepthResolution / GRID_RESOLUTION];
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < GRID_RESOLUTION - 1; ++x) {
    indices[index++] = x * GRID_RESOLUTION;
    for (uint32_t y = 0; y < GRID_RESOLUTION; ++y) {
      indices[index++] = x * GRID_RESOLUTION + y;
      indices[index++] = (x + 1) * GRID_RESOLUTION + y;
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
