////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DepthExtractor.hpp"

#include "../Utility.hpp"
#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/convert.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

DepthExtractor::DepthExtractor(std::shared_ptr<DisplayNode> displayNode,
    std::shared_ptr<cs::core::SolarSystem>                  solarSystem,
    std::shared_ptr<cs::core::TimeControl>                  timeControl)
    : mDisplayNode(displayNode)
    , mSolarSystem(solarSystem)
    , mTimeControl(timeControl)
    , mOut(GL_TEXTURE_2D) {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mVistaNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) + 9);

  glGenBuffers(1, &mPBO);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  glBufferStorage(
      GL_PIXEL_PACK_BUFFER, sizeof(float) * mResolution * mResolution, nullptr, GL_MAP_READ_BIT);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  auto        shader  = glCreateShader(GL_COMPUTE_SHADER);
  const char* pSource = GET_DEPTH_COMP.c_str();
  glShaderSource(shader, 1, &pSource, nullptr);
  glCompileShader(shader);

  int rvalue = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &rvalue);
  if (rvalue != GL_TRUE) {
    auto log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetShaderInfoLog(shader, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));
    glDeleteShader(shader);
    throw std::runtime_error(std::string("ERROR: Failed to compile shader\n") + log);
  }

  mDepthComputeShader = glCreateProgram();
  glAttachShader(mDepthComputeShader, shader);
  glLinkProgram(mDepthComputeShader);
  glDeleteShader(shader);

  glGetProgramiv(mDepthComputeShader, GL_LINK_STATUS, &rvalue);
  if (rvalue != GL_TRUE) {
    auto log_length = 0;
    glGetProgramiv(mDepthComputeShader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetProgramInfoLog(mDepthComputeShader, log_length, nullptr, v.data());
    std::string log(begin(v), end(v));

    throw std::runtime_error(std::string("ERROR: Failed to link compute shader\n") + log);
  }

  mUniforms.mTopCorner    = glGetUniformLocation(mDepthComputeShader, "uTopCorner");
  mUniforms.mBottomCorner = glGetUniformLocation(mDepthComputeShader, "uBottomCorner");
  mUniforms.mRadius       = glGetUniformLocation(mDepthComputeShader, "uRadius");
  mUniforms.mNearDistance = glGetUniformLocation(mDepthComputeShader, "uNear");
  mUniforms.mFarDistance  = glGetUniformLocation(mDepthComputeShader, "uFar");

  // Create textures for depth buffer of previous render pass
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    const auto [buffer, success] =
        mDepthBufferData.try_emplace(viewport.second, GL_TEXTURE_RECTANGLE);
    if (success) {
      buffer->second.Bind();
      buffer->second.SetWrapS(GL_CLAMP);
      buffer->second.SetWrapT(GL_CLAMP);
      buffer->second.SetMinFilter(GL_NEAREST);
      buffer->second.SetMagFilter(GL_NEAREST);
      buffer->second.Unbind();
    }
  }

  mOut.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, mResolution, mResolution, 0, GL_RED, GL_FLOAT, nullptr);
  mOut.Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DepthExtractor::~DepthExtractor() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mVistaNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DepthExtractor::setEnabled(bool enabled) {
  mEnabled = enabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::vector<float>> DepthExtractor::getDepthBuffer(int resolution) {
  if (resolution == mResolution) {
    if (mPBOFence.has_value()) {
      // Wait for max 50ms
      int sync = glClientWaitSync(mPBOFence.value(), 0, 50 * 1000000);
      if (sync == GL_ALREADY_SIGNALED || sync == GL_CONDITION_SATISFIED) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
        float* data = static_cast<float*>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));
        std::vector<float> buffer(data, data + resolution * resolution);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        return std::move(buffer);
      }
    }
  } else {
    mResolution = resolution;

    glDeleteBuffers(1, &mPBO);
    glGenBuffers(1, &mPBO);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
    glBufferStorage(
        GL_PIXEL_PACK_BUFFER, sizeof(float) * mResolution * mResolution, nullptr, GL_MAP_READ_BIT);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    mOut.Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, mResolution, mResolution, 0, GL_RED, GL_FLOAT, nullptr);
    mOut.Unbind();
  }
  return {};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DepthExtractor::Do() {
  if (!mEnabled) {
    return true;
  }

  // Get field of view and aspect ratio
  glm::mat4 glMatP;
  glGetFloatv(GL_PROJECTION_MATRIX, reinterpret_cast<GLfloat*>(&glMatP));

  float aspect  = glMatP[1][1] / glMatP[0][0];
  float fovYRad = 2.f * atan(1.f / glMatP[1][1]);
  float fovXRad = fovYRad * aspect;
  fovXRad *= 0.9f;

  // Get edges of body as screen space coordinates
  Utility::CameraParams crop =
      Utility::calculateCameraParams(static_cast<float>(mDisplayNode->getRadii()[0]),
          mDisplayNode->getRelativeTransform(
              mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver()),
          fovYRad, fovXRad);

  // Get near and far distance
  double                                      near, far;
  VistaProjection::VistaProjectionProperties* projectionProperties =
      GetVistaSystem()
          ->GetDisplayManager()
          ->GetCurrentRenderInfo()
          ->m_pViewport->GetProjection()
          ->GetProjectionProperties();
  projectionProperties->GetClippingRange(near, far);

  // copy depth buffer from previous rendering
  std::array<GLint, 4> iViewport{};
  glGetIntegerv(GL_VIEWPORT, iViewport.data());

  auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
  VistaTexture& depthBuffer = mDepthBufferData.at(viewport);

  depthBuffer.Bind();
  glCopyTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_DEPTH_COMPONENT, iViewport[0], iViewport[1],
      iViewport[2], iViewport[3], 0);
  depthBuffer.Unbind();

  glUseProgram(mDepthComputeShader);

  depthBuffer.Bind(GL_TEXTURE0);
  glBindImageTexture(1, mOut.GetId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

  glUniform2f(mUniforms.mBottomCorner, crop.mLeft, crop.mBottom);
  glUniform2f(mUniforms.mTopCorner, crop.mRight, crop.mTop);
  glUniform1f(mUniforms.mRadius, static_cast<float>(mDisplayNode->getRadii()[0]));
  glUniform1f(mUniforms.mNearDistance,
      static_cast<float>(
          near * mSolarSystem->getObserver().getAnchorScale() / mDisplayNode->getAnchorScale()));
  glUniform1f(mUniforms.mFarDistance,
      static_cast<float>(
          far * mSolarSystem->getObserver().getAnchorScale() / mDisplayNode->getAnchorScale()));

  glDispatchCompute(mResolution, mResolution, 1);

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  mOut.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, nullptr);
  mOut.Unbind();
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  mPBOFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DepthExtractor::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
