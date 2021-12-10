////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DisplayNode.hpp"

#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

DisplayNode::DisplayNode(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : mShape(shape)
    , mTexture(GL_TEXTURE_2D)
    , mDepthTexture(GL_TEXTURE_2D)
    , mShaderDirty(true)
    , mOut(GL_TEXTURE_2D) {
  settings->initAnchor(*this, anchor);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mVistaNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) + 10);

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

DisplayNode::~DisplayNode() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mVistaNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setEnabled(bool enabled) {
  mEnabled = enabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> DisplayNode::getDepthBuffer(int resolution) {
  if (resolution == mResolution) {
    if (mPBOFence.has_value()) {
      int sync = glClientWaitSync(mPBOFence.value(), 0, 50 * 1000000);
      if (sync == GL_ALREADY_SIGNALED || sync == GL_CONDITION_SATISFIED) {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
        float* data = static_cast<float*>(glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY));
        std::vector<float> buffer(data, data + resolution * resolution);
        glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        return buffer;
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
  std::vector<float> buffer(resolution * resolution, INFINITY);
  return buffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setTexture(uint8_t* texture, int width, int height) {
  mTexture.UploadTexture(width, height, texture, false, GL_RGBA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setTexture(float* texture, int width, int height) {
  mTexture.UploadTexture(width, height, texture, false, GL_RGBA, GL_FLOAT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setDepthTexture(float* texture, int width, int height) {
  // VistaTexture does not support upload with different internal format than GL_RGBA8, so we upload
  // the texture manually.
  mDepthTexture.Bind();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, texture);
  glTexParameteri(mDepthTexture.GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  mDepthTexture.Unbind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setTransform(glm::mat4 transform) {
  mTransform = std::move(transform);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setRendererMatrices(glm::mat4 modelView, glm::mat4 projection) {
  mRendererModelView  = std::move(modelView);
  mRendererProjection = std::move(projection);
  mRendererMVP        = mRendererProjection * mRendererModelView;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setUseDepth(bool useDepth) {
  mUseDepth = useDepth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setDrawDepth(bool drawDepth) {
  mDrawDepth = drawDepth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 DisplayNode::getRadii() const {
  return mRadii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::Do() {
  if (!mEnabled || !getIsInExistence() || !pVisible.get()) {
    return true;
  }

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

  glDispatchCompute(mResolution, mResolution, 1);

  glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);

  glBindBuffer(GL_PIXEL_PACK_BUFFER, mPBO);
  mOut.Bind();
  glGetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_FLOAT, nullptr);
  mOut.Unbind();
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  mPBOFence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

  return DoImpl();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
