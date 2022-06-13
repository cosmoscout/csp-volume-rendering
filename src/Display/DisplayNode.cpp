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
    , mShaderDirty(true) {
  settings->initAnchor(*this, anchor);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mVistaNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::ePlanets) - 10);
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

void DisplayNode::setRendererMatrices(glm::mat4 modelView, glm::mat4 projection, bool inside) {
  mRendererModelView  = std::move(modelView);
  mRendererProjection = std::move(projection);
  mRendererMVP        = mRendererProjection * mRendererModelView;
  mInside             = inside;
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
  return DoImpl();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
