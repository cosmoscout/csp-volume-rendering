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

DisplayNode::DisplayNode(VolumeShape shape, std::shared_ptr<cs::core::Settings> settings,
    std::string anchor, int depthResolution)
    : mShape(shape)
    , mTexture(GL_TEXTURE_2D)
    , pDepthValues(std::vector<float>(depthResolution * depthResolution))
    , mDepthResolution(depthResolution)
    , mShaderDirty(true) {
  settings->initAnchor(*this, anchor);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mVistaNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
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

void DisplayNode::setTexture(std::vector<uint8_t>& texture, int width, int height) {
  mTexture.UploadTexture(width, height, texture.data(), false, GL_RGBA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setTexture(uint8_t* texture, int width, int height) {
  mTexture.UploadTexture(width, height, texture, false, GL_RGBA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setDepthTexture(std::vector<float>& texture, int width, int height) {
  mDepthResolution = width;
  pDepthValues.set(texture);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setTransform(glm::mat4 transform) {
  mTransform = transform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DisplayNode::setMVPMatrix(glm::mat4 mvp) {
  mRendererMVP = mvp;
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

bool DisplayNode::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
