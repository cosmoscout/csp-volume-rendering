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

#include "glm/gtc/epsilon.hpp"
#include "glm/gtx/quaternion.hpp"

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

DisplayNode::DisplayNode(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : mShape(shape)
    , mShaderDirty(true) {
  settings->initAnchor(*this, anchor);

  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mVistaNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::eStars) + 1);
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

void DisplayNode::setImage(std::unique_ptr<Renderer::RenderedImage> image) {
  mImage = std::move(image);
  if (mTexture.size() != mImage->getLayerCount()) {
    mTexture.resize(mImage->getLayerCount());
  }
  for (auto i = 0u; i < mImage->getLayerCount(); ++i) {
    if (!mTexture[i]) {
      mTexture[i] = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
      mTexture[i]->SetWrapS(GL_CLAMP_TO_BORDER);
      mTexture[i]->SetWrapT(GL_CLAMP_TO_BORDER);
    }
    /*mTexture[i]->UploadTexture(mImage->getResolution(), mImage->getResolution(),
       mImage->getColorData(i), false, GL_RGBA, GL_FLOAT);*/
    mTexture[i]->Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mImage->getResolution(), mImage->getResolution(), 0,
        GL_RGBA, GL_FLOAT, mImage->getColorData(i));
    glTexParameteri(mTexture[i]->GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    mTexture[i]->Unbind();
  }

  if (mDepthTexture.size() != mImage->getLayerCount()) {
    mDepthTexture.resize(mImage->getLayerCount());
  }
  for (auto i = 0u; i < mImage->getLayerCount(); ++i) {
    if (!mDepthTexture[i]) {
      mDepthTexture[i] = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
      // mDepthTexture[i]->SetWrapS(GL_CLAMP_TO_BORDER);
      // mDepthTexture[i]->SetWrapT(GL_CLAMP_TO_BORDER);
    }
    // VistaTexture does not support upload with different internal format than GL_RGBA8, so we
    // upload the texture manually.
    mDepthTexture[i]->Bind();
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, mImage->getResolution(), mImage->getResolution(), 0, GL_RED,
        GL_FLOAT, mImage->getDepthData(i));
    glTexParameteri(mDepthTexture[i]->GetTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    mDepthTexture[i]->Unbind();
  }

  mTransform = glm::toMat4(glm::toQuat(mImage->getCameraTransform()));

  mRendererModelView  = mImage->getModelView();
  mRendererProjection = mImage->getProjection();
  mRendererMVP        = mRendererProjection * mRendererModelView;
  mInside             = mImage->isInside();
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

void DisplayNode::setHoleFillingLevel(int value) {
  mHoleFillingLevel = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 DisplayNode::getRadii() const {
  return mRadii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 DisplayNode::getVistaModelView() const {
  return mVistaModelView;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::Do() {
  if (!mEnabled || !getIsInExistence() || !pVisible.get()) {
    return true;
  }

  std::array<GLfloat, 16> glMatMV{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  mVistaModelView = glm::make_mat4(glMatMV.data());

  return DoImpl();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
