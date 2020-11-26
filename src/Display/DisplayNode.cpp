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

DisplayNode::DisplayNode(VolumeShape shape, VistaSceneGraph* sceneGraph,
    std::string const& centerName, std::string const& frameName, double startExistence,
    double endExistence, glm::dvec3 radii, int depthResolution)
    : cs::scene::CelestialBody(centerName, frameName, startExistence, endExistence)
    , mShape(shape)
    , mVistaSceneGraph(sceneGraph)
    , mRadii(radii)
    , mTexture(GL_TEXTURE_2D)
    , pDepthValues(std::vector<float>(depthResolution * depthResolution))
    , mDepthResolution(depthResolution)
    , mShaderDirty(true) {
  pVisibleRadius = mRadii[0];

  mVistaNode.reset(mVistaSceneGraph->NewOpenGLNode(mVistaSceneGraph->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVistaNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DisplayNode::~DisplayNode() {
  mVistaSceneGraph->GetRoot()->DisconnectChild(mVistaNode.get());
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

double DisplayNode::getHeight(glm::dvec2 lngLat) const {
  return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::getIntersection(
    glm::dvec3 const& rayPos, glm::dvec3 const& rayDir, glm::dvec3& pos) const {
  if (!mEnabled) {
    return false;
  }

  glm::dmat4 transform = glm::inverse(getWorldTransform());

  // Transform ray into planet coordinate system.
  glm::dvec4 origin(rayPos, 1.0);
  origin = transform * origin;

  glm::dvec4 direction(rayDir, 0.0);
  direction = transform * direction;
  direction = glm::normalize(direction);

  switch (mShape) {
  case VolumeShape::eCubic: {
    glm::dvec3 minBounds = -mRadii * (sqrt(3.) / 3.);
    glm::dvec3 maxBounds = mRadii * (sqrt(3.) / 3.);

    glm::dvec4 invDir = 1. / direction;
    double     txmin, txmax, tymin, tymax, tzmin, tzmax;
    double     tmin, tmax;
    if (invDir.x >= 0) {
      txmin = (minBounds[0] - origin.x) * invDir.x;
      txmax = (maxBounds[0] - origin.x) * invDir.x;
    } else {
      txmin = (maxBounds[0] - origin.x) * invDir.x;
      txmax = (minBounds[0] - origin.x) * invDir.x;
    }
    if (invDir.y >= 0) {
      tymin = (minBounds[1] - origin.y) * invDir.y;
      tymax = (maxBounds[1] - origin.y) * invDir.y;
    } else {
      tymin = (maxBounds[1] - origin.y) * invDir.y;
      tymax = (minBounds[1] - origin.y) * invDir.y;
    }

    if ((txmin > tymax) || (tymin > txmax)) {
      return false;
    }
    tmin = fmax(txmin, tymin);
    tmax = fmin(txmax, tymax);

    if (invDir.z >= 0) {
      tzmin = (minBounds[2] - origin.z) * invDir.z;
      tzmax = (maxBounds[2] - origin.z) * invDir.z;
    } else {
      tzmin = (maxBounds[2] - origin.z) * invDir.z;
      tzmax = (minBounds[2] - origin.z) * invDir.z;
    }

    if ((tmin > tzmax) || (tzmin > tmax)) {
      return false;
    }
    tmin = fmax(tmin, tzmin);

    pos = origin + direction * tmin;
    break;
  }
  case VolumeShape::eSpherical: {
    double b    = glm::dot(origin, direction);
    double c    = glm::dot(origin, origin) - mRadii[0] * mRadii[0];
    double fDet = b * b - c;

    if (fDet < 0.0) {
      return false;
    }

    fDet = std::sqrt(fDet);

    pos = (origin + direction * (-b - fDet));
    break;
  }
  default:
    return false;
  }

  double     localScale = getAnchorScale();
  glm::dvec3 localPos   = getAnchorPosition();
  glm::dquat localRot   = getAnchorRotation();

  double     angle = glm::angle(localRot);
  glm::dvec3 axis  = glm::axis(localRot);

  glm::dmat4 mat(1.0);
  mat = glm::translate(mat, localPos);
  mat = glm::rotate(mat, angle, axis);
  mat = glm::scale(mat, glm::dvec3(localScale, localScale, localScale));

  pos = mat * glm::dvec4(pos, 1);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DisplayNode::GetBoundingBox(VistaBoundingBox& bb) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
