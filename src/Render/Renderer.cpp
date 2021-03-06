////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Renderer.hpp"

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* RendererException::what() const noexcept {
  return "Failed to initialize Renderer.";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::Renderer(
    std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape)
    : mDataManager(dataManager)
    , mStructure(structure)
    , mShape(shape) {
  if (shape == VolumeShape::eInvalid) {
    logger().error("Invalid volume shape given in settings! Should be one of 'cubic', "
                   "'shperical'.");
    throw RendererException();
  }
  if (structure == VolumeStructure::eInvalid) {
    logger().error("Invalid volume structure given in settings! Should be one of 'structured', "
                   "'unstructured'.");
    throw RendererException();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setResolution(int resolution) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mResolution = resolution;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setSamplingRate(float samplingRate) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mSamplingRate = samplingRate;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setDepthMode(DepthMode depthMode) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mDepthMode = depthMode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setDenoiseColor(bool denoiseColor) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mDenoiseColor = denoiseColor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setDenoiseDepth(bool denoiseDepth) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mDenoiseDepth = denoiseDepth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setTransferFunction(std::vector<glm::vec4> transferFunction) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mTransferFunction = transferFunction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setDensityScale(float densityScale) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mDensityScale = densityScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setShading(bool shading) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mShading = shading;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setAmbientLight(float strength) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mAmbientLight = strength;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setSunDirection(glm::vec3 sunDirection) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mSunDirection = sunDirection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setSunStrength(float strength) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mSunStrength = strength;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<Renderer::RenderedImage> Renderer::getFrame(glm::mat4 cameraTransform) {
  std::scoped_lock lock(mParameterMutex);
  return std::async(std::launch::async, &Renderer::getFrameImpl, this, cameraTransform, mParameters,
      mDataManager->getState());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
