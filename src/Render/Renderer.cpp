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
  mParameters.mWorld.mDepthMode = depthMode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setMaxRenderPasses(int maxRenderPasses) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mMaxRenderPasses = maxRenderPasses;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setMaxLod(int maxLod) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mMaxLod = maxLod;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::clearMaxLod() {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mMaxLod = std::nullopt;
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
  mParameters.mWorld.mVolume.mTransferFunction = transferFunction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setDensityScale(float densityScale) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mVolume.mDensityScale = densityScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setScalarFilters(std::vector<ScalarFilter> filters) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mScalarFilters = filters;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setShading(bool shading) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mLights.mShading = shading;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setAmbientLight(float strength) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mLights.mAmbientStrength = strength;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setSunDirection(glm::vec3 sunDirection) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mLights.mSunDirection = sunDirection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setSunStrength(float strength) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mLights.mSunStrength = strength;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setCoreEnabled(bool value) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mCore.mEnable = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setCoreScalar(std::string value) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mCore.mScalar = std::move(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setCoreRadius(float value) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mCore.mRadius = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setPathlinesEnabled(bool enable) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mPathlines.mEnable = enable;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setPathlineSize(float value) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mPathlines.mLineSize = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::setPathlineScalarFilters(std::vector<ScalarFilter> const& value) {
  std::scoped_lock lock(mParameterMutex);
  mParameters.mWorld.mPathlines.mScalarFilters = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::unique_ptr<Renderer::RenderedImage>> Renderer::getFrame(
    glm::mat4 const& cameraTransform) {
  std::scoped_lock lock(mParameterMutex);
  return std::async(std::launch::async, &Renderer::getFrameImpl, this, cameraTransform, mParameters,
      mDataManager->getState());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Renderer::RenderedImage::operator==(const RenderedImage& other) const {
  return mResolution == other.mResolution &&
         glm::all(glm::epsilonEqual(mCameraTransform[0], other.mCameraTransform[0], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[1], other.mCameraTransform[1], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[2], other.mCameraTransform[2], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[3], other.mCameraTransform[3], 0.0001f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::RenderedImage::RenderedImage(bool valid, int resolution, glm::mat4 cameraTransform,
    glm::mat4 modelView, glm::mat4 projection)
    : mValid(valid)
    , mResolution(resolution)
    , mCameraTransform(cameraTransform)
    , mModelView(modelView)
    , mProjection(projection) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Renderer::RenderedImage::isValid() const {
  return mValid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Renderer::RenderedImage::setValid(bool value) {
  mValid = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int Renderer::RenderedImage::getResolution() const {
  return mResolution;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 const& Renderer::RenderedImage::getCameraTransform() const {
  return mCameraTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 const& Renderer::RenderedImage::getModelView() const {
  return mModelView;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 const& Renderer::RenderedImage::getProjection() const {
  return mProjection;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::CopiedImage::CopiedImage(RenderedImage& other)
    : RenderedImage(other)
    , mColorData(other.getColorData(),
          other.getColorData() + other.getResolution() * other.getResolution() * 4)
    , mDepthData(other.getDepthData(),
          other.getDepthData() + other.getResolution() * other.getResolution()) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::CopiedImage& Renderer::CopiedImage::operator=(RenderedImage& other) {
  RenderedImage::operator=(other);
  mColorData             = std::vector<float>(other.getColorData(),
      other.getColorData() + other.getResolution() * other.getResolution() * 4);
  mDepthData             = std::vector<float>(
      other.getDepthData(), other.getDepthData() + other.getResolution() * other.getResolution());
  return *this;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float* Renderer::CopiedImage::getColorData() {
  return mColorData.data();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float* Renderer::CopiedImage::getDepthData() {
  return mDepthData.data();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
