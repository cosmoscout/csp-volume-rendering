////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../Utility.hpp"
#include "../logger.hpp"

#include "../../../../src/cs-gui/internal/ScopedTimer.hpp"
#include "../../../../src/cs-utils/convert.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vtkCellData.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkStructuredPoints.h>

#include <rkcommon/math/vec.h>

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>
#include <ospray/ospray_cpp/ext/rkcommon.h>

#include <cmath>
#include <exception>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(
    std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape)
    : Renderer(dataManager, structure, shape) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::~OSPRayRenderer() {
  mCache.mVolumes.clear();
  mRenderFuture.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float OSPRayRenderer::getProgress() {
  std::scoped_lock lock(mRenderFutureMutex);
  if (mRenderFuture.has_value()) {
    return mRenderFuture->progress();
  } else {
    return 1.f;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::preloadData(
    DataManager::State const& state, std::optional<DataManager::State> const& coreState) {
  if (mCache.mVolumes.find(state) == mCache.mVolumes.end()) {
    int lod = mDataManager->getMinLod(state);
    mCache.mVolumes[state][lod] =
        std::async(std::launch::async, [this, state, lod]() { return loadVolume(state, lod); });
    if (coreState.has_value() && !(coreState.value() == state)) {
      mCache.mVolumes[coreState.value()][lod] = std::async(std::launch::async,
          [this, coreState, lod]() { return loadVolume(coreState.value(), lod); });
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::cancelRendering() {
  std::scoped_lock lock(mRenderFutureMutex);
  if (mRenderFuture.has_value()) {
    mRenderFuture->cancel();
    mRenderingCancelled = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Renderer::RenderedImage> OSPRayRenderer::getFrameImpl(
    glm::mat4 const& cameraTransform, std::optional<std::vector<float>>&& maxDepth,
    Parameters parameters, DataManager::State const& dataState) {
  // Shift filter attribute indices by one, because the scalar list used by the renderer is shifted
  // by one so that the active scalar can be placed at index 0.
  for (ScalarFilter& filter : parameters.mScalarFilters) {
    filter.mAttrIndex += 1;
  }

  mRenderingCancelled = false;
  try {
    Volume const& volume = getVolume(dataState, parameters.mMaxLod);
    Cache::State  state{cameraTransform, parameters, dataState, volume.mLod};
    if (mCache.mState == state && mFrameBufferAccumulationPasses >= parameters.mMaxRenderPasses) {
      return {};
    }
    if (!(parameters.mWorld == mCache.mState.mParameters.mWorld &&
            dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
      updateWorld(volume, parameters, dataState);
    }
    if (!(cameraTransform == mCache.mState.mCameraTransform &&
            dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
      mCache.mCamera = getCamera(1, cameraTransform);
    }
    renderFrame(mCache.mWorld, mCache.mCamera.mOsprayCamera, std::move(maxDepth), parameters,
        !(mCache.mState == state));
    RenderedImage renderedImage(
        mCache.mFrameBuffer, mCache.mCamera, volume.mHeight, parameters, cameraTransform);
    renderedImage.setValid(!mRenderingCancelled);
    mCache.mState = state;
    return std::make_unique<RenderedImage>(std::move(renderedImage));
  } catch (const std::exception&) { return {}; }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Volume const& OSPRayRenderer::getVolume(
    DataManager::State const& state, std::optional<int> const& maxLod) {
  int         lod        = mDataManager->getMaxLod(state, maxLod);
  auto const& stateCache = mCache.mVolumes.find(state);
  if (stateCache == mCache.mVolumes.end()) {
    mCache.mVolumes[state][lod] =
        std::async(std::launch::deferred, [this, state, lod]() { return loadVolume(state, lod); });
  } else {
    auto const& cachedVolume = stateCache->second.find(lod);
    if (cachedVolume == stateCache->second.end()) {
      mCache.mVolumes[state][lod] =
          std::async(std::launch::async, [this, state, lod]() { return loadVolume(state, lod); });
    }
    for (auto const& cacheEntry : stateCache->second) {
      if (cacheEntry.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        if (!maxLod.has_value() || cacheEntry.first <= maxLod) {
          lod = cacheEntry.first;
        }
      }
    }
  }
  return mCache.mVolumes[state][lod].get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Volume OSPRayRenderer::loadVolume(DataManager::State const& state, int lod) {
  Volume                      volume;
  vtkSmartPointer<vtkDataSet> volumeData = mDataManager->getData(state, lod);
  switch (mStructure) {
  case VolumeStructure::eUnstructured:
    volume.mOsprayData = OSPRayUtility::createOSPRayVolume(
        vtkUnstructuredGrid::SafeDownCast(volumeData), state.mScalar.mType);
    break;
  case VolumeStructure::eStructured: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData =
        OSPRayUtility::createOSPRayVolume(vtkImageData::SafeDownCast(volumeData), scalars);
    break;
  }
  case VolumeStructure::eRectilinear: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData =
        OSPRayUtility::createOSPRayVolume(vtkRectilinearGrid::SafeDownCast(volumeData), scalars);
    break;
  }
  case VolumeStructure::eStructuredSpherical: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData = OSPRayUtility::createOSPRayVolume(
        vtkStructuredGrid::SafeDownCast(volumeData), scalars, mDataManager->getMetadata());
    break;
  }
  case VolumeStructure::eRectilinearSpherical: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData = OSPRayUtility::createOSPRayVolume(
        vtkRectilinearGrid::SafeDownCast(volumeData), scalars, mDataManager->getMetadata());
    break;
  }
  case VolumeStructure::eImageSpherical: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData = OSPRayUtility::createOSPRayVolume(
        vtkImageData::SafeDownCast(volumeData), scalars, mDataManager->getMetadata());
    break;
  }
  case VolumeStructure::eInvalid:
    throw std::runtime_error("Trying to load volume with unknown/invalid structure!");
  }
  volume.mHeight = getHeight(volumeData);
  volume.mLod    = lod;
  return volume;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float OSPRayRenderer::getHeight(vtkSmartPointer<vtkDataSet> data) {
  data->ComputeBounds();
  std::vector<float> bounds(6);
  for (int i = 0; i < 6; i++) {
    bounds[i] = (float)data->GetBounds()[i];
  }
  float x = (bounds[1] - bounds[0]) / 2;
  float y = (bounds[3] - bounds[2]) / 2;
  float z = (bounds[5] - bounds[4]) / 2;
  float height;

  switch (mShape) {
  case VolumeShape::eCubic: {
    float diagonal = sqrtf(x * x + y * y + z * z);
    height         = diagonal;
    break;
  }
  case VolumeShape::eSpherical: {
    height = fmax(fmax(x, y), z);
    break;
  }
  case VolumeShape::eInvalid:
  default: {
    height = 0;
    break;
  }
  }
  return height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::TransferFunction OSPRayRenderer::getTransferFunction(
    DataManager::State const& state, Parameters const& parameters) {
  std::array<double, 2> range = mDataManager->getScalarRange(state.mScalar);
  if (parameters.mWorld.mVolume.mTransferFunction.size() > 0) {
    return OSPRayUtility::createOSPRayTransferFunction(static_cast<float>(range[0]),
        static_cast<float>(range[1]), parameters.mWorld.mVolume.mTransferFunction);
  } else {
    return OSPRayUtility::createOSPRayTransferFunction();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::updateWorld(
    Volume const& volume, Parameters const& parameters, DataManager::State const& dataState) {
  bool updateGroup = false;

  ospray::cpp::TransferFunction transferFunction = getTransferFunction(dataState, parameters);

  if (!(dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
    mCache.mVolumeModel.setParam("volume", volume.mOsprayData);
    mCache.mVolumeModel.setParam("transferFunction", transferFunction);
    mCache.mVolumeModel.commit();
    updateGroup = true;
  }

  if (!(dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod &&
          parameters.mWorld.mCore == mCache.mState.mParameters.mWorld.mCore)) {
    if (parameters.mWorld.mCore.mEnable) {
      DataManager::State scalarState = dataState;
      auto               scalar =
          std::find_if(mDataManager->pScalars.get().begin(), mDataManager->pScalars.get().end(),
              [&parameters](Scalar s) { return s.getId() == parameters.mWorld.mCore.mScalar; });

      if (scalar != mDataManager->pScalars.get().end()) {
        scalarState.mScalar = *scalar;
        Volume coreVolume   = getVolume(scalarState, volume.mLod);

        std::array<double, 2> valueRange    = mDataManager->getScalarRange(scalarState.mScalar);
        rkcommon::math::vec2f valueRangeOsp = {
            static_cast<float>(valueRange[1]) * 0.8f, static_cast<float>(valueRange[1]) * 0.98f};
        mCache.mCoreTransferFunction.setParam("valueRange", valueRangeOsp);
        mCache.mCoreTransferFunction.commit();

        mCache.mCoreTexture.setParam("volume", coreVolume.mOsprayData);
        mCache.mCoreTexture.commit();

        mCache.mCoreMaterial.removeParam("kd");
        mCache.mCoreMaterial.setParam("map_kd", mCache.mCoreTexture);
        mCache.mCoreMaterial.commit();
      } else {
        rkcommon::math::vec3f color = {0.8f, 0.8f, 0.8f};
        mCache.mCoreMaterial.removeParam("map_kd");
        mCache.mCoreMaterial.setParam("kd", color);
        mCache.mCoreMaterial.commit();
      }

      mCache.mCore.setParam("radius", parameters.mWorld.mCore.mRadius);
      mCache.mCore.commit();
    }
  }

  if (!(parameters.mWorld.mVolume == mCache.mState.mParameters.mWorld.mVolume &&
          parameters.mWorld.mLights.mShading ==
              mCache.mState.mParameters.mWorld.mLights.mShading)) {
    mCache.mVolumeModel.setParam("transferFunction", transferFunction);
    mCache.mVolumeModel.setParam("densityScale", parameters.mWorld.mVolume.mDensityScale);
    mCache.mVolumeModel.setParam(
        "gradientShadingScale", parameters.mWorld.mLights.mShading ? 1.f : 0.f);
    mCache.mVolumeModel.commit();
  }

  bool pathlinesPresent = true;
  if (parameters.mWorld.mPathlines.mEnable &&
      !(parameters.mWorld.mPathlines == mCache.mState.mParameters.mWorld.mPathlines)) {
    std::vector<uint32_t> indices =
        mDataManager->getPathlines().getIndices(parameters.mWorld.mPathlines.mScalarFilters);
    if (indices.size() <= 0) {
      pathlinesPresent = false;
    } else {
      std::vector<rkcommon::math::vec4f> vertices =
          mDataManager->getPathlines().getVertices(parameters.mWorld.mPathlines.mLineSize);
      std::vector<rkcommon::math::vec2f> texCoords =
          mDataManager->getPathlines().getTexCoords("point_ParticleAge", "point_ParticleAge");

      mCache.mPathlines.setParam("type", OSP_FLAT);
      mCache.mPathlines.setParam("basis", OSP_LINEAR);
      mCache.mPathlines.setParam("vertex.position_radius", ospray::cpp::Data(vertices));
      mCache.mPathlines.setParam("vertex.texcoord", ospray::cpp::Data(texCoords));
      mCache.mPathlines.setParam("index", ospray::cpp::Data(indices));
      mCache.mPathlines.commit();

      updateGroup = true;
    }
  }

  if (parameters.mWorld.mPathlines.mEnable &&
      !(parameters.mWorld.mPathlinesTexture == mCache.mState.mParameters.mWorld.mPathlinesTexture &&
          parameters.mWorld.mPathlines == mCache.mState.mParameters.mWorld.mPathlines)) {
    std::vector<uint8_t> pixels(256 * 256 * 4);
    for (int x = 0; x < 256; x++) {
      for (int y = 0; y < 256; y++) {
        pixels[4 * (y * 256 + x) + 0] = x < 128 ? x * 2 : 255;
        pixels[4 * (y * 256 + x) + 1] = 255 - std::abs((x - 128) * 2);
        pixels[4 * (y * 256 + x) + 2] = x > 128 ? 255 - (x - 128) * 2 : 255;
        pixels[4 * (y * 256 + x) + 3] = 255;
      }
    }
    ospray::cpp::Data texData(pixels.data(), OSP_VEC4UC, rkcommon::math::vec2i{256, 256});

    ospray::cpp::Texture tex("texture2d");
    tex.setParam("format", OSPTextureFormat::OSP_TEXTURE_RGBA8);
    tex.setParam("data", texData);
    tex.commit();

    ospray::cpp::Material mat("scivis", "obj");
    mat.setParam("map_kd", tex);
    mat.setParam("d", 1.f);
    mat.commit();

    mCache.mPathlinesModel.setParam("material", mat);
    mCache.mPathlinesModel.commit();
  }

  if (!(parameters.mWorld.mDepthMode == mCache.mState.mParameters.mWorld.mDepthMode &&
          parameters.mWorld.mPathlines.mEnable ==
              mCache.mState.mParameters.mWorld.mPathlines.mEnable &&
          parameters.mWorld.mCore.mEnable == mCache.mState.mParameters.mWorld.mCore.mEnable)) {
    std::vector<ospray::cpp::GeometricModel> geometries;
    if (parameters.mWorld.mDepthMode == DepthMode::eIsosurface) {
      ospray::cpp::Geometry isosurface("isosurface");
      isosurface.setParam("isovalue", 0.8f);
      isosurface.setParam("volume", mCache.mVolumeModel.handle());
      isosurface.commit();

      rkcommon::math::vec4f       color{1.f, 1.f, 1.f, 0.f};
      ospray::cpp::GeometricModel isoModel(isosurface);
      isoModel.setParam("color", ospray::cpp::Data(color));
      isoModel.commit();

      geometries.push_back(isoModel);
    }
    if (parameters.mWorld.mCore.mEnable) {
      geometries.push_back(mCache.mCoreModel);
    }
    if (parameters.mWorld.mPathlines.mEnable && pathlinesPresent) {
      geometries.push_back(mCache.mPathlinesModel);
    }
    if (geometries.size() > 0) {
      mCache.mGroup.setParam("geometry", ospray::cpp::Data(geometries));
    } else {
      mCache.mGroup.setParam("geometry", NULL);
    }
    updateGroup = true;
  }

  if (updateGroup) {
    mCache.mGroup.commit();

    rkcommon::math::affine3f transform = rkcommon::math::affine3f::scale(1.f / volume.mHeight);
    mCache.mInstance.setParam("xfm", transform);
    mCache.mInstance.commit();
  }

  if (!(parameters.mWorld.mLights == mCache.mState.mParameters.mWorld.mLights)) {
    mCache.mAmbientLight.setParam("intensity",
        parameters.mWorld.mLights.mShading ? parameters.mWorld.mLights.mAmbientStrength : 1);
    mCache.mAmbientLight.setParam("color", rkcommon::math::vec3f(1, 1, 1));
    mCache.mAmbientLight.commit();

    mCache.mSunLight.setParam("intensity",
        parameters.mWorld.mLights.mShading ? parameters.mWorld.mLights.mSunStrength : 0);
    mCache.mSunLight.setParam("color", rkcommon::math::vec3f(1, 1, 1));
    mCache.mSunLight.setParam(
        "direction", rkcommon::math::vec3f{-parameters.mWorld.mLights.mSunDirection[0],
                         -parameters.mWorld.mLights.mSunDirection[1],
                         -parameters.mWorld.mLights.mSunDirection[2]});
    mCache.mSunLight.setParam("angularDiameter", .53f);
    mCache.mSunLight.commit();
  }

  mCache.mWorld.commit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Camera OSPRayRenderer::getCamera(float volumeHeight, glm::mat4 observerTransform) {
  // Scale observer transform according to the size of the volume
  observerTransform[3] =
      observerTransform[3] * glm::vec4(volumeHeight, volumeHeight, volumeHeight, 1);

  float                 fov = 3.141f / 2;
  Utility::CameraParams crop =
      Utility::calculateCameraParams(volumeHeight, observerTransform, fov, fov);

  rkcommon::math::vec3f camPosOsp{crop.mPos.x, crop.mPos.y, crop.mPos.z};
  rkcommon::math::vec3f camUpOsp{crop.mUp.x, crop.mUp.y, crop.mUp.z};
  rkcommon::math::vec3f camViewOsp{crop.mForward.x, crop.mForward.y, crop.mForward.z};

  rkcommon::math::vec2f camImageStartOsp{crop.mLeft, crop.mBottom};
  rkcommon::math::vec2f camImageEndOsp{crop.mRight, crop.mTop};

  // TODO Get ViSTA stereo mode
  // osprayCamera.setParam("stereoMode", OSP_STEREO_NONE or OSP_STEREO_SIDE_BY_SIDE);
  ospray::cpp::Camera osprayCamera("perspective");
  osprayCamera.setParam("aspect", 1);
  osprayCamera.setParam("position", camPosOsp);
  osprayCamera.setParam("up", camUpOsp);
  osprayCamera.setParam("direction", camViewOsp);
  osprayCamera.setParam("fovy", cs::utils::convert::toDegrees(fov));
  osprayCamera.setParam("imageStart", camImageStartOsp);
  osprayCamera.setParam("imageEnd", camImageEndOsp);
  osprayCamera.commit();

  // TODO store modelview and projection for left and right eye
  Camera camera;
  camera.mOsprayCamera = osprayCamera;
  camera.mModelView    = crop.mModelView;
  camera.mProjection   = crop.mProjection;
  camera.mInside       = crop.mInside;
  return camera;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::renderFrame(ospray::cpp::World const& world, ospray::cpp::Camera const& camera,
    std::optional<std::vector<float>>&& maxDepth, Parameters const& parameters,
    bool resetAccumulation) {

  // cs::gui::detail::ScopedTimer timer("myVolumeRendering");
  ospray::cpp::Renderer renderer("volume_depth");

  if (maxDepth.has_value()) {
    ospray::cpp::Texture maxDepthTex("texture2d");
    maxDepthTex.setParam("format", OSP_TEXTURE_R32F);
    maxDepthTex.setParam(
        "data", ospray::cpp::SharedData(maxDepth.value().data(), OSP_FLOAT,
                    rkcommon::math::vec2i(parameters.mResolution, parameters.mResolution)));
    maxDepthTex.commit();

    renderer.setParam("map_maxDepth", maxDepthTex);
  }

  const void* filtersPtr = parameters.mScalarFilters.data();
  renderer.setParam("scalarFilters", OSP_VOID_PTR, &filtersPtr);
  renderer.setParam("aoSamples", 4);
  renderer.setParam("shadows", false);
  renderer.setParam("volumeSamplingRate", parameters.mSamplingRate);
  renderer.setParam("depthMode", (int)parameters.mWorld.mDepthMode);
  renderer.setParam("numScalarFilters", (int)parameters.mScalarFilters.size());
  renderer.commit();

  if (resetAccumulation) {
    int channels = OSP_FB_COLOR | OSP_FB_DEPTH | OSP_FB_ACCUM;

    mCache.mFrameBuffer = ospray::cpp::FrameBuffer(
        parameters.mResolution, parameters.mResolution, OSP_FB_RGBA32F, channels);
    mCache.mFrameBuffer.clear();
    mCache.mFrameBuffer.commit();
    mFrameBufferAccumulationPasses = 0;
  }

  {
    std::scoped_lock lock(mRenderFutureMutex);
    mRenderFuture = mCache.mFrameBuffer.renderFrame(renderer, camera, world);
    mFrameBufferAccumulationPasses++;
  }
  mRenderFuture->wait();
  {
    std::scoped_lock lock(mRenderFutureMutex);
    mRenderFuture.reset();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Cache::Cache()
    : mPathlines("curve")
    , mPathlinesModel(mPathlines)
    , mVolume("structuredRegular")
    , mVolumeModel(mVolume)
    , mCore("sphere")
    , mCoreModel(mCore)
    , mCoreTexture("volume")
    , mCoreMaterial("scivis", "obj")
    , mCoreTransferFunction("piecewiseLinear")
    , mInstance(mGroup)
    , mAmbientLight("ambient")
    , mSunLight("distant") {
  mCore.setParam("sphere.position", ospray::cpp::Data(rkcommon::math::vec3f(0, 0, 0)));
  mCore.commit();

  std::vector<rkcommon::math::vec3f> color = {
      rkcommon::math::vec3f(0.f, 0.f, 0.f), rkcommon::math::vec3f(1.f, 1.f, 1.f)};
  std::vector<float> opacity = {1.f, 1.f};

  mCoreTransferFunction.setParam("color", ospray::cpp::Data(color));
  mCoreTransferFunction.setParam("opacity", ospray::cpp::Data(opacity));
  mCoreTransferFunction.commit();

  // Commits have to be done after volume is set
  mCoreTexture.setParam("transferFunction", mCoreTransferFunction);

  mCoreModel.setParam("material", mCoreMaterial);
  mCoreModel.commit();

  mGroup.setParam("volume", ospray::cpp::Data(mVolumeModel));

  std::vector<ospray::cpp::Light> lights{mAmbientLight, mSunLight};
  mWorld.setParam("instance", ospray::cpp::Data(mInstance));
  mWorld.setParam("light", ospray::cpp::Data(lights));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::RenderedImage::RenderedImage(ospray::cpp::FrameBuffer frame, Camera const& camera,
    float volumeHeight, Parameters const& parameters, glm::mat4 const& cameraTransform)
    : Renderer::RenderedImage(true, parameters.mResolution, cameraTransform, camera.mModelView,
          camera.mProjection, camera.mInside)
    , mFrame(std::move(frame)) {
  mColorData = (float*)mFrame.map(OSP_FB_COLOR);
  mDepthData = (float*)mFrame.map(OSP_FB_DEPTH);

  std::thread colorDenoising;
  std::thread depthDenoising;
  if (parameters.mDenoiseColor) {
    colorDenoising = std::thread([this, parameters]() {
      OSPRayUtility::denoiseImage(mColorData, 4, parameters.mResolution);
    });
  }
  if (parameters.mWorld.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
    depthDenoising = std::thread([this, parameters]() {
      std::vector<float> depthGrayscale = OSPRayUtility::depthToGrayscale(mDepthData, mResolution);
      OSPRayUtility::denoiseImage(depthGrayscale.data(), 3, parameters.mResolution);
      OSPRayUtility::grayscaleToDepth(depthGrayscale, mDepthData);
    });
  }

  if (parameters.mDenoiseColor) {
    colorDenoising.join();
  }
  if (parameters.mWorld.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
    depthDenoising.join();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::RenderedImage::~RenderedImage() {
  if (mFrame.handle() != nullptr) {
    mFrame.unmap(mColorData);
    mFrame.unmap(mDepthData);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::RenderedImage::RenderedImage(RenderedImage&& other)
    : Renderer::RenderedImage(std::move(other)) {
  std::swap(other.mFrame, mFrame);
  std::swap(other.mColorData, mColorData);
  std::swap(other.mDepthData, mDepthData);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float* OSPRayRenderer::RenderedImage::getColorData() {
  return mColorData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float* OSPRayRenderer::RenderedImage::getDepthData() {
  return mDepthData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
