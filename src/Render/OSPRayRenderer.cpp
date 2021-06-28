////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../logger.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vtkCellData.h>
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

void OSPRayRenderer::preloadData(DataManager::State state) {
  if (mCache.mVolumes.find(state) == mCache.mVolumes.end()) {
    mCache.mVolumes[state][mDataManager->getMaxLod(state)] =
        std::async(std::launch::async, [this, state]() { return loadVolume(state); });
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

Renderer::RenderedImage OSPRayRenderer::getFrameImpl(
    glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) {
  mRenderingCancelled = false;
  RenderedImage renderedImage;
  try {
    const Volume& volume = getVolume(dataState);
    Cache::State  state{cameraTransform, parameters, dataState, volume.mLod};
    if (mCache.mState == state && mFrameBufferAccumulationPasses >= parameters.mMaxRenderPasses) {
      renderedImage.mValid = false;
      return renderedImage;
    }
    if (!(parameters.mWorld == mCache.mState.mParameters.mWorld &&
            dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
      updateWorld(volume, parameters, dataState);
    }
    if (!(cameraTransform == mCache.mState.mCameraTransform &&
            dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
      mCache.mCamera = getCamera(volume.mHeight, cameraTransform);
    }
    ospray::cpp::FrameBuffer frame = renderFrame(
        mCache.mWorld, mCache.mCamera.mOsprayCamera, parameters, !(mCache.mState == state));
    renderedImage        = extractImageData(frame, mCache.mCamera, volume.mHeight, parameters);
    renderedImage.mValid = !mRenderingCancelled;

    mCache.mState = state;
  } catch (const std::exception&) { renderedImage.mValid = false; }
  return renderedImage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const OSPRayRenderer::Volume& OSPRayRenderer::getVolume(DataManager::State state) {
  int         maxLod     = mDataManager->getMaxLod(state);
  auto const& stateCache = mCache.mVolumes.find(state);
  if (stateCache == mCache.mVolumes.end()) {
    mCache.mVolumes[state][maxLod] = std::async(std::launch::deferred, [this, state, maxLod]() {
      Volume v = loadVolume(state);
      v.mLod   = maxLod;
      return v;
    });
  } else {
    auto const& cachedVolume = stateCache->second.find(maxLod);
    if (cachedVolume == stateCache->second.end()) {
      mCache.mVolumes[state][maxLod] = std::async(std::launch::async, [this, state, maxLod]() {
        Volume v = loadVolume(state);
        v.mLod   = maxLod;
        return v;
      });
    }
    for (auto const& cacheEntry : stateCache->second) {
      if (cacheEntry.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        maxLod = cacheEntry.first;
      }
    }
  }
  return mCache.mVolumes[state][maxLod].get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Volume OSPRayRenderer::loadVolume(DataManager::State state) {
  Volume                      volume;
  vtkSmartPointer<vtkDataSet> volumeData = mDataManager->getData(state);
  switch (mStructure) {
  case VolumeStructure::eUnstructured:
    volume.mOsprayData = OSPRayUtility::createOSPRayVolume(
        vtkUnstructuredGrid::SafeDownCast(volumeData), state.mScalar.mType);
    break;
  case VolumeStructure::eStructured: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData =
        OSPRayUtility::createOSPRayVolume(vtkStructuredPoints::SafeDownCast(volumeData), scalars);
    break;
  }
  case VolumeStructure::eStructuredSpherical: {
    std::vector<Scalar> scalars = mDataManager->pScalars.get();
    scalars.insert(scalars.begin(), state.mScalar);
    volume.mOsprayData =
        OSPRayUtility::createOSPRayVolume(vtkStructuredGrid::SafeDownCast(volumeData), scalars);
    break;
  }
  case VolumeStructure::eInvalid:
    throw std::runtime_error("Trying to load volume with unknown/invalid structure!");
  }
  volume.mHeight       = getHeight(volumeData);
  volume.mScalarBounds = mDataManager->getScalarRange(state.mScalar);
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
    const Volume& volume, const Parameters& parameters) {
  if (parameters.mWorld.mVolume.mTransferFunction.size() > 0) {
    return OSPRayUtility::createOSPRayTransferFunction((float)volume.mScalarBounds[0],
        (float)volume.mScalarBounds[1], parameters.mWorld.mVolume.mTransferFunction);
  } else {
    return OSPRayUtility::createOSPRayTransferFunction();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::updateWorld(
    const Volume& volume, const Parameters& parameters, const DataManager::State& dataState) {
  bool updateGroup = false;

  ospray::cpp::TransferFunction transferFunction = getTransferFunction(volume, parameters);

  if (!(dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod)) {
    mCache.mVolumeModel.setParam("volume", volume.mOsprayData);
    updateGroup = true;
  }

  if (!(dataState == mCache.mState.mDataState && volume.mLod == mCache.mState.mVolumeLod &&
          parameters.mWorld.mVolume == mCache.mState.mParameters.mWorld.mVolume &&
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
      std::vector<rkcommon::math::vec2f> texCoords = mDataManager->getPathlines().getTexCoords(
          parameters.mWorld.mPathlines.mActiveScalar, "point_ParticleAge");

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
        // TODO Get max age from somewhere else
        float age = ((float)y / 256.f) * 109.f;
        if (age < parameters.mWorld.mPathlinesTexture.mLength) {
          pixels[4 * (y * 256 + x) + 0] = x < 128 ? x * 2 : 255;
          pixels[4 * (y * 256 + x) + 1] = 255 - std::abs((x - 128) * 2);
          pixels[4 * (y * 256 + x) + 2] = x > 128 ? 255 - (x - 128) * 2 : 255;
          pixels[4 * (y * 256 + x) + 3] = 255;
        } else {
          pixels[4 * (y * 256 + x) + 0] = 0;
          pixels[4 * (y * 256 + x) + 1] = 0;
          pixels[4 * (y * 256 + x) + 2] = 0;
          pixels[4 * (y * 256 + x) + 3] = 0;
        }
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
          parameters.mWorld.mPathlines == mCache.mState.mParameters.mWorld.mPathlines)) {
    std::vector<ospray::cpp::GeometricModel> geometries;
    geometries.push_back(mCache.mCoreModel);
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
    if (parameters.mWorld.mPathlines.mEnable && pathlinesPresent) {
      geometries.push_back(mCache.mPathlinesModel);
    }
    mCache.mGroup.setParam("geometry", ospray::cpp::Data(geometries));

    updateGroup = true;
  }

  if (updateGroup) {
    mCache.mGroup.commit();
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

  // Define vertical field of view for ospray camera
  float fov    = 90;
  float fovRad = fov / 180 * (float)M_PI;

  // Create camera transform looking along negative z
  glm::mat4 cameraTransform(1);
  cameraTransform[2][2] = -1;

  // Move camera to observer position relative to planet
  cameraTransform = observerTransform * cameraTransform;

  // Get base vectors of rotated coordinate system
  glm::vec3 camRight(cameraTransform[0]);
  camRight = glm::normalize(camRight);
  glm::vec3 camUp(cameraTransform[1]);
  camUp = glm::normalize(camUp);
  glm::vec3 camDir(cameraTransform[2]);
  camDir = glm::normalize(camDir);
  glm::vec3 camPos(cameraTransform[3]);

  // Get position of camera in rotated coordinate system
  float camXLen = glm::dot(camPos, camRight);
  float camYLen = glm::dot(camPos, camUp);
  float camZLen = glm::dot(camPos, camDir);

  // Get angle between camera position and forward vector
  float cameraAngleX = atan(camXLen / camZLen);
  float cameraAngleY = atan(camYLen / camZLen);

  // Get angle between ray towards center of volume and ray at edge of volume
  float modelAngleX = asin(volumeHeight / sqrt(camXLen * camXLen + camZLen * camZLen));
  float modelAngleY = asin(volumeHeight / sqrt(camYLen * camYLen + camZLen * camZLen));

  // Get angle between rays at edges of volume and forward vector
  float leftAngle, rightAngle, downAngle, upAngle;
  if (!isnan(modelAngleX) && !isnan(modelAngleY)) {
    leftAngle  = cameraAngleX - modelAngleX;
    rightAngle = cameraAngleX + modelAngleX;
    downAngle  = cameraAngleY - modelAngleY;
    upAngle    = cameraAngleY + modelAngleY;
  } else {
    // If the camera is inside the volume the model angles will be NaN,
    // so the angles are set to the edges of the field of view
    leftAngle  = -fovRad / 2;
    rightAngle = fovRad / 2;
    downAngle  = -fovRad / 2;
    upAngle    = fovRad / 2;
  }

  // Get edges of volume in image space coordinates
  float leftPercent  = 0.5f + tan(leftAngle) / (2 * tan(fovRad / 2));
  float rightPercent = 0.5f + tan(rightAngle) / (2 * tan(fovRad / 2));
  float downPercent  = 0.5f + tan(downAngle) / (2 * tan(fovRad / 2));
  float upPercent    = 0.5f + tan(upAngle) / (2 * tan(fovRad / 2));

  rkcommon::math::vec3f camPosOsp{camPos.x, camPos.y, camPos.z};
  rkcommon::math::vec3f camUpOsp{camUp.x, camUp.y, camUp.z};
  rkcommon::math::vec3f camViewOsp{camDir.x, camDir.y, camDir.z};

  rkcommon::math::vec2f camImageStartOsp{leftPercent, downPercent};
  rkcommon::math::vec2f camImageEndOsp{rightPercent, upPercent};

  ospray::cpp::Camera osprayCamera("perspective");
  osprayCamera.setParam("aspect", 1);
  osprayCamera.setParam("position", camPosOsp);
  osprayCamera.setParam("up", camUpOsp);
  osprayCamera.setParam("direction", camViewOsp);
  osprayCamera.setParam("fovy", fov);
  osprayCamera.setParam("imageStart", camImageStartOsp);
  osprayCamera.setParam("imageEnd", camImageEndOsp);
  osprayCamera.commit();

  glm::mat4 view =
      glm::translate(glm::mat4(1.f), -glm::vec3(camXLen, camYLen, -camZLen) / volumeHeight);

  float nearClip = -camZLen / volumeHeight - 1;
  float farClip  = -camZLen / volumeHeight + 1;
  if (nearClip < 0) {
    nearClip = 0.00001f;
  }
  float     leftClip  = tan(leftAngle) * nearClip;
  float     rightClip = tan(rightAngle) * nearClip;
  float     downClip  = tan(downAngle) * nearClip;
  float     upClip    = tan(upAngle) * nearClip;
  glm::mat4 projection(0);
  projection[0][0] = 2 * nearClip / (rightClip - leftClip);
  projection[1][1] = 2 * nearClip / (upClip - downClip);
  projection[2][0] = (rightClip + leftClip) / (rightClip - leftClip);
  projection[2][1] = (upClip + downClip) / (upClip - downClip);
  projection[2][2] = -(farClip + nearClip) / (farClip - nearClip);
  projection[2][3] = -1;
  projection[3][2] = -2 * farClip * nearClip / (farClip - nearClip);

  Camera camera;
  camera.mOsprayCamera         = osprayCamera;
  camera.mPositionRotated      = glm::vec3(camXLen, camYLen, -camZLen) / volumeHeight;
  camera.mTransformationMatrix = projection * view;
  return camera;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::FrameBuffer OSPRayRenderer::renderFrame(ospray::cpp::World& world,
    ospray::cpp::Camera& camera, Parameters const& parameters, bool resetAccumulation) {
  ospray::cpp::Renderer renderer("volume_depth");
  renderer.setParam("aoSamples", 0);
  renderer.setParam("shadows", false);
  renderer.setParam("volumeSamplingRate", parameters.mSamplingRate);
  renderer.setParam("depthMode", (int)parameters.mWorld.mDepthMode);
  const void* filtersPtr = parameters.mScalarFilters.data();
  renderer.setParam("scalarFilters", OSP_VOID_PTR, &filtersPtr);
  renderer.setParam("numScalarFilters", (int)parameters.mScalarFilters.size());
  renderer.commit();

  if (resetAccumulation) {
    int channels = OSP_FB_COLOR | OSP_FB_ACCUM;
    if (parameters.mWorld.mDepthMode != DepthMode::eNone) {
      channels |= OSP_FB_DEPTH;
    }

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
  return mCache.mFrameBuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::RenderedImage OSPRayRenderer::extractImageData(ospray::cpp::FrameBuffer& frame,
    const Camera& camera, float volumeHeight, const Parameters& parameters) {
  float*             colorFrame = (float*)frame.map(OSP_FB_COLOR);
  std::vector<float> colorData(
      colorFrame, colorFrame + 4 * parameters.mResolution * parameters.mResolution);
  frame.unmap(colorFrame);
  std::vector<float> depthData(parameters.mResolution * parameters.mResolution, INFINITY);

  if (parameters.mWorld.mDepthMode != DepthMode::eNone) {
    float* depthFrame = (float*)frame.map(OSP_FB_DEPTH);
    depthData         = std::vector<float>(depthFrame, depthFrame + depthData.size());
    frame.unmap(depthFrame);
  }

  depthData = normalizeDepthData(depthData, camera, volumeHeight, parameters);

  std::future<std::vector<float>> futureColor;
  std::future<std::vector<float>> futureDepth;
  if (parameters.mDenoiseColor) {
    futureColor = std::async(std::launch::deferred, [parameters, &colorData]() {
      std::vector<float> data = OSPRayUtility::denoiseImage(colorData, 4, parameters.mResolution);
      return data;
    });
  }
  if (parameters.mWorld.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
    futureDepth = std::async(std::launch::deferred, [parameters, &depthData]() {
      std::vector<float> depthGrayscale = OSPRayUtility::depthToGrayscale(depthData);
      std::vector<float> denoised =
          OSPRayUtility::denoiseImage(depthGrayscale, 3, parameters.mResolution);
      std::vector<float> data = OSPRayUtility::grayscaleToDepth(denoised);
      return data;
    });
  }

  if (parameters.mDenoiseColor) {
    colorData = futureColor.get();
  }
  if (parameters.mWorld.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
    depthData = futureDepth.get();
  }

  std::vector<uint8_t> colorDataInt(4 * parameters.mResolution * parameters.mResolution);
  for (size_t i = 0; i < colorDataInt.size(); i++) {
    colorDataInt[i] = (uint8_t)(colorData[i] * 255);
  }

  Renderer::RenderedImage renderedImage;
  renderedImage.mColorData = colorDataInt;
  renderedImage.mDepthData = depthData;
  renderedImage.mMVP       = camera.mTransformationMatrix;
  renderedImage.mValid     = true;
  return renderedImage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> OSPRayRenderer::normalizeDepthData(std::vector<float> data, const Camera& camera,
    float volumeHeight, const Parameters& parameters) {
  std::vector<float> depthData(parameters.mResolution * parameters.mResolution);

  for (size_t i = 0; i < depthData.size(); i++) {
    float val = data[i];
    if (val == INFINITY) {
      depthData[i] = -camera.mTransformationMatrix[3][2] / camera.mTransformationMatrix[2][2];
    } else {
      val /= volumeHeight;
      int       x          = i % parameters.mResolution;
      float     ndcX       = ((float)x / parameters.mResolution - 0.5f) * 2;
      int       y          = (int)(i / parameters.mResolution);
      float     ndcY       = ((float)y / parameters.mResolution - 0.5f) * 2;
      glm::vec4 posPixClip = glm::vec4(ndcX, ndcY, 0, 1);
      glm::vec4 posPix     = glm::inverse(camera.mTransformationMatrix) * posPixClip;
      glm::vec3 posPixNorm = glm::vec3(posPix) * (1 / posPix.w);
      glm::vec3 pos =
          val * glm::normalize(posPixNorm - camera.mPositionRotated) + camera.mPositionRotated;
      glm::vec4 posClip     = camera.mTransformationMatrix * glm::vec4(pos, 1);
      glm::vec3 posClipNorm = glm::vec3(posClip) * (1 / posClip.w);
      depthData[i]          = posClipNorm.z;
    }
  }

  return depthData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Cache::Cache()
    : mPathlines("curve")
    , mPathlinesModel(mPathlines)
    , mVolume("structuredRegular")
    , mVolumeModel(mVolume)
    , mCore("sphere")
    , mCoreModel(mCore)
    , mClip("plane")
    , mClipModel(mClip)
    , mInstance(mGroup)
    , mAmbientLight("ambient")
    , mSunLight("distant") {
  mCore.setParam("sphere.position", ospray::cpp::Data(rkcommon::math::vec3f(0, 0, 0)));
  mCore.setParam("radius", 3470.f);
  mCore.commit();

  mCoreModel.setParam("color", ospray::cpp::Data(rkcommon::math::vec4f(.25f, .25f, .25f, 1.f)));
  mCoreModel.commit();

  mClip.setParam("plane.coefficients", ospray::cpp::Data(rkcommon::math::vec4f(0, 0, -1, 0)));
  mClip.commit();

  mClipModel.commit();

  mGroup.setParam("volume", ospray::cpp::Data(mVolumeModel));
  mGroup.setParam("clippingGeometry", ospray::cpp::Data(mClipModel));

  std::vector<ospray::cpp::Light> lights{mAmbientLight, mSunLight};
  mWorld.setParam("instance", ospray::cpp::Data(mInstance));
  mWorld.setParam("light", ospray::cpp::Data(lights));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
