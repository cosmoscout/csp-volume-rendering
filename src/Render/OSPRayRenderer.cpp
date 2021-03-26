////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../logger.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
  OSPRayUtility::initOSPRay();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::~OSPRayRenderer() {
  mCachedVolumes.clear();
  mRenderFuture.reset();

  ospShutdown();
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
  if (mCachedVolumes.find(state) == mCachedVolumes.end()) {
    mCachedVolumes[state] =
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
    const Volume&            volume = getVolume(dataState);
    ospray::cpp::World       world  = getWorld(volume, parameters);
    Camera                   camera = getCamera(volume.mHeight, cameraTransform);
    ospray::cpp::FrameBuffer frame  = renderFrame(world, camera.mOsprayCamera, parameters);
    renderedImage                   = extractImageData(frame, camera, volume.mHeight, parameters);
    renderedImage.mValid            = !mRenderingCancelled;
  } catch (const std::exception&) { renderedImage.mValid = false; }
  return renderedImage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const OSPRayRenderer::Volume& OSPRayRenderer::getVolume(DataManager::State state) {
  auto cachedVolume = mCachedVolumes.find(state);
  if (cachedVolume == mCachedVolumes.end()) {
    mCachedVolumes[state] =
        std::async(std::launch::deferred, [this, state]() { return loadVolume(state); });
  }
  return mCachedVolumes[state].get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::Volume OSPRayRenderer::loadVolume(DataManager::State state) {
  Volume                      volume;
  vtkSmartPointer<vtkDataSet> volumeData = mDataManager->getData(state);
  switch (mStructure) {
  case VolumeStructure::eUnstructured:
    volume.mOsprayData = OSPRayUtility::createOSPRayVolumeUnstructured(
        vtkUnstructuredGrid::SafeDownCast(volumeData));
    break;
  case VolumeStructure::eStructured:
    volume.mOsprayData =
        OSPRayUtility::createOSPRayVolumeStructured(vtkStructuredPoints::SafeDownCast(volumeData));
    break;
  case VolumeStructure::eInvalid:
    throw std::runtime_error("Trying to load volume with unknown/invalid structure!");
  }
  volume.mHeight       = getHeight(volumeData);
  volume.mScalarBounds = getScalarBounds(volumeData);
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

std::array<float, 2> OSPRayRenderer::getScalarBounds(vtkSmartPointer<vtkDataSet> data) {
  std::array<float, 2> bounds;
  bounds[0] = (float)data->GetPointData()->GetScalars()->GetRange()[0];
  bounds[1] = (float)data->GetPointData()->GetScalars()->GetRange()[1];
  return bounds;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::TransferFunction OSPRayRenderer::getTransferFunction(
    const Volume& volume, const Parameters& parameters) {
  if (parameters.mTransferFunction.size() > 0) {
    return OSPRayUtility::createOSPRayTransferFunction(
        volume.mScalarBounds[0], volume.mScalarBounds[1], parameters.mTransferFunction);
  } else {
    return OSPRayUtility::createOSPRayTransferFunction();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::World OSPRayRenderer::getWorld(const Volume& volume, const Parameters& parameters) {
  ospray::cpp::TransferFunction transferFunction = getTransferFunction(volume, parameters);

  ospray::cpp::VolumetricModel volumetricModel(volume.mOsprayData);
  volumetricModel.setParam("transferFunction", transferFunction);
  volumetricModel.setParam("densityScale", parameters.mDensityScale);
  volumetricModel.setParam("gradientShadingScale", parameters.mShading ? 1.f : 0.f);
  volumetricModel.commit();

  ospray::cpp::Group group;
  group.setParam("volume", ospray::cpp::Data(volumetricModel));
  if (parameters.mDepthMode == DepthMode::eIsosurface) {
    ospray::cpp::Geometry isosurface("isosurface");
    isosurface.setParam("isovalue", 0.8f);
    isosurface.setParam("volume", volumetricModel.handle());
    isosurface.commit();

    rkcommon::math::vec4f       color{1.f, 1.f, 1.f, 0.f};
    ospray::cpp::GeometricModel isoModel(isosurface);
    isoModel.setParam("color", ospray::cpp::Data(color));
    isoModel.commit();

    group.setParam("geometry", ospray::cpp::Data(isoModel));
  }
  group.commit();

  ospray::cpp::Instance instance(group);
  instance.commit();

  std::vector<ospray::cpp::Light> lights;

  ospray::cpp::Light light("ambient");
  light.setParam("intensity", parameters.mShading ? parameters.mAmbientLight : 1);
  light.setParam("color", rkcommon::math::vec3f(1, 1, 1));
  light.commit();
  lights.push_back(light);

  if (parameters.mShading) {
    ospray::cpp::Light sun("distant");
    sun.setParam("intensity", parameters.mSunStrength);
    sun.setParam("color", rkcommon::math::vec3f(1, 1, 1));
    sun.setParam("direction", rkcommon::math::vec3f{-parameters.mSunDirection[0],
                                  -parameters.mSunDirection[1], -parameters.mSunDirection[2]});
    sun.setParam("angularDiameter", .53f);
    sun.commit();
    lights.push_back(sun);
  }

  ospray::cpp::World world;
  world.setParam("instance", ospray::cpp::Data(instance));
  world.setParam("light", ospray::cpp::Data(lights));
  world.commit();

  return world;
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

ospray::cpp::FrameBuffer OSPRayRenderer::renderFrame(
    ospray::cpp::World& world, ospray::cpp::Camera& camera, const Parameters& parameters) {
  ospray::cpp::Renderer renderer("volume_depth");
  renderer.setParam("aoSamples", 0);
  renderer.setParam("shadows", false);
  renderer.setParam("volumeSamplingRate", parameters.mSamplingRate);
  renderer.setParam("depthMode", (int)parameters.mDepthMode);
  renderer.commit();

  int channels = OSP_FB_COLOR;
  if (parameters.mDepthMode != DepthMode::eNone) {
    channels |= OSP_FB_DEPTH;
  }

  ospray::cpp::FrameBuffer framebuffer(
      parameters.mResolution, parameters.mResolution, OSP_FB_RGBA32F, channels);
  framebuffer.clear();
  framebuffer.commit();

  {
    std::scoped_lock lock(mRenderFutureMutex);
    mRenderFuture = framebuffer.renderFrame(renderer, camera, world);
  }
  mRenderFuture->wait();
  {
    std::scoped_lock lock(mRenderFutureMutex);
    mRenderFuture.reset();
  }
  return framebuffer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::RenderedImage OSPRayRenderer::extractImageData(ospray::cpp::FrameBuffer& frame,
    const Camera& camera, float volumeHeight, const Parameters& parameters) {
  float*             colorFrame = (float*)frame.map(OSP_FB_COLOR);
  std::vector<float> colorData(
      colorFrame, colorFrame + 4 * parameters.mResolution * parameters.mResolution);
  frame.unmap(colorFrame);
  std::vector<float> depthData(parameters.mResolution * parameters.mResolution, INFINITY);

  if (parameters.mDepthMode != DepthMode::eNone) {
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
  if (parameters.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
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
  if (parameters.mDepthMode != DepthMode::eNone && parameters.mDenoiseDepth) {
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

} // namespace csp::volumerendering
