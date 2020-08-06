////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../logger.hpp"
#include "OSPRayUtility.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vtk-8.2/vtkPointData.h>

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>

#include <exception>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer()
    : Renderer() {
  OSPRayUtility::initOSPRay();
  mTransferFunction = std::async(
      std::launch::deferred, [] { return OSPRayUtility::createOSPRayTransferFunction(); });
  mFov.connectAndTouch([this](float fov) { recalculateCameraDistances(); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::~OSPRayRenderer() {
  ospShutdown();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::setTransferFunction(std::vector<glm::vec4> colors) {
  mTransferFunction = std::async(std::launch::deferred, [this, colors]() {
    vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
    volumeData->GetPointData()->SetActiveScalars("T");
    return OSPRayUtility::createOSPRayTransferFunction(
        volumeData->GetPointData()->GetScalars()->GetRange()[0],
        volumeData->GetPointData()->GetScalars()->GetRange()[1], colors);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> OSPRayRenderer::getFrame(
    glm::mat4 cameraRotation, float samplingRate, Renderer::DepthMode depthMode, bool denoiseColor,
    bool denoiseDepth) {
  mRendering = true;
  return std::async(std::launch::async, [this, cameraRotation, samplingRate, depthMode,
                                            denoiseColor, denoiseDepth]() {
    if (!mVolume.has_value()) {
      vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
      mVolume = OSPRayUtility::createOSPRayVolume(volumeData, "T");
    }

    ospray::cpp::Camera camera = OSPRayUtility::createOSPRayCamera(
        mResolution.get(), mResolution.get(), mFov.get(), mCameraDistance.get(), cameraRotation);

    ospray::cpp::VolumetricModel volumetricModel(*mVolume);
    volumetricModel.setParam("transferFunction", mTransferFunction.get());
    volumetricModel.commit();

    ospray::cpp::Group group;
    group.setParam("volume", ospray::cpp::Data(volumetricModel));
    if (depthMode == Renderer::DepthMode::eIsosurface) {
      ospray::cpp::Geometry isosurface("isosurface");
      isosurface.setParam("isovalue", 0.8f);
      isosurface.setParam("volume", volumetricModel.handle());
      isosurface.commit();

      ospcommon::math::vec4f      color{1.f, 1.f, 1.f, 0.f};
      ospray::cpp::GeometricModel isoModel(isosurface);
      isoModel.setParam("color", ospray::cpp::Data(color));
      isoModel.commit();

      group.setParam("geometry", ospray::cpp::Data(isoModel));
    }
    group.commit();

    ospray::cpp::World world = OSPRayUtility::createOSPRayWorld(group);

    ospray::cpp::Renderer renderer("volume_depth");
    renderer.setParam("aoSamples", 0);
    renderer.setParam("volumeSamplingRate", samplingRate);
    renderer.setParam("maxPathLength", 1);
    renderer.setParam("depthMode", (int)depthMode);
    renderer.commit();

    ospcommon::math::vec2i imgSize;
    imgSize.x    = mResolution.get();
    imgSize.y    = mResolution.get();
    int channels = OSP_FB_COLOR;
    if (depthMode != Renderer::DepthMode::eNone) {
      channels |= OSP_FB_DEPTH;
    }

    ospray::cpp::FrameBuffer framebuffer(imgSize, OSP_FB_RGBA32F, channels);
    framebuffer.clear();

    if (denoiseColor) {
      ospray::cpp::ImageOperation denoise("denoiser");
      framebuffer.setParam("imageOperation", ospray::cpp::Data(denoise));
    } else {
      framebuffer.removeParam("imageOperation");
    }

    framebuffer.commit();

    ospray::cpp::Future renderFuture = framebuffer.renderFrame(renderer, camera, world);
    renderFuture.wait();
    logger().trace("Rendered for {}s", renderFuture.duration());

    float*               colorFrame = (float*)framebuffer.map(OSP_FB_COLOR);
    std::vector<uint8_t> frameData(4 * mResolution.get() * mResolution.get());
    std::vector<float>   depthData(mResolution.get() * mResolution.get(), INFINITY);

    for (int i = 0; i < frameData.size(); i++) {
      frameData[i] = colorFrame[i] * 255;
    }

    if (depthMode != Renderer::DepthMode::eNone) {
      float* depthFrame = (float*)framebuffer.map(OSP_FB_DEPTH);
      depthData         = std::vector(depthFrame, depthFrame + depthData.size());
    }

    glm::mat4 projection = glm::perspective(mFov.get() / 180 * (float)M_PI, 1.f,
        mNormalizedCameraDistance - 1.f, mNormalizedCameraDistance + 1.f);
    glm::mat4 modelView =
        glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -mNormalizedCameraDistance));
    glm::mat4 transform = projection * modelView;

    depthData = normalizeDepthBuffer(depthData, transform);
    if (depthMode != Renderer::DepthMode::eNone && denoiseDepth) {
      auto               timer          = std::chrono::high_resolution_clock::now();
      std::vector<float> depthGrayscale = OSPRayUtility::depthToGrayscale(depthData);
      std::vector<float> denoised = OSPRayUtility::denoiseImage(depthGrayscale, mResolution.get());
      depthData                   = OSPRayUtility::grayscaleToDepth(denoised);
      logger().trace("Denoised depth for {}s",
          (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
    }
    frameData.insert(frameData.end(), (uint8_t*)depthData.data(),
        (uint8_t*)depthData.data() + 4 * mResolution.get() * mResolution.get());

    std::tuple<std::vector<uint8_t>, glm::mat4> result(frameData, transform);
    mRendering = false;
    return result;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::recalculateCameraDistances() {
  mCameraDistance = std::async(std::launch::deferred, [this]() {
    getData()->GetPoints()->ComputeBounds();
    float modelHeight =
        (getData()->GetPoints()->GetBounds()[3] - getData()->GetPoints()->GetBounds()[2]) / 2;
    return modelHeight / sin(mFov.get() / 180 * (float)M_PI / 2);
  });

  mNormalizedCameraDistance = 1 / sin(mFov.get() / 180 * (float)M_PI / 2);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> OSPRayRenderer::normalizeDepthBuffer(std::vector<float> buffer, glm::mat4 mvp) {
  std::vector<float> depthData(mResolution.get() * mResolution.get());

  for (int i = 0; i < depthData.size(); i++) {
    float val = buffer[i];
    if (val == INFINITY) {
      depthData[i] = mvp[2][3] / mvp[3][3];
    } else {
      int   x               = i % mResolution.get();
      float ndcX            = ((float)x / mResolution.get() - 0.5f) * 2;
      int   y               = i / mResolution.get();
      float ndcY            = ((float)y / mResolution.get() - 0.5f) * 2;
      float normalizedDist  = val / mCameraDistance.get() * mNormalizedCameraDistance;
      float normalizedDepth = sqrtf(normalizedDist * normalizedDist - ndcX * ndcX + ndcY * ndcY);
      float normalizedZ     = normalizedDepth - mNormalizedCameraDistance;
      depthData[i] = (normalizedZ * mvp[2][2] + mvp[2][3]) / (normalizedZ * mvp[3][2] + mvp[3][3]);
    }
  }

  return depthData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
