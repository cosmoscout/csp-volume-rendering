////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../logger.hpp"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vtk-8.2/vtkPointData.h>
#include <vtk-8.2/vtkStructuredPoints.h>

#include <rkcommon/math/vec.h>

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>
#include <ospray/ospray_cpp/ext/rkcommon.h>

#include <exception>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(std::shared_ptr<DataManager> dataManager,
    Renderer::VolumeStructure structure, Renderer::VolumeShape shape)
    : Renderer(dataManager, structure, shape) {
  OSPRayUtility::initOSPRay();
  mTransferFunction = std::async(
      std::launch::deferred, [] { return OSPRayUtility::createOSPRayTransferFunction(); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::~OSPRayRenderer() {
  ospShutdown();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::setTransferFunction(std::vector<glm::vec4> colors) {
  mTransferFunction = std::async(std::launch::deferred, [this, colors]() {
    vtkSmartPointer<vtkDataSet> volumeData = mDataManager->getData();
    volumeData->GetPointData()->SetActiveScalars("B_Mag");
    return OSPRayUtility::createOSPRayTransferFunction(
        (float)volumeData->GetPointData()->GetScalars()->GetRange()[0],
        (float)volumeData->GetPointData()->GetScalars()->GetRange()[1], colors);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> OSPRayRenderer::getFrame(int resolution,
    glm::mat4 cameraTransform, float samplingRate, Renderer::DepthMode depthMode, bool denoiseColor,
    bool denoiseDepth, bool shading) {
  mRendering = true;
  return std::async(std::launch::async, [this, resolution, cameraTransform, samplingRate, depthMode,
                                            denoiseColor, denoiseDepth, shading]() {
    bool recreateVolume = mDataManager->isDirty();

    std::vector<float> bounds(6);

    switch (mStructure) {
    case VolumeStructure::eUnstructured: {
      vtkSmartPointer<vtkUnstructuredGrid> volumeData =
          vtkUnstructuredGrid::SafeDownCast(mDataManager->getData());
      if (recreateVolume) {
        mVolume = OSPRayUtility::createOSPRayVolumeUnstructured(volumeData, "T");
      }
      volumeData->GetPoints()->ComputeBounds();
      for (int i = 0; i < 6; i++) {
        bounds[i] = (float)volumeData->GetPoints()->GetBounds()[i];
      }
      break;
    }
    case VolumeStructure::eStructured: {
      vtkSmartPointer<vtkStructuredPoints> volumeData =
          vtkStructuredPoints::SafeDownCast(mDataManager->getData());
      if (recreateVolume) {
        mVolume = OSPRayUtility::createOSPRayVolumeStructured(volumeData, "B_Mag");
      }
      volumeData->ComputeBounds();
      for (int i = 0; i < 6; i++) {
        bounds[i] = (float)volumeData->GetBounds()[i];
      }
      break;
    }
    }

    glm::mat4 cameraTransformScaled = cameraTransform;
    float     x                     = (bounds[1] - bounds[0]) / 2;
    float     y                     = (bounds[3] - bounds[2]) / 2;
    float     z                     = (bounds[5] - bounds[4]) / 2;
    float     height;

    switch (mShape) {
    case Renderer::VolumeShape::eCubic: {
      float diagonal           = sqrtf(x * x + y * y + z * z);
      cameraTransformScaled[3] = cameraTransform[3] * glm::vec4(diagonal, diagonal, diagonal, 1.f);
      height                   = diagonal;
      break;
    }
    case Renderer::VolumeShape::eSpherical: {
      cameraTransformScaled[3] = cameraTransform[3] * glm::vec4(x, y, z, 1);
      height                   = x;
      break;
    }
    }

    OSPRayUtility::Camera camera = OSPRayUtility::createOSPRayCamera(height, cameraTransformScaled);

    ospray::cpp::VolumetricModel volumetricModel(*mVolume);
    volumetricModel.setParam("transferFunction", mTransferFunction.get());
    volumetricModel.setParam("gradientShadingScale", shading ? 1.f : 0.f);
    volumetricModel.commit();

    ospray::cpp::Group group;
    group.setParam("volume", ospray::cpp::Data(volumetricModel));
    if (depthMode == Renderer::DepthMode::eIsosurface) {
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

    ospray::cpp::World world = OSPRayUtility::createOSPRayWorld(group);

    ospray::cpp::Renderer renderer("volume_depth");
    renderer.setParam("aoSamples", 0);
    renderer.setParam("shadows", false);
    renderer.setParam("volumeSamplingRate", samplingRate);
    renderer.setParam("depthMode", (int)depthMode);
    renderer.commit();

    int channels = OSP_FB_COLOR;
    if (depthMode != Renderer::DepthMode::eNone) {
      channels |= OSP_FB_DEPTH;
    }

    ospray::cpp::FrameBuffer framebuffer(resolution, resolution, OSP_FB_RGBA32F, channels);
    framebuffer.clear();

    framebuffer.commit();

    ospray::cpp::Future renderFuture =
        framebuffer.renderFrame(renderer, camera.osprayCamera, world);
    renderFuture.wait();
    logger().trace("Rendered for {}s", renderFuture.duration());

    std::vector<float>   colorData((float*)framebuffer.map(OSP_FB_COLOR),
        (float*)framebuffer.map(OSP_FB_COLOR) + 4 * resolution * resolution);
    std::vector<float>   depthData(resolution * resolution, INFINITY);
    std::vector<uint8_t> frameData(4 * resolution * resolution);

    if (depthMode != Renderer::DepthMode::eNone) {
      float* depthFrame = (float*)framebuffer.map(OSP_FB_DEPTH);
      depthData         = std::vector(depthFrame, depthFrame + depthData.size());
    }

    depthData = OSPRayUtility::normalizeDepthBuffer(
        resolution, depthData, (bounds[1] - bounds[0]) / 2, camera);

    auto timer = std::chrono::high_resolution_clock::now();

    std::future<std::vector<float>> futureColor;
    std::future<std::vector<float>> futureDepth;
    if (denoiseColor) {
      futureColor = std::async(std::launch::deferred, [resolution, &colorData]() {
        auto               timer = std::chrono::high_resolution_clock::now();
        std::vector<float> data  = OSPRayUtility::denoiseImage(colorData, 4, resolution);
        logger().trace("Denoised color for {}s",
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
        return data;
      });
    }
    if (depthMode != Renderer::DepthMode::eNone && denoiseDepth) {
      futureDepth = std::async(std::launch::deferred, [resolution, &depthData]() {
        auto               timer          = std::chrono::high_resolution_clock::now();
        std::vector<float> depthGrayscale = OSPRayUtility::depthToGrayscale(depthData);
        std::vector<float> denoised = OSPRayUtility::denoiseImage(depthGrayscale, 3, resolution);
        std::vector<float> data     = OSPRayUtility::grayscaleToDepth(denoised);
        logger().trace("Denoised depth for {}s",
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
        return data;
      });
    }

    if (denoiseColor) {
      colorData = futureColor.get();
    }
    if (depthMode != Renderer::DepthMode::eNone && denoiseDepth) {
      depthData = futureDepth.get();
    }
    logger().trace("Denoising: {}s",
        (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
    timer = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < frameData.size(); i++) {
      frameData[i] = (uint8_t)(colorData[i] * 255);
    }
    frameData.insert(frameData.end(), (uint8_t*)depthData.data(),
        (uint8_t*)depthData.data() + 4 * resolution * resolution);

    std::tuple<std::vector<uint8_t>, glm::mat4> result(frameData, camera.transformationMatrix);
    mRendering = false;
    return result;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
