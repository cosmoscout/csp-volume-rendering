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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(std::string path)
    : Renderer(path) {
  OSPRayUtility::initOSPRay();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(std::string path, int timestep)
    : Renderer(path, timestep) {
  OSPRayUtility::initOSPRay();
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
    glm::mat4 cameraRotation, int resolution, float samplingRate, Renderer::DepthMode depthMode,
    bool denoise) {
  return std::async(std::launch::async, [this, cameraRotation, resolution, samplingRate, depthMode,
                                            denoise]() {
    if (!mVolume.has_value()) {
      vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
      mVolume = OSPRayUtility::createOSPRayVolume(volumeData, "T");
    }

    getData()->GetPoints()->ComputeBounds();
    float fov = 22;
    float modelHeight =
        (getData()->GetPoints()->GetBounds()[3] - getData()->GetPoints()->GetBounds()[2]) / 2;
    float cameraDistance = modelHeight / sin(fov / 180 * (float)M_PI / 2);

    ospray::cpp::Camera camera = OSPRayUtility::createOSPRayCamera(
        resolution, resolution, fov, cameraDistance, cameraRotation);

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

    ospray::cpp::Instance instance(group);
    instance.commit();

    ospray::cpp::Light light("ambient");
    light.commit();

    ospray::cpp::World world;
    world.setParam("instance", ospray::cpp::Data(instance));
    world.setParam("light", ospray::cpp::Data(light));
    world.commit();

    ospray::cpp::Renderer renderer("scivis");
    renderer.setParam("aoSamples", 0);
    renderer.setParam("volumeSamplingRate", samplingRate);
    renderer.setParam("maxPathLength", 1);
    renderer.commit();

    ospcommon::math::vec2i imgSize;
    imgSize.x    = resolution;
    imgSize.y    = resolution;
    int channels = OSP_FB_COLOR;
    if (depthMode == Renderer::DepthMode::eIsosurface) {
      channels |= OSP_FB_DEPTH;
    }

    ospray::cpp::FrameBuffer framebuffer(imgSize, OSP_FB_RGBA32F, channels);
    framebuffer.clear();

    if (denoise) {
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
    std::vector<uint8_t> frameData(4 * resolution * resolution);
    std::vector<float>   depthData(resolution * resolution);

    for (int i = 0; i < frameData.size(); i++) {
      frameData[i] = colorFrame[i] * 255;
    }

    float normalizedCameraDistance = 1 / sin(fov / 180 * (float)M_PI / 2);
    if (depthMode == Renderer::DepthMode::eIsosurface) {
      float* depthFrame = (float*)framebuffer.map(OSP_FB_DEPTH);

      float maxAbsDepth = 0;
      for (int i = 0; i < depthData.size(); i++) {
        float val = depthFrame[i] - cameraDistance;
        if (val != INFINITY && fabsf(val) > maxAbsDepth) {
          maxAbsDepth = fabsf(val);
        }
      }
      float normalizedMaxAbsDepth = maxAbsDepth / cameraDistance * normalizedCameraDistance;

      for (int i = 0; i < depthData.size(); i++) {
        float val = depthFrame[i];
        if (val == INFINITY) {
          depthData[i] = 0;
        } else {
          int   x               = i % resolution;
          float ndcX            = ((float)x / resolution - 0.5f) * 2;
          int   y               = i / resolution;
          float ndcY            = ((float)y / resolution - 0.5f) * 2;
          float normalizedDepth = val / cameraDistance * normalizedCameraDistance;
          float normalizedZ = sqrtf(normalizedDepth * normalizedDepth - ndcX * ndcX + ndcY * ndcY);
          depthData[i]      = (normalizedZ - normalizedCameraDistance) / normalizedMaxAbsDepth;
        }
      }
    }
    frameData.insert(frameData.end(), (uint8_t*)depthData.data(),
        (uint8_t*)depthData.data() + 4 * resolution * resolution);

    glm::mat4 projection = glm::perspective(fov / 180 * (float)M_PI, 1.f,
        normalizedCameraDistance - 1.f, normalizedCameraDistance + 1.f);
    glm::mat4 modelView =
        glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, -normalizedCameraDistance));

    std::tuple<std::vector<uint8_t>, glm::mat4> result(frameData, projection * modelView);
    return result;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
