////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "../logger.hpp"
#include "OSPRayUtility.hpp"

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

std::future<std::vector<uint8_t>> OSPRayRenderer::getFrame(
    glm::mat4 cameraRotation, int resolution, float samplingRate, Renderer::DepthMode depthMode) {
  return std::async(
      std::launch::async, [this, cameraRotation, resolution, samplingRate, depthMode]() {
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
          isosurface.setParam("isovalue", 0.9f);
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
        renderer.commit();

        ospcommon::math::vec2i imgSize;
        imgSize.x    = resolution;
        imgSize.y    = resolution;
        int channels = OSP_FB_COLOR;
        if (depthMode == Renderer::DepthMode::eIsosurface) {
          channels |= OSP_FB_DEPTH;
        }

        ospray::cpp::FrameBuffer framebuffer(imgSize, OSP_FB_RGBA8, channels);
        framebuffer.clear();

        ospray::cpp::ImageOperation d("denoiser");
        framebuffer.setParam("imageOperation", ospray::cpp::Data(d));

        ospray::cpp::Future renderFuture = framebuffer.renderFrame(renderer, camera, world);
        renderFuture.wait();
        logger().trace("Rendered for {}s", renderFuture.duration());

        uint8_t*             colorFrame = (uint8_t*)framebuffer.map(OSP_FB_COLOR);
        std::vector<uint8_t> frameData(colorFrame, colorFrame + 4 * resolution * resolution);
        std::vector<float>   depthData(resolution * resolution);

        if (depthMode == Renderer::DepthMode::eIsosurface) {
          float* depthFrame = (float*)framebuffer.map(OSP_FB_DEPTH);

          for (int i = 0; i < depthData.size(); i++) {
            float val = depthFrame[i];
            if (val == INFINITY) {
              depthData[i] = 0;
            } else {
              depthData[i] = val - cameraDistance;
            }
          }
        }
        frameData.insert(frameData.end(), (uint8_t*)depthData.data(),
            (uint8_t*)depthData.data() + 4 * resolution * resolution);
        return frameData;
      });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
