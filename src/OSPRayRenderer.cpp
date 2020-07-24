////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "OSPRayUtility.hpp"
#include "logger.hpp"

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>

#include <exception>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer()
    : Renderer() {
  initOSPRay();
  mTransferFunction = OSPRayUtility::createOSPRayTransferFunction();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(std::string path)
    : Renderer(path) {
  initOSPRay();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer(std::string path, int timestep)
    : Renderer(path, timestep) {
  initOSPRay();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::~OSPRayRenderer() {
  ospShutdown();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::setTransferFunction(std::vector<glm::vec4> colors) {
  mTransferFunction = OSPRayUtility::createOSPRayTransferFunction(0.85f, 1.f, colors);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::vector<uint8_t>> OSPRayRenderer::getFrame(
    glm::mat4 cameraRotation, int resolution, float samplingRate) {
  return std::async(std::launch::async, [this, cameraRotation, resolution, samplingRate]() {
    if (!mVolume.has_value()) {
      vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
      mVolume = OSPRayUtility::createOSPRayVolume(volumeData, "T");
    }
    ospray::cpp::Camera camera =
        OSPRayUtility::createOSPRayCamera(resolution, resolution, 22, 2.2, cameraRotation);

    ospray::cpp::VolumetricModel volumetricModel(*mVolume);
    volumetricModel.setParam("transferFunction", mTransferFunction);
    volumetricModel.commit();

    ospray::cpp::Group group;
    group.setParam("volume", ospray::cpp::Data(volumetricModel));
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

    ospray::cpp::FrameBuffer framebuffer(imgSize, OSP_FB_RGBA8, channels);
    framebuffer.clear();

    ospray::cpp::Future renderFuture = framebuffer.renderFrame(renderer, camera, world);
    renderFuture.wait();
    logger().trace("Rendered for {}s", renderFuture.duration());

    void*                frame = framebuffer.map(OSP_FB_COLOR);
    std::vector<uint8_t> frameData((uint8_t*)frame, (uint8_t*)frame + 4 * resolution * resolution);
    return frameData;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::initOSPRay() {
  int         argc = 1;
  std::string argStr("--osp::vv");
  const char* arg = argStr.c_str();

  OSPError init_error = ospInit(&argc, &arg);
  if (init_error != OSP_NO_ERROR) {
    logger().error("OSPRay Initialization failed: {}", init_error);
    throw std::runtime_error("OSPRay Initialization failed.");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
