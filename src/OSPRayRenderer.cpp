////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayRenderer.hpp"

#include "OSPRayUtility.hpp"
#include "logger.hpp"

#include <vtk-8.2/vtkPointData.h>

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>

#include <exception>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

OSPRayRenderer::OSPRayRenderer()
    : Renderer() {
  OSPRayUtility::initOSPRay();
  mTransferFunction = OSPRayUtility::createOSPRayTransferFunction();
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
  vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
  volumeData->GetPointData()->SetActiveScalars("T");
  mTransferFunction = OSPRayUtility::createOSPRayTransferFunction(
      volumeData->GetPointData()->GetScalars()->GetRange()[0],
      volumeData->GetPointData()->GetScalars()->GetRange()[1], colors);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::vector<uint8_t>> OSPRayRenderer::getFrame(
    glm::mat4 cameraRotation, int resolution, float samplingRate) {
  return std::async(std::launch::async, [this, cameraRotation, resolution, samplingRate]() {
    if (!mVolume.has_value()) {
      vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
      mVolume = OSPRayUtility::createOSPRayVolume(volumeData, "T");
    }
    getData()->GetPoints()->ComputeBounds();
    ospray::cpp::Camera camera = OSPRayUtility::createOSPRayCamera(resolution, resolution, 22,
        (getData()->GetPoints()->GetBounds()[3] - getData()->GetPoints()->GetBounds()[2]) / 2,
        cameraRotation);

    ospray::cpp::VolumetricModel volumetricModel(*mVolume);
    volumetricModel.setParam("transferFunction", mTransferFunction);
    volumetricModel.commit();

    std::vector<float>    isovalues = {0.9f};
    ospray::cpp::Geometry isosurface("isosurface");
    isosurface.setParam("isovalue", ospray::cpp::Data(isovalues));
    isosurface.setParam("volume", volumetricModel.handle());
    isosurface.commit();

    ospray::cpp::GeometricModel isoModel(isosurface);
    isoModel.commit();

    ospray::cpp::Group group;
    group.setParam("volume", ospray::cpp::Data(volumetricModel));
    group.setParam("geometry", ospray::cpp::Data(isoModel));
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
    int channels = OSP_FB_COLOR | OSP_FB_DEPTH;

    ospray::cpp::FrameBuffer framebuffer(imgSize, OSP_FB_RGBA8, channels);
    framebuffer.clear();

    ospray::cpp::ImageOperation d("denoiser");
    framebuffer.setParam("imageOperation", ospray::cpp::Data(d));

    ospray::cpp::Future renderFuture = framebuffer.renderFrame(renderer, camera, world);
    renderFuture.wait();
    logger().trace("Rendered for {}s", renderFuture.duration());

    void*                colorFrame = framebuffer.map(OSP_FB_COLOR);
    void*                depthFrame = framebuffer.map(OSP_FB_DEPTH);
    std::vector<uint8_t> frameData(
        (uint8_t*)colorFrame, (uint8_t*)colorFrame + 4 * resolution * resolution);
    frameData.insert(frameData.end(), (uint8_t*)depthFrame, (uint8_t*)depthFrame + 4 * resolution * resolution);
    return frameData;
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
