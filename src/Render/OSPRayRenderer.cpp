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

#include <rkcommon/math/vec.h>

#include <ospray/ospray.h>
#include <ospray/ospray_cpp.h>
#include <ospray/ospray_cpp/ext/rkcommon.h>

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

OSPRayRenderer::~OSPRayRenderer() {
  ospShutdown();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void OSPRayRenderer::setTransferFunction(std::vector<glm::vec4> colors) {
  mTransferFunction = std::async(std::launch::deferred, [this, colors]() {
    vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
    volumeData->GetPointData()->SetActiveScalars("T");
    return OSPRayUtility::createOSPRayTransferFunction(
        (float)volumeData->GetPointData()->GetScalars()->GetRange()[0],
        (float)volumeData->GetPointData()->GetScalars()->GetRange()[1], colors);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> OSPRayRenderer::getFrame(
    glm::mat4 cameraTransform, float samplingRate, Renderer::DepthMode depthMode, bool denoiseColor,
    bool denoiseDepth, bool shading) {
  mRendering = true;
  return std::async(std::launch::async,
      [this, cameraTransform, samplingRate, depthMode, denoiseColor, denoiseDepth, shading]() {
        if (!mVolume.has_value()) {
          vtkSmartPointer<vtkUnstructuredGrid> volumeData = getData();
          mVolume = OSPRayUtility::createOSPRayVolume(volumeData, "T");
        }

        getData()->GetPoints()->ComputeBounds();
        std::vector<float> bounds(6);
        for (int i = 0; i < 6; i++) {
          bounds[i] = (float)getData()->GetPoints()->GetBounds()[i];
        }
        glm::mat4 cameraTransformScaled = cameraTransform;
        cameraTransformScaled[3] =
            cameraTransform[3] * glm::vec4((bounds[1] - bounds[0]) / 2, (bounds[3] - bounds[2]) / 2,
                                     (bounds[5] - bounds[4]) / 2, 1);

        OSPRayUtility::Camera camera = OSPRayUtility::createOSPRayCamera(
            (bounds[1] - bounds[0]) / 2, cameraTransformScaled);

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

        ospray::cpp::FrameBuffer framebuffer(
            mResolution.get(), mResolution.get(), OSP_FB_RGBA32F, channels);
        framebuffer.clear();

        framebuffer.commit();

        ospray::cpp::Future renderFuture =
            framebuffer.renderFrame(renderer, camera.osprayCamera, world);
        renderFuture.wait();
        logger().trace("Rendered for {}s", renderFuture.duration());

        std::vector<float>   colorData((float*)framebuffer.map(OSP_FB_COLOR),
            (float*)framebuffer.map(OSP_FB_COLOR) + 4 * mResolution.get() * mResolution.get());
        std::vector<float>   depthData(mResolution.get() * mResolution.get(), INFINITY);
        std::vector<uint8_t> frameData(4 * mResolution.get() * mResolution.get());

        if (depthMode != Renderer::DepthMode::eNone) {
          float* depthFrame = (float*)framebuffer.map(OSP_FB_DEPTH);
          depthData         = std::vector(depthFrame, depthFrame + depthData.size());
        }

        depthData = normalizeDepthBuffer(depthData, (bounds[1] - bounds[0]) / 2, camera);

        auto timer = std::chrono::high_resolution_clock::now();

        std::future<std::vector<float>> futureColor;
        std::future<std::vector<float>> futureDepth;
        if (denoiseColor) {
          futureColor = std::async(std::launch::deferred, [this, &colorData]() {
            auto               timer = std::chrono::high_resolution_clock::now();
            std::vector<float> data  = OSPRayUtility::denoiseImage(colorData, 4, mResolution.get());
            logger().trace("Denoised color for {}s",
                (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
            return data;
          });
        }
        if (depthMode != Renderer::DepthMode::eNone && denoiseDepth) {
          futureDepth = std::async(std::launch::deferred, [this, &depthData]() {
            auto               timer          = std::chrono::high_resolution_clock::now();
            std::vector<float> depthGrayscale = OSPRayUtility::depthToGrayscale(depthData);
            std::vector<float> denoised =
                OSPRayUtility::denoiseImage(depthGrayscale, 3, mResolution.get());
            std::vector<float> data = OSPRayUtility::grayscaleToDepth(denoised);
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
            (uint8_t*)depthData.data() + 4 * mResolution.get() * mResolution.get());

        std::tuple<std::vector<uint8_t>, glm::mat4> result(frameData, camera.transformationMatrix);
        mRendering = false;
        return result;
      });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> OSPRayRenderer::normalizeDepthBuffer(
    std::vector<float> buffer, float modelRadius, OSPRayUtility::Camera camera) {
  std::vector<float> depthData(mResolution.get() * mResolution.get());

  for (int i = 0; i < depthData.size(); i++) {
    float val = buffer[i];
    if (val == INFINITY) {
      depthData[i] = -camera.transformationMatrix[3][2] / camera.transformationMatrix[2][2];
    } else {
      val /= modelRadius;
      int       x          = i % mResolution.get();
      float     ndcX       = ((float)x / mResolution.get() - 0.5f) * 2;
      int       y          = i / mResolution.get();
      float     ndcY       = ((float)y / mResolution.get() - 0.5f) * 2;
      glm::vec4 posPixClip = glm::vec4(ndcX, ndcY, 0, 1);
      glm::vec4 posPix     = glm::inverse(camera.transformationMatrix) * posPixClip;
      glm::vec3 posPixNorm = posPix.xyz * (1 / posPix.w);
      glm::vec3 pos =
          val * glm::normalize(posPixNorm - camera.positionRotated) + camera.positionRotated;
      glm::vec4 posClip     = camera.transformationMatrix * glm::vec4(pos, 1);
      glm::vec3 posClipNorm = posClip.xyz * (1 / posClip.w);
      depthData[i]          = posClipNorm.z;
    }
  }

  return depthData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
