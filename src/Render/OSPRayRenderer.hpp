////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
#define CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP

#include "OSPRayUtility.hpp"
#include "Renderer.hpp"

#include <ospray/ospray_cpp.h>
// Boost defines a function called likely so this macro from platform.h has to be undeffed
#undef likely

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

class OSPRayRenderer : public Renderer {
 public:
  OSPRayRenderer(std::shared_ptr<DataManager> dataManager, Renderer::VolumeStructure structure,
      Renderer::VolumeShape shape);
  ~OSPRayRenderer();

  OSPRayRenderer(const OSPRayRenderer& other) = delete;
  OSPRayRenderer& operator=(const OSPRayRenderer& other) = delete;

  std::future<Renderer::RenderedImage> getFrame(glm::mat4 cameraTransform) override;

 private:
  struct Volume {
    ospray::cpp::Volume  mOsprayData;
    float                mHeight;
    std::array<float, 2> mScalarBounds;
  };

  struct Camera {
    ospray::cpp::Camera mOsprayCamera;
    glm::vec3           mPositionRotated;
    glm::mat4           mTransformationMatrix;
  };

  std::optional<ospray::cpp::Volume> mVolume;

  Volume               getVolume(Renderer::VolumeShape shape);
  float                getHeight(vtkSmartPointer<vtkDataSet> data, Renderer::VolumeShape shape);
  std::array<float, 2> getScalarBounds(vtkSmartPointer<vtkDataSet> data);
  ospray::cpp::TransferFunction getTransferFunction(Volume& volume, Parameters& parameters);
  ospray::cpp::World            getWorld(Volume& volume, Parameters& parameters);
  OSPRayRenderer::Camera        getCamera(float volumeHeight, glm::mat4 observerTransform);
  ospray::cpp::FrameBuffer      renderFrame(
           ospray::cpp::World world, ospray::cpp::Camera camera, Parameters& parameters);
  Renderer::RenderedImage extractImageData(
      ospray::cpp::FrameBuffer frame, Camera camera, float volumeHeight, Parameters& parameters);
  std::vector<float> normalizeDepthData(
      std::vector<float> data, Camera camera, float volumeHeight, Parameters& parameters);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
