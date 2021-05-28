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

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

/// Renderer imeplementation that uses Intel OSPRay for rendering volumes.
class OSPRayRenderer : public Renderer {
 public:
  OSPRayRenderer(
      std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape);
  ~OSPRayRenderer();

  OSPRayRenderer(const OSPRayRenderer& other) = delete;
  OSPRayRenderer& operator=(const OSPRayRenderer& other) = delete;

  float getProgress() override;
  void  preloadData(DataManager::State state) override;
  void  cancelRendering() override;

 private:
  struct Volume {
    ospray::cpp::Volume   mOsprayData;
    float                 mHeight;
    std::array<double, 2> mScalarBounds;
  };

  struct Camera {
    ospray::cpp::Camera mOsprayCamera;
    glm::vec3           mPositionRotated;
    glm::mat4           mTransformationMatrix;
  };

  std::map<DataManager::State, std::map<int, std::shared_future<Volume>>> mCachedVolumes;

  ospray::cpp::World                         mCachedWorld;
  DataManager::State                         mCachedState;
  Parameters                                 mCachedParameters;
  std::optional<ospray::cpp::GeometricModel> mPathlinesModel;

  std::optional<ospray::cpp::Future> mRenderFuture;
  bool                               mRenderingCancelled;
  std::mutex                         mRenderFutureMutex;

  RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) override;

  const Volume&                 getVolume(DataManager::State state);
  Volume                        loadVolume(DataManager::State state);
  float                         getHeight(vtkSmartPointer<vtkDataSet> data);
  ospray::cpp::TransferFunction getTransferFunction(
      const Volume& volume, const Parameters& parameters);
  ospray::cpp::World       getWorld(const Volume& volume, const Parameters& parameters);
  OSPRayRenderer::Camera   getCamera(float volumeHeight, glm::mat4 observerTransform);
  ospray::cpp::FrameBuffer renderFrame(
      ospray::cpp::World& world, ospray::cpp::Camera& camera, const Parameters& parameters);
  Renderer::RenderedImage extractImageData(ospray::cpp::FrameBuffer& frame, const Camera& camera,
      float volumeHeight, const Parameters& parameters);
  std::vector<float>      normalizeDepthData(std::vector<float> data, const Camera& camera,
           float volumeHeight, const Parameters& parameters);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
