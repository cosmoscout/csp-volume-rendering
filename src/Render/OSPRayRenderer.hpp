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

class OSPRayInitializer {
 public:
  OSPRayInitializer() {
    OSPRayUtility::initOSPRay();
  }

  ~OSPRayInitializer() {
    ospShutdown();
  }
};

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
  OSPRayInitializer mInitializer;

  struct Volume {
    ospray::cpp::Volume   mOsprayData;
    float                 mHeight;
    std::array<double, 2> mScalarBounds;
    int                   mLod;
  };

  struct Camera {
    ospray::cpp::Camera mOsprayCamera;
    glm::vec3           mPositionRotated;
    glm::mat4           mTransformationMatrix;
  };

  int mFrameBufferAccumulationPasses;

  struct Cache {
    std::map<DataManager::State, std::map<int, std::shared_future<Volume>>> mVolumes;

    Camera                       mCamera;
    ospray::cpp::FrameBuffer     mFrameBuffer;
    ospray::cpp::World           mWorld;
    ospray::cpp::Geometry        mPathlines;
    ospray::cpp::GeometricModel  mPathlinesModel;
    ospray::cpp::Volume          mVolume;
    ospray::cpp::VolumetricModel mVolumeModel;
    ospray::cpp::Geometry        mCore;
    ospray::cpp::GeometricModel  mCoreModel;
    ospray::cpp::Texture         mCoreTexture;
    ospray::cpp::Material        mCoreMaterial;
    ospray::cpp::Geometry        mClip;
    ospray::cpp::GeometricModel  mClipModel;
    ospray::cpp::Group           mGroup;
    ospray::cpp::Instance        mInstance;
    ospray::cpp::Light           mAmbientLight;
    ospray::cpp::Light           mSunLight;

    struct State {
      glm::mat4          mCameraTransform;
      Parameters         mParameters;
      DataManager::State mDataState;
      int                mVolumeLod;

      bool operator==(const State& other) const {
        return mCameraTransform == other.mCameraTransform && mParameters == other.mParameters &&
               mDataState == other.mDataState && mVolumeLod == other.mVolumeLod;
      }
    } mState;

    Cache();
  } mCache;

  std::optional<ospray::cpp::Future> mRenderFuture;
  bool                               mRenderingCancelled;
  std::mutex                         mRenderFutureMutex;

  RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) override;

  const Volume&                 getVolume(DataManager::State state, std::optional<int> maxLod);
  Volume                        loadVolume(DataManager::State state, int lod);
  float                         getHeight(vtkSmartPointer<vtkDataSet> data);
  ospray::cpp::TransferFunction getTransferFunction(
      const Volume& volume, const Parameters& parameters);
  void updateWorld(
      const Volume& volume, const Parameters& parameters, const DataManager::State& dataState);
  OSPRayRenderer::Camera   getCamera(float volumeHeight, glm::mat4 observerTransform);
  ospray::cpp::FrameBuffer renderFrame(ospray::cpp::World& world, ospray::cpp::Camera& camera,
      Parameters const& parameters, bool resetAccumulation);
  Renderer::RenderedImage  extractImageData(ospray::cpp::FrameBuffer& frame, const Camera& camera,
       float volumeHeight, const Parameters& parameters);
  std::vector<float>       normalizeDepthData(std::vector<float> data, const Camera& camera,
            float volumeHeight, const Parameters& parameters);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
