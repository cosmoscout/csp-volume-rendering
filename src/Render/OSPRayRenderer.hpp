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
  void  preloadData(
       DataManager::State const& state, std::optional<DataManager::State> const& coreState) override;
  void cancelRendering() override;

  void clearCache() override;

 private:
  OSPRayInitializer mInitializer;

  struct Volume {
    ospray::cpp::Volume mOsprayData;
    float               mHeight;
    int                 mLod;
  };

  struct Camera {
    ospray::cpp::Camera mOsprayCamera;
    glm::vec3           mPositionRotated;
    glm::mat4           mModelView;
    glm::mat4           mProjection;
    bool                mInside;
  };

  class RenderedImage : public Renderer::RenderedImage {
   public:
    RenderedImage(Camera const& camera, float volumeHeight, Parameters const& parameters,
        glm::mat4 const& cameraTransform);
    RenderedImage(ospray::cpp::FrameBuffer frame, Camera const& camera, float volumeHeight,
        Parameters const& parameters, glm::mat4 const& cameraTransform);
    ~RenderedImage() override;

    RenderedImage(RenderedImage const& other) = delete;
    RenderedImage& operator=(RenderedImage const& other) = delete;

    RenderedImage(RenderedImage&& other);
    RenderedImage& operator=(RenderedImage&& other) = delete;

    float* getColorData(int layer = 0) override;
    float* getDepthData(int layer = 0) override;

    void addLayer(ospray::cpp::FrameBuffer frame);

   private:
    std::vector<float*> mColorData;
    std::vector<float*> mDepthData;

    std::vector<ospray::cpp::FrameBuffer> mFrame;

    bool mDenoiseColor;
    bool mDenoiseDepth;
  };

  int mFrameBufferAccumulationPasses = 0;

  struct Cache {
    std::map<DataManager::State, std::map<int, std::shared_future<Volume>>> mVolumes;
    std::shared_future<Volume>                                              mCurrentVolume;
    DataManager::State                                                      mCurrentVolumeState;

    Camera                                mCamera;
    std::vector<ospray::cpp::FrameBuffer> mFrameBuffer;
    ospray::cpp::World                    mWorld;
    ospray::cpp::Geometry                 mPathlines;
    ospray::cpp::GeometricModel           mPathlinesModel;
    ospray::cpp::Volume                   mVolume;
    ospray::cpp::VolumetricModel          mVolumeModel;
    ospray::cpp::Geometry                 mCore;
    ospray::cpp::GeometricModel           mCoreModel;
    ospray::cpp::Texture                  mCoreTexture;
    ospray::cpp::Material                 mCoreMaterial;
    ospray::cpp::TransferFunction         mCoreTransferFunction;
    ospray::cpp::Group                    mGroup;
    ospray::cpp::Instance                 mInstance;
    ospray::cpp::Light                    mAmbientLight;
    ospray::cpp::Light                    mSunLight;

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

  std::vector<std::optional<ospray::cpp::Future>> mRenderFuture;
  bool                                            mRenderingCancelled = false;
  std::mutex                                      mRenderFutureMutex;

  std::unique_ptr<Renderer::RenderedImage> getFrameImpl(glm::mat4 const& cameraTransform,
      std::optional<std::vector<float>>&& maxDepth, Parameters parameters,
      DataManager::State const& dataState) override;

  const Volume& getVolume(DataManager::State const& state, std::optional<int> const& maxLod);
  Volume        loadVolume(DataManager::State const& state, int lod);
  float         getHeight(vtkSmartPointer<vtkDataSet> data);
  ospray::cpp::TransferFunction getTransferFunction(
      DataManager::State const& state, Parameters const& parameters);
  void updateWorld(
      Volume const& volume, Parameters const& parameters, DataManager::State const& dataState);
  OSPRayRenderer::Camera         getCamera(float volumeHeight, glm::mat4 observerTransform);
  std::unique_ptr<RenderedImage> renderFrame(std::unique_ptr<RenderedImage> image,
      ospray::cpp::World const& world, OSPRayRenderer::Camera const& camera,
      std::optional<std::vector<float>>&& maxDepth, Parameters const& parameters,
      bool resetAccumulation);
  void renderLayer(ospray::cpp::World const& world, OSPRayRenderer::Camera const& camera,
      std::optional<std::vector<float>>& maxDepth, Parameters const& parameters,
      bool resetAccumulation, int layer);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_OSPRAYRENDERER_HPP
