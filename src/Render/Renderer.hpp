////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_RENDERER_HPP
#define CSP_VOLUME_RENDERING_RENDERER_HPP

#include "../Data/DataManager.hpp"

#include "../../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <string>

namespace csp::volumerendering {

class Renderer {
 public:
  enum class VolumeStructure { eInvalid = -1, eStructured, eUnstructured };
  enum class VolumeShape { eInvalid = -1, eCubic, eSpherical };
  enum class DepthMode {
    eNone           = 0,
    eIsosurface     = 1,
    eFirstHit       = 2,
    eLastHit        = 4,
    eThreshold      = 8,
    eMultiThreshold = 16
  };

  struct RenderedImage {
    std::vector<uint8_t> mColorData;
    std::vector<float>   mDepthData;
    glm::mat4            mMVP;
    bool                 mValid;
  };

  Renderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape);
  virtual ~Renderer() = default;

  void setResolution(int resolution);
  void setSamplingRate(float samplingRate);
  void setDepthMode(Renderer::DepthMode depthMode);

  void setDenoiseColor(bool denoiseColor);
  void setDenoiseDepth(bool denoiseDepth);

  void setTransferFunction(std::vector<glm::vec4> transferFunction);
  void setDensityScale(float densityScale);

  void setShading(bool shading);
  void setAmbientLight(float strength);
  void setSunDirection(glm::vec3 sunDirection);
  void setSunStrength(float strength);

  std::future<Renderer::RenderedImage> getFrame(glm::mat4 cameraTransform);
  virtual float                                getProgress()                         = 0;
  virtual void                                 preloadData(DataManager::State state) = 0;
  virtual void                                 cancelRendering()                     = 0;

 protected:
  struct Parameters {
    int                 mResolution;
    float               mSamplingRate;
    Renderer::DepthMode mDepthMode;

    bool mDenoiseColor;
    bool mDenoiseDepth;

    std::vector<glm::vec4> mTransferFunction;
    float                  mDensityScale;

    bool      mShading;
    float     mAmbientLight;
    glm::vec3 mSunDirection;
    float     mSunStrength;
  };

  std::shared_ptr<DataManager> mDataManager;

  const VolumeStructure mStructure;
  const VolumeShape     mShape;

  virtual RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) = 0;

 private:
  std::mutex mParameterMutex;
  Parameters mParameters;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
