////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_RENDERER_HPP
#define CSP_VOLUME_RENDERING_RENDERER_HPP

#include "DataManager.hpp"

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
  };

  Renderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape);

  void setResolution(int resolution);
  void setSamplingRate(float samplingRate);
  void setDepthMode(Renderer::DepthMode depthMode);

  void setDenoiseColor(bool denoiseColor);
  void setDenoiseDepth(bool denoiseDepth);

  void setTransferFunction(std::vector<glm::vec4> transferFunction);
  void setShading(bool shading);
  void setAmbientLight(float strength);
  void setSunDirection(glm::vec3 sunDirection);

  virtual std::future<Renderer::RenderedImage> getFrame(glm::mat4 cameraTransform) = 0;

 protected:
  struct Parameters {
    int                 mResolution;
    float               mSamplingRate;
    Renderer::DepthMode mDepthMode;

    bool mDenoiseColor;
    bool mDenoiseDepth;

    std::vector<glm::vec4> mTransferFunction;
    bool                   mShading;
    float                  mAmbientLight;
    glm::vec3              mSunDirection;
  };

  std::shared_ptr<DataManager> mDataManager;

  VolumeStructure mStructure;
  VolumeShape     mShape;

  Parameters mParameters;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
