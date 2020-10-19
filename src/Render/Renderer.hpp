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

  Renderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape);

  virtual void setTransferFunction(std::vector<glm::vec4> colors) = 0;

  virtual std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> getFrame(int resolution,
      glm::mat4 cameraTransform, float samplingRate, DepthMode depthMode, bool denoiseColor,
      bool denoiseDepth, bool shading) = 0;

 protected:
  cs::utils::DefaultProperty<bool> mRendering{false};

  std::shared_ptr<DataManager> mDataManager;

  VolumeStructure mStructure;
  VolumeShape     mShape;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
