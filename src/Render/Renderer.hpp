////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_RENDERER_HPP
#define CSP_VOLUME_RENDERING_RENDERER_HPP

#include "DataManager.hpp"

#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <string>

namespace csp::volumerendering {

class Renderer {
 public:
  enum class DepthMode { eNone, eIsosurface, eFirstHit, eLastHit };

  Renderer();
  Renderer(std::string path);
  Renderer(std::string path, int timestep);

  void setData(std::string path, int timestep);
  void setTime(int timestep);
  void setFile(std::string path);

  void setFov(float fov);
  void setResolution(int resolution);

  virtual void setTransferFunction(std::vector<glm::vec4> colors) = 0;

  virtual std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> getFrame(
      glm::mat4 cameraRotation, float samplingRate, DepthMode depthMode, bool denoise) = 0;

 protected:
  vtkSmartPointer<vtkUnstructuredGrid> getData();

  cs::utils::DefaultProperty<bool> mRendering{false};

  cs::utils::DefaultProperty<float> mFov{22};
  cs::utils::DefaultProperty<int>   mResolution{256};

 private:
  void updateData();

  DataManager mDataManager;
  std::string mCurrentFile;
  int         mCurrentTimestep;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
