////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_RENDERER_HPP
#define CSP_VOLUME_RENDERING_RENDERER_HPP

#include "DataManager.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <string>

namespace csp::volumerendering {

class Renderer {
 public:
  enum class DepthMode { eNone, eIsosurface };

  Renderer();
  Renderer(std::string path);
  Renderer(std::string path, int timestep);

  void setData(std::string path, int timestep);
  void setTime(int timestep);
  void setFile(std::string path);

  virtual void setTransferFunction(std::vector<glm::vec4> colors) = 0;

  virtual std::future<std::vector<uint8_t>> getFrame(glm::mat4 cameraRotation, int resolution,
      float samplingRate, DepthMode depthMode, bool denoise) = 0;

 protected:
  vtkSmartPointer<vtkUnstructuredGrid> getData();

 private:
  void updateData();

  DataManager mDataManager;
  std::string mCurrentFile;
  int         mCurrentTimestep;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
