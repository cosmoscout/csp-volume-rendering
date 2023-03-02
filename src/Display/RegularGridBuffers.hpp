////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_REGULAR_GRID_HPP
#define CSP_VOLUME_RENDERING_REGULAR_GRID_HPP

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering {

class RegularGridBuffers {
 public:
  RegularGridBuffers(uint32_t resolution);

  void Draw();

 private:
  uint32_t mResolution;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaBufferObject      mIBO;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_BILLBOARD_HPP
