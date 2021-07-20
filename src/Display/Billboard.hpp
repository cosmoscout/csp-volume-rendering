////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_BILLBOARD_HPP
#define CSP_VOLUME_RENDERING_BILLBOARD_HPP

#include "DisplayNode.hpp"

#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering {

/// DisplayNode implementation, that displays rendered volume images on a continuous plane mesh.
/// The mesh has a constant amount of vertices equal to GRID_RESOLUTION^2
/// (defined in Billboard.cpp).
class Billboard : public DisplayNode {
 public:
  Billboard(VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor);

  Billboard(Billboard const& other) = delete;
  Billboard(Billboard&& other)      = default;

  Billboard& operator=(Billboard const& other) = delete;
  Billboard& operator=(Billboard&& other) = default;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;

 private:
  void createBuffers();

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaBufferObject      mIBO;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_BILLBOARD_HPP
