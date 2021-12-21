////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DEPTHEXTRACTOR_HPP
#define CSP_VOLUME_RENDERING_DEPTHEXTRACTOR_HPP

#include "../Enums.hpp"
#include "DisplayNode.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaTexture.h>

#include "../../../../src/cs-core/Settings.hpp"
#include "../../../../src/cs-core/SolarSystem.hpp"
#include "../../../../src/cs-core/TimeControl.hpp"

namespace csp::volumerendering {

class DepthExtractor : public IVistaOpenGLDraw {
 public:
  DepthExtractor(std::shared_ptr<DisplayNode> displayNode,
      std::shared_ptr<cs::core::SolarSystem>  solarSystem,
      std::shared_ptr<cs::core::TimeControl>  timeControl);
  ~DepthExtractor();

  void setEnabled(bool enabled);

  /// Returns last frame's depth buffer of the OpenGL render pipeline.
  std::vector<float> getDepthBuffer(int resolution);

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  std::shared_ptr<VistaOpenGLNode>       mVistaNode;
  std::shared_ptr<DisplayNode>           mDisplayNode;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  bool mEnabled = true;

  int mResolution = 512;

  GLuint       mDepthComputeShader = 0;
  GLuint       mPBO                = 0;
  VistaTexture mOut;
  /// Store one buffer per viewport
  std::unordered_map<VistaViewport*, VistaTexture> mDepthBufferData;

  std::optional<GLsync> mPBOFence;

  struct {
    GLint mBottomCorner = 0;
    GLint mTopCorner    = 0;
    GLint mNearDistance = 0;
    GLint mFarDistance  = 0;
  } mUniforms;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DEPTHEXTRACTOR_HPP
