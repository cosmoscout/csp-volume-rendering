////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_PLUGIN_HPP
#define CSP_VOLUME_RENDERING_PLUGIN_HPP

#include "Render/Renderer.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <map>
#include <string>

namespace csp::volumerendering {

class Billboard;

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Volume {
      std::string mPath;
    };

    cs::utils::DefaultProperty<bool>  mPredictiveRendering{true};
    cs::utils::DefaultProperty<int>   mResolution{256};
    cs::utils::DefaultProperty<float> mSamplingRate{0.005};
    std::map<std::string, Volume>     mVolumes;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  struct Frame {
    int                    mResolution;
    float                  mSamplingRate;
    glm::dquat             mCameraRotation;
    std::vector<glm::vec4> mTransferFunction;

    std::vector<uint8_t> mFrameData;

    bool operator==(const Frame& other);
  };

  void requestFrame(glm::dquat cameraRotation);
  void tryReuseFrame(glm::dquat cameraRotation);
  void displayFrame(Frame& frame);

  Settings mPluginSettings;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;

  int              mLastFrameInterval;
  std::vector<int> mFrameIntervals;
  int              mFrameIntervalsLength = 1;
  int              mFrameIntervalsIndex  = 0;

  glm::dquat              mLastCameraRotation;
  std::vector<glm::dquat> mCameraRotations;
  int                     mCameraRotationsLength = 15;
  int                     mCameraRotationsIndex  = 0;

  std::unique_ptr<Renderer>        mRenderer;
  std::shared_ptr<Billboard>       mBillboard;
  std::shared_ptr<VistaOpenGLNode> mVolumeNode;

  bool                              mGettingFrame;
  std::future<std::vector<uint8_t>> mFutureFrameData;
  Frame                             mNextFrame;
  Frame                             mRenderingFrame;
  Frame                             mDisplayedFrame;
  std::vector<Frame>                mRenderedFrames;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_PLUGIN_HPP
