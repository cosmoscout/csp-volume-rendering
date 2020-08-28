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

    cs::utils::DefaultProperty<bool>                mRequestImages{true};
    cs::utils::DefaultProperty<bool>                mPredictiveRendering{false};
    cs::utils::DefaultProperty<bool>                mReuseImages{false};
    cs::utils::DefaultProperty<bool>                mDepthData{true};
    cs::utils::DefaultProperty<bool>                mDrawDepth{false};
    cs::utils::DefaultProperty<Renderer::DepthMode> mDepthMode{Renderer::DepthMode::eNone};
    cs::utils::DefaultProperty<bool>                mDenoiseColor{true};
    cs::utils::DefaultProperty<bool>                mDenoiseDepth{true};
    cs::utils::DefaultProperty<int>                 mResolution{256};
    cs::utils::DefaultProperty<float>               mSamplingRate{0.005};
    cs::utils::DefaultProperty<float>               mFov{22};
    std::map<std::string, Volume>                   mVolumes;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  struct Frame {
    int                    mResolution;
    float                  mSamplingRate;
    float                  mFov;
    glm::mat4              mCameraTransform;
    std::vector<glm::vec4> mTransferFunction;
    Renderer::DepthMode    mDepthMode;
    bool                   mDenoiseColor;
    bool                   mDenoiseDepth;

    std::vector<uint8_t> mFrameData;
    glm::mat4            mModelViewProjection;

    bool operator==(const Frame& other);
  };

  void requestFrame(glm::mat4 cameraTransform);
  void tryReuseFrame(glm::mat4 cameraTransform);
  void displayFrame(Frame& frame);

  Settings mPluginSettings;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;

  int              mLastFrameInterval;
  std::vector<int> mFrameIntervals;
  int              mFrameIntervalsLength = 1;
  int              mFrameIntervalsIndex  = 0;

  glm::mat4              mLastCameraTransform;
  std::vector<glm::mat4> mCameraTransforms;
  int                    mCameraTransformsLength = 15;
  int                    mCameraTransformsIndex  = 0;

  std::unique_ptr<Renderer>        mRenderer;
  std::shared_ptr<Billboard>       mBillboard;
  std::shared_ptr<VistaOpenGLNode> mVolumeNode;

  std::future<std::tuple<std::vector<uint8_t>, glm::mat4>> mFutureFrameData;

  bool               mGettingFrame;
  Frame              mNextFrame;
  Frame              mRenderingFrame;
  Frame              mDisplayedFrame;
  std::vector<Frame> mRenderedFrames;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_PLUGIN_HPP
