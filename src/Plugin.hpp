////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_PLUGIN_HPP
#define CSP_VOLUME_RENDERING_PLUGIN_HPP

#include "Render/DataManager.hpp"
#include "Render/Renderer.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <map>
#include <optional>
#include <string>

namespace csp::volumerendering {

class Billboard;
class PointsForwardWarped;

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    enum class DisplayMode { eMesh, ePoints };

    // Data settings
    cs::utils::Property<std::string>                 mVolumeDataPath;
    cs::utils::Property<std::string>                 mVolumeDataPattern;
    cs::utils::Property<DataManager::VolumeFileType> mVolumeDataType;
    cs::utils::Property<Renderer::VolumeStructure>   mVolumeStructure;
    cs::utils::Property<Renderer::VolumeShape>       mVolumeShape;

    // Rendering settings
    cs::utils::DefaultProperty<bool>                mRequestImages{true};
    cs::utils::DefaultProperty<int>                 mResolution{256};
    cs::utils::DefaultProperty<float>               mSamplingRate{0.05f};
    cs::utils::DefaultProperty<float>               mSunStrength{1.f};
    cs::utils::DefaultProperty<float>               mDensityScale{1.f};
    cs::utils::DefaultProperty<bool>                mDenoiseColor{true};
    cs::utils::DefaultProperty<bool>                mDenoiseDepth{true};
    cs::utils::DefaultProperty<Renderer::DepthMode> mDepthMode{Renderer::DepthMode::eNone};

    // Display settings
    cs::utils::DefaultProperty<bool>        mPredictiveRendering{false};
    cs::utils::DefaultProperty<bool>        mReuseImages{false};
    cs::utils::DefaultProperty<bool>        mDepthData{true};
    cs::utils::DefaultProperty<bool>        mDrawDepth{false};
    cs::utils::DefaultProperty<DisplayMode> mDisplayMode{DisplayMode::eMesh};

    // Transform settings
    cs::utils::Property<std::string>       mAnchor;
    cs::utils::DefaultProperty<glm::dvec3> mPosition{glm::dvec3(0, 0, 0)};
    cs::utils::DefaultProperty<double>     mScale{1.};
    cs::utils::DefaultProperty<glm::dquat> mRotation{glm::dquat(0, 0, 0, 1)};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  struct Frame {
    glm::mat4            mCameraTransform;
    glm::mat4            mModelViewProjection;
    int                  mResolution;
    std::vector<uint8_t> mColorImage;
    std::vector<float>   mDepthImage;

    bool operator==(const Frame& other);
  };

  enum class RenderState { eWaitForData, eIdle, ePaused, eRenderingImage };

  void initUI();
  void onLoad();
  void registerUICallbacks();
  void connectSettings();

  bool tryRequestFrame();

  glm::mat4 getCurrentCameraTransform();
  glm::mat4 predictCameraTransform(glm::mat4 currentTransform);

  void receiveFrame();
  void displayFrame(Frame& frame);
  void displayFrame(Frame& frame, Settings::DisplayMode displayMode);
  void tryReuseFrame(glm::mat4 cameraTransform);

  void exportTransferFunction(std::string const& path, std::string const& jsonTransferFunction);
  void importTransferFunction(std::string const& path);
  void updateAvailableTransferFunctions();

  Settings mPluginSettings;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;

  int              mLastFrameInterval;
  std::vector<int> mFrameIntervals;
  int              mFrameIntervalsLength = 3;
  int              mFrameIntervalsIndex  = 0;

  glm::mat4              mLastCameraTransform;
  std::vector<glm::mat4> mCameraTransforms;
  int                    mCameraTransformsLength = 15;
  int                    mCameraTransformsIndex  = 0;

  std::unique_ptr<Renderer>            mRenderer;
  std::shared_ptr<DataManager>         mDataManager;
  std::shared_ptr<PointsForwardWarped> mPoints;
  std::shared_ptr<Billboard>           mBillboard;
  std::shared_ptr<VistaOpenGLNode>     mPointsNode;
  std::shared_ptr<VistaOpenGLNode>     mBillboardNode;

  std::future<Renderer::RenderedImage> mFutureFrameData;

  RenderState mRenderState = RenderState::eWaitForData;

  bool mFrameInvalid;
  bool mParametersDirty;

  Frame                mNextFrame;
  Frame                mRenderingFrame;
  std::optional<Frame> mDisplayedFrame;
  std::vector<Frame>   mRenderedFrames;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_PLUGIN_HPP
