////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_PLUGIN_HPP
#define CSP_VOLUME_RENDERING_PLUGIN_HPP

#include "Data/DataManager.hpp"
#include "Display/DisplayNode.hpp"
#include "Enums.hpp"
#include "Render/Renderer.hpp"

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

#include <glm/gtc/type_ptr.hpp>

#include <future>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>

namespace csp::volumerendering {

template <typename T>
constexpr int SETTINGS_COUNT = 0;
template <>
inline constexpr int SETTINGS_COUNT<bool> = 9;
template <>
inline constexpr int SETTINGS_COUNT<int> = 1;
template <>
inline constexpr int SETTINGS_COUNT<float> = 9;
template <>
inline constexpr int SETTINGS_COUNT<std::string> = 1;

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    // Data settings
    cs::utils::Property<std::string>        mVolumeDataPath;
    cs::utils::Property<std::string>        mVolumeDataPattern;
    cs::utils::Property<VolumeFileType>     mVolumeDataType;
    cs::utils::Property<VolumeStructure>    mVolumeStructure;
    cs::utils::Property<VolumeShape>        mVolumeShape;
    cs::utils::DefaultProperty<std::string> mActiveScalar{""};

    // Rendering settings
    struct Rendering {
      cs::utils::DefaultProperty<int> mMaxPasses{10};
    } mRendering;
    cs::utils::DefaultProperty<bool>        mRequestImages{true};
    cs::utils::DefaultProperty<int>         mResolution{256};
    cs::utils::DefaultProperty<float>       mSamplingRate{0.05f};
    cs::utils::DefaultProperty<float>       mDensityScale{1.f};
    cs::utils::DefaultProperty<bool>        mDenoiseColor{true};
    cs::utils::DefaultProperty<bool>        mDenoiseDepth{true};
    cs::utils::DefaultProperty<DepthMode>   mDepthMode{DepthMode::eNone};
    cs::utils::DefaultProperty<std::string> mTransferFunction{"BlackBody.json"};

    struct Lighting {
      cs::utils::DefaultProperty<bool>  mEnabled{false};
      cs::utils::DefaultProperty<float> mSunStrength{1.f};
      cs::utils::DefaultProperty<float> mAmbientStrength{.5f};
    } mLighting;

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
    cs::utils::DefaultProperty<glm::dquat> mRotation{glm::dquat(1, 0, 0, 0)};

    struct Core {
      cs::utils::DefaultProperty<bool>        mEnabled{true};
      cs::utils::DefaultProperty<std::string> mScalar{""};
      cs::utils::Property<float>              mRadius;
    };
    std::optional<Core> mCore;

    struct Pathlines {
      cs::utils::Property<std::string>        mPath;
      cs::utils::DefaultProperty<bool>        mEnabled{true};
      cs::utils::DefaultProperty<float>       mLineOpacity{1.f};
      cs::utils::DefaultProperty<float>       mLineSize{1.f};
      cs::utils::DefaultProperty<float>       mLength{250.f};
      cs::utils::DefaultProperty<std::string> mActiveScalar{""};
    };
    std::optional<Pathlines> mPathlines;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  template <typename T>
  struct Setting {
   public:
    using Setter = void (Renderer::*)(T);
    using Target = std::reference_wrapper<cs::utils::Property<T>>;

    inline constexpr Setting()
        : mName("")
        , mComment("") {
    }

    inline constexpr Setting(std::string_view name, std::string_view comment)
        : mName(name)
        , mComment(comment) {
    }

    inline constexpr Setting(std::string_view name, std::string_view comment, Target target)
        : mName(name)
        , mComment(comment)
        , mTarget(target) {
    }

    inline constexpr Setting(
        std::string_view name, std::string_view comment, Target target, Setter setter)
        : mName(name)
        , mComment(comment)
        , mTarget(target)
        , mSetter(setter) {
    }

    static constexpr std::array<Setting<T>, SETTINGS_COUNT<T>> getSettings(
        Settings& pluginSettings){};

    std::string_view      mName;
    std::string_view      mComment;
    std::optional<Target> mTarget;
    std::optional<Setter> mSetter;
  };

  struct Frame {
    glm::mat4 mCameraTransform;
    int       mResolution;
  };

  enum class RenderState { eWaitForData, eIdle, ePaused, eRenderingImage };

  void initUI();
  void onLoad();
  void registerUICallbacks();
  void connectSettings();

  std::function<void(bool)>        getCallbackHandler(Setting<bool>::Target const& target);
  std::function<void(double)>      getCallbackHandler(Setting<int>::Target const& target);
  std::function<void(double)>      getCallbackHandler(Setting<float>::Target const& target);
  std::function<void(std::string)> getCallbackHandler(Setting<std::string>::Target const& target);

  template <typename T>
  void registerUICallback(Setting<T> const& setting) {
    if (std::string(setting.mName) == "") {
      return;
    }
    if (setting.mTarget.has_value()) {
      Setting<T>::Target target(setting.mTarget.value());
      mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
          std::string(setting.mComment), getCallbackHandler(target));
    } else {
      if constexpr (std::is_same_v<T, bool>) {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), std::function([](bool value) {}));
      } else if constexpr (std::is_same_v<T, int> || std::is_same_v<T, float>) {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), std::function([](double value) {}));
      } else if constexpr (std::is_same_v<T, std::string>) {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), std::function([](std::string value) {}));
      }
    }
  };

  void setValueInUI(std::string name, bool value);
  void setValueInUI(std::string name, int value);
  void setValueInUI(std::string name, float value);
  void setValueInUI(std::string name, std::string const& value);

  template <typename T>
  void connectSetting(Setting<T> const& setting) {
    if (std::string(setting.mName) == "" || !setting.mTarget.has_value()) {
      return;
    }
    std::string name(setting.mName);
    if (setting.mSetter.has_value()) {
      Setting<T>::Setter setter(setting.mSetter.value());
      setting.mTarget.value().get().connectAndTouch([this, name, setter](T value) {
        setValueInUI("volumeRendering." + name, value);
        invalidateCache();
        // Call the setter on mRenderer with value
        (mRenderer.get()->*setter)(value);
      });
    } else {
      setting.mTarget.value().get().connectAndTouch(
          [this, name](T value) { setValueInUI("volumeRendering." + name, value); });
    }
  };

  bool tryRequestFrame();

  glm::mat4 getCurrentCameraTransform();
  glm::mat4 predictCameraTransform(glm::mat4 currentTransform);

  void showRenderProgress();

  void receiveFrame();
  void displayFrame(std::unique_ptr<Renderer::RenderedImage> frame);
  void displayFrame(std::unique_ptr<Renderer::RenderedImage> frame, DisplayMode displayMode);
  void tryReuseFrame(glm::mat4 cameraTransform);

  void invalidateCache();

  std::vector<ScalarFilter> parseScalarFilters(
      std::string const& json, std::vector<Scalar> const& scalars);

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

  std::unique_ptr<Renderer>                           mRenderer;
  std::shared_ptr<DataManager>                        mDataManager;
  std::map<DisplayMode, std::shared_ptr<DisplayNode>> mDisplayNodes;
  std::shared_ptr<DisplayNode>                        mActiveDisplay;

  RenderState mRenderState = RenderState::eWaitForData;

  bool mFrameInvalid;
  bool mParametersDirty;

  struct ImagePtrHasher {
    size_t operator()(std::unique_ptr<Renderer::RenderedImage> const& imagePtr) const {
      return std::hash<Renderer::RenderedImage>{}(*imagePtr);
    }
  };

  struct ImagePtrEqual {
    bool operator()(std::unique_ptr<Renderer::RenderedImage> const& left,
        std::unique_ptr<Renderer::RenderedImage> const&             right) const {
      return *left == *right;
    }
  };

  Frame                                                 mNextFrame;
  Frame                                                 mRenderingFrame;
  std::future<std::unique_ptr<Renderer::RenderedImage>> mFutureFrameData;
  std::unique_ptr<Renderer::RenderedImage>              mDisplayedImage;
  std::unordered_set<std::unique_ptr<Renderer::RenderedImage>, ImagePtrHasher, ImagePtrEqual>
      mRenderedImages;
};

template <>
constexpr std::array<Plugin::Setting<bool>, SETTINGS_COUNT<bool>>
Plugin::Setting<bool>::getSettings(Settings& pluginSettings);
template <>
constexpr std::array<Plugin::Setting<int>, SETTINGS_COUNT<int>> Plugin::Setting<int>::getSettings(
    Settings& pluginSettings);
template <>
constexpr std::array<Plugin::Setting<float>, SETTINGS_COUNT<float>>
Plugin::Setting<float>::getSettings(Settings& pluginSettings);
template <>
constexpr std::array<Plugin::Setting<std::string>, SETTINGS_COUNT<std::string>>
Plugin::Setting<std::string>::getSettings(Settings& pluginSettings);

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_PLUGIN_HPP
