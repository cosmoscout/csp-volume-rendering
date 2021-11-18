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
#include "Settings.hpp"

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

/// Type transformation to get the corresponding type passed in UI callbacks.
template <typename T>
struct get_ui_type {
  using type = void;
};
template <typename T>
using get_ui_type_t = typename get_ui_type<T>::type;

template <>
struct get_ui_type<bool> {
  using type = bool;
};
template <>
struct get_ui_type<std::string> {
  using type = std::string;
};
template <>
struct get_ui_type<int> {
  using type = double;
};
template <>
struct get_ui_type<float> {
  using type = double;
};

/// Amount of different setting parameters per type.
template <typename T>
constexpr int SETTINGS_COUNT = 0;
template <>
inline constexpr int SETTINGS_COUNT<bool> = 11;
template <>
inline constexpr int SETTINGS_COUNT<int> = 2;
template <>
inline constexpr int SETTINGS_COUNT<float> = 7;
template <>
inline constexpr int SETTINGS_COUNT<std::string> = 2;
template <>
inline constexpr int SETTINGS_COUNT<DisplayMode> = 1;
template <>
inline constexpr int SETTINGS_COUNT<DepthMode> = 1;

class Plugin : public cs::core::PluginBase {
 public:
  void init() override;
  void deInit() override;
  void update() override;

 private:
  /// Template struct for managing a single setting parameter.
  template <typename T>
  struct Setting {
   public:
    using Setter = void (Renderer::*)(T);
    using Target = std::reference_wrapper<cs::utils::Property<T>>;
    using Action = std::conditional_t<std::is_arithmetic_v<T> || std::is_enum_v<T>,
        void (Plugin::*)(T), void (Plugin::*)(T const&)>;

    inline constexpr Setting()
        : mName("")
        , mComment("") {
    }

    inline constexpr Setting(std::string_view name, std::string_view comment,
        std::optional<Target> target = {}, std::optional<Setter> setter = {},
        std::optional<Action> action = {})
        : mName(name)
        , mComment(comment)
        , mTarget(target)
        , mSetter(setter)
        , mAction(action) {
    }

    /// Returns an array of all Setting objects of type T.
    static constexpr std::array<Setting<T>, SETTINGS_COUNT<T>> getSettings(
        Settings& pluginSettings){};

    std::string_view      mName;    /// Name of the parameter's UI callback
    std::string_view      mComment; /// Description of the parameter
    std::optional<Target> mTarget;  /// Property in the Settings struct
    std::optional<Setter> mSetter;  /// Setter method on the Renderer class
    std::optional<Action> mAction;  /// Setter method on the Plugin class
  };

  struct Frame {
    glm::mat4 mCameraTransform;
    int       mResolution;
  };

  enum class RenderState { eWaitForData, eIdle, ePaused, eRenderingImage };

  void initUI();
  void onLoad();
  void registerAllUICallbacks();
  void connectAllSettings();

  template <typename T>
  void connectSettings() {
    auto settings = Setting<T>::getSettings(mPluginSettings);
    for (auto const& setting : settings) {
      connectSetting(setting);
    }
  };

  template <typename T>
  void registerUICallbacks() {
    auto settings = Setting<T>::getSettings(mPluginSettings);
    for (auto const& setting : settings) {
      registerUICallback(setting);
    }
  };

  /// Returns a callback handler function that assigns the callback's parameter
  /// to a target property.
  template <typename T>
  std::function<void(get_ui_type_t<T>)> getCallbackHandler(
      typename Setting<T>::Target const& target) {
    if constexpr (std::is_same_v<get_ui_type_t<T>, T>) {
      return std::function([target](get_ui_type_t<T> value) { target.get() = value; });
    } else if constexpr (std::is_same_v<T, int>) {
      return std::function(
          [target](get_ui_type_t<T> value) { target.get() = (int)std::lround(value); });
    } else if constexpr (std::is_same_v<T, float>) {
      return std::function([target](get_ui_type_t<T> value) { target.get() = (float)value; });
    } else {
      static_assert(false, "Unhandled type for getCallbackHandler");
    }
  };

  /// If the setting specifies a callback name, the callback is registered with the gui manager.
  /// The callback handler sets the setting's target property to the value passed to the callback.
  /// For Enums, one callback is registered for each value between [Enum]::First and [Enum]::Last.
  template <typename T>
  void registerUICallback(Setting<T> const& setting) {
    if (std::string(setting.mName) == "") {
      return;
    }
    if (setting.mTarget.has_value()) {
      Setting<T>::Target target(setting.mTarget.value());
      if constexpr (std::is_enum_v<T>) {
        for (int i = static_cast<int>(T::First); i <= static_cast<int>(T::Last); i++) {
          mGuiManager->getGui()->registerCallback(
              "volumeRendering." + std::string(setting.mName) + std::to_string(i),
              std::string(setting.mComment),
              std::function([target, i]() { target.get() = static_cast<T>(i); }));
        }
      } else {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), getCallbackHandler<T>(target));
      }
    } else {
      if constexpr (std::is_void_v<get_ui_type_t<T>>) {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), std::function([]() {}));
      } else {
        mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
            std::string(setting.mComment), std::function([](get_ui_type_t<T> value) {}));
      }
    }
  };

  /// Sets the value of the UI component with callback 'name' to 'value'.
  /// Assumes checkboxes are used for boolean values, sliders for numerical values, dropdowns for
  /// strings and radio buttons for enums.
  template <typename T>
  void setValueInUI(std::string                                                     name,
      std::conditional_t<std::is_arithmetic_v<T> || std::is_enum_v<T>, T, T const&> value) {
    if constexpr (std::is_same_v<T, bool>) {
      mGuiManager->setCheckboxValue(name, value);
    } else if constexpr (std::is_arithmetic_v<T>) {
      mGuiManager->setSliderValue(name, value);
    } else if constexpr (std::is_same_v<T, std::string>) {
      mGuiManager->getGui()->callJavascript("CosmoScout.gui.setDropdownValue", name, value);
    } else if constexpr (std::is_enum_v<T>) {
      mGuiManager->setRadioChecked(
          "volumeRendering." + name + std::to_string(static_cast<int>(value)));
    } else {
      static_assert(false, "Unhandled type for setValueInUI");
    }
  };

  /// Connects a handler function to a setting's target property.
  /// The handler will update the corresponding UI component, pass the value to the active renderer
  /// and execute a setter method.
  template <typename T>
  void connectSetting(Setting<T> const& setting) {
    if (std::string(setting.mName) == "" || !setting.mTarget.has_value()) {
      return;
    }
    std::string                       name(setting.mName);
    std::optional<Setting<T>::Setter> setter(setting.mSetter);
    std::optional<Setting<T>::Action> action(setting.mAction);
    setting.mTarget.value().get().connectAndTouch([this, name, setter, action](T value) {
      setValueInUI<T>("volumeRendering." + name, value);
      if (setter.has_value()) {
        invalidateCache();
        // Call the setter on mRenderer with value
        (mRenderer.get()->*setter.value())(value);
      }
      if (action.has_value()) {
        (this->*action.value())(value);
      }
    });
  };

  void setResolution(int value);
  void setDepthData(bool value);
  void setDrawDepth(bool value);
  void setScalar(std::string const& value);
  void setDisplayMode(DisplayMode value);

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

  bool mAnimating = false;

  std::future<std::vector<float>> mDataSample;
  int                             mSampleCount = 0;
  bool                            mResetTfHandles;

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
template <>
constexpr std::array<Plugin::Setting<DisplayMode>, SETTINGS_COUNT<DisplayMode>>
Plugin::Setting<DisplayMode>::getSettings(Settings& pluginSettings);
template <>
constexpr std::array<Plugin::Setting<DepthMode>, SETTINGS_COUNT<DepthMode>>
Plugin::Setting<DepthMode>::getSettings(Settings& pluginSettings);

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_PLUGIN_HPP
