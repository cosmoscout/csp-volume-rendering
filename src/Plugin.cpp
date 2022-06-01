////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Display/Billboard.hpp"
#include "Display/PointsForwardWarped.hpp"
#include "Render/OSPRayRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/ColorMap.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <vtkOutputWindow.h>
#include <vtkSmartPointer.h>

#include "glm/gtc/epsilon.hpp"
#include "glm/gtx/quaternion.hpp"

#include <numeric>
#include <thread>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::volumerendering::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  std::thread keepAlive([]() {
    while (1)
      std::this_thread::sleep_for(std::chrono::seconds(1));
  });
  keepAlive.detach();

  // Makes sure that the vtk output window is created on the main thread. If it is created on
  // another thread and that thread stops, all calls to DisplayText etc. from any other thread will
  // block indefinitely.
  vtkOutputWindow::GetInstance()->DisplayDebugText("");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-volume-rendering"] = mPluginSettings; });

  onLoad();
  registerAllUICallbacks();
  initUI();
  connectAllSettings();

  // Init buffers for predictive rendering
  mFrameIntervals.resize(mFrameIntervalsLength);
  mCameraTransforms.resize(mCameraTransformsLength);

  logger().info("Loading plugin done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  for (auto const& node : mDisplayNodes) {
    mSolarSystem->unregisterAnchor(node.second);
  }

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  mNextFrame.mCameraTransform = getCurrentCameraTransform();

  if (mDataManager->isReady() && mSampleCount < 20 && !mAnimating) {
    if (mDataSample.valid() &&
        mDataSample.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
      std::vector<float> sample = mDataSample.get();
      if (sample.size() > 0) {
        nlohmann::json sampleJson = sample;
        if (mSampleCount == 0) {
          nlohmann::json extents =
              mDataManager->getScalarRange(mDataManager->getState().mScalar.getId());
          mGuiManager->getGui()->callJavascript(
              "CosmoScout.volumeRendering.transferFunctionEditor.setData", sampleJson,
              mResetTfHandles, extents);
        } else {
          mGuiManager->getGui()->callJavascript(
              "CosmoScout.volumeRendering.transferFunctionEditor.addData", sampleJson, false);
        }
        mSampleCount++;
      }
    } else {
      mDataSample =
          mDataManager->getSample(mDataManager->getState(), std::chrono::milliseconds(10));
    }
  }

  switch (mRenderState) {
  case RenderState::eWaitForData:
    if (mDataManager->isReady()) {
      mRenderState = RenderState::eIdle;
      if (!mPluginSettings.mData.mMetadata.has_value()) {
        mPluginSettings.mData.mMetadata = mDataManager->calculateMetadata();
      }
      // Make sure that the correct scalar is selected in the DataManager
      mPluginSettings.mData.mActiveScalar.touch();
    }
    break;
  case RenderState::ePaused:
    if (mPluginSettings.mRendering.mRequestImages.get()) {
      mRenderState = RenderState::eIdle;
    }
    break;
  case RenderState::eIdle:
    if (!mPluginSettings.mRendering.mRequestImages.get()) {
      mRenderState = RenderState::ePaused;
    } else if (tryRequestFrame()) {
      mRenderState = RenderState::eRenderingImage;
    }
    break;
  case RenderState::eRenderingImage:
    showRenderProgress();
    if (!mPluginSettings.mRendering.mRequestImages.get()) {
      mRenderer->cancelRendering();
      mRenderState = RenderState::ePaused;
    } else if (mFutureFrameData.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      receiveFrame();
      if (!tryRequestFrame()) {
        mRenderState = RenderState::eIdle;
      }
    }
    break;
  }

  if (mPluginSettings.mDisplay.mReuseImages.get() && mRenderedImages.size() > 0) {
    tryReuseFrame(mNextFrame.mCameraTransform);
  }

  mLastFrameInterval++;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Get plugin settings
  from_json(mAllSettings->mPlugins.at("csp-volume-rendering"), mPluginSettings);

  // Init data manager and volume renderer
  mRenderState = RenderState::eWaitForData;

  mDataManager = std::make_shared<DataManager>(mPluginSettings.mData);
  if (mPluginSettings.mPathlines.has_value()) {
    mDataManager->addPathlines(mPluginSettings.mPathlines.value());
  }

  mRenderer = std::make_unique<OSPRayRenderer>(
      mDataManager, mPluginSettings.mData.mStructure.get(), mPluginSettings.mData.mShape.get());

  // If the volume representations already exist, remove them from the solar system
  for (auto const& node : mDisplayNodes) {
    mSolarSystem->unregisterAnchor(node.second);
  }
  mDisplayNodes.clear();
  mActiveDisplay.reset();

  // Init volume representation
  auto anchor = mAllSettings->mAnchors.find(mPluginSettings.mTransform.mAnchor.get());
  if (anchor == mAllSettings->mAnchors.end()) {
    logger().error("No anchor with name '{}' found!", mPluginSettings.mTransform.mAnchor.get());
    throw std::runtime_error("Failed to initialize CelestialObjects.");
  }

  auto existence                    = anchor->second.mExistence;
  mDisplayNodes[DisplayMode::eMesh] = std::make_shared<Billboard>(
      mPluginSettings.mData.mShape.get(), mAllSettings, mPluginSettings.mTransform.mAnchor.get());
  mDisplayNodes[DisplayMode::ePoints] = std::make_shared<PointsForwardWarped>(
      mPluginSettings.mData.mShape.get(), mAllSettings, mPluginSettings.mTransform.mAnchor.get());

  mDepthExtractor = std::make_unique<DepthExtractor>(
      mDisplayNodes[DisplayMode::eMesh], mSolarSystem, mTimeControl);

  for (auto const& node : mDisplayNodes) {
    node.second->setAnchorPosition(mPluginSettings.mTransform.mPosition.get());
    node.second->setAnchorScale(mPluginSettings.mTransform.mScale.get());
    node.second->setAnchorRotation(
        cs::utils::convert::toRadians(mPluginSettings.mTransform.mRotation.get()));

    mSolarSystem->registerAnchor(node.second);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerAllUICallbacks() {
  registerUICallbacks<bool>();
  registerUICallbacks<int>();
  registerUICallbacks<float>();
  registerUICallbacks<std::string>();
  registerUICallbacks<DisplayMode>();
  registerUICallbacks<DepthMode>();

  // Rendering settings
  mGuiManager->getGui()->registerCallback("volumeRendering.cancel",
      "If an image is currently rendered, cancel it.",
      std::function([this]() { mRenderer->cancelRendering(); }));

  // Data settings
  mGuiManager->getGui()->registerCallback("volumeRendering.setTimestep",
      "Sets the timestep of the rendered volume images.", std::function([this](double value) {
        invalidateCache();
        mDataManager->setTimestep((int)std::lround(value));
        mSampleCount    = 0;
        mResetTfHandles = false;
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.preloadTimestep",
      "Prepares the renderer for rendering.", std::function([this](double value) {
        DataManager::State state = mDataManager->getState();
        state.mTimestep          = (int)std::lround(value);
        std::optional<DataManager::State> coreState;
        if (mPluginSettings.mCore.has_value()) {
          coreState = state;
          auto scalar =
              std::find_if(mDataManager->pScalars.get().begin(), mDataManager->pScalars.get().end(),
                  [this](Scalar s) { return s.getId() == mPluginSettings.mCore->mScalar.get(); });
          if (scalar != mDataManager->pScalars.get().end()) {
            coreState->mScalar = *scalar;
          } else {
            coreState = {};
          }
        }
        mRenderer->preloadData(state, coreState);
      }));

  // Transferfunction
  mGuiManager->getGui()->registerCallback("volumeRendering.setTransferFunction",
      "Sets the transfer function for rendering the volume.",
      std::function([this](std::string json) {
        invalidateCache();
        cs::graphics::ColorMap colorMap(json);
        mRenderer->setTransferFunction(colorMap.getRawData());
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setVolumeScalarFilters",
      "Sets filters for selecting which parts of the volume should be rendered.",
      std::function([this](std::string jsonString) {
        std::vector<Scalar>       scalars = mDataManager->pScalars.get();
        std::vector<ScalarFilter> filters = parseScalarFilters(jsonString, scalars);
        invalidateCache();
        mRenderer->setScalarFilters(filters);
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setTimestepAnimating",
      "Specifies, whether timesteps are currently animated (increasing automatically).",
      std::function([this](bool value) {
        invalidateCache();
        mAnimating = value;
        if (value) {
          mRenderer->setMaxLod(mDataManager->getMinLod(mDataManager->getState()));
        } else {
          mRenderer->clearMaxLod();
        }
      }));

  // Pathline settings
  if (mPluginSettings.mPathlines.has_value()) {
    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlinesScalarFilters",
        "Sets filters for selecting which parts of the pathlines should be rendered.",
        std::function([this](std::string jsonString) {
          std::vector<ScalarFilter> filters =
              parseScalarFilters(jsonString, mDataManager->getPathlines().getScalars());
          invalidateCache();
          mRenderer->setPathlineScalarFilters(filters);
        }));
  }

  // Parcoords
  mGuiManager->getGui()->registerCallback("parcoords.importBrushState",
      "Import a saved parcoords brush state.",
      std::function([this](std::string name, std::string editorId) {
        std::stringstream json;
        std::ifstream     i("../share/resources/parcoords/" + name);
        json << i.rdbuf();

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.parcoords.loadBrushState", json.str(), editorId);
      }));

  mGuiManager->getGui()->registerCallback("parcoords.exportBrushState",
      "Export the current parcoords brush state to a file.",
      std::function([this](std::string name, std::string json) {
        std::ofstream o("../share/resources/parcoords/" + name);
        o << json;

        mGuiManager->getGui()->callJavascript("CosmoScout.parcoords.addAvailableBrushState", name);
      }));

  mGuiManager->getGui()->registerCallback("parcoords.getAvailableBrushStates",
      "Requests the list of currently available parcoords brush state files.",
      std::function([this]() {
        nlohmann::json j = nlohmann::json::array();
        std::string    parcoordsPath("../share/resources/parcoords/");
        if (!boost::filesystem::exists(parcoordsPath)) {
          cs::utils::filesystem::createDirectoryRecursively(
              parcoordsPath, boost::filesystem::perms::all_all);
        }
        for (const auto& file : cs::utils::filesystem::listFiles(parcoordsPath)) {
          std::string filename = file;
          filename.erase(0, 29);
          j.push_back(filename);
        }

        mGuiManager->getGui()->callJavascript(
            "CosmoScout.parcoords.setAvailableBrushStates", j.dump());
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setRotationYaw",
      "Sets the display node rotation.", std::function([this](double value) {
        glm::dvec3 rotation                  = mPluginSettings.mTransform.mRotation.get();
        rotation.x                           = value;
        mPluginSettings.mTransform.mRotation = rotation;
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setRotationPitch",
      "Sets the display node rotation.", std::function([this](double value) {
        glm::dvec3 rotation                  = mPluginSettings.mTransform.mRotation.get();
        rotation.y                           = value;
        mPluginSettings.mTransform.mRotation = rotation;
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setRotationRoll",
      "Sets the display node rotation.", std::function([this](double value) {
        glm::dvec3 rotation                  = mPluginSettings.mTransform.mRotation.get();
        rotation.z                           = value;
        mPluginSettings.mTransform.mRotation = rotation;
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::connectAllSettings() {
  connectSettings<bool>();
  connectSettings<int>();
  connectSettings<float>();
  connectSettings<std::string>();
  connectSettings<DisplayMode>();
  connectSettings<DepthMode>();

  // Connect to plugin settings
  // Rendering settings
  mPluginSettings.mRendering.mTransferFunction.connectAndTouch([this](std::string name) {
    std::string code = "CosmoScout.volumeRendering.loadTransferFunction('" + name + "');";
    mGuiManager->addScriptToGui(code);
  });

  mPluginSettings.mTransform.mRotation.connectAndTouch([this](glm::dvec3 value) {
    mGuiManager->setSliderValue("volumeRendering.setRotationYaw", value.x);
    mGuiManager->setSliderValue("volumeRendering.setRotationPitch", value.y);
    mGuiManager->setSliderValue("volumeRendering.setRotationRoll", value.z);
    for (auto const& node : mDisplayNodes) {
      node.second->setAnchorRotation(cs::utils::convert::toRadians(value));
    }
  });

  // Connect to data manager properties
  mDataManager->pScalars.connectAndTouch([this](std::vector<Scalar> scalars) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.clearDropdown", "volumeRendering.setScalar");
    for (Scalar scalar : scalars) {
      mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
          "volumeRendering.setScalar", scalar.getId(), scalar.mName, false);
    }
    if (scalars.size() > 0) {
      auto activeScalar = std::find_if(scalars.begin(), scalars.end(),
          [this](Scalar s) { return s.getId() == mPluginSettings.mData.mActiveScalar.get(); });
      if (activeScalar == scalars.end()) {
        activeScalar = scalars.begin();
      }
      mPluginSettings.mData.mActiveScalar.set(activeScalar->getId());
      mSampleCount    = 0;
      mResetTfHandles = true;
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setDropdownValue", "volumeRendering.setScalar", activeScalar->getId());
    }
  });
  mDataManager->pTimesteps.connectAndTouch([this](std::vector<int> timesteps) {
    nlohmann::json timestepsJson(timesteps);
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.volumeRendering.setTimesteps", timestepsJson.dump());
  });
  mDataManager->onScalarRangeUpdated().connect([this](Scalar const& scalar) {
    if (mDataManager->isReady() && scalar.getId() == mPluginSettings.mData.mActiveScalar.get()) {
      mSampleCount    = 0;
      mResetTfHandles = false;
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<bool>, SETTINGS_COUNT<bool>>
Plugin::Setting<bool>::getSettings(Settings& pluginSettings) {
  std::array<Setting<bool>, SETTINGS_COUNT<bool>> settings{
      // Rendering settings
      Setting<bool>{"setEnableRequestImages", "If disabled no new images will be rendered.",
          pluginSettings.mRendering.mRequestImages},
      Setting<bool>{"setUseMaxDepth",
          "If enabled, rendering will use CosmoScout's depth buffer as the maximum depth.",
          pluginSettings.mRendering.mUseMaxDepth, {}, &Plugin::setUseMaxDepth},
      Setting<bool>{"setEnableDenoiseColor", "Enables use of OIDN for displaying color data.",
          pluginSettings.mRendering.mDenoiseColor, &Renderer::setDenoiseColor},
      Setting<bool>{"setEnableDenoiseDepth", "Enables use of OIDN for displaying depth data.",
          pluginSettings.mRendering.mDenoiseDepth, &Renderer::setDenoiseDepth},
      // Display settings
      Setting<bool>{"setEnableDepthData", "Enables use of depth data for displaying data.",
          pluginSettings.mDisplay.mDepthData, {}, &Plugin::setDepthData},
      Setting<bool>{"setEnableDrawDepth",
          "Enables displaying the depth buffer instead of the color buffer.",
          pluginSettings.mDisplay.mDrawDepth, {}, &Plugin::setDrawDepth},
      Setting<bool>{"setEnablePredictiveRendering",
          "Enables predicting the next camera position for rendering.",
          pluginSettings.mDisplay.mPredictiveRendering},
      Setting<bool>{"setEnableReuseImages", "Enables reuse of previously rendered images.",
          pluginSettings.mDisplay.mReuseImages},
      // Lighting settings
      Setting<bool>{"setEnableLighting", "Enables/Disables shading.",
          pluginSettings.mLighting.mEnabled, &Renderer::setShading},
      // Core settings
      pluginSettings.mCore.has_value()
          ? Setting<bool>{"setEnableCore", "Enable/disable rendering of core",
                pluginSettings.mCore->mEnabled, &Renderer::setCoreEnabled}
          : Setting<bool>{},
      // Pathline settings
      pluginSettings.mPathlines.has_value()
          ? Setting<bool>{"setEnablePathlines", "Enable/disable rendering of pathlines.",
                pluginSettings.mPathlines->mEnabled, &Renderer::setPathlinesEnabled}
          : Setting<bool>{},
      Setting<bool>{
          "setEnablePathlinesParcoords",
          "Use a separate parallel coordinate diagram for the pathlines.",
      },
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<int>, SETTINGS_COUNT<int>> Plugin::Setting<int>::getSettings(
    Settings& pluginSettings) {
  std::array<Setting<int>, SETTINGS_COUNT<int>> settings{
      // Rendering settings
      Setting<int>{"setMaxRenderPasses",
          "Sets the maximum number of render passes for constant rendering parameters.",
          pluginSettings.mRendering.mMaxPasses, &Renderer::setMaxRenderPasses},
      Setting<int>{"setResolution", "Sets the resolution of the rendered volume images.",
          pluginSettings.mRendering.mResolution, &Renderer::setResolution, &Plugin::setResolution},
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<float>, SETTINGS_COUNT<float>>
Plugin::Setting<float>::getSettings(Settings& pluginSettings) {
  std::array<Setting<float>, SETTINGS_COUNT<float>> settings{
      // Animation settings
      Setting<float>{"setAnimationSpeed", "Time units per second when animating."},
      // Rendering settings
      Setting<float>{"setSamplingRate", "Sets the sampling rate for volume rendering.",
          pluginSettings.mRendering.mSamplingRate, &Renderer::setSamplingRate},
      Setting<float>{"setDensityScale", "Sets the density scale of the volume.",
          pluginSettings.mRendering.mDensityScale, &Renderer::setDensityScale},
      // Lighting settings
      Setting<float>{"setSunStrength", "Sets the strength of the sun when shading is enabled.",
          pluginSettings.mLighting.mSunStrength, &Renderer::setSunStrength},
      Setting<float>{"setAmbientStrength",
          "Sets the strength of the ambient light when shading is enabled.",
          pluginSettings.mLighting.mAmbientStrength, &Renderer::setAmbientLight},
      // Core settings
      pluginSettings.mCore.has_value()
          ? Setting<float>{"setCoreRadius", "Sets the radius of the rendered core.",
                pluginSettings.mCore->mRadius, &Renderer::setCoreRadius}
          : Setting<float>{},
      // Pathline settings
      pluginSettings.mPathlines.has_value()
          ? Setting<float>{"setPathlineSize", "Sets the size of the rendered pathlines.",
                pluginSettings.mPathlines->mLineSize, &Renderer::setPathlineSize}
          : Setting<float>{},
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<std::string>, SETTINGS_COUNT<std::string>>
Plugin::Setting<std::string>::getSettings(Settings& pluginSettings) {
  std::array<Setting<std::string>, SETTINGS_COUNT<std::string>> settings{
      // Data settings
      Setting<std::string>{"setScalar", "Set the scalar to be rendered.",
          pluginSettings.mData.mActiveScalar, {}, &Plugin::setScalar},
      // Core settings
      pluginSettings.mCore.has_value()
          ? Setting<std::string>{"setCoreScalar", "Sets the scalar used for coloring the core.",
                pluginSettings.mCore->mScalar, &Renderer::setCoreScalar}
          : Setting<std::string>{},
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<DisplayMode>, SETTINGS_COUNT<DisplayMode>>
Plugin::Setting<DisplayMode>::getSettings(Settings& pluginSettings) {
  std::array<Setting<DisplayMode>, SETTINGS_COUNT<DisplayMode>> settings{
      Setting<DisplayMode>{"setDisplayMode", "Sets the mode for displaying the rendered images.",
          pluginSettings.mDisplay.mDisplayMode, {}, &Plugin::setDisplayMode},
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<DepthMode>, SETTINGS_COUNT<DepthMode>>
Plugin::Setting<DepthMode>::getSettings(Settings& pluginSettings) {
  std::array<Setting<DepthMode>, SETTINGS_COUNT<DepthMode>> settings{
      Setting<DepthMode>{"setDepthMode",
          "Sets the mode for determining the per pixel depth values of the volume.",
          pluginSettings.mRendering.mDepthMode, &Renderer::setDepthMode},
  };
  return settings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::initUI() {
  // Add the volume rendering user interface components to the CosmoScout user interface.
  mGuiManager->addCssToGui("css/csp-volume-rendering.css");
  mGuiManager->addCssToGui("third-party/css/d3.parcoords.css");
  mGuiManager->addPluginTabToSideBarFromHTML(
      "Volume Rendering", "blur_circular", "../share/resources/gui/volume_rendering_tab.html");
  mGuiManager->addScriptToGuiFromJS(
      "../share/resources/gui/third-party/js/parcoords.standalone.js");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/parcoords.js");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-volume-rendering.js");
  mGuiManager->addScriptToGui(
      "CosmoScout.volumeRendering.initParcoords(`" + mDataManager->getCsvData() + "`, `" +
      (mPluginSettings.mPathlines ? mDataManager->getPathlines().getCsvData() : "") + "`);");

  if (mPluginSettings.mPathlines.has_value()) {
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.volumeRendering.enableSettingsSection", "pathlines");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::setResolution(int value) {
  mNextFrame.mResolution = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setUseMaxDepth(bool value) {
  mDepthExtractor->setEnabled(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::setDepthData(bool value) {
  for (auto const& node : mDisplayNodes) {
    node.second->setUseDepth(value);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::setDrawDepth(bool value) {
  for (auto const& node : mDisplayNodes) {
    node.second->setDrawDepth(value);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::setScalar(std::string const& value) {
  invalidateCache();
  if (mDataManager->isReady()) {
    mDataManager->setActiveScalar(value);
    mSampleCount    = 0;
    mResetTfHandles = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::setDisplayMode(DisplayMode value) {
  for (auto const& node : mDisplayNodes) {
    if (node.first == value) {
      node.second->setEnabled(true);
    } else {
      node.second->setEnabled(false);
    }
  }
  mActiveDisplay = mDisplayNodes.find(value)->second;

  if (mDisplayedImage) {
    displayFrame(std::move(mDisplayedImage), value);
  } else {
    // TODO What to do here?
    mParametersDirty = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::tryRequestFrame() {
  if (mActiveDisplay->pVisible.get()) {
    cs::utils::FrameTimings::ScopedTimer timer("Request frame");

    std::optional<std::vector<float>> depthBuffer;
    if (mPluginSettings.mRendering.mUseMaxDepth.get()) {
      depthBuffer = mDepthExtractor->getDepthBuffer(mPluginSettings.mRendering.mResolution.get());
      if (!depthBuffer.has_value()) {
        return false;
      }
    }

    mRenderingFrame = mNextFrame;
    glm::vec4 dir = glm::vec4(mSolarSystem->getSunDirection(mActiveDisplay->getWorldPosition()), 1);
    dir           = dir * glm::inverse(mRenderingFrame.mCameraTransform);
    mRenderer->setSunDirection(dir);
    mFutureFrameData =
        mRenderer->getFrame(mRenderingFrame.mCameraTransform, std::move(depthBuffer));
    mLastFrameInterval = 0;
    mParametersDirty   = false;
    mFrameInvalid      = false;
    return true;
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 Plugin::getCurrentCameraTransform() {
  glm::mat4 currentCameraTransform = mActiveDisplay->getRelativeTransform(
      mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver());
  float     scale = (float)mActiveDisplay->getRelativeScale(mSolarSystem->getObserver());
  glm::vec3 r     = cs::core::SolarSystem::getRadii(mActiveDisplay->getCenterName());
  currentCameraTransform =
      glm::scale(currentCameraTransform, glm::vec3(1.f / scale, 1.f / scale, 1.f / scale));
  currentCameraTransform[3] *= glm::vec4(1 / r[0], 1 / r[1], 1 / r[2], 1);

  mCameraTransforms[mCameraTransformsIndex] =
      currentCameraTransform * glm::inverse(mLastCameraTransform);
  mLastCameraTransform = currentCameraTransform;
  if (++mCameraTransformsIndex >= mCameraTransformsLength) {
    mCameraTransformsIndex = 0;
  }

  if (mPluginSettings.mDisplay.mPredictiveRendering.get()) {
    currentCameraTransform = predictCameraTransform(currentCameraTransform);
  }
  return currentCameraTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 Plugin::predictCameraTransform(glm::mat4 currentTransform) {
  glm::mat4 predictedCameraTransform = currentTransform;
  int       meanFrameInterval =
      std::accumulate(mFrameIntervals.begin(), mFrameIntervals.end(), 0) / mFrameIntervalsLength;
  glm::vec3 predictedTranslation(0);
  glm::quat predictedRotation(1, 0, 0, 0);

  for (int i = 0; i < mCameraTransformsLength; i++) {
    predictedTranslation += glm::vec3(mCameraTransforms[i][3]) / (float)mCameraTransformsLength;
    predictedRotation =
        glm::slerp(predictedRotation, glm::quat_cast(mCameraTransforms[i]), 1.f / (i + 1));
  }

  for (int i = 0; i < meanFrameInterval; i++) {
    predictedCameraTransform = glm::mat4_cast(predictedRotation) * predictedCameraTransform;
    predictedCameraTransform = glm::translate(predictedCameraTransform, predictedTranslation);
  }
  return predictedCameraTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::showRenderProgress() {
  std::stringstream code;
  code << "CosmoScout.volumeRendering.setRenderProgress(" << mRenderer->getProgress() << ", false)";
  mGuiManager->addScriptToGui(code.str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::receiveFrame() {
  mFrameIntervals[mFrameIntervalsIndex] = mLastFrameInterval;
  if (++mFrameIntervalsIndex >= mFrameIntervalsLength) {
    mFrameIntervalsIndex = 0;
  }

  std::unique_ptr<Renderer::RenderedImage> renderedImage = mFutureFrameData.get();
  if (!renderedImage || !renderedImage->isValid()) {
    mFrameInvalid = true;
    return;
  }

  displayFrame(std::move(renderedImage));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float diffTranslations(glm::mat4 transformA, glm::mat4 transformB) {
  glm::vec3 translationA = glm::normalize(transformA[3]);
  glm::vec3 translationB = glm::normalize(transformB[3]);
  return glm::length(translationB - translationA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::tryReuseFrame(glm::mat4 cameraTransform) {
  cs::utils::FrameTimings::ScopedTimer timer("Try reuse frame");

  auto  bestFrame = mRenderedImages.begin();
  float minDiff;
  float currentDiff = INFINITY;
  if (mDisplayedImage) {
    minDiff     = diffTranslations(cameraTransform, mDisplayedImage->getCameraTransform());
    currentDiff = minDiff;
  } else {
    minDiff = diffTranslations(cameraTransform, (*bestFrame)->getCameraTransform());
  }
  for (auto img = bestFrame; img != mRenderedImages.end(); img++) {
    float diff = diffTranslations(cameraTransform, (*img)->getCameraTransform());
    if (diff < minDiff) {
      minDiff   = diff;
      bestFrame = img;
    }
  }
  if (minDiff < currentDiff) {
    std::unique_ptr<Renderer::RenderedImage> image =
        std::move(mRenderedImages.extract(bestFrame).value());
    displayFrame(std::move(image));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::invalidateCache() {
  mRenderedImages.clear();
  mDisplayedImage.reset();
  mParametersDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<ScalarFilter> csp::volumerendering::Plugin::parseScalarFilters(
    std::string const& json, std::vector<Scalar> const& scalars) {
  auto                      j = nlohmann::json::parse(json);
  std::vector<ScalarFilter> filters;
  for (auto const& [axis, value] : j.items()) {
    auto const& scalar = std::find_if(scalars.begin(), scalars.end(),
        [& axis = axis](Scalar const& s) { return s.mName == axis; });
    if (scalar != scalars.end()) {
      ScalarFilter filter;
      filter.mAttrIndex = (int)std::distance(scalars.begin(), scalar);
      filter.mMin       = value["selection"]["scaled"][1];
      filter.mMax       = value["selection"]["scaled"][0];
      filters.push_back(filter);
    }
  }
  return filters;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::displayFrame(std::unique_ptr<Renderer::RenderedImage> frame) {
  displayFrame(std::move(frame), mPluginSettings.mDisplay.mDisplayMode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::displayFrame(std::unique_ptr<Renderer::RenderedImage> frame, DisplayMode displayMode) {
  cs::utils::FrameTimings::ScopedTimer timer("Display volume frame");

  std::shared_ptr<DisplayNode> displayNode = mDisplayNodes.find(displayMode)->second;
  displayNode->setTexture(frame->getColorData(), frame->getResolution(), frame->getResolution());
  displayNode->setDepthTexture(
      frame->getDepthData(), frame->getResolution(), frame->getResolution());
  displayNode->setTransform(glm::toMat4(glm::toQuat(frame->getCameraTransform())));
  displayNode->setRendererMatrices(frame->getModelView(), frame->getProjection());

  if (mDisplayedImage) {
    mRenderedImages.insert(std::move(mDisplayedImage));
  }
  if (mPluginSettings.mDisplay.mReuseImages.get()) {
    mDisplayedImage = std::make_unique<Renderer::CopiedImage>(*frame);
  } else {
    mDisplayedImage.reset();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
