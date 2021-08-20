////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Data/NetCdfFileLoader.hpp"
#include "Data/VtkFileLoader.hpp"
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

NLOHMANN_JSON_SERIALIZE_ENUM(VolumeFileType, {
                                                 {VolumeFileType::eInvalid, nullptr},
                                                 {VolumeFileType::eVtk, "vtk"},
                                                 {VolumeFileType::eNetCdf, "netcdf"},
                                             })

NLOHMANN_JSON_SERIALIZE_ENUM(
    VolumeStructure, {
                         {VolumeStructure::eInvalid, nullptr},
                         {VolumeStructure::eStructured, "structured"},
                         {VolumeStructure::eStructuredSpherical, "structuredSpherical"},
                         {VolumeStructure::eUnstructured, "unstructured"},
                     })

NLOHMANN_JSON_SERIALIZE_ENUM(VolumeShape, {
                                              {VolumeShape::eInvalid, nullptr},
                                              {VolumeShape::eCubic, "cubic"},
                                              {VolumeShape::eSpherical, "spherical"},
                                          })

NLOHMANN_JSON_SERIALIZE_ENUM(DisplayMode, {
                                              {DisplayMode::ePoints, "points"},
                                              {DisplayMode::eMesh, "mesh"},
                                          })

NLOHMANN_JSON_SERIALIZE_ENUM(DepthMode, {
                                            {DepthMode::eNone, "none"},
                                            {DepthMode::eIsosurface, "isosurface"},
                                            {DepthMode::eFirstHit, "firstHit"},
                                            {DepthMode::eLastHit, "lastHit"},
                                            {DepthMode::eThreshold, "threshold"},
                                            {DepthMode::eMultiThreshold, "multiThreshold"},
                                        })

void from_json(nlohmann::json const& j, Plugin::Settings::Lighting& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::deserialize(j, "ambientStrength", o.mSunStrength);
}

void to_json(nlohmann::json& j, Plugin::Settings::Lighting const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::serialize(j, "ambientStrength", o.mSunStrength);
}

void from_json(nlohmann::json const& j, Plugin::Settings::Pathlines& o) {
  cs::core::Settings::deserialize(j, "path", o.mPath);
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "opacity", o.mLineOpacity);
  cs::core::Settings::deserialize(j, "size", o.mLineSize);
  cs::core::Settings::deserialize(j, "length", o.mLength);
  cs::core::Settings::deserialize(j, "activeScalar", o.mActiveScalar);
}

void to_json(nlohmann::json& j, Plugin::Settings::Pathlines const& o) {
  cs::core::Settings::serialize(j, "path", o.mPath);
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "opacity", o.mLineOpacity);
  cs::core::Settings::serialize(j, "size", o.mLineSize);
  cs::core::Settings::serialize(j, "length", o.mLength);
  cs::core::Settings::serialize(j, "activeScalar", o.mActiveScalar);
}

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  // Data settings
  cs::core::Settings::deserialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::deserialize(j, "volumeDataPattern", o.mVolumeDataPattern);
  cs::core::Settings::deserialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::deserialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::deserialize(j, "volumeShape", o.mVolumeShape);
  cs::core::Settings::deserialize(j, "activeScalar", o.mActiveScalar);

  // Rendering settings
  cs::core::Settings::deserialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::deserialize(j, "resolution", o.mResolution);
  cs::core::Settings::deserialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::deserialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::deserialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::deserialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::deserialize(j, "depthMode", o.mDepthMode);
  cs::core::Settings::deserialize(j, "transferFunction", o.mTransferFunction);

  // Display settings
  cs::core::Settings::deserialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::deserialize(j, "reuseImages", o.mReuseImages);
  cs::core::Settings::deserialize(j, "useDepth", o.mDepthData);
  cs::core::Settings::deserialize(j, "drawDepth", o.mDrawDepth);
  cs::core::Settings::deserialize(j, "displayMode", o.mDisplayMode);

  // Transform settings
  cs::core::Settings::deserialize(j, "anchor", o.mAnchor);
  cs::core::Settings::deserialize(j, "position", o.mPosition);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "rotation", o.mRotation);

  if (j.contains("lighting")) {
    cs::core::Settings::deserialize(j, "lighting", o.mLighting);
  }
  cs::core::Settings::deserialize(j, "pathlines", o.mPathlines);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  // Data settings
  cs::core::Settings::serialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::serialize(j, "volumeDataPattern", o.mVolumeDataPattern);
  cs::core::Settings::serialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::serialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::serialize(j, "volumeShape", o.mVolumeShape);
  cs::core::Settings::serialize(j, "activeScalar", o.mActiveScalar);

  // Rendering settings
  cs::core::Settings::serialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::serialize(j, "resolution", o.mResolution);
  cs::core::Settings::serialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::serialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::serialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::serialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::serialize(j, "depthMode", o.mDepthMode);
  cs::core::Settings::serialize(j, "transferFunction", o.mTransferFunction);

  // Display settings
  cs::core::Settings::serialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::serialize(j, "reuseImages", o.mReuseImages);
  cs::core::Settings::serialize(j, "useDepth", o.mDepthData);
  cs::core::Settings::serialize(j, "drawDepth", o.mDrawDepth);
  cs::core::Settings::serialize(j, "displayMode", o.mDisplayMode);

  // Transform settings
  cs::core::Settings::serialize(j, "anchor", o.mAnchor);
  cs::core::Settings::serialize(j, "position", o.mPosition);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "rotation", o.mRotation);

  cs::core::Settings::serialize(j, "lighting", o.mLighting);
  cs::core::Settings::serialize(j, "pathlines", o.mPathlines);
}

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
  registerUICallbacks();
  initUI();
  connectSettings();

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

  switch (mRenderState) {
  case RenderState::eWaitForData:
    if (mDataManager->isReady()) {
      mRenderState = RenderState::eIdle;
      // Make sure that the correct scalar is selected in the DataManager
      mPluginSettings.mActiveScalar.touch();
    }
    break;
  case RenderState::ePaused:
    if (mPluginSettings.mRequestImages.get()) {
      mRenderState = RenderState::eIdle;
    }
    break;
  case RenderState::eIdle:
    if (!mPluginSettings.mRequestImages.get()) {
      mRenderState = RenderState::ePaused;
    } else if (tryRequestFrame()) {
      mRenderState = RenderState::eRenderingImage;
    }
    break;
  case RenderState::eRenderingImage:
    showRenderProgress();
    if (!mPluginSettings.mRequestImages.get()) {
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

  if (mPluginSettings.mReuseImages.get() && mRenderedImages.size() > 0) {
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

  switch (mPluginSettings.mVolumeDataType.get()) {
  case VolumeFileType::eVtk:
    mDataManager = std::make_shared<DataManager>(mPluginSettings.mVolumeDataPath.get(),
        mPluginSettings.mVolumeDataPattern.get(), std::make_unique<VtkFileLoader>(),
        mPluginSettings.mPathlines ? std::optional(mPluginSettings.mPathlines->mPath.get())
                                   : std::nullopt);
    break;
  case VolumeFileType::eNetCdf:
    mDataManager = std::make_shared<DataManager>(mPluginSettings.mVolumeDataPath.get(),
        mPluginSettings.mVolumeDataPattern.get(), std::make_unique<NetCdfFileLoader>(),
        mPluginSettings.mPathlines ? std::optional(mPluginSettings.mPathlines->mPath.get())
                                   : std::nullopt);
    break;
  default:
    logger().error("Invalid volume data type given in settings! Should be 'vtk'.");
    throw std::runtime_error("Failed to initialize DataManager.");
    break;
  }

  mRenderer = std::make_unique<OSPRayRenderer>(
      mDataManager, mPluginSettings.mVolumeStructure.get(), mPluginSettings.mVolumeShape.get());

  // If the volume representations already exist, remove them from the solar system
  for (auto const& node : mDisplayNodes) {
    mSolarSystem->unregisterAnchor(node.second);
  }
  mDisplayNodes.clear();
  mActiveDisplay.reset();

  // Init volume representation
  auto anchor = mAllSettings->mAnchors.find(mPluginSettings.mAnchor.get());
  if (anchor == mAllSettings->mAnchors.end()) {
    logger().error("No anchor with name '{}' found!", mPluginSettings.mAnchor.get());
    throw std::runtime_error("Failed to initialize CelestialObjects.");
  }

  auto existence                    = anchor->second.mExistence;
  mDisplayNodes[DisplayMode::eMesh] = std::make_shared<Billboard>(
      mPluginSettings.mVolumeShape.get(), mAllSettings, mPluginSettings.mAnchor.get());
  mDisplayNodes[DisplayMode::ePoints] = std::make_shared<PointsForwardWarped>(
      mPluginSettings.mVolumeShape.get(), mAllSettings, mPluginSettings.mAnchor.get());

  for (auto const& node : mDisplayNodes) {
    node.second->setAnchorPosition(mPluginSettings.mPosition.get());
    node.second->setAnchorScale(mPluginSettings.mScale.get());
    node.second->setAnchorRotation(mPluginSettings.mRotation.get());

    mSolarSystem->registerAnchor(node.second);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerUICallbacks() {
  auto boolSettings = Setting<bool>::getSettings(mPluginSettings);
  for (auto const& setting : boolSettings) {
    registerUICallback(setting);
  }
  auto intSettings = Setting<int>::getSettings(mPluginSettings);
  for (auto const& setting : intSettings) {
    registerUICallback(setting);
  }
  auto floatSettings = Setting<float>::getSettings(mPluginSettings);
  for (auto const& setting : floatSettings) {
    registerUICallback(setting);
  }

  // Rendering settings
  mGuiManager->getGui()->registerCallback("volumeRendering.cancel",
      "If an image is currently rendered, cancel it.",
      std::function([this]() { mRenderer->cancelRendering(); }));

  /*registerUICallback(Setting<int>("setResolution",
      "Sets the resolution of the rendered volume images.", mPluginSettings.mResolution));
  registerUICallback(Setting<int>("setMaxRenderPasses",
      "Sets the maximum number of render passes for constant rendering parameters.",
      mPluginSettings.mRendering.mMaxPasses));*/

  mGuiManager->getGui()->registerCallback("volumeRendering.setSamplingRate",
      "Sets the sampling rate for volume rendering.",
      std::function([this](double value) { mPluginSettings.mSamplingRate = (float)value; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setSunStrength",
      "Sets the strength of the sun when shading is enabled.", std::function([this](double value) {
        mPluginSettings.mLighting.mSunStrength = (float)value;
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setAmbientStrength",
      "Sets the strength of the ambient light when shading is enabled.",
      std::function(
          [this](double value) { mPluginSettings.mLighting.mAmbientStrength = (float)value; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setDensityScale",
      "Sets the density scale of the volume.",
      std::function([this](double value) { mPluginSettings.mDensityScale = (float)value; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDenoiseColor",
      "Enables use of OIDN for displaying color data.",
      std::function([this](bool enable) { mPluginSettings.mDenoiseColor = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDenoiseDepth",
      "Enables use of OIDN for displaying depth data.",
      std::function([this](bool enable) { mPluginSettings.mDenoiseDepth = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode0",
      "Don't calculate a depth value.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eNone; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode1",
      "Calculate depth of isosurface.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eIsosurface; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode2",
      "Uses first hit as depth value.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eFirstHit; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode3",
      "Uses last hit as depth value.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eLastHit; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode4",
      "Uses depth, at which an opacity threshold was reached.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eThreshold; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode5",
      "Uses depth, at which the last of multiple opacity thresholds was reached.",
      std::function([this]() { mPluginSettings.mDepthMode = DepthMode::eMultiThreshold; }));

  // Display settings
  mGuiManager->getGui()->registerCallback("volumeRendering.setEnablePredictiveRendering",
      "Enables predicting the next camera position for rendering.",
      std::function([this](bool enable) { mPluginSettings.mPredictiveRendering = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableReuseImages",
      "Enables reuse of previously rendered images.",
      std::function([this](bool enable) { mPluginSettings.mReuseImages = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDepthData",
      "Enables use of depth data for displaying data.",
      std::function([this](bool enable) { mPluginSettings.mDepthData = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDrawDepth",
      "Enables displaying the depth buffer instead of the color buffer.",
      std::function([this](bool enable) { mPluginSettings.mDrawDepth = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setDisplayMode0",
      "Displays the rendered images on a continuous mesh.",
      std::function([this]() { mPluginSettings.mDisplayMode = DisplayMode::eMesh; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDisplayMode1",
      "Displays the rendered images on a continuous mesh.",
      std::function([this]() { mPluginSettings.mDisplayMode = DisplayMode::ePoints; }));

  // Data settings
  mGuiManager->getGui()->registerCallback("volumeRendering.setScalar",
      "Set the scalar to be rendered.",
      std::function([this](std::string scalar) { mPluginSettings.mActiveScalar = scalar; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setTimestep",
      "Sets the timestep of the rendered volume images.", std::function([this](double value) {
        invalidateCache();
        mDataManager->setTimestep((int)std::lround(value));
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.preloadTimestep",
      "Prepares the renderer for rendering.", std::function([this](double value) {
        DataManager::State state = mDataManager->getState();
        state.mTimestep          = (int)std::lround(value);
        mRenderer->preloadData(state);
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setAnimationSpeed",
      "Time units per second when animating.", std::function([](double value) {
        // Callback is only registered to suppress warnings
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
        if (value) {
          mRenderer->setMaxLod(mDataManager->getMinLod(mDataManager->getState()));
        } else {
          mRenderer->clearMaxLod();
        }
      }));

  // Pathline settings
  if (mPluginSettings.mPathlines.has_value()) {
    mGuiManager->getGui()->registerCallback("volumeRendering.setEnablePathlines",
        "Enable/disable rendering of pathlines.",
        std::function([this](bool enable) { mPluginSettings.mPathlines->mEnabled = enable; }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlineOpacity",
        "Sets the opacity of the rendered pathlines.", std::function([this](double value) {
          mPluginSettings.mPathlines->mLineOpacity = (float)value;
        }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlineSize",
        "Sets the size of the rendered pathlines.", std::function([this](double value) {
          mPluginSettings.mPathlines->mLineSize = (float)value;
        }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlineLength",
        "Sets the length of the rendered pathlines.", std::function([this](double value) {
          mPluginSettings.mPathlines->mLength = (float)value;
        }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlineActiveScalar",
        "Sets the active scalar for coloring the pathlines.",
        std::function([this](std::string value) {
          if (std::find_if(mDataManager->getPathlines().getScalars().begin(),
                  mDataManager->getPathlines().getScalars().end(), [value](Scalar s) {
                    return s.getId() == value;
                  }) != mDataManager->getPathlines().getScalars().end()) {
            mPluginSettings.mPathlines->mActiveScalar = value;
          }
        }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setPathlinesScalarFilters",
        "Sets filters for selecting which parts of the pathlines should be rendered.",
        std::function([this](std::string jsonString) {
          std::vector<ScalarFilter> filters =
              parseScalarFilters(jsonString, mDataManager->getPathlines().getScalars());
          invalidateCache();
          mRenderer->setPathlineScalarFilters(filters);
        }));

    mGuiManager->getGui()->registerCallback("volumeRendering.setEnablePathlinesParcoords",
        "Use a separate parallel coordinate diagram for the pathlines.",
        std::function([](bool value) {
          // Callback is only registered to suppress warnings
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::connectSettings() {
  auto boolSettings = Setting<bool>::getSettings(mPluginSettings);
  for (auto const& setting : boolSettings) {
    connectSetting(setting);
  }
  auto intSettings = Setting<int>::getSettings(mPluginSettings);
  for (auto const& setting : intSettings) {
    connectSetting(setting);
  }
  auto floatSettings = Setting<float>::getSettings(mPluginSettings);
  for (auto const& setting : floatSettings) {
    connectSetting(setting);
  }

  // Connect to plugin settings
  // Data settings
  mPluginSettings.mActiveScalar.connectAndTouch([this](std::string value) {
    invalidateCache();
    if (mDataManager->isReady()) {
      mDataManager->setActiveScalar(value);
      mGuiManager->getGui()->callJavascript("CosmoScout.volumeRendering.setXRange",
          mDataManager->getScalarRange(mDataManager->getState().mScalar.getId())[0],
          mDataManager->getScalarRange(mDataManager->getState().mScalar.getId())[1], true);
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setDropdownValue", "volumeRendering.setScalar", value);
    }
  });

  // Rendering settings
  mPluginSettings.mResolution.connectAndTouch([this](int value) {
    invalidateCache();
    mNextFrame.mResolution = value;
    mRenderer->setResolution(value);
    mGuiManager->setSliderValue("volumeRendering.setResolution", value);
  });
  mPluginSettings.mSamplingRate.connectAndTouch([this](float value) {
    invalidateCache();
    mRenderer->setSamplingRate(value);
    mGuiManager->setSliderValue("volumeRendering.setSamplingRate", value);
  });
  mPluginSettings.mRendering.mMaxPasses.connectAndTouch([this](int value) {
    invalidateCache();
    mRenderer->setMaxRenderPasses(value);
    mGuiManager->setSliderValue("volumeRendering.setMaxRenderPasses", value);
  });
  mPluginSettings.mDensityScale.connectAndTouch([this](float value) {
    invalidateCache();
    mRenderer->setDensityScale(value);
    mGuiManager->setSliderValue("volumeRendering.setDensityScale", value);
  });
  mPluginSettings.mDenoiseColor.connectAndTouch([this](bool enable) {
    invalidateCache();
    mRenderer->setDenoiseColor(enable);
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseColor", enable);
  });
  mPluginSettings.mDenoiseDepth.connectAndTouch([this](bool enable) {
    invalidateCache();
    mRenderer->setDenoiseDepth(enable);
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseDepth", enable);
  });
  mPluginSettings.mDepthMode.connectAndTouch([this](DepthMode drawMode) {
    invalidateCache();
    mRenderer->setDepthMode(drawMode);
    if (drawMode == DepthMode::eNone) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode0");
    } else if (drawMode == DepthMode::eIsosurface) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode1");
    } else if (drawMode == DepthMode::eFirstHit) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode2");
    } else if (drawMode == DepthMode::eLastHit) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode3");
    } else if (drawMode == DepthMode::eThreshold) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode4");
    } else if (drawMode == DepthMode::eMultiThreshold) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode5");
    }
  });
  mPluginSettings.mTransferFunction.connectAndTouch([this](std::string name) {
    std::string code = "CosmoScout.volumeRendering.loadTransferFunction('" + name + "');";
    mGuiManager->addScriptToGui(code);
  });

  // Lighting settings
  mPluginSettings.mLighting.mSunStrength.connectAndTouch([this](float value) {
    invalidateCache();
    mRenderer->setSunStrength(value);
    mGuiManager->setSliderValue("volumeRendering.setSunStrength", value);
  });
  mPluginSettings.mLighting.mAmbientStrength.connectAndTouch([this](float value) {
    invalidateCache();
    mRenderer->setAmbientLight(value);
    mGuiManager->setSliderValue("volumeRendering.setAmbientStrength", value);
  });

  // Display settings
  mPluginSettings.mPredictiveRendering.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnablePredictiveRendering", enable);
  });
  mPluginSettings.mReuseImages.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableReuseImages", enable);
  });
  mPluginSettings.mDepthData.connectAndTouch([this](bool enable) {
    for (auto const& node : mDisplayNodes) {
      node.second->setUseDepth(enable);
    }
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDepthData", enable);
  });
  mPluginSettings.mDrawDepth.connectAndTouch([this](bool enable) {
    for (auto const& node : mDisplayNodes) {
      node.second->setDrawDepth(enable);
    }
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDrawDepth", enable);
  });
  mPluginSettings.mDisplayMode.connectAndTouch([this](DisplayMode displayMode) {
    for (auto const& node : mDisplayNodes) {
      if (node.first == displayMode) {
        node.second->setEnabled(true);
      } else {
        node.second->setEnabled(false);
      }
    }
    mActiveDisplay = mDisplayNodes.find(displayMode)->second;

    if (displayMode == DisplayMode::eMesh) {
      mGuiManager->setRadioChecked("volumeRendering.setDisplayMode0");
    } else if (displayMode == DisplayMode::ePoints) {
      mGuiManager->setRadioChecked("volumeRendering.setDisplayMode1");
    }
    if (mDisplayedImage) {
      displayFrame(std::move(mDisplayedImage), displayMode);
    } else {
      // TODO What to do here?
      mParametersDirty = true;
    }
  });

  // Pathline settings
  if (mPluginSettings.mPathlines.has_value()) {
    mPluginSettings.mPathlines->mEnabled.connectAndTouch([this](bool enable) {
      invalidateCache();
      mRenderer->setPathlinesEnabled(enable);
      mGuiManager->setCheckboxValue("volumeRendering.setEnablePathlines", enable);
    });
    mPluginSettings.mPathlines->mLineOpacity.connectAndTouch([this](float value) {
      invalidateCache();
      mRenderer->setPathlineOpacity(value);
      mGuiManager->setSliderValue("volumeRendering.setPathlineOpacity", value);
    });
    mPluginSettings.mPathlines->mLineSize.connectAndTouch([this](float value) {
      invalidateCache();
      mRenderer->setPathlineSize(value);
      mGuiManager->setSliderValue("volumeRendering.setPathlineSize", value);
    });
    mPluginSettings.mPathlines->mLength.connectAndTouch([this](float value) {
      invalidateCache();
      mRenderer->setPathlineLength(value);
      mGuiManager->setSliderValue("volumeRendering.setPathlineLength", value);
    });
    mPluginSettings.mPathlines->mActiveScalar.connectAndTouch([this](std::string value) {
      invalidateCache();
      mRenderer->setPathlineActiveScalar(value);
    });
  }

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
          [this](Scalar s) { return s.getId() == mPluginSettings.mActiveScalar.get(); });
      if (activeScalar == scalars.end()) {
        activeScalar = scalars.begin();
      }
      mPluginSettings.mActiveScalar.set(activeScalar->getId());
      mGuiManager->getGui()->callJavascript("CosmoScout.volumeRendering.setXRange",
          mDataManager->getScalarRange(activeScalar->getId())[0],
          mDataManager->getScalarRange(activeScalar->getId())[1], true);
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
    if (mDataManager->isReady() && scalar.getId() == mPluginSettings.mActiveScalar.get()) {
      mGuiManager->getGui()->callJavascript("CosmoScout.volumeRendering.setXRange",
          mDataManager->getScalarRange(scalar.getId())[0],
          mDataManager->getScalarRange(scalar.getId())[1], false);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerUICallback(Setting<bool> const& setting) {
  std::reference_wrapper targetRef(setting.mTarget);
  mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
      std::string(setting.mComment),
      std::function([targetRef](bool enable) { targetRef.get() = enable; }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerUICallback(Setting<int> const& setting) {
  std::reference_wrapper targetRef(setting.mTarget);
  mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
      std::string(setting.mComment),
      std::function([targetRef](double value) { targetRef.get() = (int)std::lround(value); }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerUICallback(Setting<float> const& setting) {
  std::reference_wrapper targetRef(setting.mTarget);
  mGuiManager->getGui()->registerCallback("volumeRendering." + std::string(setting.mName),
      std::string(setting.mComment),
      std::function([targetRef](double value) { targetRef.get() = (float)value; }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::connectSetting(Setting<bool> const& setting) {
  std::string name(setting.mName);
  if (setting.mSetter.has_value()) {
    Setting<bool>::Setter setter(setting.mSetter.value());
    setting.mTarget.connectAndTouch([this, name, setter](bool value) {
      mGuiManager->setCheckboxValue("volumeRendering." + name, value);
      invalidateCache();
      // Call the setter on mRenderer with value
      (mRenderer.get()->*setter)(value);
    });
  } else {
    setting.mTarget.connectAndTouch([this, name](bool value) {
      mGuiManager->setCheckboxValue("volumeRendering." + name, value);
    });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::connectSetting(Setting<int> const& setting) {
  std::string name(setting.mName);
  if (setting.mSetter.has_value()) {
    Setting<int>::Setter setter(setting.mSetter.value());
    setting.mTarget.connectAndTouch([this, name, setter](int value) {
      mGuiManager->setSliderValue("volumeRendering." + name, value);
      invalidateCache();
      // Call the setter on mRenderer with value
      (mRenderer.get()->*setter)(value);
    });
  } else {
    setting.mTarget.connectAndTouch(
        [this, name](int value) { mGuiManager->setSliderValue("volumeRendering." + name, value); });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void csp::volumerendering::Plugin::connectSetting(Setting<float> const& setting) {
  std::string name(setting.mName);
  if (setting.mSetter.has_value()) {
    Setting<float>::Setter setter(setting.mSetter.value());
    setting.mTarget.connectAndTouch([this, name, setter](float value) {
      mGuiManager->setSliderValue("volumeRendering." + name, value);
      invalidateCache();
      // Call the setter on mRenderer with value
      (mRenderer.get()->*setter)(value);
    });
  } else {
    setting.mTarget.connectAndTouch([this, name](float value) {
      mGuiManager->setSliderValue("volumeRendering." + name, value);
    });
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<bool>, Plugin::mSettingsCount<bool>>
Plugin::Setting<bool>::getSettings(Settings& pluginSettings) {
  std::array<Setting<bool>, mSettingsCount<bool>> settings{
      Setting<bool>{"setEnableRequestImages", "If disabled no new images will be rendered.",
          pluginSettings.mRequestImages},
      Setting<bool>{"setEnableLighting", "Enables/Disables shading.",
          pluginSettings.mLighting.mEnabled, &Renderer::setShading}};
  return std::move(settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<int>, Plugin::mSettingsCount<int>>
Plugin::Setting<int>::getSettings(Settings& pluginSettings) {
  std::array<Setting<int>, mSettingsCount<int>> settings{};
  return std::move(settings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
constexpr std::array<Plugin::Setting<float>, Plugin::mSettingsCount<float>>
Plugin::Setting<float>::getSettings(Settings& pluginSettings) {
  std::array<Setting<float>, mSettingsCount<float>> settings{};
  return std::move(settings);
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

bool Plugin::tryRequestFrame() {
  if (mActiveDisplay->pVisible.get()) {
    cs::utils::FrameTimings::ScopedTimer timer("Request frame");

    mRenderingFrame = mNextFrame;
    glm::vec4 dir = glm::vec4(mSolarSystem->getSunDirection(mActiveDisplay->getWorldPosition()), 1);
    dir           = dir * glm::inverse(mRenderingFrame.mCameraTransform);
    mRenderer->setSunDirection(dir);
    mFutureFrameData   = mRenderer->getFrame(mRenderingFrame.mCameraTransform);
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

  if (mPluginSettings.mPredictiveRendering.get()) {
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
        [&axis = axis](Scalar const& s) { return s.mName == axis; });
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
  displayFrame(std::move(frame), mPluginSettings.mDisplayMode.get());
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
  if (mPluginSettings.mReuseImages.get()) {
    mDisplayedImage = std::make_unique<Renderer::CopiedImage>(*frame);
  } else {
    mDisplayedImage.reset();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
