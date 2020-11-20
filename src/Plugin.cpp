////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Data/VtkDataManager.hpp"
#include "Display/Billboard.hpp"
#include "Display/PointsForwardWarped.hpp"
#include "Render/OSPRayRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/ColorMap.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <vtk-8.1/vtkOutputWindow.h>
#include <vtk-8.1/vtkSmartPointer.h>

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

NLOHMANN_JSON_SERIALIZE_ENUM(
    Plugin::Settings::VolumeFileType, {
                                          {Plugin::Settings::VolumeFileType::eInvalid, nullptr},
                                          {Plugin::Settings::VolumeFileType::eVtk, "vtk"},
                                      })

NLOHMANN_JSON_SERIALIZE_ENUM(
    Renderer::VolumeStructure, {
                                   {Renderer::VolumeStructure::eInvalid, nullptr},
                                   {Renderer::VolumeStructure::eStructured, "structured"},
                                   {Renderer::VolumeStructure::eUnstructured, "unstructured"},
                               })

NLOHMANN_JSON_SERIALIZE_ENUM(
    Renderer::VolumeShape, {
                               {Renderer::VolumeShape::eInvalid, nullptr},
                               {Renderer::VolumeShape::eCubic, "cubic"},
                               {Renderer::VolumeShape::eSpherical, "spherical"},
                           })

NLOHMANN_JSON_SERIALIZE_ENUM(
    Plugin::Settings::DisplayMode, {
                                       {Plugin::Settings::DisplayMode::ePoints, "points"},
                                       {Plugin::Settings::DisplayMode::eMesh, "mesh"},
                                   })

NLOHMANN_JSON_SERIALIZE_ENUM(
    Renderer::DepthMode, {
                             {Renderer::DepthMode::eNone, "none"},
                             {Renderer::DepthMode::eIsosurface, "isosurface"},
                             {Renderer::DepthMode::eFirstHit, "firstHit"},
                             {Renderer::DepthMode::eLastHit, "lastHit"},
                             {Renderer::DepthMode::eThreshold, "threshold"},
                             {Renderer::DepthMode::eMultiThreshold, "multiThreshold"},
                         })

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  // Data settings
  cs::core::Settings::deserialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::deserialize(j, "volumeDataPattern", o.mVolumeDataPattern);
  cs::core::Settings::deserialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::deserialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::deserialize(j, "volumeShape", o.mVolumeShape);

  // Rendering settings
  cs::core::Settings::deserialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::deserialize(j, "resolution", o.mResolution);
  cs::core::Settings::deserialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::deserialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::deserialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::deserialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::deserialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::deserialize(j, "depthMode", o.mDepthMode);

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
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  // Data settings
  cs::core::Settings::serialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::serialize(j, "volumeDataPattern", o.mVolumeDataPattern);
  cs::core::Settings::serialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::serialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::serialize(j, "volumeShape", o.mVolumeShape);

  // Rendering settings
  cs::core::Settings::serialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::serialize(j, "resolution", o.mResolution);
  cs::core::Settings::serialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::serialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::serialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::serialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::serialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::serialize(j, "depthMode", o.mDepthMode);

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Frame::operator==(const Frame& other) {
  return mResolution == other.mResolution &&
         glm::all(glm::epsilonEqual(mCameraTransform[0], other.mCameraTransform[0], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[1], other.mCameraTransform[1], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[2], other.mCameraTransform[2], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[3], other.mCameraTransform[3], 0.0001f));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  std::thread keepAlive([]() {
    while (1)
      std::this_thread::sleep_for(std::chrono::seconds(1));
  });
  keepAlive.detach();

  // Print vtk output to the console instead of opening a window
  vtkSmartPointer<vtkOutputWindow> outputWindow = vtkSmartPointer<vtkOutputWindow>::New();
  vtkOutputWindow::SetInstance(outputWindow);

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
  mAllSettings->mGraphics.pEnableLighting.disconnect(mLightingConnection);
  mAllSettings->mGraphics.pAmbientBrightness.disconnect(mAmbientConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  mNextFrame.mCameraTransform = getCurrentCameraTransform();

  switch (mRenderState) {
  case RenderState::eWaitForData:
    if (mDataManager->isReady()) {
      mRenderState = RenderState::eIdle;
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

  if (mPluginSettings.mReuseImages.get() && mRenderedFrames.size() > 0) {
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
  case Settings::VolumeFileType::eVtk:
    mDataManager = std::make_shared<VtkDataManager>(
        mPluginSettings.mVolumeDataPath.get(), mPluginSettings.mVolumeDataPattern.get());
    break;
  default:
    logger().error("Invalid volume data type given in settings! Should be 'vtk'.");
    throw std::exception("Failed to initialize DataManager.");
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
    throw std::exception("Failed to initialize CelestialObjects.");
  }

  auto [tStartExistence, tEndExistence] = anchor->second.getExistence();
  mDisplayNodes[Plugin::Settings::DisplayMode::eMesh] =
      std::make_shared<Billboard>(mSceneGraph, anchor->second.mCenter, anchor->second.mFrame,
          tStartExistence, tEndExistence, cs::core::SolarSystem::getRadii(anchor->second.mCenter));
  mDisplayNodes[Plugin::Settings::DisplayMode::ePoints] = std::make_shared<PointsForwardWarped>(
      mSceneGraph, anchor->second.mCenter, anchor->second.mFrame, tStartExistence, tEndExistence,
      cs::core::SolarSystem::getRadii(anchor->second.mCenter));

  for (auto const& node : mDisplayNodes) {
    node.second->setAnchorPosition(mPluginSettings.mPosition.get());
    node.second->setAnchorScale(mPluginSettings.mScale.get());
    node.second->setAnchorRotation(mPluginSettings.mRotation.get());

    mSolarSystem->registerAnchor(node.second);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::registerUICallbacks() {
  // Rendering settings
  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableRequestImages",
      "If disabled no new images will be rendered.",
      std::function([this](bool enable) { mPluginSettings.mRequestImages = enable; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.cancel",
      "If an image is currently rendered, cancel it.",
      std::function([this]() { mRenderer->cancelRendering(); }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setResolution",
      "Sets the resolution of the rendered volume images.", std::function([this](double value) {
        mPluginSettings.mResolution = (int)std::lround(value);
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setSamplingRate",
      "Sets the sampling rate for volume rendering.",
      std::function([this](double value) { mPluginSettings.mSamplingRate = (float)value; }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setSunStrength",
      "Sets the strength of the sun when shading is enabled.",
      std::function([this](double value) { mPluginSettings.mSunStrength = (float)value; }));

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
      std::function([this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eNone; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode1",
      "Calculate depth of isosurface.",
      std::function([this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eIsosurface; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode2",
      "Uses first hit as depth value.",
      std::function([this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eFirstHit; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode3",
      "Uses last hit as depth value.",
      std::function([this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eLastHit; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode4",
      "Uses depth, at which an opacity threshold was reached.",
      std::function([this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eThreshold; }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode5",
      "Uses depth, at which the last of multiple opacity thresholds was reached.",
      std::function(
          [this]() { mPluginSettings.mDepthMode = Renderer::DepthMode::eMultiThreshold; }));

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
      "Displays the rendered images on a continuous mesh.", std::function([this]() {
        mPluginSettings.mDisplayMode = Plugin::Settings::DisplayMode::eMesh;
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDisplayMode1",
      "Displays the rendered images on a continuous mesh.", std::function([this]() {
        mPluginSettings.mDisplayMode = Plugin::Settings::DisplayMode::ePoints;
      }));

  // Data settings
  mGuiManager->getGui()->registerCallback("volumeRendering.setScalar",
      "Set the scalar to be rendered.", std::function([this](std::string scalar) {
        mRenderedFrames.clear();
        mDataManager->setActiveScalar(scalar);
        mParametersDirty = true;
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setTimestep",
      "Sets the timestep of the rendered volume images.", std::function([this](double value) {
        mRenderedFrames.clear();
        mDataManager->setTimestep((int)std::lround(value));
        mParametersDirty = true;
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.preloadTimestep",
      "Prepares the renderer for rendering.", std::function([this](double value) {
        DataManager::State state = mDataManager->getState();
        state.mTimestep          = (int)std::lround(value);
        mRenderer->preloadData(state);
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.setAnimationSpeed",
      "Time units per second when animating.", std::function([this](double value) {
        // Callback is only registered to suppress warnings
      }));

  // Transferfunction
  mGuiManager->getGui()->registerCallback("volumeRendering.setTransferFunction",
      "Sets the transfer function for rendering the volume.",
      std::function([this](std::string json) {
        mRenderedFrames.clear();
        cs::graphics::ColorMap colorMap(json, false);
        mRenderer->setTransferFunction(colorMap.getRawData());
        mParametersDirty = true;
      }));

  mGuiManager->getGui()->registerCallback("volumeRendering.importTransferFunction",
      "Import a saved transfer function.",
      std::function([this](std::string name) { importTransferFunction(name); }));
  mGuiManager->getGui()->registerCallback("volumeRendering.exportTransferFunction",
      "Export the current transfer function to a file.",
      std::function([this](std::string name, std::string jsonTransferFunction) {
        exportTransferFunction(name, jsonTransferFunction);
      }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::connectSettings() {
  // Connect to data manager properties
  mDataManager->pScalars.connectAndTouch([this](std::vector<std::string> scalars) {
    for (std::string scalar : scalars) {
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.addDropdownValue", "volumeRendering.setScalar", scalar, scalar, false);
    }
    if (scalars.size() > 0) {
      mGuiManager->getGui()->callJavascript(
          "CosmoScout.gui.setDropdownValue", "volumeRendering.setScalar", scalars[0]);
    }
  });
  mDataManager->pTimesteps.connectAndTouch([this](std::vector<int> timesteps) {
    nlohmann::json timestepsJson(timesteps);
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.volumeRendering.setTimesteps", timestepsJson.dump());
  });

  // Connect to global CosmoScout graphics settings
  mLightingConnection =
      mAllSettings->mGraphics.pEnableLighting.connectAndTouch([this](bool enable) {
        mRenderedFrames.clear();
        mRenderer->setShading(enable);
        mParametersDirty = true;
      });
  mAmbientConnection =
      mAllSettings->mGraphics.pAmbientBrightness.connectAndTouch([this](float value) {
        mRenderedFrames.clear();
        mRenderer->setAmbientLight(value);
        mParametersDirty = true;
      });

  // Connect to plugin settings
  // Rendering settings
  mPluginSettings.mRequestImages.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableRequestImages", enable);
  });
  mPluginSettings.mResolution.connectAndTouch([this](int value) {
    mRenderedFrames.clear();
    mNextFrame.mResolution = value;
    mRenderer->setResolution(value);
    mParametersDirty = true;
    mGuiManager->setSliderValue("volumeRendering.setResolution", value);
  });
  mPluginSettings.mSamplingRate.connectAndTouch([this](float value) {
    mRenderedFrames.clear();
    mRenderer->setSamplingRate(value);
    mParametersDirty = true;
    mGuiManager->setSliderValue("volumeRendering.setSamplingRate", value);
  });
  mPluginSettings.mSunStrength.connectAndTouch([this](float value) {
    mRenderedFrames.clear();
    mRenderer->setSunStrength(value);
    mParametersDirty = true;
    mGuiManager->setSliderValue("volumeRendering.setSunStrength", value);
  });
  mPluginSettings.mDensityScale.connectAndTouch([this](float value) {
    mRenderedFrames.clear();
    mRenderer->setDensityScale(value);
    mParametersDirty = true;
    mGuiManager->setSliderValue("volumeRendering.setDensityScale", value);
  });
  mPluginSettings.mDenoiseColor.connectAndTouch([this](bool enable) {
    mRenderedFrames.clear();
    mRenderer->setDenoiseColor(enable);
    mParametersDirty = true;
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseColor", enable);
  });
  mPluginSettings.mDenoiseDepth.connectAndTouch([this](bool enable) {
    mRenderedFrames.clear();
    mRenderer->setDenoiseDepth(enable);
    mParametersDirty = true;
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseDepth", enable);
  });
  mPluginSettings.mDepthMode.connectAndTouch([this](Renderer::DepthMode drawMode) {
    mRenderedFrames.clear();
    mRenderer->setDepthMode(drawMode);
    mParametersDirty = true;
    if (drawMode == Renderer::DepthMode::eNone) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode0");
    } else if (drawMode == Renderer::DepthMode::eIsosurface) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode1");
    } else if (drawMode == Renderer::DepthMode::eFirstHit) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode2");
    } else if (drawMode == Renderer::DepthMode::eLastHit) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode3");
    } else if (drawMode == Renderer::DepthMode::eThreshold) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode4");
    } else if (drawMode == Renderer::DepthMode::eMultiThreshold) {
      mGuiManager->setRadioChecked("volumeRendering.setDepthMode5");
    }
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
  mPluginSettings.mDisplayMode.connectAndTouch([this](Settings::DisplayMode displayMode) {
    for (auto const& node : mDisplayNodes) {
      if (node.first == displayMode) {
        node.second->setEnabled(true);
      } else {
        node.second->setEnabled(false);
      }
    }
    mActiveDisplay = mDisplayNodes.find(displayMode)->second;

    if (displayMode == Settings::DisplayMode::eMesh) {
      mGuiManager->setRadioChecked("volumeRendering.setDisplayMode0");
    } else if (displayMode == Settings::DisplayMode::ePoints) {
      mGuiManager->setRadioChecked("volumeRendering.setDisplayMode1");
    }
    if (mDisplayedFrame.has_value()) {
      displayFrame(*mDisplayedFrame, displayMode);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::initUI() {
  // Add the volume rendering user interface components to the CosmoScout user interface.
  mGuiManager->addCssToGui("css/csp-volume-rendering.css");
  mGuiManager->addPluginTabToSideBarFromHTML(
      "Volume Rendering", "blur_circular", "../share/resources/gui/volume_rendering_tab.html");
  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-volume-rendering.js");

  updateAvailableTransferFunctions();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::tryRequestFrame() {
  if ((!(mNextFrame == mRenderingFrame) || mParametersDirty || mFrameInvalid) &&
      mActiveDisplay->pVisible.get()) {
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
    predictedTranslation += glm::vec3(mCameraTransforms[i][3].xyz) / (float)mCameraTransformsLength;
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

  Renderer::RenderedImage renderedImage = mFutureFrameData.get();
  if (!renderedImage.mValid) {
    mFrameInvalid = true;
    return;
  }

  mRenderingFrame.mColorImage          = renderedImage.mColorData;
  mRenderingFrame.mDepthImage          = renderedImage.mDepthData;
  mRenderingFrame.mModelViewProjection = renderedImage.mMVP;
  mRenderedFrames.push_back(mRenderingFrame);

  displayFrame(mRenderingFrame);
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

  Frame bestFrame = mRenderedFrames.back();
  float minDiff   = diffTranslations(cameraTransform, bestFrame.mCameraTransform);
  for (const Frame& f : mRenderedFrames) {
    float diff = diffTranslations(cameraTransform, f.mCameraTransform);
    if (diff < minDiff) {
      minDiff   = diff;
      bestFrame = f;
    }
  }
  if (mDisplayedFrame.has_value() && !(bestFrame == *mDisplayedFrame) && minDiff > 0) {
    displayFrame(bestFrame);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::displayFrame(Frame& frame) {
  displayFrame(frame, mPluginSettings.mDisplayMode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::displayFrame(Frame& frame, Settings::DisplayMode displayMode) {
  cs::utils::FrameTimings::ScopedTimer timer("Display volume frame");

  std::shared_ptr<DisplayNode> displayNode = mDisplayNodes.find(displayMode)->second;
  displayNode->setTexture(frame.mColorImage, frame.mResolution, frame.mResolution);
  displayNode->setDepthTexture(frame.mDepthImage, frame.mResolution, frame.mResolution);
  displayNode->setTransform(glm::toMat4(glm::toQuat(frame.mCameraTransform)));
  displayNode->setMVPMatrix(frame.mModelViewProjection);

  mDisplayedFrame = frame;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::exportTransferFunction(
    std::string const& path, std::string const& jsonTransferFunction) {
  std::ofstream o("../share/resources/transferfunctions/" + path);
  o << jsonTransferFunction;

  updateAvailableTransferFunctions();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::importTransferFunction(std::string const& path) {
  std::stringstream jsonTransferFunction;
  std::ifstream     i("../share/resources/transferfunctions/" + path);
  jsonTransferFunction << i.rdbuf();
  std::string code =
      "CosmoScout.volumeRendering.loadTransferFunction(`" + jsonTransferFunction.str() + "`);";
  mGuiManager->addScriptToGui(code);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::updateAvailableTransferFunctions() {
  nlohmann::json j;
  for (const auto& file :
      cs::utils::filesystem::listFiles("../share/resources/transferfunctions/")) {
    std::string filename = file;
    filename.erase(0, 37);
    j.push_back(filename);
  }

  std::string code =
      "CosmoScout.volumeRendering.setAvailableTransferFunctions(`" + j.dump() + "`);";
  mGuiManager->addScriptToGui(code);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
