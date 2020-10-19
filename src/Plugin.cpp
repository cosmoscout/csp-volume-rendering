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
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/ColorMap.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

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
    DataManager::VolumeFileType, {
                                     {DataManager::VolumeFileType::eInvalid, nullptr},
                                     {DataManager::VolumeFileType::eGaia, "gaia"},
                                     {DataManager::VolumeFileType::eVtk, "vtk"},
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

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::deserialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::deserialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::deserialize(j, "volumeShape", o.mVolumeShape);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "volumeDataPath", o.mVolumeDataPath);
  cs::core::Settings::serialize(j, "volumeDataType", o.mVolumeDataType);
  cs::core::Settings::serialize(j, "volumeStructure", o.mVolumeStructure);
  cs::core::Settings::serialize(j, "volumeShape", o.mVolumeShape);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Frame::operator==(const Frame& other) {
  return mResolution == other.mResolution &&
         glm::all(glm::epsilonEqual(mCameraTransform[0], other.mCameraTransform[0], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[1], other.mCameraTransform[1], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[2], other.mCameraTransform[2], 0.0001f)) &&
         glm::all(glm::epsilonEqual(mCameraTransform[3], other.mCameraTransform[3], 0.0001f)) &&
         mSamplingRate == other.mSamplingRate && mTransferFunction == other.mTransferFunction &&
         mDenoiseColor == other.mDenoiseColor && mDenoiseDepth == other.mDenoiseDepth &&
         mDepthMode == other.mDepthMode && mShading == other.mShading;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection =
      mAllSettings->onLoad().connect([this]() { logger().info("Settings loaded."); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-volume-rendering"] = mPluginSettings; });

  // Add the volume rendering user interface components to the CosmoScout user interface.
  mGuiManager->addPluginTabToSideBarFromHTML(
      "Volume Rendering", "blur_circular", "../share/resources/gui/volume_rendering_tab.html");

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-volume-rendering.js");

  mGuiManager->getGui()->registerCallback("volumeRendering.setResolution",
      "Sets the resolution of the rendered volume images.",
      std::function([this](double value) { mPluginSettings.mResolution = value; }));
  mPluginSettings.mResolution.connectAndTouch(
      [this](int value) { mGuiManager->setSliderValue("volumeRendering.setResolution", value); });

  mGuiManager->getGui()->registerCallback("volumeRendering.setSamplingRate",
      "Sets the sampling rate for volume rendering.",
      std::function([this](double value) { mPluginSettings.mSamplingRate = value; }));
  mPluginSettings.mSamplingRate.connectAndTouch([this](float value) {
    mGuiManager->setSliderValue("volumeRendering.setSamplingRate", value);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableShading",
      "If enabled gradient shading will be used.", std::function([this](bool enable) {
        mPluginSettings.mShading = enable;
        mRenderedFrames.clear();
      }));
  mPluginSettings.mShading.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableShading", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableRequestImages",
      "If disabled no new images will be rendered.",
      std::function([this](bool enable) { mPluginSettings.mRequestImages = enable; }));
  mPluginSettings.mRequestImages.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableRequestImages", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnablePredictiveRendering",
      "Enables predicting the next camera position for rendering.",
      std::function([this](bool enable) { mPluginSettings.mPredictiveRendering = enable; }));
  mPluginSettings.mPredictiveRendering.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnablePredictiveRendering", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableReuseImages",
      "Enables reuse of previously rendered images.",
      std::function([this](bool enable) { mPluginSettings.mReuseImages = enable; }));
  mPluginSettings.mReuseImages.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableReuseImages", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDepthData",
      "Enables use of depth data for displaying data.", std::function([this](bool enable) {
        mPluginSettings.mDepthData = enable;
        mBillboard->setUseDepth(enable);
        mPoints->setUseDepth(enable);
      }));
  mPluginSettings.mDepthData.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDepthData", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDrawDepth",
      "Enables displaying the depth buffer instead of the color buffer.",
      std::function([this](bool enable) {
        mPluginSettings.mDrawDepth = enable;
        mBillboard->setDrawDepth(enable);
        mPoints->setDrawDepth(enable);
      }));
  mPluginSettings.mDrawDepth.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDrawDepth", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDenoiseColor",
      "Enables use of OIDN for displaying color data.",
      std::function([this](bool enable) { mPluginSettings.mDenoiseColor = enable; }));
  mPluginSettings.mDenoiseColor.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseColor", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDenoiseDepth",
      "Enables use of OIDN for displaying depth data.",
      std::function([this](bool enable) { mPluginSettings.mDenoiseDepth = enable; }));
  mPluginSettings.mDenoiseDepth.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoiseDepth", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setTransferFunction",
      "Sets the transfer function for rendering the volume.",
      std::function([this](std::string json) {
        cs::graphics::ColorMap colorMap(json, false);
        mRenderer->setTransferFunction(colorMap.getRawData());
        mNextFrame.mTransferFunction = colorMap.getRawData();
        mRenderedFrames.clear();
      }));

  mGuiManager->getGui()->registerCallback(
      "volumeRendering.setDepthMode0", "Don't calculate a depth value.", std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eNone;
        mRenderedFrames.clear();
      }));
  mGuiManager->getGui()->registerCallback(
      "volumeRendering.setDepthMode1", "Calculate depth of isosurface.", std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eIsosurface;
        mRenderedFrames.clear();
      }));
  mGuiManager->getGui()->registerCallback(
      "volumeRendering.setDepthMode2", "Uses first hit as depth value.", std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eFirstHit;
        mRenderedFrames.clear();
      }));
  mGuiManager->getGui()->registerCallback(
      "volumeRendering.setDepthMode3", "Uses last hit as depth value.", std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eLastHit;
        mRenderedFrames.clear();
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode4",
      "Uses depth, at which an opacity threshold was reached.", std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eThreshold;
        mRenderedFrames.clear();
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDepthMode5",
      "Uses depth, at which the last of multiple opacity thresholds was reached.",
      std::function([this]() {
        mPluginSettings.mDepthMode = Renderer::DepthMode::eMultiThreshold;
        mRenderedFrames.clear();
      }));
  mPluginSettings.mDepthMode.connect([this](Renderer::DepthMode drawMode) {
    if (drawMode == Renderer::DepthMode::eNone) {
      mGuiManager->setRadioChecked("stars.setDrawMode0");
    } else if (drawMode == Renderer::DepthMode::eIsosurface) {
      mGuiManager->setRadioChecked("stars.setDrawMode1");
    } else if (drawMode == Renderer::DepthMode::eFirstHit) {
      mGuiManager->setRadioChecked("stars.setDrawMode2");
    } else if (drawMode == Renderer::DepthMode::eLastHit) {
      mGuiManager->setRadioChecked("stars.setDrawMode3");
    } else if (drawMode == Renderer::DepthMode::eThreshold) {
      mGuiManager->setRadioChecked("stars.setDrawMode4");
    } else if (drawMode == Renderer::DepthMode::eMultiThreshold) {
      mGuiManager->setRadioChecked("stars.setDrawMode5");
    }
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setDisplayMode0",
      "Displays the rendered images on a continuous mesh.", std::function([this]() {
        mBillboard->setEnabled(true);
        mPoints->setEnabled(false);
      }));
  mGuiManager->getGui()->registerCallback("volumeRendering.setDisplayMode1",
      "Displays the rendered images on a continuous mesh.", std::function([this]() {
        mBillboard->setEnabled(false);
        mPoints->setEnabled(true);
      }));
  mPluginSettings.mDisplayMode.connect([this](Settings::DisplayMode displayMode) {
    if (displayMode == Settings::DisplayMode::eMesh) {
      mGuiManager->setRadioChecked("stars.setDisplayMode0");
    } else if (displayMode == Settings::DisplayMode::ePoints) {
      mGuiManager->setRadioChecked("stars.setDisplayMode1");
    }
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.importTransferFunction",
      "Import a saved transfer function.",
      std::function([this](std::string name) { importTransferFunction(name); }));
  mGuiManager->getGui()->registerCallback("volumeRendering.exportTransferFunction",
      "Export the current transfer function to a file.",
      std::function([this](std::string name, std::string jsonTransferFunction) {
        exportTransferFunction(name, jsonTransferFunction);
      }));
  updateAvailableTransferFunctions();

  // Load settings
  from_json(mAllSettings->mPlugins.at("csp-volume-rendering"), mPluginSettings);

  // Init data manager and volume renderer
  mDataManager = std::make_unique<DataManager>(
      mPluginSettings.mVolumeDataPath.get(), mPluginSettings.mVolumeDataType.get());
  mRenderer = std::make_unique<OSPRayRenderer>(
      mDataManager, mPluginSettings.mVolumeStructure.get(), mPluginSettings.mVolumeShape.get());
  // mDataManager->setTimestep(14564);
  mRenderState = RenderState::eRequestImage;

  // Init volume representation
  auto anchor                           = mAllSettings->mAnchors.find("Mars");
  auto [tStartExistence, tEndExistence] = anchor->second.getExistence();
  mBillboard = std::make_shared<Billboard>(anchor->second.mCenter, anchor->second.mFrame,
      tStartExistence, tEndExistence, cs::core::SolarSystem::getRadii(anchor->second.mCenter));
  mPoints    = std::make_shared<PointsForwardWarped>(anchor->second.mCenter, anchor->second.mFrame,
      tStartExistence, tEndExistence, cs::core::SolarSystem::getRadii(anchor->second.mCenter));

  // Add volume representation to solar system and scene graph
  mSolarSystem->registerAnchor(mBillboard);
  mBillboardNode.reset(mSceneGraph->NewOpenGLNode(mSceneGraph->GetRoot(), mBillboard.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mBillboardNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));
  mSolarSystem->registerAnchor(mPoints);
  mPointsNode.reset(mSceneGraph->NewOpenGLNode(mSceneGraph->GetRoot(), mPoints.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mPointsNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  mBillboard->setEnabled(true);
  mPoints->setEnabled(false);

  // Init buffers for predictive rendering
  mFrameIntervals.resize(mFrameIntervalsLength);
  mCameraTransforms.resize(mCameraTransformsLength);

  logger().info("Loading plugin done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mSolarSystem->unregisterAnchor(mBillboard);
  mSolarSystem->unregisterAnchor(mPoints);
  mSceneGraph->GetRoot()->DisconnectChild(mBillboardNode.get());
  mSceneGraph->GetRoot()->DisconnectChild(mPointsNode.get());

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  glm::mat4 currentCameraTransform = mBillboard->getRelativeTransform(
      mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver());
  float     scale = (float)mBillboard->getRelativeScale(mSolarSystem->getObserver());
  glm::vec3 r     = cs::core::SolarSystem::getRadii(mBillboard->getCenterName());
  currentCameraTransform =
      glm::scale(currentCameraTransform, glm::vec3(1.f / scale, 1.f / scale, 1.f / scale));
  currentCameraTransform[3] *= glm::vec4(1 / r[0], 1 / r[1], 1 / r[2], 1);

  if (mRenderState == RenderState::eRenderingImage) {
    if (mFutureFrameData.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      std::tie(mRenderingFrame.mFrameData, mRenderingFrame.mModelViewProjection) =
          mFutureFrameData.get();
      mRenderedFrames.push_back(mRenderingFrame);

      displayFrame(mRenderingFrame);

      mRenderState                          = RenderState::eRequestImage;
      mFrameIntervals[mFrameIntervalsIndex] = mLastFrameInterval;
      if (++mFrameIntervalsIndex >= mFrameIntervalsLength) {
        mFrameIntervalsIndex = 0;
      }
    }
  }
  if (mRenderState == RenderState::eRequestImage && mPluginSettings.mRequestImages.get()) {
    requestFrame(currentCameraTransform);
  }

  if (mPluginSettings.mReuseImages.get() && mRenderedFrames.size() > 0) {
    tryReuseFrame(currentCameraTransform);
  }

  mLastFrameInterval++;
  mCameraTransforms[mCameraTransformsIndex] =
      currentCameraTransform * glm::inverse(mLastCameraTransform);
  mLastCameraTransform = currentCameraTransform;
  if (++mCameraTransformsIndex >= mCameraTransformsLength) {
    mCameraTransformsIndex = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::requestFrame(glm::mat4 cameraTransform) {
  cs::utils::FrameTimings::ScopedTimer timer("Request frame");

  if (mPluginSettings.mPredictiveRendering.get()) {
    glm::mat4 predictedCameraTransform = cameraTransform;
    int       meanFrameInterval =
        std::accumulate(mFrameIntervals.begin(), mFrameIntervals.end(), 0) / mFrameIntervalsLength;
    glm::vec3 predictedTranslation(0);
    glm::quat predictedRotation(1, 0, 0, 0);

    for (int i = 0; i < mCameraTransformsLength; i++) {
      predictedTranslation +=
          glm::vec3(mCameraTransforms[i][3].xyz) / (float)mCameraTransformsLength;
      predictedRotation =
          glm::slerp(predictedRotation, glm::quat_cast(mCameraTransforms[i]), 1.f / (i + 1));
    }

    for (int i = 0; i < meanFrameInterval; i++) {
      predictedCameraTransform = glm::mat4_cast(predictedRotation) * predictedCameraTransform;
      predictedCameraTransform = glm::translate(predictedCameraTransform, predictedTranslation);
    }

    mNextFrame.mCameraTransform = predictedCameraTransform;
  } else {
    mNextFrame.mCameraTransform = cameraTransform;
  }

  mNextFrame.mResolution   = mPluginSettings.mResolution.get();
  mNextFrame.mSamplingRate = mPluginSettings.mSamplingRate.get();
  mNextFrame.mDepthMode    = mPluginSettings.mDepthMode.get();
  mNextFrame.mDenoiseColor = mPluginSettings.mDenoiseColor.get();
  mNextFrame.mDenoiseDepth = mPluginSettings.mDenoiseDepth.get();
  mNextFrame.mShading      = mPluginSettings.mShading.get();

  if (!(mNextFrame == mRenderingFrame)) {
    mRenderingFrame    = mNextFrame;
    mFutureFrameData   = mRenderer->getFrame(mNextFrame.mResolution, cameraTransform,
        mRenderingFrame.mSamplingRate, mRenderingFrame.mDepthMode, mNextFrame.mDenoiseColor,
        mNextFrame.mDenoiseDepth, mNextFrame.mShading);
    mRenderState       = RenderState::eRenderingImage;
    mLastFrameInterval = 0;
  }
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
    double diff = diffTranslations(cameraTransform, f.mCameraTransform);
    if (diff < minDiff) {
      minDiff   = diff;
      bestFrame = f;
    }
  }
  if (minDiff > 0) {
    displayFrame(bestFrame);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::displayFrame(Frame& frame) {
  if (!(frame == mDisplayedFrame)) {
    cs::utils::FrameTimings::ScopedTimer timer("Display volume frame");

    std::vector<uint8_t> colorData(frame.mFrameData.begin(),
        frame.mFrameData.begin() + 4 * frame.mResolution * frame.mResolution);
    std::vector<float>   depthData(
        (float*)frame.mFrameData.data() + frame.mResolution * frame.mResolution,
        (float*)frame.mFrameData.data() + 2 * frame.mResolution * frame.mResolution);

    mBillboard->setTexture(colorData, frame.mResolution, frame.mResolution);
    mPoints->setTexture(colorData, frame.mResolution, frame.mResolution);
    mBillboard->setDepthTexture(depthData, frame.mResolution, frame.mResolution);
    mPoints->setDepthTexture(depthData, frame.mResolution, frame.mResolution);
    mBillboard->setTransform(glm::toMat4(glm::toQuat(frame.mCameraTransform)));
    mPoints->setTransform(glm::toMat4(glm::toQuat(frame.mCameraTransform)));

    mBillboard->setMVPMatrix(frame.mModelViewProjection);
    mPoints->setMVPMatrix(frame.mModelViewProjection);
    mDisplayedFrame = frame;
  }
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
