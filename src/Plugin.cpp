////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Billboard.hpp"
#include "OSPRayRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-graphics/ColorMap.hpp"

#include <VistaKernel/GraphicsManager/VistaGroupNode.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

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

void from_json(nlohmann::json const& j, Plugin::Settings::Volume& o) {
  cs::core::Settings::deserialize(j, "path", o.mPath);
}

void to_json(nlohmann::json& j, Plugin::Settings::Volume const& o) {
  cs::core::Settings::serialize(j, "path", o.mPath);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::deserialize(j, "resolution", o.mResolution);
  cs::core::Settings::deserialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::deserialize(j, "volumes", o.mVolumes);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::serialize(j, "resolution", o.mResolution);
  cs::core::Settings::serialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::serialize(j, "volumes", o.mVolumes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Plugin::Frame::operator==(const Frame& other) {
  return mResolution == other.mResolution && mCameraTransform == other.mCameraTransform &&
         mSamplingRate == other.mSamplingRate;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection =
      mAllSettings->onLoad().connect([this]() { logger().info("Settings loaded."); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-volume-rendering"] = mPluginSettings; });

  // Add the stars user interface components to the CosmoScout user interface.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Volume Rendering", "blur_circular", "../share/resources/gui/volume_rendering_settings.html");

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

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnablePredictiveRendering",
      "Enables predicting the next camera position for rendering.",
      std::function([this](bool enable) { mPluginSettings.mPredictiveRendering = enable; }));
  mPluginSettings.mPredictiveRendering.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnablePredictiveRendering", enable);
  });

  // Init volume renderer
  mRenderer = std::make_unique<OSPRayRenderer>();
  mRenderer->setData("", 0);
  mGettingFrame = false;

  cs::graphics::ColorMap colorMap("../share/resources/transferfunctions/Volume.json");
  mRenderer->setTransferFunction(colorMap.getRawData());

  // Init volume representation
  auto anchor                           = mAllSettings->mAnchors.find("Mars");
  auto [tStartExistence, tEndExistence] = anchor->second.getExistence();
  mBillboard = std::make_shared<Billboard>(anchor->second.mCenter, anchor->second.mFrame,
      tStartExistence, tEndExistence, cs::core::SolarSystem::getRadii(anchor->second.mCenter));

  // Add volume representation to solar system and scene graph
  mSolarSystem->registerAnchor(mBillboard);
  mVolumeNode.reset(mSceneGraph->NewOpenGLNode(mSceneGraph->GetRoot(), mBillboard.get()));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mVolumeNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  // Init buffers for predictive rendering
  mFrameIntervals.resize(mFrameIntervalsLength);
  mCameraRotations.resize(mCameraRotationsLength);

  logger().info("Loading plugin done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mSolarSystem->unregisterAnchor(mBillboard);
  mSceneGraph->GetRoot()->DisconnectChild(mVolumeNode.get());

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
  glm::dquat currentCameraRotation = mBillboard->getRelativeRotation(
      mTimeControl->pSimulationTime.get(), mSolarSystem->getObserver());

  if (mGettingFrame) {
    if (mFutureFrameData.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      mCurrentFrame.mFrameData = mFutureFrameData.get();

      mBillboard->setTexture(
          mCurrentFrame.mFrameData, mCurrentFrame.mResolution, mCurrentFrame.mResolution);
      mBillboard->setTransform(mCurrentFrame.mCameraTransform);

      mGettingFrame                         = false;
      mFrameIntervals[mFrameIntervalsIndex] = mLastFrameInterval;
      if (++mFrameIntervalsIndex >= mFrameIntervalsLength) {
        mFrameIntervalsIndex = 0;
      }
    }
  } else {
    Frame newFrame;

    if (mPluginSettings.mPredictiveRendering.get()) {
      glm::dquat predictedCameraRotation = currentCameraRotation;
      for (int i = 0; i < std::accumulate(mFrameIntervals.begin(), mFrameIntervals.end(), 0) /
                              mFrameIntervalsLength;
           i++) {
        predictedCameraRotation = std::accumulate(mCameraRotations.begin(), mCameraRotations.end(),
                                     glm::dquat(0, 0, 0, 0)) /
                                  (double)mCameraRotationsLength * predictedCameraRotation;
      }

      newFrame.mCameraTransform = glm::toMat4(predictedCameraRotation);
    } else {
      newFrame.mCameraTransform = glm::toMat4(currentCameraRotation);
    }

    newFrame.mResolution   = mPluginSettings.mResolution.get();
    newFrame.mSamplingRate = mPluginSettings.mSamplingRate.get();

    if (!(newFrame == mCurrentFrame)) {
      mCurrentFrame    = newFrame;
      mFutureFrameData = mRenderer->getFrame(
          mCurrentFrame.mCameraTransform, mCurrentFrame.mResolution, mCurrentFrame.mSamplingRate);
      mGettingFrame      = true;
      mLastFrameInterval = 0;
    }
  }

  mLastFrameInterval++;
  mCameraRotations[mCameraRotationsIndex] = currentCameraRotation * glm::inverse(mLastCameraRotation);
  mLastCameraRotation                     = currentCameraRotation;
  if (++mCameraRotationsIndex >= mCameraRotationsLength) {
    mCameraRotationsIndex = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
