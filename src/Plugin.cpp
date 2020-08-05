////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "Display/Billboard.hpp"
#include "Render/OSPRayRenderer.hpp"
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
  return mResolution == other.mResolution && mCameraRotation == other.mCameraRotation &&
         mSamplingRate == other.mSamplingRate && mTransferFunction == other.mTransferFunction &&
         mHasDenoise == other.mHasDenoise && mDepthMode == other.mDepthMode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection =
      mAllSettings->onLoad().connect([this]() { logger().info("Settings loaded."); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-volume-rendering"] = mPluginSettings; });

  // Init volume renderer
  mRenderer = std::make_unique<OSPRayRenderer>();
  mRenderer->setData("C:/Users/frit_jn/Documents/GAIA_Data/PX_OUT_mars_14564", 0);
  mGettingFrame = false;

  // Add the volume rendering user interface components to the CosmoScout user interface.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Volume Rendering", "blur_circular", "../share/resources/gui/volume_rendering_settings.html");

  mGuiManager->addScriptToGuiFromJS("../share/resources/gui/js/csp-volume-rendering.js");

  mGuiManager->getGui()->registerCallback("volumeRendering.setResolution",
      "Sets the resolution of the rendered volume images.", std::function([this](double value) {
        mPluginSettings.mResolution = value;
        mRenderer->setResolution(value);
      }));
  mPluginSettings.mResolution.connectAndTouch(
      [this](int value) { mGuiManager->setSliderValue("volumeRendering.setResolution", value); });

  mGuiManager->getGui()->registerCallback("volumeRendering.setSamplingRate",
      "Sets the sampling rate for volume rendering.",
      std::function([this](double value) { mPluginSettings.mSamplingRate = value; }));
  mPluginSettings.mSamplingRate.connectAndTouch([this](float value) {
    mGuiManager->setSliderValue("volumeRendering.setSamplingRate", value);
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
      }));
  mPluginSettings.mDepthData.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDepthData", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDrawDepth",
      "Enables displaying the depth buffer instead of the color buffer.",
      std::function([this](bool enable) {
        mPluginSettings.mDrawDepth = enable;
        mBillboard->setDrawDepth(enable);
      }));
  mPluginSettings.mDrawDepth.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDrawDepth", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setEnableDenoise",
      "Enables use of OIDN for displaying data.",
      std::function([this](bool enable) { mPluginSettings.mDenoise = enable; }));
  mPluginSettings.mDenoise.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("volumeRendering.setEnableDenoise", enable);
  });

  mGuiManager->getGui()->registerCallback("volumeRendering.setTransferFunction",
      "Sets the transfer function for rendering the volume.",
      std::function([this](std::string json) {
        cs::graphics::ColorMap colorMap(json, false);
        mRenderer->setTransferFunction(colorMap.getRawData());
        mNextFrame.mTransferFunction = colorMap.getRawData();
        mRenderedFrames.clear();
      }));

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
  mPluginSettings.mDepthMode.connect([this](Renderer::DepthMode drawMode) {
    if (drawMode == Renderer::DepthMode::eNone) {
      mGuiManager->setRadioChecked("stars.setDrawMode0");
    } else if (drawMode == Renderer::DepthMode::eIsosurface) {
      mGuiManager->setRadioChecked("stars.setDrawMode1");
    } else if (drawMode == Renderer::DepthMode::eFirstHit) {
      mGuiManager->setRadioChecked("stars.setDrawMode2");
    } else if (drawMode == Renderer::DepthMode::eLastHit) {
      mGuiManager->setRadioChecked("stars.setDrawMode3");
    }
  });

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
      std::tie(mRenderingFrame.mFrameData, mRenderingFrame.mModelViewProjection) =
          mFutureFrameData.get();
      mRenderedFrames.push_back(mRenderingFrame);

      displayFrame(mRenderingFrame);

      mGettingFrame                         = false;
      mFrameIntervals[mFrameIntervalsIndex] = mLastFrameInterval;
      if (++mFrameIntervalsIndex >= mFrameIntervalsLength) {
        mFrameIntervalsIndex = 0;
      }
    }
  } else {
    if (mPluginSettings.mRequestImages.get()) {
      requestFrame(currentCameraRotation);
    }
  }

  if (mPluginSettings.mReuseImages.get() && mRenderedFrames.size() > 0) {
    tryReuseFrame(currentCameraRotation);
  }

  mLastFrameInterval++;
  mCameraRotations[mCameraRotationsIndex] =
      currentCameraRotation * glm::conjugate(mLastCameraRotation);
  mLastCameraRotation = currentCameraRotation;
  if (++mCameraRotationsIndex >= mCameraRotationsLength) {
    mCameraRotationsIndex = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::requestFrame(glm::dquat cameraRotation) {
  cs::utils::FrameTimings::ScopedTimer timer("Request frame");

  if (mPluginSettings.mPredictiveRendering.get()) {
    glm::dquat predictedCameraRotation = cameraRotation;
    for (int i = 0; i < std::accumulate(mFrameIntervals.begin(), mFrameIntervals.end(), 0) /
                            mFrameIntervalsLength;
         i++) {
      predictedCameraRotation = std::accumulate(mCameraRotations.begin(), mCameraRotations.end(),
                                    glm::dquat(0, 0, 0, 0)) /
                                (double)mCameraRotationsLength * predictedCameraRotation;
    }

    mNextFrame.mCameraRotation = predictedCameraRotation;
  } else {
    mNextFrame.mCameraRotation = cameraRotation;
  }

  mNextFrame.mResolution   = mPluginSettings.mResolution.get();
  mNextFrame.mSamplingRate = mPluginSettings.mSamplingRate.get();
  mNextFrame.mDepthMode    = mPluginSettings.mDepthMode.get();
  mNextFrame.mHasDenoise   = mPluginSettings.mDenoise.get();

  if (!(mNextFrame == mRenderingFrame)) {
    mRenderingFrame    = mNextFrame;
    mFutureFrameData   = mRenderer->getFrame(glm::toMat4(mRenderingFrame.mCameraRotation),
        mRenderingFrame.mSamplingRate, mRenderingFrame.mDepthMode, mNextFrame.mHasDenoise);
    mGettingFrame      = true;
    mLastFrameInterval = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::tryReuseFrame(glm::dquat cameraRotation) {
  cs::utils::FrameTimings::ScopedTimer timer("Try reuse frame");

  double minDiff   = 0;
  Frame& bestFrame = mRenderedFrames[0];
  for (Frame& f : mRenderedFrames) {
    double diff = glm::extractRealComponent(f.mCameraRotation * glm::conjugate(cameraRotation));
    if (diff < minDiff) {
      minDiff   = diff;
      bestFrame = f;
    }
  }
  if (glm::extractRealComponent(
          mDisplayedFrame.mCameraRotation * glm::conjugate(bestFrame.mCameraRotation)) > -0.99) {
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
    mBillboard->setDepthTexture(depthData, frame.mResolution, frame.mResolution);
    mBillboard->setTransform(glm::toMat4(frame.mCameraRotation));
    mBillboard->setMVPMatrix(frame.mModelViewProjection);
    mDisplayedFrame = frame;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
