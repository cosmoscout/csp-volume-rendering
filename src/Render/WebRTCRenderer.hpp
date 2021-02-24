////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
#define CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP

#include "Renderer.hpp"

#include "../../../../src/cs-core/GuiManager.hpp"
#include "../../../../src/cs-utils/Signal.hpp"

#include <gst/gst.h>
#include <gst/sdp/sdp.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#include <libsoup/soup.h>

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

class SignallingServer {
 public:
  SignallingServer(std::string const& url, std::string peerId);
  ~SignallingServer();

  void send(std::string const& text);

  cs::utils::Signal<> const&                         onPeerConnected() const;
  cs::utils::Signal<std::string, std::string> const& onSdpReceived() const;
  cs::utils::Signal<std::string, gint64> const&      onIceReceived() const;

 private:
  static void onServerConnected(SoupSession* session, GAsyncResult* res, SignallingServer* pThis);
  static void onServerClosed(SoupSession* session, SignallingServer* pThis);
  static void onServerMessage(SoupWebsocketConnection* conn, SoupWebsocketDataType type,
      GBytes* message, SignallingServer* pThis);

  gboolean registerWithServer();
  gboolean setupCall();

  std::unique_ptr<SoupWebsocketConnection> wsConnection;

  std::string mOurId;
  std::string mPeerId;

  cs::utils::Signal<>                         mOnPeerConnected;
  cs::utils::Signal<std::string, std::string> mOnSdpReceived;
  cs::utils::Signal<std::string, gint64>      mOnIceReceived;
};

class WebRTCStream {
 public:
  WebRTCStream();
  ~WebRTCStream();

  std::optional<std::vector<uint8_t>> getSample(int resolution);

 private:
  static void onOfferSet(GstPromise* promise, WebRTCStream* pThis);
  static void onAnswerCreated(GstPromise* promise, WebRTCStream* pThis);
  static void onOfferCreated(GstPromise* promise, WebRTCStream* pThis);
  static void onNegotiationNeeded(GstElement* element, WebRTCStream* pThis);
  static void onIceCandidate(
      GstElement* webrtc, guint mlineindex, gchar* candidate, WebRTCStream* pThis);
  static void onIceGatheringStateNotify(
      GstElement* webrtcbin, GParamSpec* pspec, WebRTCStream* pThis);

  static void onIncomingStream(GstElement* webrtc, GstPad* pad, WebRTCStream* pThis);
  static void onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, WebRTCStream* pThis);

  void onOfferReceived(GstSDPMessage* sdp);
  void onAnswerReceived(GstSDPMessage* sdp);
  void sendSdpToPeer(GstWebRTCSessionDescription* desc);
  void handleVideoStream(GstPad* pad);

  gboolean startPipeline();

  bool mCreateOffer = true;

  SignallingServer mSignallingServer;

  std::thread mMainLoop;

  std::unique_ptr<GstElement, std::function<void(GstElement*)>> mAppSink;
};

class WebRTCRenderer : public Renderer {
 public:
  WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
      VolumeShape shape, std::shared_ptr<cs::core::GuiManager> guiManager);
  ~WebRTCRenderer();

  float getProgress() override;
  void  preloadData(DataManager::State state) override;
  void  cancelRendering() override;

 private:
  RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) override;

  glm::mat4 getOSPRayMVP(float volumeHeight, glm::mat4 observerTransform);

  WebRTCStream mStream;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_WEBRTCRENDERER_HPP
