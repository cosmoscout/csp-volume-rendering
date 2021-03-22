////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Connection.hpp"

#include "../../Enums.hpp"
#include "../../logger.hpp"

#include <sstream>

namespace csp::volumerendering::webrtc {

////////////////////////////////////////////////////////////////////////////////////////////////////

Connection::Connection(std::string signallingUrl)
    : mSignallingServer(std::make_unique<SignallingServer>(std::move(signallingUrl))) {
  mSignallingServer->onConnected().connect([this]() {
    mState = PeerCallState::eNegotiating;
    // Start negotiation (exchange SDP and ICE candidates)
    if (!createWebrtcbin()) {
      mState = PeerCallState::eError;
      logger().error("ERROR: failed to start pipeline");
    }
  });
  mSignallingServer->onSdpReceived().connect([this](std::string type, std::string text) {
    GstSDPMessage* sdp;
    int            ret = gst_sdp_message_new(&sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);
    ret = gst_sdp_message_parse_buffer((guint8*)text.data(), static_cast<guint>(text.size()), sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);

    if (type == "answer") {
      logger().trace("Received answer.");
      onAnswerReceived(sdp);
    } else {
      logger().trace("Received offer.");
      onOfferReceived(sdp);
    }
  });
  mSignallingServer->onIceReceived().connect([this](std::string text, guint64 spdmLineIndex) {
    // Add ice candidate sent by remote peer
    g_signal_emit_by_name(mWebrtcBin.get(), "add-ice-candidate", spdmLineIndex, text.c_str());
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Connection::~Connection() {
  mSignallingServer.reset();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<GstPointer<GstElement> const&> const& Connection::onWebrtcbinCreated() const {
  return mOnWebrtcbinCreated;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<std::shared_ptr<DataChannel>> const& Connection::onDataChannelConnected() const {
  return mOnDataChannelConnected;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<std::shared_ptr<GstPad>> const& Connection::onVideoStreamConnected() const {
  return mOnVideoStreamConnected;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onOfferSet(GstPromise* promisePtr, Connection* pThis) {
  GstPointer<GstPromise> promise(promisePtr);
  promise.reset(gst_promise_new_with_change_func(
      reinterpret_cast<GstPromiseChangeFunc>(&Connection::onAnswerCreated), pThis, NULL));
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-answer", NULL, promise.release());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onAnswerCreated(GstPromise* promisePtr, Connection* pThis) {
  GstPointer<GstPromise>       promise(promisePtr);
  GstWebRTCSessionDescription* answer = NULL;
  const GstStructure*          reply;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise.get()), ==, GST_PROMISE_RESULT_REPLIED);
  reply = gst_promise_get_reply(promise.get());
  gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, NULL);

  promise.reset(gst_promise_new());
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", answer, promise.get());
  gst_promise_interrupt(promise.get());

  pThis->sendSdpToPeer(answer);
  gst_webrtc_session_description_free(answer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onOfferCreated(GstPromise* promisePtr, Connection* pThis) {
  GstPointer<GstPromise>       promise(promisePtr);
  GstWebRTCSessionDescription* offer = NULL;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise.get()), ==, GST_PROMISE_RESULT_REPLIED);
  const GstStructure* reply = gst_promise_get_reply(promise.get());
  gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);

  promise.reset(gst_promise_new());
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", offer, promise.get());
  gst_promise_interrupt(promise.get());

  // Send offer to peer
  pThis->sendSdpToPeer(offer);
  gst_webrtc_session_description_free(offer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onNegotiationNeeded(GstElement* element, Connection* pThis) {
  pThis->mState = PeerCallState::eNegotiating;

  if (pThis->mCreateOffer) {
    GstPointer<GstPromise> promise(gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Connection::onOfferCreated), pThis, NULL));
    g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-offer", NULL, promise.release());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onIceCandidate(
    GstElement* webrtc, guint mlineindex, gchar* candidate, Connection* pThis) {
  if (pThis->mState < PeerCallState::eNegotiating) {
    pThis->mState = PeerCallState::eError;
    logger().error("Can't send ICE, not in call");
    return;
  }

  pThis->mSignallingServer->sendIce(mlineindex, candidate);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onIceGatheringStateNotify(
    GstElement* webrtcbin, GParamSpec* pspec, Connection* pThis) {
  GstWebRTCICEGatheringState ice_gather_state;
  const gchar*               new_state = "unknown";

  g_object_get(webrtcbin, "ice-gathering-state", &ice_gather_state, NULL);
  switch (ice_gather_state) {
  case GST_WEBRTC_ICE_GATHERING_STATE_NEW:
    new_state = "new";
    break;
  case GST_WEBRTC_ICE_GATHERING_STATE_GATHERING:
    new_state = "gathering";
    break;
  case GST_WEBRTC_ICE_GATHERING_STATE_COMPLETE:
    new_state = "complete";
    break;
  }
  logger().trace("ICE gathering state changed to {}.", new_state);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onDataChannel(
    GstElement* webrtc, GstWebRTCDataChannel* data_channel, Connection* pThis) {
  auto dc = std::make_shared<DataChannel>(data_channel);
  pThis->mOnDataChannelConnected.emit(dc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onIncomingStream(GstElement* webrtc, GstPad* pad, Connection* pThis) {
  if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
    return;

  GstPointer<GstPad> padPtr(GST_PAD(gst_object_ref(pad)));
  pThis->mOnVideoStreamConnected.emit(std::move(padPtr));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onOfferReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* offer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  g_assert_nonnull(offer);

  // Set remote description on our pipeline
  {
    GstPointer<GstPromise> promise(gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Connection::onOfferSet), this, NULL));
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", offer, promise.release());
    gst_webrtc_session_description_free(offer);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::onAnswerReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* answer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
  g_assert_nonnull(answer);

  // Set remote description on our pipeline
  {
    GstPointer<GstPromise> promise(gst_promise_new());
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", answer, promise.get());
    gst_promise_interrupt(promise.get());
  }
  mState = PeerCallState::eStarted;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Connection::sendSdpToPeer(GstWebRTCSessionDescription* desc) {
  if (mState < PeerCallState::eNegotiating) {
    mState = PeerCallState::eError;
    logger().error("Can't send SDP to peer, not in call");
    return;
  }

  mSignallingServer->sendSdp(desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Connection::createWebrtcbin() {
  mWebrtcBin.reset(
      GST_ELEMENT(gst_object_ref_sink(gst_element_factory_make("webrtcbin", "sendrecv"))));
  g_assert_nonnull(mWebrtcBin.get());
  g_object_set(mWebrtcBin.get(), "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);
  g_object_set(mWebrtcBin.get(), "stun-server", "stun://stun.l.google.com:19302", NULL);

  for (int i = 0; i < 2; i++) {
    GstPointer<GstCaps> caps(
        gst_caps_from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96"));
    GstWebRTCRTPTransceiver* transceiverPtr = NULL;
    g_signal_emit_by_name(mWebrtcBin.get(), "add-transceiver",
        GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, caps.get(), &transceiverPtr);
    GstPointer<GstWebRTCRTPTransceiver> transceiver(transceiverPtr);
  }

  // This is the gstwebrtc entry point where we create the offer and so on.
  // It will be called when the pipeline goes to PLAYING.
  g_signal_connect(
      mWebrtcBin.get(), "on-negotiation-needed", G_CALLBACK(Connection::onNegotiationNeeded), this);
  // We need to transmit this ICE candidate to the browser via the websockets
  // signalling server. Incoming ice candidates from the browser need to be
  // added by us too, see on_server_message()
  g_signal_connect(
      mWebrtcBin.get(), "on-ice-candidate", G_CALLBACK(Connection::onIceCandidate), this);
  g_signal_connect(mWebrtcBin.get(), "notify::ice-gathering-state",
      G_CALLBACK(Connection::onIceGatheringStateNotify), this);

  g_signal_connect(
      mWebrtcBin.get(), "on-data-channel", G_CALLBACK(Connection::onDataChannel), this);
  // Incoming streams will be exposed via this signal
  g_signal_connect(mWebrtcBin.get(), "pad-added", G_CALLBACK(Connection::onIncomingStream), this);

  mOnWebrtcbinCreated.emit(mWebrtcBin);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
