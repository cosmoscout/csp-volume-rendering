////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Stream.hpp"

#include "../../logger.hpp"

#include <json-glib/json-glib.h>

#include <sstream>

namespace {

static gchar* get_string_from_json_object(JsonObject* object) {
  JsonNode*      root;
  JsonGenerator* generator;
  gchar*         text;

  // Make it the root node
  root      = json_node_init_object(json_node_alloc(), object);
  generator = json_generator_new();
  json_generator_set_root(generator, root);
  text = json_generator_to_data(generator, NULL);

  // Release everything
  g_object_unref(generator);
  json_node_free(root);
  return text;
}

gboolean check_plugins() {
  guint        i;
  gboolean     ret;
  GstPlugin*   plugin;
  GstRegistry* registry;
  const gchar* needed[] = {"opus", "vpx", "nice", "webrtc", "dtls", "srtp", "rtpmanager",
      "videoconvert", "coreelements", "app", "videoscale", NULL};

  registry = gst_registry_get();
  ret      = TRUE;
  for (i = 0; i < g_strv_length((gchar**)needed); i++) {
    plugin = gst_registry_find_plugin(registry, needed[i]);
    if (!plugin) {
      csp::volumerendering::logger().error("Required gstreamer plugin '{}' not found!", needed[i]);
      ret = FALSE;
      continue;
    }
    gst_object_unref(plugin);
  }
  return ret;
}

} // namespace

namespace csp::volumerendering::webrtc {

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::Stream()
    : mSignallingServer("wss://webrtc.nirbheek.in:8443", "1234") {
  GError* error = NULL;

  if (!gst_init_check(nullptr, nullptr, &error) || !check_plugins()) {
    throw std::runtime_error("Could not initialize GStreamer");
  }

  mSignallingServer.onPeerConnected().connect([this]() {
    mState = PeerCallState::eNegotiating;
    // Start negotiation (exchange SDP and ICE candidates)
    if (!startPipeline()) {
      mState = PeerCallState::eError;
      logger().error("ERROR: failed to start pipeline");
    }
  });
  mSignallingServer.onSdpReceived().connect([this](std::string type, std::string text) {
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
  mSignallingServer.onIceReceived().connect([this](std::string text, guint64 spdmLineIndex) {
    // Add ice candidate sent by remote peer
    g_signal_emit_by_name(mWebrtcBin.get(), "add-ice-candidate", spdmLineIndex, text.c_str());
  });

  mMainLoop = std::unique_ptr<GMainLoop, std::function<void(GMainLoop*)>>(
      g_main_loop_new(NULL, FALSE), [](GMainLoop* loop) {
        g_main_loop_quit(loop);
        g_main_loop_unref(loop);
      });

  mMainLoopThread = std::thread([this]() { g_main_loop_run(mMainLoop.get()); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::~Stream() {
  mMainLoop.reset();
  mPipeline.reset();
  mMainLoopThread.join();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::sendMessage(std::string const& message) {
  if (mSendChannel) {
    mSendChannel->send(message);
  } else if (mReceiveChannel) {
    mReceiveChannel->send(message);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::vector<uint8_t>> Stream::getSample(int resolution) {
  if (!mAppSink) {
    return {};
  }
  GstSample* sample = NULL;
  if (resolution != mResolution) {
    mResolution = resolution;
    std::stringstream capsStr;
    capsStr << "video/x-raw,format=RGBA,width=" << mResolution << ",height=" << mResolution;
    GstCaps* caps = gst_caps_from_string(capsStr.str().c_str());
    g_object_set(mCapsFilter.get(), "caps", caps, NULL);
    gst_caps_unref(caps);

    // Drop samples with incorrect resolution
    GstCaps* sampleCaps;
    do {
      if (sample) {
        gst_sample_unref(sample);
      }
      g_signal_emit_by_name(mAppSink.get(), "pull-sample", &sample, NULL);
      sampleCaps = gst_sample_get_caps(sample);
    } while (!gst_caps_is_subset(sampleCaps, caps));
  } else {
    g_signal_emit_by_name(mAppSink.get(), "pull-sample", &sample, NULL);
  }
  GstBuffer* buf = gst_sample_get_buffer(sample);
  gst_sample_unref(sample);
  if (!buf) {
    return {};
  }

  std::vector<uint8_t> image(resolution * resolution * 4);
  gst_buffer_extract(buf, 0, image.data(), resolution * resolution * 4);
  return image;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferSet(GstPromise* promise, Stream* pThis) {
  gst_promise_unref(promise);
  promise = gst_promise_new_with_change_func(
      reinterpret_cast<GstPromiseChangeFunc>(&Stream::onAnswerCreated), pThis, NULL);
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-answer", NULL, promise);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onAnswerCreated(GstPromise* promise, Stream* pThis) {
  GstWebRTCSessionDescription* answer = NULL;
  const GstStructure*          reply;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise), ==, GST_PROMISE_RESULT_REPLIED);
  reply = gst_promise_get_reply(promise);
  gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, NULL);
  gst_promise_unref(promise);

  promise = gst_promise_new();
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", answer, promise);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  pThis->sendSdpToPeer(answer);
  gst_webrtc_session_description_free(answer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferCreated(GstPromise* promise, Stream* pThis) {
  GstWebRTCSessionDescription* offer = NULL;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise), ==, GST_PROMISE_RESULT_REPLIED);
  const GstStructure* reply = gst_promise_get_reply(promise);
  gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
  gst_promise_unref(promise);

  promise = gst_promise_new();
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", offer, promise);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  // Send offer to peer
  pThis->sendSdpToPeer(offer);
  gst_webrtc_session_description_free(offer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onNegotiationNeeded(GstElement* element, Stream* pThis) {
  pThis->mState = PeerCallState::eNegotiating;

  if (pThis->mCreateOffer) {
    GstPromise* promise = gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Stream::onOfferCreated), pThis, NULL);
    g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-offer", NULL, promise);
  } else {
    pThis->mSignallingServer.send("OFFER_REQUEST");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIceCandidate(GstElement* webrtc, guint mlineindex, gchar* candidate, Stream* pThis) {
  gchar*      text;
  JsonObject *ice, *msg;

  if (pThis->mState < PeerCallState::eNegotiating) {
    pThis->mState = PeerCallState::eError;
    logger().error("Can't send ICE, not in call");
    return;
  }

  ice = json_object_new();
  json_object_set_string_member(ice, "candidate", candidate);
  json_object_set_int_member(ice, "sdpMLineIndex", mlineindex);
  msg = json_object_new();
  json_object_set_object_member(msg, "ice", ice);
  text = get_string_from_json_object(msg);
  json_object_unref(msg);

  pThis->mSignallingServer.send(text);
  g_free(text);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIceGatheringStateNotify(GstElement* webrtcbin, GParamSpec* pspec, Stream* pThis) {
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

void Stream::onDataChannel(GstElement* webrtc, GObject* data_channel, Stream* pThis) {
  pThis->mReceiveChannel = std::make_unique<DataChannel>(data_channel);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIncomingStream(GstElement* webrtc, GstPad* pad, Stream* pThis) {
  GstElement* decodebin;
  GstPad*     sinkpad;

  if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
    return;

  decodebin = gst_element_factory_make("decodebin", NULL);
  g_signal_connect(decodebin, "pad-added", G_CALLBACK(Stream::onIncomingDecodebinStream), pThis);
  gst_bin_add(GST_BIN(pThis->mPipeline.get()), decodebin);
  gst_element_sync_state_with_parent(decodebin);

  sinkpad = gst_element_get_static_pad(decodebin, "sink");
  gst_pad_link(pad, sinkpad);
  gst_object_unref(sinkpad);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, Stream* pThis) {
  GstCaps*     caps;
  const gchar* name;

  if (!gst_pad_has_current_caps(pad)) {
    logger().warn("Pad '{}' has no caps, can't do anything, ignoring!", GST_PAD_NAME(pad));
    return;
  }

  caps = gst_pad_get_current_caps(pad);
  name = gst_structure_get_name(gst_caps_get_structure(caps, 0));

  if (g_str_has_prefix(name, "video")) {
    pThis->handleVideoStream(pad);
  } else {
    logger().warn("Unknown pad '{}', ignoring!", GST_PAD_NAME(pad));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* offer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  g_assert_nonnull(offer);

  // Set remote description on our pipeline
  {
    GstPromise* promise = gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Stream::onOfferSet), this, NULL);
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", offer, promise);
  }
  gst_webrtc_session_description_free(offer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onAnswerReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* answer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
  g_assert_nonnull(answer);

  // Set remote description on our pipeline
  {
    GstPromise* promise = gst_promise_new();
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", answer, promise);
    gst_promise_interrupt(promise);
    gst_promise_unref(promise);
  }
  mState = PeerCallState::eStarted;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::sendSdpToPeer(GstWebRTCSessionDescription* desc) {
  gchar*      text;
  JsonObject *msg, *sdp;

  if (mState < PeerCallState::eNegotiating) {
    mState = PeerCallState::eError;
    logger().error("Can't send SDP to peer, not in call");
    return;
  }

  text = gst_sdp_message_as_text(desc->sdp);
  sdp  = json_object_new();

  if (desc->type == GST_WEBRTC_SDP_TYPE_OFFER) {
    logger().trace("Sending offer.");
    json_object_set_string_member(sdp, "type", "offer");
  } else if (desc->type == GST_WEBRTC_SDP_TYPE_ANSWER) {
    logger().trace("Sending answer.");
    json_object_set_string_member(sdp, "type", "answer");
  } else {
    g_assert_not_reached();
  }

  json_object_set_string_member(sdp, "sdp", text);
  g_free(text);

  msg = json_object_new();
  json_object_set_object_member(msg, "sdp", sdp);
  text = get_string_from_json_object(msg);
  json_object_unref(msg);

  mSignallingServer.send(text);
  g_free(text);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::handleVideoStream(GstPad* pad) {
  GError* error = NULL;

  logger().trace("Trying to handle video stream.");

  GstElement* bin = gst_parse_bin_from_description(
      "queue ! videoconvert ! videoscale add-borders=false ! capsfilter "
      "caps=video/x-raw,format=RGBA,width=512,height=512 name=capsfilter ! "
      "appsink drop=true max-buffers=1 name=framecapture",
      TRUE, &error);
  if (error) {
    logger().error("Failed to parse launch: {}!", error->message);
    g_error_free(error);
    return;
  }

  mAppSink = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_bin_get_by_name(GST_BIN(bin), "framecapture"),
      [](GstElement* element) { gst_object_unref(element); });
  mCapsFilter = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_bin_get_by_name(GST_BIN(bin), "capsfilter"),
      [](GstElement* element) { gst_object_unref(element); });

  gst_bin_add_many(GST_BIN(mPipeline.get()), bin, NULL);
  gst_element_sync_state_with_parent(bin);

  GstPad*          binpad = gst_element_get_static_pad(bin, "sink");
  GstPadLinkReturn ret    = gst_pad_link(pad, binpad);
  g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean Stream::startPipeline() {
  mPipeline = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_pipeline_new("pipeline"), [](GstElement* pipeline) {
        gst_element_set_state(GST_ELEMENT(pipeline), GST_STATE_NULL);
        logger().trace("Pipeline stopped.");
        gst_object_unref(pipeline);
      });
  g_assert_nonnull(mPipeline.get());

  mWebrtcBin = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_element_factory_make("webrtcbin", "sendrecv"), [](GstElement*) {});
  g_assert_nonnull(mWebrtcBin.get());
  g_object_set(mWebrtcBin.get(), "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);
  g_object_set(mWebrtcBin.get(), "stun-server", "stun://stun.l.google.com:19302", NULL);

  gst_bin_add_many(GST_BIN(mPipeline.get()), mWebrtcBin.get(), NULL);

  gst_element_sync_state_with_parent(mWebrtcBin.get());

  GstWebRTCRTPTransceiver* transceiver;
  GstCaps*                 caps =
      gst_caps_from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96");
  g_signal_emit_by_name(mWebrtcBin.get(), "add-transceiver",
      GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, caps, &transceiver);
  gst_caps_unref(caps);
  gst_object_unref(transceiver);

  // This is the gstwebrtc entry point where we create the offer and so on.
  // It will be called when the pipeline goes to PLAYING.
  g_signal_connect(
      mWebrtcBin.get(), "on-negotiation-needed", G_CALLBACK(Stream::onNegotiationNeeded), this);
  // We need to transmit this ICE candidate to the browser via the websockets
  // signalling server. Incoming ice candidates from the browser need to be
  // added by us too, see on_server_message()
  g_signal_connect(mWebrtcBin.get(), "on-ice-candidate", G_CALLBACK(Stream::onIceCandidate), this);
  g_signal_connect(mWebrtcBin.get(), "notify::ice-gathering-state",
      G_CALLBACK(Stream::onIceGatheringStateNotify), this);

  gst_element_set_state(mPipeline.get(), GST_STATE_READY);

  try {
    mSendChannel = std::make_unique<DataChannel>(mWebrtcBin.get());
  } catch (std::exception const& e) { logger().warn(e.what()); }

  g_signal_connect(mWebrtcBin.get(), "on-data-channel", G_CALLBACK(Stream::onDataChannel), this);
  // Incoming streams will be exposed via this signal
  g_signal_connect(mWebrtcBin.get(), "pad-added", G_CALLBACK(Stream::onIncomingStream), this);

  logger().trace("Starting pipeline.");
  GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(mPipeline.get()), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    // TODO
  }

  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
