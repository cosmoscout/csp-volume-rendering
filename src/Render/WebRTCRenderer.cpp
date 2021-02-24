////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebRTCRenderer.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/convert.hpp"

#include <json-glib/json-glib.h>

#include <glm/gtc/matrix_transform.hpp>

#include <vector>

namespace {

/*
 * Author: Nirbheek Chauhan <nirbheek@centricular.com>
 */

enum AppState {
  APP_STATE_UNKNOWN = 0,
  APP_STATE_ERROR   = 1, /* generic error */
  SERVER_CONNECTING = 1000,
  SERVER_CONNECTION_ERROR,
  SERVER_CONNECTED, /* Ready to register */
  SERVER_REGISTERING = 2000,
  SERVER_REGISTRATION_ERROR,
  SERVER_REGISTERED, /* Ready to call a peer */
  SERVER_CLOSED,     /* server connection closed by us or the server */
  PEER_CONNECTING = 3000,
  PEER_CONNECTION_ERROR,
  PEER_CONNECTED,
  PEER_CALL_NEGOTIATING = 4000,
  PEER_CALL_STARTED,
  PEER_CALL_STOPPING,
  PEER_CALL_STOPPED,
  PEER_CALL_ERROR,
};

static GObject *send_channel, *receive_channel;

static enum AppState app_state = static_cast<AppState>(0);

#define RTP_CAPS_OPUS "application/x-rtp,media=audio,encoding-name=OPUS,payload="
#define RTP_CAPS_VP8 "application/x-rtp,media=video,encoding-name=VP8,payload="

static gboolean cleanup_and_quit_loop(const gchar* msg, enum AppState state) {
  if (msg)
    gst_printerr("%s\n", msg);
  if (state > 0)
    app_state = state;

  /* To allow usage as a GSourceFunc */
  return G_SOURCE_REMOVE;
}

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

static void data_channel_on_error(GObject* dc, gpointer user_data) {
  cleanup_and_quit_loop("Data channel error", static_cast<AppState>(0));
}

static void data_channel_on_open(GObject* dc, gpointer user_data) {
  GBytes* bytes = g_bytes_new("data", strlen("data"));
  gst_print("data channel opened\n");
  g_signal_emit_by_name(dc, "send-string", "Hi! from GStreamer");
  g_signal_emit_by_name(dc, "send-data", bytes);
  g_bytes_unref(bytes);
}

static void data_channel_on_close(GObject* dc, gpointer user_data) {
  cleanup_and_quit_loop("Data channel closed", static_cast<AppState>(0));
}

static void data_channel_on_message_string(GObject* dc, gchar* str, gpointer user_data) {
  gst_print("Received data channel message: %s\n", str);
}

static void connect_data_channel_signals(GObject* data_channel) {
  g_signal_connect(data_channel, "on-error", G_CALLBACK(data_channel_on_error), NULL);
  g_signal_connect(data_channel, "on-open", G_CALLBACK(data_channel_on_open), NULL);
  g_signal_connect(data_channel, "on-close", G_CALLBACK(data_channel_on_close), NULL);
  g_signal_connect(
      data_channel, "on-message-string", G_CALLBACK(data_channel_on_message_string), NULL);
}

static void on_data_channel(GstElement* webrtc, GObject* data_channel, gpointer user_data) {
  connect_data_channel_signals(data_channel);
  receive_channel = data_channel;
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

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

SignallingServer::SignallingServer(std::string const& url, std::string peerId)
    : mOurId(std::to_string(g_random_int_range(10, 10000)))
    , mPeerId(std::move(peerId)) {
  SoupLogger*  soupLogger;
  SoupMessage* message;
  SoupSession* session;
  const char*  https_aliases[] = {"wss", NULL};

  session = soup_session_new_with_options(SOUP_SESSION_SSL_STRICT, false,
      SOUP_SESSION_SSL_USE_SYSTEM_CA_FILE, TRUE, SOUP_SESSION_HTTPS_ALIASES, https_aliases, NULL);

  soupLogger = soup_logger_new(SOUP_LOGGER_LOG_BODY, -1);
  soup_session_add_feature(session, SOUP_SESSION_FEATURE(soupLogger));
  g_object_unref(soupLogger);

  message = soup_message_new(SOUP_METHOD_GET, url.c_str());

  logger().trace("Connecting to signalling server...");

  // Once connected, we will register
  soup_session_websocket_connect_async(session, message, NULL, NULL, NULL,
      reinterpret_cast<GAsyncReadyCallback>(&SignallingServer::onServerConnected), this);
  app_state = SERVER_CONNECTING;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SignallingServer::~SignallingServer() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::send(std::string const& text) {
  soup_websocket_connection_send_text(wsConnection.get(), text.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<> const& SignallingServer::onPeerConnected() const {
  return mOnPeerConnected;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<std::string, std::string> const& SignallingServer::onSdpReceived() const {
  return mOnSdpReceived;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<std::string, gint64> const& SignallingServer::onIceReceived() const {
  return mOnIceReceived;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerConnected(
    SoupSession* session, GAsyncResult* res, SignallingServer* pThis) {
  GError* error = NULL;

  pThis->wsConnection =
      std::unique_ptr<SoupWebsocketConnection, std::function<void(SoupWebsocketConnection*)>>(
          soup_session_websocket_connect_finish(session, res, &error),
          [](SoupWebsocketConnection* conn) {
            if (soup_websocket_connection_get_state(conn) == SOUP_WEBSOCKET_STATE_OPEN) {
              soup_websocket_connection_close(conn, 1000, "");
              // TODO wait until 'closed' fired
            }
            g_object_unref(conn);
          });
  if (error) {
    cleanup_and_quit_loop(error->message, SERVER_CONNECTION_ERROR);
    g_error_free(error);
    return;
  }

  g_assert_nonnull(pThis->wsConnection.get());

  logger().trace("Connected to signalling server.");
  app_state = SERVER_CONNECTED;

  g_signal_connect(
      pThis->wsConnection.get(), "closed", G_CALLBACK(SignallingServer::onServerMessage), pThis);
  g_signal_connect(
      pThis->wsConnection.get(), "message", G_CALLBACK(SignallingServer::onServerMessage), pThis);

  // Register with the server so it knows about us and can accept commands
  pThis->registerWithServer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerClosed(SoupSession* session, SignallingServer* pThis) {
  app_state = SERVER_CLOSED;
  cleanup_and_quit_loop("Server connection closed", static_cast<AppState>(0));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerMessage(SoupWebsocketConnection* conn, SoupWebsocketDataType type,
    GBytes* message, SignallingServer* pThis) {
  std::string text;

  switch (type) {
  case SOUP_WEBSOCKET_DATA_BINARY:
    logger().warn("Received unknown binary message, ignoring!");
    return;
  case SOUP_WEBSOCKET_DATA_TEXT: {
    gsize        size;
    const gchar* data = (gchar*)g_bytes_get_data(message, &size);
    text              = std::string(data, size);
    break;
  }
  default:
    g_assert_not_reached();
  }

  if (text == "HELLO") {
    // Server has accepted our registration, we are ready to send commands
    if (app_state != SERVER_REGISTERING) {
      cleanup_and_quit_loop("ERROR: Received HELLO when not registering", APP_STATE_ERROR);
      return;
    }
    app_state = SERVER_REGISTERED;
    logger().trace("Registered with server.");
    // Ask signalling server to connect us with a specific peer
    if (!pThis->setupCall()) {
      cleanup_and_quit_loop("ERROR: Failed to setup call", PEER_CALL_ERROR);
      return;
    }
  } else if (text == "SESSION_OK") {
    // The call initiated by us has been setup by the server; now we can start negotiation
    if (app_state != PEER_CONNECTING) {
      cleanup_and_quit_loop("ERROR: Received SESSION_OK when not calling", PEER_CONNECTION_ERROR);
      return;
    }

    app_state = PEER_CONNECTED;
    pThis->mOnPeerConnected.emit();
  } else if (text.rfind("ERROR", 0) == 0) {
    // Handle errors
    switch (app_state) {
    case SERVER_CONNECTING:
      app_state = SERVER_CONNECTION_ERROR;
      break;
    case SERVER_REGISTERING:
      app_state = SERVER_REGISTRATION_ERROR;
      break;
    case PEER_CONNECTING:
      app_state = PEER_CONNECTION_ERROR;
      break;
    case PEER_CONNECTED:
    case PEER_CALL_NEGOTIATING:
      app_state = PEER_CALL_ERROR;
      break;
    default:
      app_state = APP_STATE_ERROR;
    }
    cleanup_and_quit_loop(text.c_str(), static_cast<AppState>(0));
  } else {
    // Look for JSON messages containing SDP and ICE candidates
    JsonNode*   root;
    JsonObject *object, *child;
    JsonParser* parser = json_parser_new();
    if (!json_parser_load_from_data(parser, text.c_str(), -1, NULL)) {
      logger().warn("Unknown message '{}', ignoring!", text);
      g_object_unref(parser);
      return;
    }

    root = json_parser_get_root(parser);
    if (!JSON_NODE_HOLDS_OBJECT(root)) {
      logger().warn("Unknown json message '{}', ignoring!", text);
      g_object_unref(parser);
      return;
    }

    object = json_node_get_object(root);
    // Check type of JSON message
    if (json_object_has_member(object, "sdp")) {
      const gchar *text, *sdptype;

      g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

      child = json_object_get_object_member(object, "sdp");

      if (!json_object_has_member(child, "type")) {
        cleanup_and_quit_loop("ERROR: received SDP without 'type'", PEER_CALL_ERROR);
        return;
      }

      sdptype = json_object_get_string_member(child, "type");
      text    = json_object_get_string_member(child, "sdp");
      pThis->mOnSdpReceived.emit(sdptype, text);
    } else if (json_object_has_member(object, "ice")) {
      const gchar* candidate;
      gint64       sdpmlineindex;

      child         = json_object_get_object_member(object, "ice");
      candidate     = json_object_get_string_member(child, "candidate");
      sdpmlineindex = json_object_get_int_member(child, "sdpMLineIndex");
      pThis->mOnIceReceived.emit(candidate, sdpmlineindex);
    } else {
      gst_printerr("Ignoring unknown JSON message:\n%s\n", text);
    }
    g_object_unref(parser);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean SignallingServer::registerWithServer() {
  if (soup_websocket_connection_get_state(wsConnection.get()) != SOUP_WEBSOCKET_STATE_OPEN)
    return FALSE;

  // Register with the server with a random integer id. Reply will be received
  // by onServerMessage()
  logger().trace("Registering id '{}' with server.", mOurId);
  app_state = SERVER_REGISTERING;
  send("HELLO " + mOurId);
  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean SignallingServer::setupCall() {
  if (soup_websocket_connection_get_state(wsConnection.get()) != SOUP_WEBSOCKET_STATE_OPEN)
    return FALSE;

  logger().trace("Setting up call with '{}'.", mPeerId);
  app_state = PEER_CONNECTING;
  send("SESSION " + mPeerId);
  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCStream::WebRTCStream()
    : mSignallingServer("wss://webrtc.nirbheek.in:8443", "1234") {
  GError* error = NULL;

  if (!gst_init_check(nullptr, nullptr, &error) || !check_plugins()) {
    // TODO
  }

  mSignallingServer.onPeerConnected().connect([this]() {
    // Start negotiation (exchange SDP and ICE candidates)
    if (!startPipeline())
      cleanup_and_quit_loop("ERROR: failed to start pipeline", PEER_CALL_ERROR);
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

WebRTCStream::~WebRTCStream() {
  mMainLoop.reset();
  mPipeline.reset();
  mMainLoopThread.join();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::vector<uint8_t>> WebRTCStream::getSample(int resolution) {
  if (!mAppSink) {
    return {};
  }
  GstSample* sample;
  g_signal_emit_by_name(mAppSink.get(), "pull-sample", &sample, NULL);
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

void WebRTCStream::onOfferSet(GstPromise* promise, WebRTCStream* pThis) {
  gst_promise_unref(promise);
  promise = gst_promise_new_with_change_func(
      reinterpret_cast<GstPromiseChangeFunc>(&WebRTCStream::onAnswerCreated), pThis, NULL);
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-answer", NULL, promise);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCStream::onAnswerCreated(GstPromise* promise, WebRTCStream* pThis) {
  GstWebRTCSessionDescription* answer = NULL;
  const GstStructure*          reply;

  g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

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

void WebRTCStream::onOfferCreated(GstPromise* promise, WebRTCStream* pThis) {
  GstWebRTCSessionDescription* offer = NULL;

  g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

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

void WebRTCStream::onNegotiationNeeded(GstElement* element, WebRTCStream* pThis) {
  app_state = PEER_CALL_NEGOTIATING;

  if (pThis->mCreateOffer) {
    GstPromise* promise = gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&WebRTCStream::onOfferCreated), pThis, NULL);
    g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-offer", NULL, promise);
  } else {
    pThis->mSignallingServer.send("OFFER_REQUEST");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCStream::onIceCandidate(
    GstElement* webrtc, guint mlineindex, gchar* candidate, WebRTCStream* pThis) {
  gchar*      text;
  JsonObject *ice, *msg;

  if (app_state < PEER_CALL_NEGOTIATING) {
    cleanup_and_quit_loop("Can't send ICE, not in call", APP_STATE_ERROR);
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

void WebRTCStream::onIceGatheringStateNotify(
    GstElement* webrtcbin, GParamSpec* pspec, WebRTCStream* pThis) {
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

void WebRTCStream::onIncomingStream(GstElement* webrtc, GstPad* pad, WebRTCStream* pThis) {
  GstElement* decodebin;
  GstPad*     sinkpad;

  if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
    return;

  decodebin = gst_element_factory_make("decodebin", NULL);
  g_signal_connect(
      decodebin, "pad-added", G_CALLBACK(WebRTCStream::onIncomingDecodebinStream), pThis);
  gst_bin_add(GST_BIN(pThis->mPipeline.get()), decodebin);
  gst_element_sync_state_with_parent(decodebin);

  sinkpad = gst_element_get_static_pad(decodebin, "sink");
  gst_pad_link(pad, sinkpad);
  gst_object_unref(sinkpad);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCStream::onIncomingDecodebinStream(
    GstElement* decodebin, GstPad* pad, WebRTCStream* pThis) {
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

void WebRTCStream::onOfferReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* offer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  g_assert_nonnull(offer);

  // Set remote description on our pipeline
  {
    GstPromise* promise = gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&WebRTCStream::onOfferSet), this, NULL);
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", offer, promise);
  }
  gst_webrtc_session_description_free(offer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCStream::onAnswerReceived(GstSDPMessage* sdp) {
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
  app_state = PEER_CALL_STARTED;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCStream::sendSdpToPeer(GstWebRTCSessionDescription* desc) {
  gchar*      text;
  JsonObject *msg, *sdp;

  if (app_state < PEER_CALL_NEGOTIATING) {
    cleanup_and_quit_loop("Can't send SDP to peer, not in call", APP_STATE_ERROR);
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

void WebRTCStream::handleVideoStream(GstPad* pad) {
  GError* error = NULL;

  logger().trace("Trying to handle video stream.");

  GstElement* bin = gst_parse_bin_from_description(
      "queue ! videoconvert ! videoscale add-borders=false ! capsfilter "
      "caps=video/x-raw,format=RGBA,width=512,height=512 ! "
      "appsink emit-signals=true drop=true max-buffers=1 name=framecapture",
      TRUE, &error);
  if (error) {
    logger().error("Failed to parse launch: {}!", error->message);
    g_error_free(error);
    return;
  }

  mAppSink = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_bin_get_by_name(GST_BIN(bin), "framecapture"),
      [](GstElement* appsink) { gst_object_unref(appsink); });

  gst_bin_add_many(GST_BIN(mPipeline.get()), bin, NULL);
  gst_element_sync_state_with_parent(bin);

  GstPad*          binpad = gst_element_get_static_pad(bin, "sink");
  GstPadLinkReturn ret    = gst_pad_link(pad, binpad);
  g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean WebRTCStream::startPipeline() {
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
  g_signal_emit_by_name(mWebrtcBin.get(), "add-transceiver",
      GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, gst_caps_from_string(RTP_CAPS_VP8 "96"),
      &transceiver);
  gst_object_unref(transceiver);

  // This is the gstwebrtc entry point where we create the offer and so on.
  // It will be called when the pipeline goes to PLAYING.
  g_signal_connect(mWebrtcBin.get(), "on-negotiation-needed",
      G_CALLBACK(WebRTCStream::onNegotiationNeeded), this);
  // We need to transmit this ICE candidate to the browser via the websockets
  // signalling server. Incoming ice candidates from the browser need to be
  // added by us too, see on_server_message()
  g_signal_connect(
      mWebrtcBin.get(), "on-ice-candidate", G_CALLBACK(WebRTCStream::onIceCandidate), this);
  g_signal_connect(mWebrtcBin.get(), "notify::ice-gathering-state",
      G_CALLBACK(WebRTCStream::onIceGatheringStateNotify), this);

  gst_element_set_state(mPipeline.get(), GST_STATE_READY);

  g_signal_emit_by_name(mWebrtcBin.get(), "create-data-channel", "channel", NULL, &send_channel);
  if (send_channel) {
    logger().trace("Created data channel.");
    connect_data_channel_signals(send_channel);
  } else {
    logger().warn("Could not create data channel, is usrsctp available?");
  }

  g_signal_connect(mWebrtcBin.get(), "on-data-channel", G_CALLBACK(on_data_channel), NULL);
  // Incoming streams will be exposed via this signal
  g_signal_connect(mWebrtcBin.get(), "pad-added", G_CALLBACK(WebRTCStream::onIncomingStream), this);

  logger().trace("Starting pipeline.");
  GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(mPipeline.get()), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    // TODO
  }

  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCRenderer::WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
    VolumeShape shape, std::shared_ptr<cs::core::GuiManager> guiManager)
    : Renderer(dataManager, structure, shape) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCRenderer::~WebRTCRenderer(){};

////////////////////////////////////////////////////////////////////////////////////////////////////

float WebRTCRenderer::getProgress() {
  // TODO Implement
  return 0.0f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCRenderer::preloadData(DataManager::State state) {
  // TODO Implement
  logger().warn("Preloading not implemented yet");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void WebRTCRenderer::cancelRendering() {
  // TODO Implement
  logger().warn("Canceling not implemented yet");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Renderer::RenderedImage WebRTCRenderer::getFrameImpl(
    glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) {
  std::optional<std::vector<uint8_t>> image = mStream.getSample(parameters.mResolution);

  if (!image.has_value()) {
    RenderedImage failed;
    failed.mValid = false;
    return failed;
  }

  RenderedImage result;
  result.mColorData = std::move(image.value());
  result.mDepthData = std::vector<float>(parameters.mResolution * parameters.mResolution);
  result.mMVP       = getOSPRayMVP(512., cameraTransform);
  result.mValid     = true;
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 WebRTCRenderer::getOSPRayMVP(float volumeHeight, glm::mat4 observerTransform) {
  // Scale observer transform according to the size of the volume
  observerTransform[3] =
      observerTransform[3] * glm::vec4(volumeHeight, volumeHeight, volumeHeight, 1);

  // Define vertical field of view for ospray camera
  float fov    = 90;
  float fovRad = cs::utils::convert::toRadians(fov);

  // Create camera transform looking along negative z
  glm::mat4 cameraTransform(1);
  cameraTransform[2][2] = -1;

  // Move camera to observer position relative to planet
  cameraTransform = observerTransform * cameraTransform;

  // Get base vectors of rotated coordinate system
  glm::vec3 camRight(cameraTransform[0]);
  camRight = glm::normalize(camRight);
  glm::vec3 camUp(cameraTransform[1]);
  camUp = glm::normalize(camUp);
  glm::vec3 camDir(cameraTransform[2]);
  camDir = glm::normalize(camDir);
  glm::vec3 camPos(cameraTransform[3]);

  // Get position of camera in rotated coordinate system
  float camXLen = glm::dot(camPos, camRight);
  float camYLen = glm::dot(camPos, camUp);
  float camZLen = glm::dot(camPos, camDir);

  // Get angle between camera position and forward vector
  float cameraAngleX = atan(camXLen / camZLen);
  float cameraAngleY = atan(camYLen / camZLen);

  // Get angle between ray towards center of volume and ray at edge of volume
  float modelAngleX = asin(volumeHeight / sqrt(camXLen * camXLen + camZLen * camZLen));
  float modelAngleY = asin(volumeHeight / sqrt(camYLen * camYLen + camZLen * camZLen));

  // Get angle between rays at edges of volume and forward vector
  float leftAngle, rightAngle, downAngle, upAngle;
  if (!isnan(modelAngleX) && !isnan(modelAngleY)) {
    leftAngle  = cameraAngleX - modelAngleX;
    rightAngle = cameraAngleX + modelAngleX;
    downAngle  = cameraAngleY - modelAngleY;
    upAngle    = cameraAngleY + modelAngleY;
  } else {
    // If the camera is inside the volume the model angles will be NaN,
    // so the angles are set to the edges of the field of view
    leftAngle  = -fovRad / 2;
    rightAngle = fovRad / 2;
    downAngle  = -fovRad / 2;
    upAngle    = fovRad / 2;
  }

  glm::mat4 view =
      glm::translate(glm::mat4(1.f), -glm::vec3(camXLen, camYLen, -camZLen) / volumeHeight);

  float nearClip = -camZLen / volumeHeight - 1;
  float farClip  = -camZLen / volumeHeight + 1;
  if (nearClip < 0) {
    nearClip = 0.00001f;
  }
  float     leftClip  = tan(leftAngle) * nearClip;
  float     rightClip = tan(rightAngle) * nearClip;
  float     downClip  = tan(downAngle) * nearClip;
  float     upClip    = tan(upAngle) * nearClip;
  glm::mat4 projection(0);
  projection[0][0] = 2 * nearClip / (rightClip - leftClip);
  projection[1][1] = 2 * nearClip / (upClip - downClip);
  projection[2][0] = (rightClip + leftClip) / (rightClip - leftClip);
  projection[2][1] = (upClip + downClip) / (upClip - downClip);
  projection[2][2] = -(farClip + nearClip) / (farClip - nearClip);
  projection[2][3] = -1;
  projection[3][2] = -2 * farClip * nearClip / (farClip - nearClip);

  return projection * view;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
