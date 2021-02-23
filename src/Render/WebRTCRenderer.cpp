////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebRTCRenderer.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/convert.hpp"

#include <gst/gst.h>
#include <gst/sdp/sdp.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

/* For signalling */
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

static GMainLoop*  loop;
static GstElement *pipe1, *webrtc1 = NULL;
static GObject *   send_channel, *receive_channel;

static enum AppState app_state         = static_cast<AppState>(0);
static gboolean      disable_ssl       = TRUE;
static gboolean      remote_is_offerer = FALSE;

std::unique_ptr<GstElement, std::function<void(GstElement*)>> mAppSink;
csp::volumerendering::SignallingServer mSignallingServer("wss://webrtc.nirbheek.in:8443", "1234");

#define STUN_SERVER " stun-server=stun://stun.l.google.com:19302 "
#define RTP_CAPS_OPUS "application/x-rtp,media=audio,encoding-name=OPUS,payload="
#define RTP_CAPS_VP8 "application/x-rtp,media=video,encoding-name=VP8,payload="

static gboolean cleanup_and_quit_loop(const gchar* msg, enum AppState state) {
  if (msg)
    gst_printerr("%s\n", msg);
  if (state > 0)
    app_state = state;

  if (loop) {
    g_main_loop_quit(loop);
    g_clear_pointer(&loop, g_main_loop_unref);
  }

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

static void handle_media_stream(GstPad* pad, GstElement* pipe) {
  GstPad*          binpad;
  GstElement*      bin;
  GstPadLinkReturn ret;
  GError*          error = NULL;

  gst_println("Trying to handle stream");

  bin = gst_parse_bin_from_description(
      "queue ! videoconvert ! videoscale add-borders=false ! capsfilter "
      "caps=video/x-raw,format=RGBA,width=512,height=512 ! "
      "appsink emit-signals=true drop=true max-buffers=1 name=framecapture",
      TRUE, &error);
  if (error) {
    gst_printerr("Failed to parse launch: %s\n", error->message);
    g_error_free(error);
  }

  mAppSink = std::unique_ptr<GstElement, std::function<void(GstElement*)>>(
      gst_bin_get_by_name(GST_BIN(bin), "framecapture"),
      [](GstElement* appsink) { gst_object_unref(appsink); });

  gst_bin_add_many(GST_BIN(pipe), bin, NULL);
  gst_element_sync_state_with_parent(bin);

  binpad = gst_element_get_static_pad(bin, "sink");
  ret    = gst_pad_link(pad, binpad);
  gst_println("Link result: %i", ret);
  g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);
}

static void on_incoming_decodebin_stream(GstElement* decodebin, GstPad* pad, GstElement* pipe) {
  GstCaps*     caps;
  const gchar* name;

  if (!gst_pad_has_current_caps(pad)) {
    gst_printerr("Pad '%s' has no caps, can't do anything, ignoring\n", GST_PAD_NAME(pad));
    return;
  }

  caps = gst_pad_get_current_caps(pad);
  name = gst_structure_get_name(gst_caps_get_structure(caps, 0));

  if (g_str_has_prefix(name, "video")) {
    handle_media_stream(pad, pipe);
  } else {
    gst_printerr("Unknown pad %s, ignoring", GST_PAD_NAME(pad));
  }
}

static void on_incoming_stream(GstElement* webrtc, GstPad* pad, GstElement* pipe) {
  GstElement* decodebin;
  GstPad*     sinkpad;

  if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
    return;

  decodebin = gst_element_factory_make("decodebin", NULL);
  g_signal_connect(decodebin, "pad-added", G_CALLBACK(on_incoming_decodebin_stream), pipe);
  gst_bin_add(GST_BIN(pipe), decodebin);
  gst_element_sync_state_with_parent(decodebin);

  sinkpad = gst_element_get_static_pad(decodebin, "sink");
  gst_pad_link(pad, sinkpad);
  gst_object_unref(sinkpad);
}

static void send_ice_candidate_message(GstElement* webrtc G_GNUC_UNUSED, guint mlineindex,
    gchar* candidate, gpointer user_data G_GNUC_UNUSED) {
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

  mSignallingServer.send(text);
  g_free(text);
}

static void send_sdp_to_peer(GstWebRTCSessionDescription* desc) {
  gchar*      text;
  JsonObject *msg, *sdp;

  if (app_state < PEER_CALL_NEGOTIATING) {
    cleanup_and_quit_loop("Can't send SDP to peer, not in call", APP_STATE_ERROR);
    return;
  }

  text = gst_sdp_message_as_text(desc->sdp);
  sdp  = json_object_new();

  if (desc->type == GST_WEBRTC_SDP_TYPE_OFFER) {
    gst_println("Sending offer.");
    json_object_set_string_member(sdp, "type", "offer");
  } else if (desc->type == GST_WEBRTC_SDP_TYPE_ANSWER) {
    gst_println("Sending answer.");
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

/* Offer created by our pipeline, to be sent to the peer */
static void on_offer_created(GstPromise* promise, gpointer user_data) {
  GstWebRTCSessionDescription* offer = NULL;
  const GstStructure*          reply;

  g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

  g_assert_cmphex(gst_promise_wait(promise), ==, GST_PROMISE_RESULT_REPLIED);
  reply = gst_promise_get_reply(promise);
  gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
  gst_promise_unref(promise);

  promise = gst_promise_new();
  g_signal_emit_by_name(webrtc1, "set-local-description", offer, promise);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  /* Send offer to peer */
  send_sdp_to_peer(offer);
  gst_webrtc_session_description_free(offer);
}

static void on_negotiation_needed(GstElement* element, gpointer user_data) {
  gboolean create_offer = GPOINTER_TO_INT(user_data);
  app_state             = PEER_CALL_NEGOTIATING;

  if (remote_is_offerer) {
    mSignallingServer.send("OFFER_REQUEST");
  } else if (create_offer) {
    GstPromise* promise = gst_promise_new_with_change_func(on_offer_created, NULL, NULL);
    g_signal_emit_by_name(webrtc1, "create-offer", NULL, promise);
  }
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

static void on_ice_gathering_state_notify(
    GstElement* webrtcbin, GParamSpec* pspec, gpointer user_data) {
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
  gst_print("ICE gathering state changed to %s\n", new_state);
}

static gboolean start_pipeline(gboolean create_offer) {
  GstStateChangeReturn ret;

  pipe1 = gst_pipeline_new("pipeline");
  g_assert_nonnull(pipe1);

  webrtc1 = gst_element_factory_make("webrtcbin", "sendrecv");
  g_assert_nonnull(webrtc1);
  g_object_set(webrtc1, "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);
  g_object_set(webrtc1, "stun-server", "stun://stun.l.google.com:19302", NULL);

  gst_bin_add_many(GST_BIN(pipe1), webrtc1, NULL);

  gst_element_sync_state_with_parent(webrtc1);

  GstWebRTCRTPTransceiver* trans;
  g_signal_emit_by_name(webrtc1, "add-transceiver", GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY,
      gst_caps_from_string(RTP_CAPS_VP8 "96"), &trans);
  gst_object_unref(trans);

  /* This is the gstwebrtc entry point where we create the offer and so on. It
   * will be called when the pipeline goes to PLAYING. */
  g_signal_connect(webrtc1, "on-negotiation-needed", G_CALLBACK(on_negotiation_needed),
      GINT_TO_POINTER(create_offer));
  /* We need to transmit this ICE candidate to the browser via the websockets
   * signalling server. Incoming ice candidates from the browser need to be
   * added by us too, see on_server_message() */
  g_signal_connect(webrtc1, "on-ice-candidate", G_CALLBACK(send_ice_candidate_message), NULL);
  g_signal_connect(
      webrtc1, "notify::ice-gathering-state", G_CALLBACK(on_ice_gathering_state_notify), NULL);

  gst_element_set_state(pipe1, GST_STATE_READY);

  g_signal_emit_by_name(webrtc1, "create-data-channel", "channel", NULL, &send_channel);
  if (send_channel) {
    gst_print("Created data channel\n");
    connect_data_channel_signals(send_channel);
  } else {
    gst_print("Could not create data channel, is usrsctp available?\n");
  }

  g_signal_connect(webrtc1, "on-data-channel", G_CALLBACK(on_data_channel), NULL);
  /* Incoming streams will be exposed via this signal */
  g_signal_connect(webrtc1, "pad-added", G_CALLBACK(on_incoming_stream), pipe1);

  gst_print("Starting pipeline\n");
  ret = gst_element_set_state(GST_ELEMENT(pipe1), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE)
    goto err;

  return TRUE;

err:
  if (pipe1)
    g_clear_object(&pipe1);
  if (webrtc1)
    webrtc1 = NULL;
  return FALSE;
}

static void on_server_closed(
    SoupWebsocketConnection* conn G_GNUC_UNUSED, gpointer user_data G_GNUC_UNUSED) {
  app_state = SERVER_CLOSED;
  cleanup_and_quit_loop("Server connection closed", static_cast<AppState>(0));
}

/* Answer created by our pipeline, to be sent to the peer */
static void on_answer_created(GstPromise* promise, gpointer user_data) {
  GstWebRTCSessionDescription* answer = NULL;
  const GstStructure*          reply;

  g_assert_cmphex(app_state, ==, PEER_CALL_NEGOTIATING);

  g_assert_cmphex(gst_promise_wait(promise), ==, GST_PROMISE_RESULT_REPLIED);
  reply = gst_promise_get_reply(promise);
  gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, NULL);
  gst_promise_unref(promise);

  promise = gst_promise_new();
  g_signal_emit_by_name(webrtc1, "set-local-description", answer, promise);
  gst_promise_interrupt(promise);
  gst_promise_unref(promise);

  /* Send answer to peer */
  send_sdp_to_peer(answer);
  gst_webrtc_session_description_free(answer);
}

static void on_offer_set(GstPromise* promise, gpointer user_data) {
  gst_promise_unref(promise);
  promise = gst_promise_new_with_change_func(on_answer_created, NULL, NULL);
  g_signal_emit_by_name(webrtc1, "create-answer", NULL, promise);
}

static void on_offer_received(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* offer = NULL;
  GstPromise*                  promise;

  offer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  g_assert_nonnull(offer);

  /* Set remote description on our pipeline */
  {
    promise = gst_promise_new_with_change_func(on_offer_set, NULL, NULL);
    g_signal_emit_by_name(webrtc1, "set-remote-description", offer, promise);
  }
  gst_webrtc_session_description_free(offer);
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
  if (wsConnection) {
    if (soup_websocket_connection_get_state(wsConnection.get()) == SOUP_WEBSOCKET_STATE_OPEN) {
      soup_websocket_connection_close(wsConnection.get(), 1000, "");
      // TODO wait until 'closed' fired
    }
    g_object_unref(wsConnection.get());
  }
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

  pThis->wsConnection.reset(soup_session_websocket_connect_finish(session, res, &error));
  if (error) {
    cleanup_and_quit_loop(error->message, SERVER_CONNECTION_ERROR);
    g_error_free(error);
    return;
  }

  g_assert_nonnull(pThis->wsConnection.get());

  logger().trace("Connected to signalling server.");
  app_state = SERVER_CONNECTED;

  g_signal_connect(pThis->wsConnection.get(), "closed", G_CALLBACK(on_server_closed), NULL);
  g_signal_connect(
      pThis->wsConnection.get(), "message", G_CALLBACK(SignallingServer::onServerMessage), pThis);

  // Register with the server so it knows about us and can accept commands
  pThis->registerWithServer();
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

WebRTCRenderer::WebRTCRenderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure,
    VolumeShape shape, std::shared_ptr<cs::core::GuiManager> guiManager)
    : Renderer(dataManager, structure, shape) {
  GError* error = NULL;

  if (!gst_init_check(nullptr, nullptr, &error) || !check_plugins()) {
    // TODO
  }

  mSignallingServer.onPeerConnected().connect([]() {
    // Start negotiation (exchange SDP and ICE candidates)
    if (!start_pipeline(TRUE))
      cleanup_and_quit_loop("ERROR: failed to start pipeline", PEER_CALL_ERROR);
  });
  mSignallingServer.onSdpReceived().connect([](std::string type, std::string text) {
    GstSDPMessage* sdp;
    int            ret = gst_sdp_message_new(&sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);
    ret = gst_sdp_message_parse_buffer((guint8*)text.data(), static_cast<guint>(text.size()), sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);

    if (type == "answer") {
      logger().trace("Received answer.");
      GstWebRTCSessionDescription* answer =
          gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
      g_assert_nonnull(answer);

      // Set remote description on our pipeline
      {
        GstPromise* promise = gst_promise_new();
        g_signal_emit_by_name(webrtc1, "set-remote-description", answer, promise);
        gst_promise_interrupt(promise);
        gst_promise_unref(promise);
      }
      app_state = PEER_CALL_STARTED;
    } else {
      logger().trace("Received offer.");
      on_offer_received(sdp);
    }
  });
  mSignallingServer.onIceReceived().connect([](std::string text, guint64 spdmLineIndex) {
    // Add ice candidate sent by remote peer
    g_signal_emit_by_name(webrtc1, "add-ice-candidate", spdmLineIndex, text.c_str());
  });

  loop = g_main_loop_new(NULL, FALSE);

  mMainLoop = std::thread([]() {
    g_main_loop_run(loop);

    if (loop)
      g_main_loop_unref(loop);

    if (pipe1) {
      gst_element_set_state(GST_ELEMENT(pipe1), GST_STATE_NULL);
      gst_print("Pipeline stopped\n");
      gst_object_unref(pipe1);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebRTCRenderer::~WebRTCRenderer() {
  cleanup_and_quit_loop(nullptr, PEER_CALL_STOPPED);

  mMainLoop.join();
};

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
  RenderedImage invalidResult;
  invalidResult.mValid = false;
  if (!mAppSink) {
    return invalidResult;
  }
  GstSample* sample;
  g_signal_emit_by_name(mAppSink.get(), "pull-sample", &sample, NULL);
  GstBuffer* buf = gst_sample_get_buffer(sample);
  gst_sample_unref(sample);
  if (!buf) {
    return invalidResult;
  }

  RenderedImage result;
  result.mColorData = std::vector<uint8_t>(parameters.mResolution * parameters.mResolution * 4);
  gst_buffer_extract(
      buf, 0, result.mColorData.data(), parameters.mResolution * parameters.mResolution * 4);

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
