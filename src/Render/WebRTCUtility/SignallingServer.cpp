////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SignallingServer.hpp"

#include "../../logger.hpp"

#include <json-glib/json-glib.h>

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

} // namespace

namespace csp::volumerendering::webrtc {

////////////////////////////////////////////////////////////////////////////////////////////////////

SignallingServer::SignallingServer(std::string const& url) {
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

  soup_session_websocket_connect_async(session, message, NULL, NULL, NULL,
      reinterpret_cast<GAsyncReadyCallback>(&SignallingServer::onServerConnected), this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SignallingServer::~SignallingServer() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::send(std::string const& text) {
  soup_websocket_connection_send_text(wsConnection.get(), text.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::sendSdp(GstWebRTCSessionDescription* desc) {
  JsonObject* sdp = json_object_new();

  if (desc->type == GST_WEBRTC_SDP_TYPE_OFFER) {
    logger().trace("Sending offer.");
    json_object_set_string_member(sdp, "type", "offer");
  } else if (desc->type == GST_WEBRTC_SDP_TYPE_ANSWER) {
    logger().trace("Sending answer.");
    json_object_set_string_member(sdp, "type", "answer");
  } else {
    g_assert_not_reached();
  }

  gchar* sdpText = gst_sdp_message_as_text(desc->sdp);
  json_object_set_string_member(sdp, "sdp", sdpText);
  g_free(sdpText);

  JsonObject* msg = json_object_new();
  json_object_set_object_member(msg, "data", sdp);
  json_object_set_string_member(msg, "type", "sdp");
  gchar* text = get_string_from_json_object(msg);
  json_object_unref(msg);

  send(text);
  g_free(text);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::sendIce(guint mlineindex, gchar* candidate) {
  JsonObject* ice = json_object_new();
  json_object_set_string_member(ice, "candidate", candidate);
  json_object_set_int_member(ice, "sdpMLineIndex", mlineindex);
  JsonObject* msg = json_object_new();
  json_object_set_object_member(msg, "data", ice);
  json_object_set_string_member(msg, "type", "ice");
  gchar* text = get_string_from_json_object(msg);
  json_object_unref(msg);

  send(text);
  g_free(text);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<> const& SignallingServer::onConnected() const {
  return mOnConnected;
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
    logger().error(error->message);
    g_error_free(error);
    return;
  }

  g_assert_nonnull(pThis->wsConnection.get());

  logger().trace("Connected to signalling server.");
  pThis->mOnConnected.emit();

  g_signal_connect(
      pThis->wsConnection.get(), "closed", G_CALLBACK(SignallingServer::onServerMessage), pThis);
  g_signal_connect(
      pThis->wsConnection.get(), "message", G_CALLBACK(SignallingServer::onServerMessage), pThis);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerClosed(SoupSession* session, SignallingServer* pThis) {
  logger().error("Server connection closed");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerMessage(SoupWebsocketConnection* conn,
    SoupWebsocketDataType dataType, GBytes* message, SignallingServer* pThis) {
  std::string text;

  switch (dataType) {
  case SOUP_WEBSOCKET_DATA_BINARY: {
    logger().warn("Received unknown binary message, ignoring!");
    return;
  }
  case SOUP_WEBSOCKET_DATA_TEXT: {
    gsize        size;
    const gchar* data = (gchar*)g_bytes_get_data(message, &size);
    text              = std::string(data, size);
    break;
  }
  default:
    g_assert_not_reached();
  }

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
  if (!json_object_has_member(object, "type")) {
    logger().warn("No type given in json message '{}', ignoring!", text);
    g_object_unref(parser);
    return;
  }
  std::string type(json_object_get_string_member(object, "type"));

  if (type == "sdp") {
    const gchar *text, *sdptype;

    child = json_object_get_object_member(object, "data");

    if (!json_object_has_member(child, "type")) {
      logger().error("ERROR: received SDP without 'type'");
      g_object_unref(parser);
      return;
    }

    sdptype = json_object_get_string_member(child, "type");
    text    = json_object_get_string_member(child, "sdp");
    pThis->mOnSdpReceived.emit(sdptype, text);
  } else if (type == "ice") {
    const gchar* candidate;
    gint64       sdpmlineindex;

    child         = json_object_get_object_member(object, "data");
    candidate     = json_object_get_string_member(child, "candidate");
    sdpmlineindex = json_object_get_int_member(child, "sdpMLineIndex");
    pThis->mOnIceReceived.emit(candidate, sdpmlineindex);
  } else {
    logger().warn("Unknown json message '{}', ignoring!", text);
  }
  g_object_unref(parser);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
