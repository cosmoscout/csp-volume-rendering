////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SignallingServer.hpp"

#include "../../logger.hpp"

#include <json-glib/json-glib.h>

namespace csp::volumerendering::webrtc {

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
  mState = ConnectionState::eServerConnecting;
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
    pThis->mState = ConnectionState::eServerConnectionError;
    logger().error(error->message);
    g_error_free(error);
    return;
  }

  g_assert_nonnull(pThis->wsConnection.get());

  logger().trace("Connected to signalling server.");
  pThis->mState = ConnectionState::eServerConnected;

  g_signal_connect(
      pThis->wsConnection.get(), "closed", G_CALLBACK(SignallingServer::onServerMessage), pThis);
  g_signal_connect(
      pThis->wsConnection.get(), "message", G_CALLBACK(SignallingServer::onServerMessage), pThis);

  // Register with the server so it knows about us and can accept commands
  pThis->registerWithServer();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerClosed(SoupSession* session, SignallingServer* pThis) {
  pThis->mState = ConnectionState::eServerClosed;
  logger().error("Server connection closed");
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
    if (pThis->mState != ConnectionState::eServerRegistering) {
      pThis->mState = ConnectionState::eError;
      logger().error("ERROR: Received HELLO when not registering");
      return;
    }
    pThis->mState = ConnectionState::eServerRegistered;
    logger().trace("Registered with server.");
    // Ask signalling server to connect us with a specific peer
    if (!pThis->setupCall()) {
      pThis->mState = ConnectionState::ePeerCallError;
      logger().error("ERROR: Failed to setup call");
      return;
    }
  } else if (text == "SESSION_OK") {
    // The call initiated by us has been setup by the server; now we can start negotiation
    if (pThis->mState != ConnectionState::ePeerConnecting) {
      pThis->mState = ConnectionState::ePeerConnectionError;
      logger().error("ERROR: Received SESSION_OK when not calling");
      return;
    }

    pThis->mState = ConnectionState::ePeerConnected;
    pThis->mOnPeerConnected.emit();
  } else if (text.rfind("ERROR", 0) == 0) {
    // Handle errors
    switch (pThis->mState) {
    case ConnectionState::eServerConnecting:
      pThis->mState = ConnectionState::eServerConnectionError;
      break;
    case ConnectionState::eServerRegistering:
      pThis->mState = ConnectionState::eServerRegistrationError;
      break;
    case ConnectionState::ePeerConnecting:
      pThis->mState = ConnectionState::ePeerConnectionError;
      break;
    case ConnectionState::ePeerConnected:
      pThis->mState = ConnectionState::ePeerCallError;
      break;
    default:
      pThis->mState = ConnectionState::eError;
    }
    logger().error(text.c_str());
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

      g_assert_cmphex(static_cast<guint64>(pThis->mState), ==,
          static_cast<guint64>(ConnectionState::ePeerConnected));

      child = json_object_get_object_member(object, "sdp");

      if (!json_object_has_member(child, "type")) {
        pThis->mState = ConnectionState::ePeerCallError;
        logger().error("ERROR: received SDP without 'type'");
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
  mState = ConnectionState::eServerRegistering;
  send("HELLO " + mOurId);
  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean SignallingServer::setupCall() {
  if (soup_websocket_connection_get_state(wsConnection.get()) != SOUP_WEBSOCKET_STATE_OPEN)
    return FALSE;

  logger().trace("Setting up call with '{}'.", mPeerId);
  mState = ConnectionState::ePeerConnecting;
  send("SESSION " + mPeerId);
  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
