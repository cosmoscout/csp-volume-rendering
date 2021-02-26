////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SignallingServer.hpp"

#include "../../logger.hpp"

#include <nlohmann/json.hpp>

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
  soup_websocket_connection_send_text(mWebsocketConn.get(), text.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::sendSdp(GstWebRTCSessionDescription* desc) {
  nlohmann::json msg;
  msg["type"] = "sdp";

  if (desc->type == GST_WEBRTC_SDP_TYPE_OFFER) {
    logger().trace("Sending offer.");
    msg["data"]["type"] = "offer";
  } else if (desc->type == GST_WEBRTC_SDP_TYPE_ANSWER) {
    logger().trace("Sending answer.");
    msg["data"]["type"] = "answer";
  } else {
    g_assert_not_reached();
  }

  gchar* sdpText     = gst_sdp_message_as_text(desc->sdp);
  msg["data"]["sdp"] = sdpText;
  g_free(sdpText);

  send(msg.dump());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::sendIce(guint mlineindex, gchar* candidate) {
  nlohmann::json msg;
  msg["type"]                  = "ice";
  msg["data"]["candidate"]     = candidate;
  msg["data"]["sdpMLineIndex"] = mlineindex;

  send(msg.dump());
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

  pThis->mWebsocketConn =
      std::unique_ptr<SoupWebsocketConnection, std::function<void(SoupWebsocketConnection*)>>(
          soup_session_websocket_connect_finish(session, res, &error),
          [pThis](SoupWebsocketConnection* conn) {
            if (soup_websocket_connection_get_state(conn) == SOUP_WEBSOCKET_STATE_OPEN) {
              std::unique_lock<std::mutex> lock(pThis->mClosedMutex);
              // Code should be SOUP_WEBSOCKET_CLOSE_NORMAL, but this leads to a server side crash
              // when connecting to the sendonly gst example
              int code = 0;
              soup_websocket_connection_close(conn, code, "");
              while (!pThis->mIsClosed) {
                pThis->mClosedCV.wait(lock);
              }
            }
            g_object_unref(conn);
          });
  if (error) {
    logger().error(error->message);
    g_error_free(error);
    return;
  }

  g_assert_nonnull(pThis->mWebsocketConn.get());

  logger().trace("Connected to signalling server.");
  pThis->mIsClosed = false;
  pThis->mOnConnected.emit();

  g_signal_connect(
      pThis->mWebsocketConn.get(), "closed", G_CALLBACK(SignallingServer::onServerClosed), pThis);
  g_signal_connect(
      pThis->mWebsocketConn.get(), "message", G_CALLBACK(SignallingServer::onServerMessage), pThis);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SignallingServer::onServerClosed(SoupSession* session, SignallingServer* pThis) {
  logger().trace("Server connection closed.");
  std::lock_guard(pThis->mClosedMutex);
  pThis->mIsClosed = true;
  pThis->mClosedCV.notify_all();
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
  nlohmann::json msg;
  try {
    msg = nlohmann::json::parse(text);
  } catch (nlohmann::json::parse_error& e) {
    logger().warn("Failed to parse message '{}', ignoring! Reason: '{}'", text, e.what());
    return;
  }

  // Check type of JSON message
  auto type = msg.find("type");
  if (type == msg.end()) {
    logger().warn("No type given in json message '{}', ignoring!", text);
    return;
  }
  auto data = msg.find("data");
  if (data == msg.end() || !data->is_object()) {
    logger().warn("No data given in json message '{}', ignoring!", text);
    return;
  }

  if (*type == "sdp") {
    try {
      std::string sdpType = data->at("type");
      std::string sdp     = data->at("sdp");
      pThis->mOnSdpReceived.emit(sdpType, sdp);
    } catch (nlohmann::json::out_of_range& e) {
      logger().warn(
          "Could not parse data of json message '{}', ignoring! Reason: '{}'", text, e.what());
      return;
    }
  } else if (*type == "ice") {
    try {
      std::string candidate  = data->at("candidate");
      int         mlineIndex = data->at("sdpMLineIndex");
      pThis->mOnIceReceived.emit(candidate, mlineIndex);
    } catch (nlohmann::json::out_of_range& e) {
      logger().warn(
          "Could not parse data of json message '{}', ignoring! Reason: '{}'", text, e.what());
      return;
    }
  } else {
    logger().warn("Unknown json message '{}', ignoring!", text);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
