////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SIGNALLINGSERVER_HPP
#define CSP_VOLUME_RENDERING_SIGNALLINGSERVER_HPP

#include "../../../../src/cs-utils/Signal.hpp"

#include <gst/gst.h>
#include <gst/sdp/sdp.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#include <libsoup/soup.h>

#include <string>

namespace csp::volumerendering::webrtc {

class SignallingServer {
 public:
  SignallingServer(std::string const& url, std::string peerId);
  ~SignallingServer();

  void send(std::string const& text);

  cs::utils::Signal<> const&                         onPeerConnected() const;
  cs::utils::Signal<std::string, std::string> const& onSdpReceived() const;
  cs::utils::Signal<std::string, gint64> const&      onIceReceived() const;

 private:
  enum class ConnectionState {
    eUnknown          = 0,
    eError            = 1, // generic error
    eServerConnecting = 1000,
    eServerConnectionError,
    eServerConnected, // Ready to register
    eServerRegistering = 2000,
    eServerRegistrationError,
    eServerRegistered, // Ready to call a peer
    eServerClosed,     // server connection closed by us or the server
    ePeerConnecting = 3000,
    ePeerConnectionError,
    ePeerConnected,
    ePeerCallError,
  };

  static void onServerConnected(SoupSession* session, GAsyncResult* res, SignallingServer* pThis);
  static void onServerClosed(SoupSession* session, SignallingServer* pThis);
  static void onServerMessage(SoupWebsocketConnection* conn, SoupWebsocketDataType type,
      GBytes* message, SignallingServer* pThis);

  gboolean registerWithServer();
  gboolean setupCall();

  ConnectionState mState = ConnectionState::eUnknown;

  std::unique_ptr<SoupWebsocketConnection, std::function<void(SoupWebsocketConnection*)>>
      wsConnection;

  std::string mOurId;
  std::string mPeerId;

  cs::utils::Signal<>                         mOnPeerConnected;
  cs::utils::Signal<std::string, std::string> mOnSdpReceived;
  cs::utils::Signal<std::string, gint64>      mOnIceReceived;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_SIGNALLINGSERVER_HPP
