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
  SignallingServer(std::string const& url);
  ~SignallingServer();

  void send(std::string const& text);
  void sendSdp(GstWebRTCSessionDescription* desc);
  void sendIce(guint mlineindex, gchar* candidate);

  cs::utils::Signal<> const&                         onConnected() const;
  cs::utils::Signal<std::string, std::string> const& onSdpReceived() const;
  cs::utils::Signal<std::string, gint64> const&      onIceReceived() const;

 private:
  static void onServerConnected(SoupSession* session, GAsyncResult* res, SignallingServer* pThis);
  static void onServerClosed(SoupSession* session, SignallingServer* pThis);
  static void onServerMessage(SoupWebsocketConnection* conn, SoupWebsocketDataType type,
      GBytes* message, SignallingServer* pThis);

  std::unique_ptr<SoupWebsocketConnection, std::function<void(SoupWebsocketConnection*)>>
      wsConnection;

  cs::utils::Signal<>                         mOnConnected;
  cs::utils::Signal<std::string, std::string> mOnSdpReceived;
  cs::utils::Signal<std::string, gint64>      mOnIceReceived;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_SIGNALLINGSERVER_HPP
