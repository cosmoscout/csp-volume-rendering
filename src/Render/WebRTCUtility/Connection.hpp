////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_CONNECTION_HPP
#define CSP_VOLUME_RENDERING_CONNECTION_HPP

#include "DataChannel.hpp"
#include "GstDeleters.hpp"
#include "SignallingServer.hpp"

#include <gst/gl/gl.h>
#include <gst/gst.h>
#include <gst/sdp/sdp.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>

#include <GL/glew.h>

#include <array>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace csp::volumerendering::webrtc {

class Connection {
 public:
  Connection(std::string signallingUrl);
  ~Connection();

  cs::utils::Signal<GstPointer<GstElement> const&> const&       onWebrtcbinCreated() const;
  cs::utils::Signal<std::shared_ptr<DataChannel>> const& onDataChannelConnected() const;
  cs::utils::Signal<std::shared_ptr<GstPad>> const&      onVideoStreamConnected() const;

 private:
  enum class PeerCallState { eUnknown = 0, eNegotiating, eStarted, eError };

  static void onOfferSet(GstPromise* promisePtr, Connection* pThis);
  static void onAnswerCreated(GstPromise* promisePtr, Connection* pThis);
  static void onOfferCreated(GstPromise* promisePtr, Connection* pThis);
  static void onNegotiationNeeded(GstElement* element, Connection* pThis);
  static void onIceCandidate(
      GstElement* webrtc, guint mlineindex, gchar* candidate, Connection* pThis);
  static void onIceGatheringStateNotify(
      GstElement* webrtcbin, GParamSpec* pspec, Connection* pThis);
  static void onDataChannel(
      GstElement* webrtc, GstWebRTCDataChannel* data_channel, Connection* pThis);

  static void onIncomingStream(GstElement* webrtc, GstPad* pad, Connection* pThis);

  void onOfferReceived(GstSDPMessage* sdp);
  void onAnswerReceived(GstSDPMessage* sdp);
  void sendSdpToPeer(GstWebRTCSessionDescription* desc);

  bool createWebrtcbin();

  std::unique_ptr<SignallingServer> mSignallingServer;

  PeerCallState mState = PeerCallState::eUnknown;

  bool mCreateOffer = false;

  GstPointer<GstElement> mWebrtcBin;

  cs::utils::Signal<GstPointer<GstElement> const&> mOnWebrtcbinCreated;
  cs::utils::Signal<std::shared_ptr<DataChannel>>  mOnDataChannelConnected;
  cs::utils::Signal<std::shared_ptr<GstPad>>       mOnVideoStreamConnected;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_CONNECTION_HPP
