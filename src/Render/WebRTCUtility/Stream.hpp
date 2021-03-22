////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_STREAM_HPP
#define CSP_VOLUME_RENDERING_STREAM_HPP

#include "../../Enums.hpp"
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

class Stream {
 public:
  Stream(std::string signallingUrl);
  ~Stream();

  void                                  sendMessage(std::string const& message);
  std::optional<std::pair<int, GLsync>> getTextureId(int resolution);

  cs::utils::Signal<> const& onUncurrentRequired() const;
  cs::utils::Signal<> const& onUncurrentRelease() const;

 private:
  enum class PeerCallState { eUnknown = 0, eNegotiating, eStarted, eError };
  enum class StreamType { eColor, eAlpha };

  static void onBusSyncMessage(GstBus* bus, GstMessage* msg, Stream* pThis);

  static void onOfferSet(GstPromise* promisePtr, Stream* pThis);
  static void onAnswerCreated(GstPromise* promisePtr, Stream* pThis);
  static void onOfferCreated(GstPromise* promisePtr, Stream* pThis);
  static void onNegotiationNeeded(GstElement* element, Stream* pThis);
  static void onIceCandidate(GstElement* webrtc, guint mlineindex, gchar* candidate, Stream* pThis);
  static void onIceGatheringStateNotify(GstElement* webrtcbin, GParamSpec* pspec, Stream* pThis);
  static void onDataChannel(GstElement* webrtc, GstWebRTCDataChannel* data_channel, Stream* pThis);

  static void onIncomingStream(GstElement* webrtc, GstPad* pad, Stream* pThis);
  static void onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, Stream* pThis);

  static GstGLContext* onCreateContext(
      GstGLDisplay* display, GstGLContext* otherContext, Stream* pThis);

  void onOfferReceived(GstSDPMessage* sdp);
  void onAnswerReceived(GstSDPMessage* sdp);
  void sendSdpToPeer(GstWebRTCSessionDescription* desc);
  void handleVideoStream(GstPad* pad, StreamType type);

  gboolean startPipeline();

  GstPointer<GstCaps>   setCaps(int resolution);
  GstPointer<GstSample> getSample(int resolution);

  std::unique_ptr<SignallingServer> mSignallingServer;
  std::unique_ptr<DataChannel>      mReceiveChannel;

  PeerCallState mState = PeerCallState::eUnknown;

  bool mCreateOffer = false;

  GPointer<GMainLoop> mMainLoop;
  std::thread         mMainLoopThread;

  GstPointer<GstPipeline>                      mPipeline;
  GstPointer<GstElement>                       mWebrtcBin;
  std::map<StreamType, GstPointer<GstElement>> mDecoders;
  GstPointer<GstElement>                       mEndBin;
  GstPointer<GstElement>                       mVideoMixer;
  GstPointer<GstElement>                       mAppSink;
  GstPointer<GstElement>                       mCapsFilter;

  std::mutex mDecodersMutex;
  std::mutex mElementsMutex;

  static constexpr int                           mFrameCount = 2;
  int                                            mFrameIndex = 0;
  std::array<GstVideoFrame, mFrameCount>         mFrames;
  std::array<bool, mFrameCount>                  mFrameMapped;
  std::array<GstPointer<GstSample>, mFrameCount> mSamples;
  std::array<GstPointer<GstBuffer>, mFrameCount> mBuffers;

  int mResolution = 1;

  GstPointer<GstGLDisplay> mGstGLDisplay;
  GstPointer<GstGLContext> mGstGLContext;
  guintptr                 mGlContext;

  cs::utils::Signal<> mOnUncurrentRequired;
  cs::utils::Signal<> mOnUncurrentRelease;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_STREAM_HPP
