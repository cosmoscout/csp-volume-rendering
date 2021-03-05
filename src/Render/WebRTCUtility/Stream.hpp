////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_STREAM_HPP
#define CSP_VOLUME_RENDERING_STREAM_HPP

#include "../../Enums.hpp"
#include "DataChannel.hpp"
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

namespace {
template <typename T>
struct NoDeleter {
  inline void operator()(T* p) {
  }
};

template <typename T>
struct GstObjectDeleter {
  inline void operator()(T* p) {
    gst_object_unref(p);
  }
};

struct GstVideoFrameDeleter {
  inline void operator()(GstVideoFrame* p) {
    gst_video_frame_unmap(p);
  }
};

struct GstSampleDeleter {
  inline void operator()(GstSample* p) {
    gst_sample_unref(p);
  }
};

struct GstCapsDeleter {
  inline void operator()(GstCaps* p) {
    gst_caps_unref(p);
  }
};
} // namespace

namespace csp::volumerendering::webrtc {

class Stream {
 public:
  Stream(SampleType type);
  ~Stream();

  void                                  sendMessage(std::string const& message);
  std::optional<std::vector<uint8_t>>   getColorImage(int resolution);
  std::optional<std::pair<int, GLsync>> getTextureId(int resolution);

 private:
  enum class PeerCallState { eUnknown = 0, eNegotiating, eStarted, eError };

  static void onBusSyncMessage(GstBus* bus, GstMessage* msg, Stream* pThis);

  static void onOfferSet(GstPromise* promise, Stream* pThis);
  static void onAnswerCreated(GstPromise* promise, Stream* pThis);
  static void onOfferCreated(GstPromise* promise, Stream* pThis);
  static void onNegotiationNeeded(GstElement* element, Stream* pThis);
  static void onIceCandidate(GstElement* webrtc, guint mlineindex, gchar* candidate, Stream* pThis);
  static void onIceGatheringStateNotify(GstElement* webrtcbin, GParamSpec* pspec, Stream* pThis);
  static void onDataChannel(GstElement* webrtc, GObject* data_channel, Stream* pThis);

  static void onIncomingStream(GstElement* webrtc, GstPad* pad, Stream* pThis);
  static void onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, Stream* pThis);

  static GstGLContext* onCreateContext(
      GstGLDisplay* display, GstGLContext* otherContext, Stream* pThis);

  void onOfferReceived(GstSDPMessage* sdp);
  void onAnswerReceived(GstSDPMessage* sdp);
  void sendSdpToPeer(GstWebRTCSessionDescription* desc);
  void handleVideoStream(GstPad* pad);

  gboolean startPipeline();

  std::unique_ptr<GstCaps, GstCapsDeleter>     setCaps(int resolution, SampleType type);
  std::unique_ptr<GstSample, GstSampleDeleter> getSample(int resolution);

  cs::utils::Signal<> const& onUncurrentRequired() const;
  cs::utils::Signal<> const& onUncurrentRelease() const;

  std::unique_ptr<SignallingServer> mSignallingServer;
  std::unique_ptr<DataChannel>      mSendChannel;
  std::unique_ptr<DataChannel>      mReceiveChannel;

  PeerCallState mState = PeerCallState::eUnknown;

  bool mCreateOffer = false;

  std::unique_ptr<GMainLoop, std::function<void(GMainLoop*)>> mMainLoop;
  std::thread                                                 mMainLoopThread;

  std::unique_ptr<GstElement, std::function<void(GstElement*)>> mPipeline;
  std::unique_ptr<GstElement, NoDeleter<GstElement>>            mWebrtcBin;
  std::unique_ptr<GstElement, GstObjectDeleter<GstElement>>     mAppSink;
  std::unique_ptr<GstElement, GstObjectDeleter<GstElement>>     mCapsFilter;

  static constexpr int                                                          mFrameCount = 10;
  int                                                                           mFrameIndex = 0;
  std::array<std::unique_ptr<GstVideoFrame, GstVideoFrameDeleter>, mFrameCount> mFrames;

  int              mResolution = 1;
  const SampleType mSampleType;

  guintptr            mGlContext;
  cs::utils::Signal<> mOnUncurrentRequired;
  cs::utils::Signal<> mOnUncurrentRelease;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_STREAM_HPP