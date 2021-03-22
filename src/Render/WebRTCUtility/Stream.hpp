////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_STREAM_HPP
#define CSP_VOLUME_RENDERING_STREAM_HPP

#include "../../Enums.hpp"
#include "Connection.hpp"
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
  enum class StreamType { eColor, eAlpha };

  static void          onBusSyncMessage(GstBus* bus, GstMessage* msg, Stream* pThis);
  static void          onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, Stream* pThis);
  static GstGLContext* onCreateContext(
      GstGLDisplay* display, GstGLContext* otherContext, Stream* pThis);

  void startPipeline(GstPointer<GstElement> const& webrtcbin);
  void handleEncodedStream(std::shared_ptr<GstPad> pad);
  void handleDecodedStream(GstPad* pad, StreamType type);

  GstPointer<GstCaps>   setCaps(int resolution);
  GstPointer<GstSample> getSample(int resolution);

  std::unique_ptr<Connection>  mWebrtcConnection;
  std::shared_ptr<DataChannel> mReceiveChannel;

  GPointer<GMainLoop> mMainLoop;
  std::thread         mMainLoopThread;

  GstPointer<GstPipeline>                      mPipeline;
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
