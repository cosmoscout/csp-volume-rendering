////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Stream.hpp"

#include "../../Plugin.hpp"

#include "../../Enums.hpp"
#include "../../logger.hpp"

#include <gst/app/gstappsink.h>

#ifdef _WIN32
#include <Windows.h>
#endif

#include <sstream>

namespace csp::volumerendering::webrtc {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

gboolean check_plugins() {
  guint        i;
  gboolean     ret;
  GstRegistry* registry;
  const gchar* needed[] = {"opus", "vpx", "nice", "webrtc", "dtls", "srtp", "rtpmanager",
      "videoconvert", "coreelements", "app", "videoscale", NULL};

  registry = gst_registry_get();
  ret      = TRUE;
  for (i = 0; i < g_strv_length((gchar**)needed); i++) {
    std::unique_ptr<GstPlugin, GstObjectDeleter<GstPlugin>> plugin(
        gst_registry_find_plugin(registry, needed[i]));
    if (!plugin) {
      csp::volumerendering::logger().error("Required gstreamer plugin '{}' not found!", needed[i]);
      ret = FALSE;
      continue;
    }
  }
  return ret;
}

const char* ALPHA_FRAGMENT = R"(
#version 330
varying vec2 v_texcoord;
uniform sampler2D tex;
uniform float time;
uniform float width;
uniform float height;

#define from_rgb_bt601_offset vec3(0.0625, 0.5, 0.5)
#define from_rgb_bt601_ycoeff vec3( 0.2578125, 0.50390625, 0.09765625)
#define from_rgb_bt601_ucoeff vec3(-0.1484375,-0.28906250, 0.43750000)
#define from_rgb_bt601_vcoeff vec3( 0.4375000,-0.36718750,-0.07031250)

vec3 rgb_to_yuv (vec3 val) {
  vec3 yuv;
  yuv.r = dot(val.rgb, from_rgb_bt601_ycoeff);
  yuv.g = dot(val.rgb, from_rgb_bt601_ucoeff);
  yuv.b = dot(val.rgb, from_rgb_bt601_vcoeff);
  yuv += from_rgb_bt601_offset;
  return yuv;
}

void main () {
  vec3 rgb = vec3(texture2D(tex, v_texcoord));
  vec3 yuv = rgb_to_yuv(rgb);
  vec4 color = vec4(1, 1, 1, 1 - rgb.g);
  gl_FragColor = color;
}
)";

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::Stream(std::string signallingUrl, SampleType type)
    : mSignallingServer(std::make_unique<SignallingServer>(std::move(signallingUrl)))
#ifdef _WIN32
    , mGlContext((guintptr)wglGetCurrentContext())
#else
    , mGlContext(NULL)
#endif
    , mSampleType(std::move(type)) {
  mFrameMapped.fill(false);

  GError* error = NULL;

  static bool initialized = false;
  if (!initialized) {
    if (!gst_init_check(nullptr, nullptr, &error) || !check_plugins()) {
      throw std::runtime_error("Could not initialize GStreamer");
    }
    initialized = true;
  }

  mSignallingServer->onConnected().connect([this]() {
    mState = PeerCallState::eNegotiating;
    // Start negotiation (exchange SDP and ICE candidates)
    if (!startPipeline()) {
      mState = PeerCallState::eError;
      logger().error("ERROR: failed to start pipeline");
    }
  });
  mSignallingServer->onSdpReceived().connect([this](std::string type, std::string text) {
    GstSDPMessage* sdp;
    int            ret = gst_sdp_message_new(&sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);
    ret = gst_sdp_message_parse_buffer((guint8*)text.data(), static_cast<guint>(text.size()), sdp);
    g_assert_cmphex(ret, ==, GST_SDP_OK);

    if (type == "answer") {
      logger().trace("Received answer.");
      onAnswerReceived(sdp);
    } else {
      logger().trace("Received offer.");
      onOfferReceived(sdp);
    }
  });
  mSignallingServer->onIceReceived().connect([this](std::string text, guint64 spdmLineIndex) {
    // Add ice candidate sent by remote peer
    g_signal_emit_by_name(mWebrtcBin.get(), "add-ice-candidate", spdmLineIndex, text.c_str());
  });

  mMainLoop.reset(g_main_loop_new(NULL, FALSE));

  mMainLoopThread = std::thread([this]() { g_main_loop_run(mMainLoop.get()); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::~Stream() {
  mSignallingServer.reset();
  mPipeline.reset();
  mMainLoop.reset();
  mMainLoopThread.join();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::sendMessage(std::string const& message) {
  if (mReceiveChannel) {
    mReceiveChannel->send(message);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<GstCaps, GstCapsDeleter> Stream::setCaps(int resolution, SampleType type) {
  std::string videoType;
  switch (type) {
  case SampleType::eImageData:
    videoType = "video/x-raw";
    break;
  case SampleType::eTexId:
    videoType = "video/x-raw(memory:GLMemory)";
    break;
  }
  std::stringstream capsStr;
  capsStr << videoType << ",framerate=30/1,format=RGBA,width=" << resolution
          << ",height=" << resolution;
  std::unique_ptr<GstCaps, GstCapsDeleter> caps(gst_caps_from_string(capsStr.str().c_str()));
  g_object_set(mCapsFilter.get(), "caps", caps.get(), NULL);
  mResolution = resolution;
  return caps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<GstSample, GstSampleDeleter> Stream::getSample(int resolution) {
  {
    std::lock_guard lock(mElementsMutex);
    if (!mAppSink) {
      return {};
    }
  }

  std::unique_ptr<GstSample, GstSampleDeleter> sample;

  if (resolution != mResolution) {
    auto caps = setCaps(resolution, mSampleType);

    // Drop samples with incorrect resolution
    GstCaps* sampleCaps;
    do {
      sample.reset(gst_app_sink_pull_sample(GST_APP_SINK_CAST(mAppSink.get())));
      if (!sample) {
        break;
      }
      sampleCaps = gst_sample_get_caps(sample.get());
    } while (!gst_caps_is_subset(sampleCaps, caps.get()));
  } else {
    sample.reset(gst_app_sink_pull_sample(GST_APP_SINK_CAST(mAppSink.get())));
  }
  if (!sample) {
    // Pipeline stopped
    return {};
  }
  return sample;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::vector<uint8_t>> Stream::getColorImage(int resolution) {
  assert(mSampleType == SampleType::eImageData);
  auto sample = getSample(resolution);
  if (!sample) {
    return {};
  }

  GstBuffer* buf = gst_sample_get_buffer(sample.get());
  if (!buf) {
    return {};
  }

  std::vector<uint8_t> image(resolution * resolution * 4);
  gst_buffer_extract(buf, 0, image.data(), resolution * resolution * 4);
  return image;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::pair<int, GLsync>> Stream::getTextureId(int resolution) {
  assert(mSampleType == SampleType::eTexId);
  {
    auto sample = getSample(resolution);
    if (!sample) {
      return {};
    }

    if (mFrameMapped[mFrameIndex]) {
      gst_video_frame_unmap(&mFrames[mFrameIndex]);
    }
    if (++mFrameIndex >= mFrameCount) {
      mFrameIndex = 0;
    }

    mSamples[mFrameIndex].swap(sample);
  }

  GstBuffer* buf = gst_sample_get_buffer(mSamples[mFrameIndex].get());
  if (!buf) {
    return {};
  }
  GstVideoInfo info;
  gst_video_info_from_caps(&info, gst_sample_get_caps(mSamples[mFrameIndex].get()));
  if (!gst_video_frame_map(
          &mFrames[mFrameIndex], &info, buf, (GstMapFlags)(GST_MAP_READ | GST_MAP_GL))) {
    logger().warn("Failed to map video frame");
    mFrameMapped[mFrameIndex] = false;
    return {};
  }
  mFrameMapped[mFrameIndex] = true;

  GstMemory* mem = gst_buffer_peek_memory(buf, 0);
  if (!gst_is_gl_memory(mem)) {
    return {};
  }
  GstGLMemory* glmem = (GstGLMemory*)mem;

  GstGLSyncMeta* sync = gst_buffer_get_gl_sync_meta(buf);
  if (!sync) {
    logger().warn("Current GstBuffer has no GstGLSyncMeta!");
    return {};
  }
  gst_gl_sync_meta_set_sync_point(sync, glmem->mem.context);

  return std::make_pair<int, GLsync>((int)glmem->tex_id, (GLsync)sync->data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onBusSyncMessage(GstBus* bus, GstMessage* msg, Stream* pThis) {
  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_NEED_CONTEXT: {
    const gchar*                                   context_type;
    std::unique_ptr<GstContext, GstContextDeleter> context;

    gst_message_parse_context_type(msg, &context_type);
    logger().trace("Got need context {}", context_type);

    if (g_strcmp0(context_type, GST_GL_DISPLAY_CONTEXT_TYPE) == 0) {
#ifdef _WIN32
      if (!pThis->mGstGLDisplay) {
        pThis->mGstGLDisplay.reset(gst_gl_display_new());
        g_signal_connect(pThis->mGstGLDisplay.get(), "create-context",
            G_CALLBACK(Stream::onCreateContext), pThis);
      }

      context.reset(gst_context_new(GST_GL_DISPLAY_CONTEXT_TYPE, TRUE));
      gst_context_set_gl_display(context.get(), pThis->mGstGLDisplay.get());

      gst_element_set_context(GST_ELEMENT(msg->src), context.get());
#else
      logger().warn("Setting OpenGL display is currently only supported on Windows.");
#endif
    } else if (g_strcmp0(context_type, "gst.gl.app_context") == 0) {
#ifdef _WIN32
      if (!pThis->mGstGLContext) {
        pThis->mOnUncurrentRequired.emit();
        pThis->mGstGLContext.reset(gst_gl_context_new_wrapped(pThis->mGstGLDisplay.get(),
            pThis->mGlContext, GST_GL_PLATFORM_WGL, GST_GL_API_OPENGL3));
      }

      context.reset(gst_context_new("gst.gl.app_context", TRUE));
      GstStructure* s = gst_context_writable_structure(context.get());
      gst_structure_set(s, "context", GST_TYPE_GL_CONTEXT, pThis->mGstGLContext.get(), NULL);

      gst_element_set_context(GST_ELEMENT(msg->src), context.get());
#else
      logger().warn("Setting OpenGL context is currently only supported on Windows.");
#endif
    }
    break;
  }
  default: {
    break;
  }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferSet(GstPromise* promisePtr, Stream* pThis) {
  std::unique_ptr<GstPromise, GstPromiseDeleter> promise(promisePtr);
  promise.reset(gst_promise_new_with_change_func(
      reinterpret_cast<GstPromiseChangeFunc>(&Stream::onAnswerCreated), pThis, NULL));
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-answer", NULL, promise.release());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onAnswerCreated(GstPromise* promisePtr, Stream* pThis) {
  std::unique_ptr<GstPromise, GstPromiseDeleter> promise(promisePtr);
  GstWebRTCSessionDescription*                   answer = NULL;
  const GstStructure*                            reply;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise.get()), ==, GST_PROMISE_RESULT_REPLIED);
  reply = gst_promise_get_reply(promise.get());
  gst_structure_get(reply, "answer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &answer, NULL);

  promise.reset(gst_promise_new());
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", answer, promise.get());
  gst_promise_interrupt(promise.get());

  pThis->sendSdpToPeer(answer);
  gst_webrtc_session_description_free(answer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferCreated(GstPromise* promisePtr, Stream* pThis) {
  std::unique_ptr<GstPromise, GstPromiseDeleter> promise(promisePtr);
  GstWebRTCSessionDescription*                   offer = NULL;

  g_assert_cmphex(
      static_cast<guint64>(pThis->mState), ==, static_cast<guint64>(PeerCallState::eNegotiating));

  g_assert_cmphex(gst_promise_wait(promise.get()), ==, GST_PROMISE_RESULT_REPLIED);
  const GstStructure* reply = gst_promise_get_reply(promise.get());
  gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);

  promise.reset(gst_promise_new());
  g_signal_emit_by_name(pThis->mWebrtcBin.get(), "set-local-description", offer, promise.get());
  gst_promise_interrupt(promise.get());

  // Send offer to peer
  pThis->sendSdpToPeer(offer);
  gst_webrtc_session_description_free(offer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onNegotiationNeeded(GstElement* element, Stream* pThis) {
  pThis->mState = PeerCallState::eNegotiating;

  if (pThis->mCreateOffer) {
    std::unique_ptr<GstPromise, GstPromiseDeleter> promise(gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Stream::onOfferCreated), pThis, NULL));
    g_signal_emit_by_name(pThis->mWebrtcBin.get(), "create-offer", NULL, promise.release());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIceCandidate(GstElement* webrtc, guint mlineindex, gchar* candidate, Stream* pThis) {
  if (pThis->mState < PeerCallState::eNegotiating) {
    pThis->mState = PeerCallState::eError;
    logger().error("Can't send ICE, not in call");
    return;
  }

  pThis->mSignallingServer->sendIce(mlineindex, candidate);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIceGatheringStateNotify(GstElement* webrtcbin, GParamSpec* pspec, Stream* pThis) {
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
  logger().trace("ICE gathering state changed to {}.", new_state);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onDataChannel(GstElement* webrtc, GObject* data_channel, Stream* pThis) {
  pThis->mReceiveChannel = std::make_unique<DataChannel>(data_channel);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIncomingStream(GstElement* webrtc, GstPad* pad, Stream* pThis) {
  GstElement* decodebin;

  if (GST_PAD_DIRECTION(pad) != GST_PAD_SRC)
    return;

  std::string padName(gst_pad_get_name(pad));
  StreamType  type;
  if (padName == "src_0") {
    type = StreamType::eColor;
  } else if (padName == "src_1") {
    type = StreamType::eAlpha;
  }

  decodebin = gst_element_factory_make("decodebin", NULL);
  {
    std::lock_guard lock(pThis->mDecodersMutex);
    pThis->mDecoders.try_emplace(type, decodebin);
  }
  g_signal_connect(decodebin, "pad-added", G_CALLBACK(Stream::onIncomingDecodebinStream), pThis);
  gst_bin_add(GST_BIN(pThis->mPipeline.get()), decodebin);
  gst_element_sync_state_with_parent(decodebin);

  std::unique_ptr<GstPad, GstObjectDeleter<GstPad>> sinkpad(
      gst_element_get_static_pad(decodebin, "sink"));
  gst_pad_link(pad, sinkpad.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onIncomingDecodebinStream(GstElement* decodebin, GstPad* pad, Stream* pThis) {
  GstCaps*     caps;
  const gchar* name;

  if (!gst_pad_has_current_caps(pad)) {
    logger().warn("Pad '{}' has no caps, can't do anything, ignoring!", GST_PAD_NAME(pad));
    return;
  }

  caps = gst_pad_get_current_caps(pad);
  name = gst_structure_get_name(gst_caps_get_structure(caps, 0));

  if (g_str_has_prefix(name, "video")) {
    StreamType type;
    {
      std::lock_guard lock(pThis->mDecodersMutex);
      type = std::find_if(
          pThis->mDecoders.begin(), pThis->mDecoders.end(), [decodebin](auto const& decoder) {
            return decoder.second.get() == decodebin;
          })->first;
    }
    pThis->handleVideoStream(pad, type);
  } else {
    logger().warn("Unknown pad '{}', ignoring!", GST_PAD_NAME(pad));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GstGLContext* Stream::onCreateContext(
    GstGLDisplay* display, GstGLContext* otherContext, Stream* pThis) {
  std::unique_ptr<GstGLContext, GstObjectDeleter<GstGLContext>> newContext(
      gst_gl_context_new(display));
  if (!newContext) {
    logger().error("Failed to create GL context!");
    return NULL;
  }

  GError* error = NULL;
  if (gst_gl_context_create(newContext.get(), otherContext, &error)) {
    pThis->mOnUncurrentRelease.emit();
    return newContext.release();
  }

  logger().error("Failed to share GL context!");
  return NULL;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onOfferReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* offer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_OFFER, sdp);
  g_assert_nonnull(offer);

  // Set remote description on our pipeline
  {
    std::unique_ptr<GstPromise, GstPromiseDeleter> promise(gst_promise_new_with_change_func(
        reinterpret_cast<GstPromiseChangeFunc>(&Stream::onOfferSet), this, NULL));
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", offer, promise.release());
    gst_webrtc_session_description_free(offer);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::onAnswerReceived(GstSDPMessage* sdp) {
  GstWebRTCSessionDescription* answer =
      gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
  g_assert_nonnull(answer);

  // Set remote description on our pipeline
  {
    std::unique_ptr<GstPromise, GstPromiseDeleter> promise(gst_promise_new());
    g_signal_emit_by_name(mWebrtcBin.get(), "set-remote-description", answer, promise.get());
    gst_promise_interrupt(promise.get());
  }
  mState = PeerCallState::eStarted;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::sendSdpToPeer(GstWebRTCSessionDescription* desc) {
  if (mState < PeerCallState::eNegotiating) {
    mState = PeerCallState::eError;
    logger().error("Can't send SDP to peer, not in call");
    return;
  }

  mSignallingServer->sendSdp(desc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::handleVideoStream(GstPad* pad, StreamType type) {
  std::lock_guard lock(mElementsMutex);
  GError*         error = NULL;

  if (!mVideoMixer) {
    logger().trace("Creating mixer and appsink");
    std::string binString;
    switch (mSampleType) {
    case SampleType::eImageData:
      binString = "videomixer name=mixer "
                  "! videoscale add-borders=false "
                  "! capsfilter name=capsfilter "
                  "! appsink drop=true max-buffers=1 name=framecapture";
      break;
    case SampleType::eTexId:
      // Color conversion is done so that a GstGLSyncMeta is added to the buffers
      binString = "glvideomixerelement name=mixer "
                  "! glcolorconvert "
                  "! video/x-raw(memory:GLMemory),format=BGRA "
                  "! glcolorconvert "
                  "! capsfilter name=capsfilter caps-change-mode=delayed "
                  "! appsink name=framecapture drop=true max-buffers=1 ";
      break;
    }

    mEndBin.reset(gst_parse_bin_from_description(binString.c_str(), TRUE, &error));
    if (error) {
      logger().error("Failed to parse launch: {}!", error->message);
      g_error_free(error);
      return;
    }
    gst_element_set_name(mEndBin.get(), "bin_sink");

    mVideoMixer.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "mixer"));
    mCapsFilter.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "capsfilter"));
    mAppSink.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "framecapture"));

    setCaps(mResolution, mSampleType);

    gst_bin_add(GST_BIN(mPipeline.get()), mEndBin.get());
    gst_element_sync_state_with_parent(mEndBin.get());
  }

  logger().trace("Trying to handle video stream.");

  std::string binString;
  if (type == StreamType::eColor) {
    switch (mSampleType) {
    case SampleType::eImageData:
      binString = "queue ! videoconvert";
      break;
    case SampleType::eTexId:
      binString = "queue "
                  "! glupload "
                  "! glcolorconvert ";
      break;
    }
  }
  if (type == StreamType::eAlpha) {
    switch (mSampleType) {
    case SampleType::eImageData:
      binString = "queue ! videoconvert";
      break;
    case SampleType::eTexId:
      binString = "queue "
                  "! glupload "
                  "! glcolorconvert "
                  "! glshader fragment=\"{SHADER}\" ";
      cs::utils::replaceString(binString, "{SHADER}", ALPHA_FRAGMENT);
      break;
    }
  }

  GstElement* bin = gst_parse_bin_from_description(binString.c_str(), TRUE, &error);
  if (error) {
    logger().error("Failed to parse launch: {}!", error->message);
    g_error_free(error);
    return;
  }
  switch (type) {
  case StreamType::eColor:
    gst_element_set_name(bin, "bin_color");
    break;
  case StreamType::eAlpha:
    gst_element_set_name(bin, "bin_alpha");
    break;
  }

  gst_bin_add(GST_BIN(mPipeline.get()), bin);
  gst_element_sync_state_with_parent(bin);

  GstPad*          binSink = gst_element_get_static_pad(bin, "sink");
  GstPad*          binSrc  = gst_element_get_static_pad(bin, "src");
  GstPadLinkReturn ret;
  ret = gst_pad_link(pad, binSink);
  g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);

  std::string mixerSinkName;
  switch (type) {
  case StreamType::eColor:
    mixerSinkName = "sink_color";
    break;
  case StreamType::eAlpha:
    mixerSinkName = "sink_alpha";
    break;
  }
  GstPad* mixerSink = gst_element_get_request_pad(mVideoMixer.get(), "sink_%u");
  switch (type) {
  case StreamType::eColor:
    g_object_set(mixerSink, "blend-function-dst-alpha", GL_ONE, NULL);
    g_object_set(mixerSink, "blend-function-src-alpha", GL_ZERO, NULL);
    g_object_set(mixerSink, "blend-function-dst-rgb", GL_ZERO, NULL);
    g_object_set(mixerSink, "blend-function-src-rgb", GL_ONE, NULL);
    break;
  case StreamType::eAlpha:
    g_object_set(mixerSink, "blend-function-dst-alpha", GL_ZERO, NULL);
    g_object_set(mixerSink, "blend-function-src-alpha", GL_ONE, NULL);
    g_object_set(mixerSink, "blend-function-dst-rgb", GL_ONE, NULL);
    g_object_set(mixerSink, "blend-function-src-rgb", GL_ZERO, NULL);
    break;
  }
  GstPad* ghostSink = gst_ghost_pad_new(mixerSinkName.c_str(), mixerSink);
  gst_element_add_pad(mEndBin.get(), ghostSink);

  ret = gst_pad_link(binSrc, ghostSink);
  g_assert_cmphex(ret, ==, GST_PAD_LINK_OK);
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

gboolean Stream::startPipeline() {
  mPipeline.reset(gst_pipeline_new("pipeline"));
  g_assert_nonnull(mPipeline.get());

  GstBus* bus = gst_pipeline_get_bus(GST_PIPELINE(mPipeline.get()));
  gst_bus_enable_sync_message_emission(bus);
  g_signal_connect(bus, "sync-message", G_CALLBACK(Stream::onBusSyncMessage), this);

  mWebrtcBin.reset(gst_element_factory_make("webrtcbin", "sendrecv"));
  g_assert_nonnull(mWebrtcBin.get());
  g_object_set(mWebrtcBin.get(), "bundle-policy", GST_WEBRTC_BUNDLE_POLICY_MAX_BUNDLE, NULL);
  g_object_set(mWebrtcBin.get(), "stun-server", "stun://stun.l.google.com:19302", NULL);

  gst_bin_add_many(GST_BIN(mPipeline.get()), mWebrtcBin.get(), NULL);

  gst_element_sync_state_with_parent(mWebrtcBin.get());

  for (int i = 0; i < 2; i++) {
    std::unique_ptr<GstCaps, GstCapsDeleter> caps(
        gst_caps_from_string("application/x-rtp,media=video,encoding-name=VP8,payload=96"));
    GstWebRTCRTPTransceiver* transceiverPtr = NULL;
    g_signal_emit_by_name(mWebrtcBin.get(), "add-transceiver",
        GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, caps.get(), &transceiverPtr);
    std::unique_ptr<GstWebRTCRTPTransceiver, GstObjectDeleter<GstWebRTCRTPTransceiver>> transceiver(
        transceiverPtr);
  }

  // This is the gstwebrtc entry point where we create the offer and so on.
  // It will be called when the pipeline goes to PLAYING.
  g_signal_connect(
      mWebrtcBin.get(), "on-negotiation-needed", G_CALLBACK(Stream::onNegotiationNeeded), this);
  // We need to transmit this ICE candidate to the browser via the websockets
  // signalling server. Incoming ice candidates from the browser need to be
  // added by us too, see on_server_message()
  g_signal_connect(mWebrtcBin.get(), "on-ice-candidate", G_CALLBACK(Stream::onIceCandidate), this);
  g_signal_connect(mWebrtcBin.get(), "notify::ice-gathering-state",
      G_CALLBACK(Stream::onIceGatheringStateNotify), this);

  gst_element_set_state(mPipeline.get(), GST_STATE_READY);

  g_signal_connect(mWebrtcBin.get(), "on-data-channel", G_CALLBACK(Stream::onDataChannel), this);
  // Incoming streams will be exposed via this signal
  g_signal_connect(mWebrtcBin.get(), "pad-added", G_CALLBACK(Stream::onIncomingStream), this);

  logger().trace("Starting pipeline.");
  GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(mPipeline.get()), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    // TODO
  }

  return TRUE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<> const& Stream::onUncurrentRequired() const {
  return mOnUncurrentRequired;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<> const& Stream::onUncurrentRelease() const {
  return mOnUncurrentRelease;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
