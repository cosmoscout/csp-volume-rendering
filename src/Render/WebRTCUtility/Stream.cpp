////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Stream.hpp"

#include "../../Enums.hpp"
#include "../../logger.hpp"

#include "../../../../../src/cs-utils/utils.hpp"

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
    GstPointer<GstPlugin> plugin(gst_registry_find_plugin(registry, needed[i]));
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
  vec4 color = vec4(rgb.r, rgb.g, rgb.b, 1 - rgb.g);
  gl_FragColor = color;
}
)";

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::Stream(std::string signallingUrl)
    : mWebrtcConnection(std::make_unique<Connection>(std::move(signallingUrl)))
#ifdef _WIN32
    , mGlContext((guintptr)wglGetCurrentContext()) {
#else
    , mGlContext(NULL) {
#endif
  mFrameMapped.fill(false);

  GError* error = NULL;

  static bool initialized = false;
  if (!initialized) {
    if (!gst_init_check(nullptr, nullptr, &error) || !check_plugins()) {
      throw std::runtime_error("Could not initialize GStreamer");
    }
    initialized = true;
  }

  mWebrtcConnection->onWebrtcbinCreated().connect(
      [this](GstPointer<GstElement> const& webrtcbin) { startPipeline(webrtcbin); });
  mWebrtcConnection->onVideoStreamConnected().connect(
      [this](std::shared_ptr<GstPad> streamPad) { handleEncodedStream(streamPad); });
  mWebrtcConnection->onDataChannelConnected().connect(
      [this](std::shared_ptr<DataChannel> dc) { mReceiveChannel = dc; });

  mMainLoop.reset(g_main_loop_new(NULL, FALSE));

  mMainLoopThread = std::thread([this]() { g_main_loop_run(mMainLoop.get()); });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Stream::~Stream() {
  mWebrtcConnection.reset();
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

GstPointer<GstCaps> Stream::setCaps(int resolution) {
  std::stringstream capsStr;
  capsStr << "video/x-raw(memory:GLMemory),format=RGBA,width=" << resolution
          << ",height=" << resolution;
  GstPointer<GstCaps> caps(gst_caps_from_string(capsStr.str().c_str()));
  g_object_set(mCapsFilter.get(), "caps", caps.get(), NULL);
  mResolution = resolution;
  return caps;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GstPointer<GstSample> Stream::getSample(int resolution) {
  {
    std::lock_guard lock(mElementsMutex);
    if (!mAppSink) {
      return {};
    }
  }

  GstPointer<GstSample> sample;

  if (resolution != mResolution) {
    auto caps = setCaps(resolution);

    // Drop samples with incorrect resolution
    GstCaps* sampleCaps;
    do {
      sample.reset(gst_app_sink_pull_sample(GST_APP_SINK(mAppSink.get())));
      if (!sample) {
        break;
      }
      sampleCaps = gst_sample_get_caps(sample.get());
    } while (!gst_caps_is_subset(sampleCaps, caps.get()));
  } else {
    sample.reset(gst_app_sink_pull_sample(GST_APP_SINK(mAppSink.get())));
  }
  if (!sample) {
    // Pipeline stopped
    return {};
  }
  return sample;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::optional<std::pair<int, GLsync>> Stream::getTextureId(int resolution) {
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
    const gchar*           context_type;
    GstPointer<GstContext> context;

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
    pThis->handleDecodedStream(pad, type);
  } else {
    logger().warn("Unknown pad '{}', ignoring!", GST_PAD_NAME(pad));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GstGLContext* Stream::onCreateContext(
    GstGLDisplay* display, GstGLContext* otherContext, Stream* pThis) {
  GstPointer<GstGLContext> newContext(gst_gl_context_new(display));
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

void Stream::startPipeline(GstPointer<GstElement> const& webrtcbin) {
  mPipeline.reset(GST_PIPELINE(gst_pipeline_new("pipeline")));
  g_assert_nonnull(mPipeline.get());

  GstBus* bus = gst_pipeline_get_bus(mPipeline.get());
  gst_bus_enable_sync_message_emission(bus);
  g_signal_connect(bus, "sync-message", G_CALLBACK(Stream::onBusSyncMessage), this);

  gst_bin_add_many(GST_BIN(mPipeline.get()), webrtcbin.get(), NULL);
  gst_element_sync_state_with_parent(webrtcbin.get());

  logger().trace("Starting pipeline.");
  GstStateChangeReturn ret = gst_element_set_state(GST_ELEMENT(mPipeline.get()), GST_STATE_PLAYING);
  if (ret == GST_STATE_CHANGE_FAILURE) {
    // TODO
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::handleEncodedStream(std::shared_ptr<GstPad> pad) {
  std::string padName(gst_pad_get_name(pad.get()));
  StreamType  type;
  if (padName == "src_0") {
    type = StreamType::eColor;
  } else if (padName == "src_1") {
    type = StreamType::eAlpha;
  }

  GstPointer<GstElement> decodebin(
      GST_ELEMENT(gst_object_ref_sink(gst_element_factory_make("decodebin", NULL))));
  g_signal_connect(
      decodebin.get(), "pad-added", G_CALLBACK(Stream::onIncomingDecodebinStream), this);
  gst_bin_add(GST_BIN(mPipeline.get()), decodebin.get());
  gst_element_sync_state_with_parent(decodebin.get());

  GstPointer<GstPad> sinkpad(gst_element_get_static_pad(decodebin.get(), "sink"));
  {
    std::lock_guard lock(mDecodersMutex);
    mDecoders.insert_or_assign(type, std::move(decodebin));
  }
  gst_pad_link(pad.get(), sinkpad.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stream::handleDecodedStream(GstPad* pad, StreamType type) {
  std::lock_guard lock(mElementsMutex);
  GError*         error = NULL;

  if (!mVideoMixer) {
    logger().trace("Creating mixer and appsink");
    // Color conversion is done so that a GstGLSyncMeta is added to the buffers
    std::string binString = "glvideomixerelement name=mixer "
                            "! glcolorconvert "
                            "! video/x-raw(memory:GLMemory),format=BGRA "
                            "! glcolorconvert "
                            "! capsfilter name=capsfilter caps-change-mode=delayed "
                            "! appsink name=framecapture drop=true max-buffers=1 ";

    mEndBin.reset(GST_ELEMENT(
        gst_object_ref_sink(gst_parse_bin_from_description(binString.c_str(), TRUE, &error))));
    if (error) {
      logger().error("Failed to parse launch: {}!", error->message);
      g_error_free(error);
      return;
    }
    gst_element_set_name(mEndBin.get(), "bin_sink");

    mVideoMixer.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "mixer"));
    mCapsFilter.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "capsfilter"));
    mAppSink.reset(gst_bin_get_by_name(GST_BIN(mEndBin.get()), "framecapture"));

    setCaps(mResolution);

    gst_bin_add(GST_BIN(mPipeline.get()), mEndBin.get());
    gst_element_sync_state_with_parent(mEndBin.get());
  }

  logger().trace("Trying to handle video stream.");

  std::string binString;
  switch (type) {
  case StreamType::eColor:
    binString = "queue "
                "! glupload "
                "! glcolorconvert ";
    break;
  case StreamType::eAlpha:
    binString = "queue "
                "! glupload "
                "! glcolorconvert "
                "! glshader fragment=\"{SHADER}\" ";
    cs::utils::replaceString(binString, "{SHADER}", ALPHA_FRAGMENT);
    break;
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

cs::utils::Signal<> const& Stream::onUncurrentRequired() const {
  return mOnUncurrentRequired;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<> const& Stream::onUncurrentRelease() const {
  return mOnUncurrentRelease;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
