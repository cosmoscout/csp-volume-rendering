////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_GST_DELETERS_HPP
#define CSP_VOLUME_RENDERING_GST_DELETERS_HPP

#include <memory>

#include <gst/gst.h>
#include <gst/video/video-frame.h>

namespace csp::volumerendering::webrtc {

template <typename T>
struct GstObjectDeleter {
  inline void operator()(T* p) {
    gst_object_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstPipeline> {
  inline void operator()(GstPipeline* p) {
    gst_element_set_state(GST_ELEMENT(p), GST_STATE_NULL);
    gst_object_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstContext> {
  inline void operator()(GstContext* p) {
    gst_context_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstPromise> {
  inline void operator()(GstPromise* p) {
    gst_promise_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstSample> {
  inline void operator()(GstSample* p) {
    gst_sample_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstBuffer> {
  inline void operator()(GstBuffer* p) {
    gst_buffer_unref(p);
  }
};

template <>
struct GstObjectDeleter<GstCaps> {
  inline void operator()(GstCaps* p) {
    gst_caps_unref(p);
  }
};

template <typename T>
struct GObjectDeleter {
  inline void operator()(T* p) {
    g_object_unref(p);
  }
};

template <>
struct GObjectDeleter<GMainLoop> {
  inline void operator()(GMainLoop* p) {
    g_main_loop_quit(p);
    g_main_loop_unref(p);
  }
};

template <typename T>
using GstPointer = std::unique_ptr<T, GstObjectDeleter<T>>;
template <typename T>
using GPointer = std::unique_ptr<T, GObjectDeleter<T>>;

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_GST_DELETERS_HPP
