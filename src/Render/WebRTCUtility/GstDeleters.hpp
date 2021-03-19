////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_GST_DELETERS_HPP
#define CSP_VOLUME_RENDERING_GST_DELETERS_HPP

#include <gst/gst.h>
#include <gst/video/video-frame.h>

namespace csp::volumerendering::webrtc {

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

struct GstPipelineDeleter {
  inline void operator()(GstElement* p) {
    gst_element_set_state(p, GST_STATE_NULL);
    gst_object_unref(p);
  }
};

struct GstContextDeleter {
  inline void operator()(GstContext* p) {
    gst_context_unref(p);
  }
};

struct GstPromiseDeleter {
  inline void operator()(GstPromise* p) {
    gst_promise_unref(p);
  }
};

struct GstSampleDeleter {
  inline void operator()(GstSample* p) {
    gst_sample_unref(p);
  }
};

struct GstBufferDeleter {
  inline void operator()(GstBuffer* p) {
    gst_buffer_unref(p);
  }
};

struct GstCapsDeleter {
  inline void operator()(GstCaps* p) {
    gst_caps_unref(p);
  }
};

struct GMainLoopDeleter {
  inline void operator()(GMainLoop* p) {
    g_main_loop_quit(p);
    g_main_loop_unref(p);
  }
};

template <typename T>
struct GObjectDeleter {
  inline void operator()(T* p) {
    g_object_unref(p);
  }
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_GST_DELETERS_HPP
