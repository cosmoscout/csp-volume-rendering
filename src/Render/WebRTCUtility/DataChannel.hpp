////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATACHANNEL_HPP
#define CSP_VOLUME_RENDERING_DATACHANNEL_HPP

#include "GstDeleters.hpp"

#include <gst/gst.h>

#include <functional>
#include <string>

namespace csp::volumerendering::webrtc {

class DataChannel {
 public:
  DataChannel(GstElement* webrtc);
  DataChannel(GObject* channel);

  void send(std::string data);

 private:
  static void onError(GObject* dc, DataChannel* pThis);
  static void onOpen(GObject* dc, DataChannel* pThis);
  static void onClose(GObject* dc, DataChannel* pThis);
  static void onMessageString(GObject* dc, gchar* str, DataChannel* pThis);

  void connectSignals();

  std::unique_ptr<GObject, NoDeleter<GObject>> mChannel;
};

} // namespace csp::volumerendering::webrtc

#endif // CSP_VOLUME_RENDERING_DATACHANNEL_HPP
