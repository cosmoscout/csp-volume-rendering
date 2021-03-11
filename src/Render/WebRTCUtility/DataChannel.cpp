////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2021 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataChannel.hpp"

#include "../../logger.hpp"

namespace csp::volumerendering::webrtc {

////////////////////////////////////////////////////////////////////////////////////////////////////

DataChannel::DataChannel(GstElement* webrtc) {
  GObject* channel;
  g_signal_emit_by_name(webrtc, "create-data-channel", "channel", NULL, &channel);
  if (channel) {
    logger().trace("Created data channel.");
    mChannel.reset(channel);
    connectSignals();
  } else {
    throw std::runtime_error("Could not create data channel, is usrsctp available?");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataChannel::DataChannel(GObject* channel) {
  mChannel.reset(channel);
  connectSignals();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::send(std::string data) {
  g_signal_emit_by_name(mChannel.get(), "send-string", data.c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::onError(GObject* dc, DataChannel* pThis) {
  logger().error("Data channel error!");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::onOpen(GObject* dc, DataChannel* pThis) {
  logger().trace("Data channel opened.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::onClose(GObject* dc, DataChannel* pThis) {
  logger().trace("Data channel closed.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::onMessageString(GObject* dc, gchar* str, DataChannel* pThis) {
  logger().trace("Received data channel message: '{}'", str);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataChannel::connectSignals() {
  g_signal_connect(mChannel.get(), "on-error", G_CALLBACK(DataChannel::onError), this);
  g_signal_connect(mChannel.get(), "on-open", G_CALLBACK(DataChannel::onOpen), this);
  g_signal_connect(mChannel.get(), "on-close", G_CALLBACK(DataChannel::onClose), this);
  g_signal_connect(
      mChannel.get(), "on-message-string", G_CALLBACK(DataChannel::onMessageString), this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::webrtc
