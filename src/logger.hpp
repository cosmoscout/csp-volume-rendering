////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_LOGGER_HPP
#define CSP_VOLUME_RENDERING_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::volumerendering {

/// This creates the default singleton logger for "csp-volumerendering" when called for the first
/// time and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_LOGGER_HPP
