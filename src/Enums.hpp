////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_ENUMS_HPP
#define CSP_VOLUME_RENDERING_ENUMS_HPP

namespace csp::volumerendering {

enum class VolumeShape { eInvalid = -1, eCubic, eSpherical };
enum class VolumeStructure { eInvalid = -1, eStructured, eUnstructured };
enum class VolumeFileType { eInvalid = -1, eVtk };
enum class DisplayMode { eMesh, ePoints };
enum class DepthMode {
  eNone           = 0,
  eIsosurface     = 1,
  eFirstHit       = 2,
  eLastHit        = 4,
  eThreshold      = 8,
  eMultiThreshold = 16
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_ENUMS_HPP
