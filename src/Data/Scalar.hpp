////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SCALAR_HPP
#define CSP_VOLUME_RENDERING_SCALAR_HPP

#include "../Enums.hpp"

#include <string>

namespace csp::volumerendering {

struct ScalarFilter {
  int   mAttrIndex;
  float mMin;
  float mMax;

  bool operator==(ScalarFilter const& other) const {
    return mAttrIndex == other.mAttrIndex && mMin == other.mMin && mMax == other.mMax;
  }
};

struct Scalar {
  std::string mName;
  ScalarType  mType;

  std::string getId() const {
    std::string id;
    switch (mType) {
    case ScalarType::ePointData:
      id.append("point_");
      break;
    case ScalarType::eCellData:
      id.append("cell_");
      break;
    }
    id.append(mName);
    return id;
  }

  bool operator==(const Scalar& other) const {
    return mName == other.mName && mType == other.mType;
  }

  bool operator<(const Scalar& other) const {
    if (mName < other.mName)
      return true;
    if (other.mName < mName)
      return false;
    if (mType < other.mType)
      return true;
    if (other.mType < mType)
      return false;
    return false;
  }
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_SCALAR_HPP
