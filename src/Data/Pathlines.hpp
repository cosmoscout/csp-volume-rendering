////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_PATHLINES_HPP
#define CSP_VOLUME_RENDERING_PATHLINES_HPP

#include "Scalar.hpp"

#include <rkcommon/math/vec.h>

#include <vtkPolyData.h>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace csp::volumerendering {

class PathlinesException : public std::exception {
 public:
  const char* what() const noexcept override;
};

class Pathlines {
 public:
  Pathlines(std::string const& file);

  std::vector<Scalar> const&                          getScalars() const;
  std::map<std::string, std::array<double, 2>> const& getScalarRanges() const;

  std::string const& getCsvData() const;

  std::vector<rkcommon::math::vec4f> getVertices(float lineSize) const;
  std::vector<rkcommon::math::vec4f> getColors(
      std::string const& scalarId, float lineOpacity) const;
  std::vector<rkcommon::math::vec2f> getTexCoords(
      std::string const& xScalarId, std::string const& yScalarId) const;
  std::vector<uint32_t> getIndices(std::vector<ScalarFilter> const& filters) const;

 private:
  bool isCellValid(
      std::vector<std::pair<Scalar, ScalarFilter>> const& filters, vtkIdType cellId) const;

  std::string                                  mCsvData;
  vtkSmartPointer<vtkPolyData>                 mData;
  std::vector<Scalar>                          mScalars;
  std::map<std::string, std::array<double, 2>> mScalarRanges;
};

} // namespace csp::volumerendering

#undef likely

#endif // CSP_VOLUME_RENDERING_PATHLINES_HPP
