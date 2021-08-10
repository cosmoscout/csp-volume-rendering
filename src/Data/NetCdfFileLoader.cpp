////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "NetCdfFileLoader.hpp"

#include "../logger.hpp"

#include <vtkNetCDFCFReader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> NetCdfFileLoader::loadDataImpl(std::string const& file) {
  vtkSmartPointer<vtkDataSet> data;

  auto reader = vtkSmartPointer<vtkNetCDFCFReader>::New();
  reader->SetFileName(file.c_str());
  reader->SphericalCoordinatesOn();
  reader->SetDimensions("(lat, r, lon)");
  reader->Update();

  data = vtkDataSet::SafeDownCast(reader->GetOutput());
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
