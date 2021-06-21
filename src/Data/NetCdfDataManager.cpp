////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "NetCdfDataManager.hpp"

#include "../logger.hpp"

#include <vtkNetCDFCFReader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

NetCdfDataManager::NetCdfDataManager(
    std::string path, std::string filenamePattern, std::string pathlinesPath)
    : DataManager(path, filenamePattern, pathlinesPath) {
  initState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> NetCdfDataManager::loadDataImpl(Timestep timestep, Lod lod) {
  vtkSmartPointer<vtkDataSet> data;

  auto reader = vtkSmartPointer<vtkNetCDFCFReader>::New();
  reader->SetFileName(mFiles[timestep][lod].c_str());
  reader->SphericalCoordinatesOn();
  reader->SetDimensions("(lat, r, lon)");
  reader->Update();

  data = vtkDataSet::SafeDownCast(reader->GetOutput());
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
