////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_VTKDATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_VTKDATAMANAGER_HPP

#include "../logger.hpp"

#include "DataManager.hpp"

namespace csp::volumerendering {

/// DataManager with an implementation for reading vtk data files.
class VtkDataManager : public DataManager {
 public:
  /// Create a DataManager that can read files in the vtk data format.
  /// All files in 'path' matching 'filenamePattern' can be accessed using the DataManager.
  VtkDataManager(std::string path, std::string filenamePattern, std::string pathlinesPath);

 protected:
  vtkSmartPointer<vtkDataSet> loadDataImpl(Timestep timestep, Lod lod) override;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_VTKDATAMANAGER_HPP
