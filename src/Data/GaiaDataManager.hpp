////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_GAIADATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_GAIADATAMANAGER_HPP

#include "../logger.hpp"

#include "DataManager.hpp"

namespace csp::volumerendering {

class GaiaDataManager : public DataManager {
 public:
  /// Create a DataManager that can read files produced with the GAIA simulation code.
  /// All files in 'path' matching 'filenamePattern' can be accessed using the DataManager.
  GaiaDataManager(std::string path, std::string filenamePattern);

 protected:
  vtkSmartPointer<vtkCellArray> mGaiaCells;

  vtkSmartPointer<vtkDataSet> loadDataImpl(int timestep) override;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_GAIADATAMANAGER_HPP
