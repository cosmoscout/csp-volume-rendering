////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_DATAMANAGER_HPP

#include "vtk-8.2/vtkSmartPointer.h"
#include "vtk-8.2/vtkUnstructuredGrid.h"

#include <future>
#include <map>
#include <string>

namespace csp::volumerendering {

class DataManager {
 public:
  void                                 loadData(std::string path, int timestep);
  vtkSmartPointer<vtkUnstructuredGrid> getData(std::string path, int timestep);

 private:
  struct DataSet {
    std::string mPath;
    int         mTimestep;

    std::shared_future<vtkSmartPointer<vtkUnstructuredGrid>> mFutureData;
  };

  std::vector<DataSet> mCache;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DATAMANAGER_HPP
