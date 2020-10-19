////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_DATAMANAGER_HPP

#include "vtk-8.2/vtkDataSet.h"
#include "vtk-8.2/vtkSmartPointer.h"

#include <future>
#include <map>
#include <mutex>
#include <string>

namespace csp::volumerendering {

class DataManager {
 public:
  enum class VolumeFileType { eInvalid = -1, eGaia, eVtk };

  DataManager(std::string path, VolumeFileType type);

  void setTimestep(int timestep);
  void cacheTimestep(int timestep);
  bool isDirty();

  vtkSmartPointer<vtkDataSet> getData();

 private:
  std::string    mPath;
  VolumeFileType mType;
  int            mCurrentTimestep;
  bool           mDirty;

  std::mutex mReadMutex;

  std::map<int, std::shared_future<vtkSmartPointer<vtkDataSet>>> mCache;

  void loadData(int timestep);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DATAMANAGER_HPP
