////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_DATAMANAGER_HPP

#include "../../../src/cs-utils/Property.hpp"

#include <vtk-8.2/vtkDataSet.h>
#include <vtk-8.2/vtkSmartPointer.h>

#include <future>
#include <map>
#include <mutex>
#include <string>

namespace csp::volumerendering {

class DataManager {
 public:
  enum class VolumeFileType { eInvalid = -1, eGaia, eVtk };

  DataManager(std::string path, std::string filenamePattern, VolumeFileType type);

  cs::utils::Property<std::vector<int>>         pTimesteps;
  cs::utils::Property<std::vector<std::string>> pScalars;

  void setTimestep(int timestep);
  void cacheTimestep(int timestep);
  bool isDirty();

  void setActiveScalar(std::string scalar);

  vtkSmartPointer<vtkDataSet> getData();

 private:
  VolumeFileType mType;
  int            mCurrentTimestep;
  std::string    mActiveScalar;
  bool           mDirty;

  std::mutex mReadMutex;

  std::map<int, std::string> mTimestepFiles;

  std::map<int, std::shared_future<vtkSmartPointer<vtkDataSet>>> mCache;

  void                        loadData(int timestep);
  vtkSmartPointer<vtkDataSet> loadGaiaData(int timestep);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DATAMANAGER_HPP
