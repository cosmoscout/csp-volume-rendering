////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_DATAMANAGER_HPP

#include "../logger.hpp"

#include "../../../src/cs-utils/Property.hpp"

#include <vtk-8.2/vtkCellArray.h>
#include <vtk-8.2/vtkDataSet.h>
#include <vtk-8.2/vtkSmartPointer.h>

#include <future>
#include <map>
#include <mutex>
#include <string>

namespace csp::volumerendering {

/// The DataManager class provides an interface for accessing multiple volumetric data files in one
/// directory. Implementations for reading different kinds of data formats are supplied in
/// subclasses. The DataManager class can't be instantiated, use one of the subclasses instead.
class DataManager {
 public:
  struct State {
    int         mTimestep;
    std::string mScalar;

    bool operator<(const State& other) const {
      if (mTimestep < other.mTimestep)
        return true;
      if (other.mTimestep < mTimestep)
        return false;
      if (mScalar < other.mScalar)
        return true;
      if (other.mScalar < mScalar)
        return false;
      return false;
    }
  };

  virtual ~DataManager();

  /// List of timesteps for which files were found.
  cs::utils::Property<std::vector<int>> pTimesteps;
  /// List of available scalars in the data.
  cs::utils::Property<std::vector<std::string>> pScalars;

  /// Until this returns true queried data will have no active scalar.
  bool isReady();

  /// Sets the current timestep to the given value.
  /// Future calls to getData() will return data for this time.
  void setTimestep(int timestep);
  /// Loads data for the timestep into the cache.
  /// Does not change the current timestep.
  void cacheTimestep(int timestep);
  /// Sets the current scalar to the given value.
  /// Has to be one of the values in pScalars.
  /// Future calls to getData() will return data with this as the active scalar.
  void setActiveScalar(std::string scalar);
  /// Returns whether the current state changed since the last call to getData().
  bool isDirty();

  /// Returns the data for the current state.
  /// May block, if the requested data is not yet loaded.
  /// If isReady() returns false, the data may have no active scalar.
  /// Will throw an std::exception if the data is null.
  vtkSmartPointer<vtkDataSet> getData();
  /// Returns the data for the given state.
  /// May block, if the requested data is not yet loaded.
  /// Will throw an std::exception if the data is null.
  vtkSmartPointer<vtkDataSet> getData(State state);
  /// Returns the current state.
  /// If isReady() returns false, the scalar won't be set yet.
  State getState();

 protected:
  int         mCurrentTimestep;
  std::string mActiveScalar = "";
  bool        mDirty;

  std::mutex mReadMutex;
  std::mutex mStateMutex;
  std::mutex mDataMutex;

  std::map<int, std::string> mTimestepFiles;

  std::map<int, std::shared_future<vtkSmartPointer<vtkDataSet>>> mCache;

  std::thread mInitScalarsThread;

  DataManager(std::string path, std::string filenamePattern);

  void initState();
  void initScalars();

  void                                loadData(int timestep);
  virtual vtkSmartPointer<vtkDataSet> loadDataImpl(int timestep) = 0;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DATAMANAGER_HPP
