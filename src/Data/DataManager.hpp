////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_DATAMANAGER_HPP
#define CSP_VOLUME_RENDERING_DATAMANAGER_HPP

#include "../logger.hpp"

#include "../Enums.hpp"
#include "FileLoader.hpp"
#include "Pathlines.hpp"
#include "Scalar.hpp"

#include "../../../../src/cs-utils/Property.hpp"

#include <vtkCellArray.h>
#include <vtkDataSet.h>
#include <vtkSmartPointer.h>

#include <array>
#include <future>
#include <map>
#include <mutex>
#include <optional>
#include <string>

namespace csp::volumerendering {

class DataManagerException : public std::exception {
 public:
  const char* what() const noexcept override;
};

/// The DataManager class provides an interface for accessing multiple volumetric data files in one
/// directory. Implementations for reading different kinds of data formats are supplied in
/// subclasses. The DataManager class can't be instantiated, use one of the subclasses instead.
class DataManager {
 public:
  struct State {
    int    mTimestep;
    Scalar mScalar;

    bool operator==(const State& other) const {
      return mTimestep == other.mTimestep && mScalar == other.mScalar;
    }

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

  DataManager(std::string const& path, std::string const& filenamePattern,
      std::unique_ptr<FileLoader> fileLoader, std::optional<std::string> const& pathlinesPath = {});
  ~DataManager();

  /// List of timesteps for which files were found.
  cs::utils::Property<std::vector<int>> pTimesteps;
  /// List of available scalars in the data.
  cs::utils::Property<std::vector<Scalar>> pScalars;

  /// Until this returns true queried data will have no active scalar.
  bool isReady();

  /// This signal is called, when the scalar range for the given scalar changed.
  cs::utils::Signal<Scalar const&> const& onScalarRangeUpdated() const;
  /// Gets the min and max value for the given scalar.
  std::array<double, 2> getScalarRange(Scalar const& scalar);
  std::array<double, 2> getScalarRange(std::string scalarId);

  /// Sets the current timestep to the given value.
  /// Future calls to getData() will return data for this time.
  void setTimestep(int timestep);
  /// Loads data for the timestep into the cache.
  /// Does not change the current timestep.
  void cacheTimestep(int timestep);
  /// Sets the current scalar to the given value.
  /// Has to be one of the values in pScalars.
  /// Future calls to getData() will return data with this as the active scalar.
  void setActiveScalar(std::string scalarId);

  /// Returns the volume data as a csv string.
  std::string const& getCsvData();

  /// Returns the data for the current state.
  /// May block, if the requested data is not yet loaded.
  /// If isReady() returns false, the data may have no active scalar.
  /// Will throw an std::exception if the data is null.
  /// If present, lod has to be a level of detail value, that is available for the requested state.
  vtkSmartPointer<vtkDataSet> getData(std::optional<State> state = {}, std::optional<int> lod = {});

  /// Returns the current state.
  /// If isReady() returns false, the scalar won't be set yet.
  State getState();
  /// Returns the maximum level of detail, that can instantly be returned for the given state.
  /// Allows to optionally specify an upper bound for the level of detail.
  int getMaxLod(State state, std::optional<int> max = {});
  /// Returns the minimum level of detail, that can instantly be returned for the given state.
  int getMinLod(State state);

  Pathlines const& getPathlines() const;

 protected:
  using Timestep = int;
  using Lod      = int;

  std::unique_ptr<FileLoader> mFileLoader;
  std::unique_ptr<Pathlines>  mPathlines;

  std::string mCsvData;

  Timestep mCurrentTimestep;
  Scalar   mActiveScalar;

  std::mutex mReadMutex;
  std::mutex mScalarsMutex;
  std::mutex mStateMutex;
  std::mutex mDataMutex;

  std::condition_variable mTimestepCv;

  std::map<Timestep, std::map<Lod, std::string>> mFiles;
  std::map<std::string, std::array<double, 2>>   mScalarRanges;

  std::map<Timestep, std::map<Lod, std::shared_future<vtkSmartPointer<vtkDataSet>>>> mCache;

  std::thread mInitScalarsThread;

  cs::utils::Signal<Scalar const&> mOnScalarRangeUpdated;

  void initState();
  void initScalars();

  std::shared_future<vtkSmartPointer<vtkDataSet>> getFromCache(
      Timestep timestep, std::optional<Lod> lod = {});

  void loadData(Timestep timestep, Lod lod);
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_DATAMANAGER_HPP
