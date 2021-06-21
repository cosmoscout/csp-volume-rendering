////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataManager.hpp"

#include "../logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

#include <vtkCellData.h>
#include <vtkPointData.h>

#include <algorithm>
#include <future>
#include <regex>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DataManagerException::what() const noexcept {
  return "Failed to initialize DataManager.";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::DataManager(std::string path, std::string filenamePattern, std::string pathlinesPath)
    : mPathlines(pathlinesPath) {
  std::regex patternRegex;
  try {
    patternRegex = std::regex(".*" + filenamePattern);
  } catch (const std::regex_error& e) {
    logger().error(
        "Filename pattern '{}' is not a valid regular expression: {}", filenamePattern, e.what());
    throw DataManagerException();
  }

  bool haveLodFiles = false;
  if (patternRegex.mark_count() == 2) {
    haveLodFiles = true;
  } else if (patternRegex.mark_count() != 1) {
    logger().error(
        "Filename pattern '{}' has the wrong amount of capture groups: {}! The pattern should "
        "contain only one capture group capturing the timestep component of the filename.",
        filenamePattern, patternRegex.mark_count());
    throw DataManagerException();
  }

  std::set<std::string> files;
  try {
    files = cs::utils::filesystem::listFiles(path, patternRegex);
  } catch (const boost::filesystem::filesystem_error& e) {
    logger().error("Loading volume data from '{}' failed: {}", path, e.what());
    throw DataManagerException();
  }

  std::vector<Timestep> timesteps;
  for (std::string file : files) {
    file = std::regex_replace(file, std::regex(R"(\\)"), "/");
    std::smatch match;
    std::regex_search(file, match, patternRegex);

    Timestep timestep;
    Lod      lod;
    try {
      timestep = std::stoi(match[haveLodFiles ? 2 : 1].str());
    } catch (const std::invalid_argument&) {
      logger().error("Capture group in filename pattern '{}' does not match an integer for file "
                     "'{}': Match of group is '{}'! A suitable capture group could be '([0-9])+'.",
          filenamePattern, file, match[haveLodFiles ? 2 : 1].str());
      throw DataManagerException();
    }
    try {
      lod = haveLodFiles ? std::stoi(match[1].str()) : 0;
    } catch (const std::invalid_argument&) {
      logger().error("Capture group in filename pattern '{}' does not match an integer for file "
                     "'{}': Match of group is '{}'! A suitable capture group could be '([0-9])+'.",
          filenamePattern, file, match[1].str());
      throw DataManagerException();
    }

    timesteps.push_back(timestep);
    mFiles[timestep][lod] = file;
  }
  if (timesteps.size() == 0) {
    logger().error("No files matching '{}' found in '{}'!", filenamePattern, path);
    throw DataManagerException();
  }
  pTimesteps.set(timesteps);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::~DataManager() {
  if (mInitScalarsThread.joinable()) {
    mInitScalarsThread.join();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DataManager::isReady() {
  std::scoped_lock lock(mScalarsMutex);
  return pScalars.get().size() > 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::utils::Signal<Scalar const&> const& DataManager::onScalarRangeUpdated() const {
  return mOnScalarRangeUpdated;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<double, 2> DataManager::getScalarRange(Scalar const& scalar) {
  std::scoped_lock lock(mStateMutex);
  return mScalarRanges[scalar.getId()];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<double, 2> DataManager::getScalarRange(std::string scalarId) {
  std::scoped_lock lock(mStateMutex);
  return mScalarRanges[scalarId];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setTimestep(int timestep) {
  {
    std::scoped_lock lock(mStateMutex);
    if (std::find(pTimesteps.get().begin(), pTimesteps.get().end(), timestep) !=
        pTimesteps.get().end()) {
      mCurrentTimestep = timestep;
    } else {
      logger().warn("'{}' is not a timestep in the current dataset. '{}' will be used instead.",
          timestep, mCurrentTimestep);
      timestep = mCurrentTimestep;
    }
  }
  mTimestepCv.notify_all();
  getFromCache(timestep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::cacheTimestep(int timestep) {
  if (std::find(pTimesteps.get().begin(), pTimesteps.get().end(), timestep) ==
      pTimesteps.get().end()) {
    logger().warn(
        "'{}' is not a timestep in the current dataset. No timestep will be cached.", timestep);
    return;
  }
  getFromCache(timestep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setActiveScalar(std::string scalarId) {
  std::scoped_lock lock(mStateMutex, mScalarsMutex);
  auto             scalar = std::find_if(pScalars.get().begin(), pScalars.get().end(),
      [&scalarId](Scalar const& s) { return s.getId() == scalarId; });
  if (scalar != pScalars.get().end()) {
    mActiveScalar = *scalar;
  } else {
    logger().warn("'{}' is not a scalar in the current dataset. '{}' will be used instead.",
        scalarId, mActiveScalar.getId());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> DataManager::getData() {
  return getData(getState());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> DataManager::getData(State state) {
  std::shared_future<vtkSmartPointer<vtkDataSet>> futureData = getFromCache(state.mTimestep);
  vtkSmartPointer<vtkDataSet>                     dataset    = futureData.get();
  if (!dataset) {
    logger().error("Loaded data is null! Is the data type correctly set in the settings?");
    throw std::runtime_error("Loaded data is null.");
  }
  if (state.mScalar.mName != "") {
    switch (state.mScalar.mType) {
    case ScalarType::ePointData:
      dataset->GetPointData()->SetActiveScalars(state.mScalar.mName.c_str());
      break;
    case ScalarType::eCellData:
      dataset->GetCellData()->SetActiveScalars(state.mScalar.mName.c_str());
      break;
    }
  }
  return dataset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::State DataManager::getState() {
  std::scoped_lock lock(mStateMutex);
  State            state{mCurrentTimestep, mActiveScalar};
  return state;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int DataManager::getMaxLod(State state) {
  std::scoped_lock lock(mDataMutex);
  auto             cacheEntry = mCache.find(state.mTimestep);
  if (cacheEntry == mCache.end()) {
    return 0;
  }
  Lod maxLod = 0;
  for (auto const& lod : cacheEntry->second) {
    if (lod.second.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      maxLod = lod.first;
    }
  }
  return maxLod;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Pathlines const& DataManager::getPathlines() const {
  return mPathlines;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::initState() {
  Timestep timestep;
  {
    std::scoped_lock lock(mStateMutex);
    timestep = pTimesteps.get()[0];
  }
  setTimestep(timestep);
  mInitScalarsThread = std::thread(&DataManager::initScalars, this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::initScalars() {
  vtkSmartPointer<vtkDataSet> data;
  try {
    data = getData();
  } catch (const std::exception&) {
    logger().error("Could not load initial data for determining available scalars! "
                   "Requested data may have no active scalar.");
    return;
  }

  std::vector<Scalar> scalars;
  {
    std::scoped_lock lock(mStateMutex);
    for (int i = 0; i < data->GetPointData()->GetNumberOfArrays(); i++) {
      if (data->GetPointData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
        Scalar scalar;
        scalar.mName = data->GetPointData()->GetArrayName(i);
        scalar.mType = ScalarType::ePointData;
        data->GetPointData()
            ->GetScalars(scalar.mName.c_str())
            ->GetRange(mScalarRanges[scalar.getId()].data());
        scalars.push_back(scalar);
      }
    }
    for (int i = 0; i < data->GetCellData()->GetNumberOfArrays(); i++) {
      if (data->GetCellData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
        Scalar scalar;
        scalar.mName = data->GetCellData()->GetArrayName(i);
        scalar.mType = ScalarType::eCellData;
        data->GetCellData()
            ->GetScalars(scalar.mName.c_str())
            ->GetRange(mScalarRanges[scalar.getId()].data());
        scalars.push_back(scalar);
      }
    }
    if (scalars.size() == 0) {
      logger().error("No scalars found in volume data! Requested data may have no active scalar.");
      return;
    }
    mActiveScalar = scalars[0];
  }
  {
    std::scoped_lock lock(mScalarsMutex);
    pScalars.setWithNoEmit(scalars);
  }
  pScalars.touch();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_future<vtkSmartPointer<vtkDataSet>> DataManager::getFromCache(Timestep timestep) {
  auto lods   = mFiles.find(timestep)->second;
  Lod  maxLod = getMaxLod({timestep, mActiveScalar});

  std::scoped_lock lock(mDataMutex);
  auto             cacheEntry = mCache.find(timestep);
  if (cacheEntry == mCache.end()) {
    Lod lod = lods.lower_bound(0)->first;
    loadData(timestep, lod);
    return mCache[timestep][lod];
  }
  std::thread loadNext([this, maxLod, lods, timestep]() {
    {
      std::unique_lock<std::mutex> cvLock(mStateMutex);
      if (mTimestepCv.wait_for(cvLock, std::chrono::seconds(2),
              [this, timestep]() { return mCurrentTimestep != timestep; })) {
        return;
      }
    }
    std::scoped_lock lock(mDataMutex);
    auto             cacheEntry = mCache.find(timestep);
    auto             loadingLod = cacheEntry->second.upper_bound(maxLod);
    auto             nextLod    = lods.upper_bound(maxLod);
    if (loadingLod == cacheEntry->second.end() && nextLod != lods.end()) {
      loadData(timestep, nextLod->first);
    }
  });
  loadNext.detach();
  return mCache[timestep][maxLod];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::loadData(Timestep timestep, Lod lod) {
  logger().info("Loading data for timestep {}, level of detail {}...", timestep, lod);
  auto data = std::async(
      std::launch::async,
      [this](Timestep timestep, Lod lod) {
        std::chrono::high_resolution_clock::time_point timer;
        vtkSmartPointer<vtkDataSet>                    data;
        {
          std::scoped_lock lock(mReadMutex);
          timer = std::chrono::high_resolution_clock::now();
          data  = loadDataImpl(timestep, lod);
        }
        std::vector<Scalar> updatedScalars;
        {
          std::scoped_lock lock(mStateMutex, mScalarsMutex);
          for (auto const& scalar : pScalars.get()) {
            bool                  rangeUpdated = false;
            std::array<double, 2> range;
            switch (scalar.mType) {
            case ScalarType::ePointData:
              data->GetPointData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
              break;
            case ScalarType::eCellData:
              data->GetCellData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
              break;
            }
            if (range[0] < mScalarRanges[scalar.getId()][0]) {
              mScalarRanges[scalar.getId()][0] = range[0];
              rangeUpdated                     = true;
            }
            if (range[1] > mScalarRanges[scalar.getId()][1]) {
              mScalarRanges[scalar.getId()][1] = range[1];
              rangeUpdated                     = true;
            }
            if (rangeUpdated) {
              updatedScalars.push_back(scalar);
            }
          }
        }
        for (auto const& scalar : updatedScalars) {
          mOnScalarRangeUpdated.emit(scalar);
        }

        logger().info("Finished loading data for timestep {}, level of detail {}. Took {}s.",
            timestep, lod,
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);

        return data;
      },
      timestep, lod);
  mCache[timestep][lod] = std::move(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
