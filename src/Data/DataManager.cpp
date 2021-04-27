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

DataManager::DataManager(std::string path, std::string filenamePattern) {
  std::regex patternRegex;
  try {
    patternRegex = std::regex(".*" + filenamePattern);
  } catch (const std::regex_error& e) {
    logger().error(
        "Filename pattern '{}' is not a valid regular expression: {}", filenamePattern, e.what());
    throw DataManagerException();
  }
  if (patternRegex.mark_count() != 1) {
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

  std::vector<int> timesteps;
  for (std::string file : files) {
    file = std::regex_replace(file, std::regex(R"(\\)"), "/");
    std::smatch match;
    std::regex_search(file, match, patternRegex);

    int timestep;
    try {
      timestep = std::stoi(match[1].str());
    } catch (const std::invalid_argument&) {
      logger().error("Capture group in filename pattern '{}' does not match an integer for file "
                     "'{}': Match of group is '{}'! A suitable capture group could be '([0-9])+'.",
          filenamePattern, file, match[1].str());
      throw DataManagerException();
    }

    timesteps.push_back(timestep);
    mTimestepFiles[timestep] = file;
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
  std::scoped_lock lock(mStateMutex);
  return pScalars.get().size() > 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setTimestep(int timestep) {
  std::scoped_lock lock(mStateMutex, mDataMutex);
  if (std::find(pTimesteps.get().begin(), pTimesteps.get().end(), timestep) !=
      pTimesteps.get().end()) {
    mCurrentTimestep = timestep;
    mDirty           = true;
  } else {
    logger().warn("'{}' is not a timestep in the current dataset. '{}' will be used instead.",
        timestep, mCurrentTimestep);
  }
  if (mCache.find(timestep) == mCache.end()) {
    loadData(timestep);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::cacheTimestep(int timestep) {
  std::scoped_lock lock(mDataMutex);
  if (mCache.find(timestep) == mCache.end()) {
    loadData(timestep);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DataManager::isDirty() {
  return mDirty;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setActiveScalar(std::string scalarId) {
  std::scoped_lock lock(mStateMutex);
  auto             scalar = std::find_if(pScalars.get().begin(), pScalars.get().end(),
      [&scalarId](Scalar const& s) { return s.getId() == scalarId; });
  if (scalar != pScalars.get().end()) {
    mActiveScalar = *scalar;
    mDirty        = true;
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
  std::shared_future<vtkSmartPointer<vtkDataSet>> futureData;
  {
    std::scoped_lock lock(mDataMutex);
    mDirty          = false;
    auto cacheEntry = mCache.find(state.mTimestep);
    if (cacheEntry == mCache.end()) {
      loadData(state.mTimestep);
      cacheEntry = mCache.find(state.mTimestep);
    }
    futureData = cacheEntry->second;
  }
  vtkSmartPointer<vtkDataSet> dataset = futureData.get();
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

void DataManager::initState() {
  int timestep;
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
  for (int i = 0; i < data->GetPointData()->GetNumberOfArrays(); i++) {
    if (data->GetPointData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
      Scalar scalar;
      scalar.mName = data->GetPointData()->GetArrayName(i);
      scalar.mType = ScalarType::ePointData;
      data->GetPointData()->GetScalars(scalar.mName.c_str())->GetRange(scalar.mRange.data());
      scalars.push_back(scalar);
    }
  }
  for (int i = 0; i < data->GetCellData()->GetNumberOfArrays(); i++) {
    if (data->GetCellData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
      Scalar scalar;
      scalar.mName = data->GetCellData()->GetArrayName(i);
      scalar.mType = ScalarType::eCellData;
      data->GetCellData()->GetScalars(scalar.mName.c_str())->GetRange(scalar.mRange.data());
      scalars.push_back(scalar);
    }
  }
  if (scalars.size() == 0) {
    logger().error("No scalars found in volume data! Requested data may have no active scalar.");
    return;
  }
  std::scoped_lock lock(mStateMutex);
  mActiveScalar = scalars[0];
  pScalars.set(scalars);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::loadData(int timestep) {
  logger().info("Loading data from {}, timestep {}...", mTimestepFiles[timestep], timestep);
  auto data = std::async(
      std::launch::async,
      [this](std::string path, int timestep) {
        std::chrono::high_resolution_clock::time_point timer;
        vtkSmartPointer<vtkDataSet>                    data;
        {
          std::scoped_lock lock(mReadMutex);
          timer = std::chrono::high_resolution_clock::now();
          data  = loadDataImpl(timestep);
        }
        {
          std::scoped_lock(mStateMutex);
          std::vector<Scalar> scalars;
          bool                scalarsUpdated = false;
          for (auto scalar : pScalars.get()) {
            std::array<double, 2> range;
            switch (scalar.mType) {
            case ScalarType::ePointData:
              data->GetPointData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
              break;
            case ScalarType::eCellData:
              data->GetCellData()->GetScalars(scalar.mName.c_str())->GetRange(range.data());
              break;
            }
            if (range[0] < scalar.mRange[0]) {
              scalar.mRange[0] = range[0];
              scalarsUpdated   = true;
            }
            if (range[1] > scalar.mRange[1]) {
              scalar.mRange[1] = range[1];
              scalarsUpdated   = true;
            }
            scalars.push_back(scalar);
          }
          if (scalarsUpdated) {
            pScalars.set(scalars);
          }
        }

        logger().info("Finished loading data from {}, timestep {}. Took {}s.", path, timestep,
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);

        return data;
      },
      mTimestepFiles[timestep], timestep);
  mCache[timestep] = std::move(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
