////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataManager.hpp"

#include "../logger.hpp"

#include <ViracochaBackend/DataManager/VrcGenericDataLoader.h>

#include <algorithm>
#include <future>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::DataManager(std::string path, VolumeFileType type)
    : mPath(path)
    , mType(type) {
  setTimestep(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::setTimestep(int timestep) {
  mCurrentTimestep = timestep;
  mDirty           = true;
  if (mCache.find(timestep) == mCache.end()) {
    loadData(timestep);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::cacheTimestep(int timestep) {
  loadData(timestep);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DataManager::isDirty() {
  return mDirty;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> DataManager::getData() {
  auto                        data    = mCache.find(mCurrentTimestep);
  vtkSmartPointer<vtkDataSet> dataset = data->second.get();
  mDirty                              = false;
  return dataset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::loadData(int timestep) {
  logger().info("Loading data from {}, timestep {}...", mPath, timestep);
  auto data        = std::async(std::launch::async,
      [this](std::string path, VolumeFileType type, int timestep) {
        std::lock_guard<std::mutex> lock(mReadMutex);
        auto                        timer = std::chrono::high_resolution_clock::now();
        vtkSmartPointer<vtkDataSet> data;
        switch (type) {
        case VolumeFileType::eGaia:
          data = VrcGenericDataLoader::LoadGaiaDataSet(path.c_str(), timestep, nullptr);
          break;
        case VolumeFileType::eVtk:
          data = VrcGenericDataLoader::LoadVtkDataSet(path.c_str());
          break;
        }
        logger().info("Finished loading data from {}, timestep {}. Took {}s", path, timestep,
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);
        return data;
      },
      mPath, mType, timestep);
  mCache[timestep] = std::move(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
