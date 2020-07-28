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

void DataManager::loadData(std::string path, int timestep) {
  logger().info("Loading data from {}, timestep {}...", path, timestep);
  DataSet data;
  data.mPath         = path;
  data.mTimestep     = timestep;
  data.mFutureData   = std::async(std::launch::async,
      [](std::string path, int timestep) {
        vtkSmartPointer<vtkUnstructuredGrid> data = vtkUnstructuredGrid::SafeDownCast(
            VrcGenericDataLoader::LoadGaiaDataSet(path.c_str(), timestep, nullptr));
        logger().info("Finished loading data from {}, timestep {}.", path, timestep);
        return data;
      },
      path, timestep);
  mCache.push_back(std::move(data));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkUnstructuredGrid> DataManager::getData(std::string path, int timestep) {
  auto data = std::find_if(mCache.begin(), mCache.end(),
      [&path, &timestep](const DataSet& d) { return d.mPath == path && d.mTimestep == timestep; });

  std::shared_future<vtkSmartPointer<vtkUnstructuredGrid>> future = data->mFutureData;
  return future.get();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
