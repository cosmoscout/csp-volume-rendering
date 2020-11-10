////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataManager.hpp"

#include "../logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

#include <vtk-8.2/vtkCellData.h>
#include <vtk-8.2/vtkCellSizeFilter.h>
#include <vtk-8.2/vtkDoubleArray.h>
#include <vtk-8.2/vtkExtractSelection.h>
#include <vtk-8.2/vtkPointData.h>
#include <vtk-8.2/vtkSelection.h>
#include <vtk-8.2/vtkSelectionNode.h>
#include <vtk-8.2/vtkUnstructuredGrid.h>

#include <ViracochaBackend/DataManager/VrcGenericDataLoader.h>

#include <algorithm>
#include <future>
#include <regex>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr char* DATAMANAGER_ERROR_MESSAGE = "Failed to initialize DataManager.";

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::DataManager(std::string path, std::string filenamePattern, VolumeFileType type)
    : mType(type) {
  std::regex patternRegex;
  try {
    patternRegex = std::regex(".*" + filenamePattern);
  } catch (const std::regex_error& e) {
    logger().error(
        "Filename pattern '{}' is not a valid regular expression: {}", filenamePattern, e.what());
    throw std::exception(DATAMANAGER_ERROR_MESSAGE);
  }
  if (patternRegex.mark_count() != 1) {
    logger().error(
        "Filename pattern '{}' has the wrong amount of capture groups: {}! The pattern should "
        "contain only one capture group capturing the timestep component of the filename.",
        filenamePattern, patternRegex.mark_count());
    throw std::exception(DATAMANAGER_ERROR_MESSAGE);
  }

  std::set<std::string> files;
  try {
    files = cs::utils::filesystem::listFiles(path, patternRegex);
  } catch (const boost::filesystem::filesystem_error& e) {
    logger().error("Loading volume data from '{}' failed: {}", path, e.what());
    throw std::exception(DATAMANAGER_ERROR_MESSAGE);
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
      throw std::exception(DATAMANAGER_ERROR_MESSAGE);
    }

    timesteps.push_back(timestep);
    mTimestepFiles[timestep] = file;
  }
  if (timesteps.size() == 0) {
    logger().error("No files matching '{}' found in '{}'!", filenamePattern, path);
    throw std::exception(DATAMANAGER_ERROR_MESSAGE);
  }
  pTimesteps.set(timesteps);
  setTimestep(timesteps[0]);
  mInitScalarsThread = std::thread(&DataManager::initScalars, this);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::~DataManager() {
  mInitScalarsThread.join();
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

void DataManager::setActiveScalar(std::string scalar) {
  std::scoped_lock lock(mStateMutex);
  if (std::find(pScalars.get().begin(), pScalars.get().end(), scalar) != pScalars.get().end()) {
    mActiveScalar = scalar;
    mDirty        = true;
  } else {
    logger().warn("'{}' is not a scalar in the current dataset. '{}' will be used instead.", scalar,
        mActiveScalar);
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
  dataset->GetPointData()->SetActiveScalars(state.mScalar.c_str());
  return dataset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DataManager::State DataManager::getState() {
  std::scoped_lock lock(mStateMutex);
  State            state{mCurrentTimestep, mActiveScalar};
  return state;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::initScalars() {
  vtkSmartPointer<vtkDataSet> data = getData();
  std::vector<std::string>    scalars;
  for (int i = 0; i < data->GetPointData()->GetNumberOfArrays(); i++) {
    if (data->GetPointData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
      scalars.push_back(data->GetPointData()->GetArrayName(i));
    }
  }
  if (scalars.size() == 0) {
    logger().error("No scalars found in volume data!");
    return;
  }
  std::scoped_lock lock(mStateMutex);
  mActiveScalar = scalars[0];
  pScalars.set(scalars);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DataManager::loadData(int timestep) {
  logger().info("Loading data from {}, timestep {}...", mTimestepFiles[timestep], timestep);
  auto data        = std::async(std::launch::async,
      [this](std::string path, VolumeFileType type, int timestep) {
        std::scoped_lock            lock(mReadMutex);
        auto                        timer = std::chrono::high_resolution_clock::now();
        vtkSmartPointer<vtkDataSet> data;

        switch (type) {
        case VolumeFileType::eGaia:
          data = loadGaiaData(timestep);
          break;
        case VolumeFileType::eVtk:
          data = VrcGenericDataLoader::LoadVtkDataSet(path.c_str());
          break;
        }

        logger().info("Finished loading data from {}, timestep {}. Took {}s.", path, timestep,
            (float)(std::chrono::high_resolution_clock::now() - timer).count() / 1000000000);

        return data;
      },
      mTimestepFiles[timestep], mType, timestep);
  mCache[timestep] = std::move(data);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> DataManager::loadGaiaData(int timestep) {
  vtkSmartPointer<vtkDataSet> data;
  data =
      VrcGenericDataLoader::LoadGaiaDataSet(mTimestepFiles[timestep].c_str(), timestep, mGaiaCells);
  if (mGaiaCells == nullptr) {
    mGaiaCells = vtkUnstructuredGrid::SafeDownCast(data)->GetCells();
  }

  vtkSmartPointer<vtkCellSizeFilter> sizeFilter = vtkSmartPointer<vtkCellSizeFilter>::New();
  sizeFilter->SetComputeArea(false);
  sizeFilter->SetComputeLength(false);
  sizeFilter->SetComputeSum(false);
  sizeFilter->SetComputeVertexCount(false);
  sizeFilter->SetComputeVolume(true);
  sizeFilter->SetInputData(data);
  sizeFilter->Update();
  data = vtkDataSet::SafeDownCast(sizeFilter->GetOutput());
  data->GetCellData()->SetActiveScalars("Volume");

  vtkSmartPointer<vtkDoubleArray> thresholds = vtkSmartPointer<vtkDoubleArray>::New();
  thresholds->SetNumberOfComponents(2);
  thresholds->InsertNextTuple2(1.e-06, 2.59941e-05);

  vtkSmartPointer<vtkSelectionNode> selectionNode = vtkSmartPointer<vtkSelectionNode>::New();
  selectionNode->SetContentType(vtkSelectionNode::SelectionContent::THRESHOLDS);
  selectionNode->SetFieldType(vtkSelectionNode::SelectionField::CELL);
  selectionNode->SetSelectionList(thresholds);

  vtkSmartPointer<vtkSelection> selection = vtkSmartPointer<vtkSelection>::New();
  selection->AddNode(selectionNode);

  vtkSmartPointer<vtkExtractSelection> extract = vtkSmartPointer<vtkExtractSelection>::New();
  extract->SetInputData(0, data);
  extract->SetInputData(1, selection);
  extract->Update();
  data = vtkDataSet::SafeDownCast(extract->GetOutput());
  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
