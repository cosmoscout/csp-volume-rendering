////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataManager.hpp"

#include "../logger.hpp"

#include <vtk-8.2/vtkCellData.h>
#include <vtk-8.2/vtkCellSizeFilter.h>
#include <vtk-8.2/vtkDoubleArray.h>
#include <vtk-8.2/vtkExtractSelection.h>
#include <vtk-8.2/vtkPointData.h>
#include <vtk-8.2/vtkSelection.h>
#include <vtk-8.2/vtkSelectionNode.h>

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
  dataset->GetPointData()->SetActiveScalars("T");
  mDirty = false;
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
          data = loadGaiaData(timestep);
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

vtkSmartPointer<vtkDataSet> DataManager::loadGaiaData(int timestep) {
  vtkSmartPointer<vtkDataSet> data;
  data = VrcGenericDataLoader::LoadGaiaDataSet(mPath.c_str(), timestep, nullptr);

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
  thresholds->InsertNextTuple2(0.000001, 2.59941e-05);

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
