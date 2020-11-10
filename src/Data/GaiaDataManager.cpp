////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GaiaDataManager.hpp"

#include "../logger.hpp"

#include <vtk-8.2/vtkCellData.h>
#include <vtk-8.2/vtkCellSizeFilter.h>
#include <vtk-8.2/vtkDoubleArray.h>
#include <vtk-8.2/vtkExtractSelection.h>
#include <vtk-8.2/vtkSelection.h>
#include <vtk-8.2/vtkSelectionNode.h>
#include <vtk-8.2/vtkUnstructuredGrid.h>

#include <ViracochaBackend/DataManager/VrcGenericDataLoader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

GaiaDataManager::GaiaDataManager(std::string path, std::string filenamePattern)
    : DataManager(path, filenamePattern) {
  initState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> GaiaDataManager::loadDataImpl(int timestep) {
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
