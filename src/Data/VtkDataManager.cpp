////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VtkDataManager.hpp"

#include "../logger.hpp"

#include <vtk-8.1/vtkCellDataToPointData.h>
#include <vtk-8.1/vtkDataSetReader.h>
#include <vtk-8.1/vtkXMLFileReadTester.h>
#include <vtk-8.1/vtkXMLGenericDataObjectReader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

VtkDataManager::VtkDataManager(std::string path, std::string filenamePattern)
    : DataManager(path, filenamePattern) {
  initState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> VtkDataManager::loadDataImpl(int timestep) {
  vtkSmartPointer<vtkDataSet> data;

  auto fileTester = vtkSmartPointer<vtkXMLFileReadTester>::New();
  fileTester->SetFileName(mTimestepFiles[timestep].c_str());

  if (fileTester->TestReadFile() > 0) {
    // Is an XML File in new vtk data format
    auto reader = vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(mTimestepFiles[timestep].c_str());
    reader->Update();

    data = reader->GetOutputAsDataSet();
  } else {
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(mTimestepFiles[timestep].c_str());
    reader->ReadAllScalarsOn();
    reader->Update();

    data = reader->GetOutput();
  }

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
