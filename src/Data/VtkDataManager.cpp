////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VtkDataManager.hpp"

#include "../logger.hpp"

#include <vtkCellDataToPointData.h>
#include <vtkDataSetReader.h>
#include <vtkXMLFileReadTester.h>
#include <vtkXMLGenericDataObjectReader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

VtkDataManager::VtkDataManager(
    std::string path, std::string filenamePattern, std::string pathlinesPath)
    : DataManager(path, filenamePattern, pathlinesPath) {
  initState();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> VtkDataManager::loadDataImpl(Timestep timestep, Lod lod) {
  vtkSmartPointer<vtkDataSet> data;

  auto fileTester = vtkSmartPointer<vtkXMLFileReadTester>::New();
  fileTester->SetFileName(mFiles[timestep][lod].c_str());

  if (fileTester->TestReadFile() > 0) {
    // Is an XML File in new vtk data format
    auto reader = vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(mFiles[timestep][lod].c_str());
    reader->Update();

    data = reader->GetOutputAsDataSet();
  } else {
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(mFiles[timestep][lod].c_str());
    reader->ReadAllScalarsOn();
    reader->Update();

    data = reader->GetOutput();
  }

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
