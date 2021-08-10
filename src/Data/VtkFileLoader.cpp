////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VtkFileLoader.hpp"

#include "../logger.hpp"

#include <vtkDataSetReader.h>
#include <vtkXMLFileReadTester.h>
#include <vtkXMLGenericDataObjectReader.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

vtkSmartPointer<vtkDataSet> VtkFileLoader::loadDataImpl(std::string const& file) {
  vtkSmartPointer<vtkDataSet> data;

  auto fileTester = vtkSmartPointer<vtkXMLFileReadTester>::New();
  fileTester->SetFileName(file.c_str());

  if (fileTester->TestReadFile() > 0) {
    // Is an XML File in new vtk data format
    auto reader = vtkSmartPointer<vtkXMLGenericDataObjectReader>::New();
    reader->SetFileName(file.c_str());
    reader->Update();

    data = reader->GetOutputAsDataSet();
  } else {
    auto reader = vtkSmartPointer<vtkDataSetReader>::New();
    reader->SetFileName(file.c_str());
    reader->ReadAllScalarsOn();
    reader->Update();

    data = reader->GetOutput();
  }

  return data;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
