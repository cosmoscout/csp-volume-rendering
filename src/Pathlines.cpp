////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Pathlines.hpp"

#include "logger.hpp"

#include "../../../../src/cs-utils/filesystem.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <vtkCellArrayIterator.h>
#include <vtkCellData.h>
#include <vtkDataSetReader.h>
#include <vtkPointData.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* PathlinesException::what() const noexcept {
  return "Failed to initialize Pathlines.";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Pathlines::Pathlines(std::string const& file) {
  std::string csvFile = file;
  cs::utils::replaceString(csvFile, boost::filesystem::extension(csvFile), ".csv");
  try {
    if (boost::filesystem::exists(csvFile)) {
      mCsvData = cs::utils::filesystem::loadToString(csvFile);
    } else {
      logger().error("No csv pathline data found at '{}'!", csvFile);
      throw PathlinesException();
    }
  } catch (const boost::filesystem::filesystem_error& e) {
    logger().error("Loading csv pathline data from '{}' failed: {}", csvFile, e.what());
    throw PathlinesException();
  }

  vtkSmartPointer<vtkDataSetReader> reader = vtkSmartPointer<vtkDataSetReader>::New();
  reader->SetFileName(file.c_str());
  reader->ReadAllScalarsOn();
  reader->Update();

  mData = vtkPolyData::SafeDownCast(reader->GetOutput());

  for (int i = 0; i < mData->GetPointData()->GetNumberOfArrays(); i++) {
    if (mData->GetPointData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
      Scalar scalar;
      scalar.mName = mData->GetPointData()->GetArrayName(i);
      scalar.mType = ScalarType::ePointData;
      mData->GetPointData()
          ->GetScalars(scalar.mName.c_str())
          ->GetRange(mScalarRanges[scalar.getId()].data());
      mScalars.push_back(scalar);
    }
  }
  for (int i = 0; i < mData->GetCellData()->GetNumberOfArrays(); i++) {
    if (mData->GetCellData()->GetAbstractArray(i)->GetNumberOfComponents() == 1) {
      Scalar scalar;
      scalar.mName = mData->GetCellData()->GetArrayName(i);
      scalar.mType = ScalarType::eCellData;
      mData->GetCellData()
          ->GetScalars(scalar.mName.c_str())
          ->GetRange(mScalarRanges[scalar.getId()].data());
      mScalars.push_back(scalar);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<Scalar> const& Pathlines::getScalars() const {
  return mScalars;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<std::string, std::array<double, 2>> const& Pathlines::getScalarRanges() const {
  return mScalarRanges;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Pathlines::getCsvData() const {
  return mCsvData;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<rkcommon::math::vec4f> Pathlines::getVertices(float lineSize) const {
  std::vector<rkcommon::math::vec4f> vertices(mData->GetNumberOfPoints());

  for (int i = 0; i < mData->GetNumberOfPoints(); i++) {
    std::array<double, 3> pos;
    mData->GetPoint(i, pos.data());
    vertices[i][0] = (float)pos[0];
    vertices[i][1] = (float)pos[1];
    vertices[i][2] = (float)pos[2];
    vertices[i][3] = lineSize;
  }

  return vertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<rkcommon::math::vec4f> Pathlines::getColors(
    std::string const& scalarId, float lineOpacity) const {
  auto scalar = std::find_if(
      mScalars.begin(), mScalars.end(), [scalarId](Scalar s) { return s.getId() == scalarId; });
  std::vector<rkcommon::math::vec4f> colors(mData->GetNumberOfPoints());

  // TODO Handle cell data

  for (int i = 0; i < mData->GetNumberOfPoints(); i++) {
    double* value = mData->GetPointData()->GetScalars(scalar->mName.c_str())->GetTuple(i);
    double  min   = mScalarRanges.find(scalarId)->second[0];
    double  max   = mScalarRanges.find(scalarId)->second[1];
    double  norm  = (*value - min) / (max - min);
    // Red - Black - Blue
    /*colors[i] = rkcommon::math::vec4f{norm > 0.5 ? ((float)norm - 0.5f) * 2.f : 0.f, 0.f,
        norm < 0.5 ? -((float)norm - 0.5f) * 2.f : 0.f, .5f};*/
    // Red - White - Blue
    colors[i] = rkcommon::math::vec4f{norm < 0.5 ? (float)norm * 2.f : 1.f,
        1.f - std::abs(((float)norm - 0.5f) * 2.f),
        norm > 0.5 ? 1.f - ((float)norm - 0.5f) * 2.f : 1.f, lineOpacity};
  }

  return colors;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<rkcommon::math::vec2f> Pathlines::getTexCoords(
    std::string const& xScalarId, std::string const& yScalarId) const {
  std::array<std::string, 2> scalarIds = {xScalarId, yScalarId};
  std::array<Scalar, 2>      scalars;
  auto                       xScalar = std::find_if(
      mScalars.begin(), mScalars.end(), [xScalarId](Scalar s) { return s.getId() == xScalarId; });
  if (xScalar != mScalars.end()) {
    scalars[0] = *xScalar;
  }
  auto yScalar = std::find_if(
      mScalars.begin(), mScalars.end(), [yScalarId](Scalar s) { return s.getId() == yScalarId; });
  if (yScalar != mScalars.end()) {
    scalars[1] = *yScalar;
  }

  std::vector<rkcommon::math::vec2f> texCoords(mData->GetNumberOfPoints());

  for (size_t axis = 0; axis < scalars.size(); axis++) {
    if (scalarIds[axis] == "") {
      for (size_t i = 0; i < texCoords.size(); i++) {
        texCoords[i][axis] = 0.5;
      }
      continue;
    }

    double min = mScalarRanges.find(scalarIds[axis])->second[0];
    double max = mScalarRanges.find(scalarIds[axis])->second[1];
    switch (scalars[axis].mType) {
    case ScalarType::ePointData: {
      for (int i = 0; i < mData->GetNumberOfPoints(); i++) {
        double* value = mData->GetPointData()->GetScalars(scalars[axis].mName.c_str())->GetTuple(i);
        texCoords[i][axis] = (float)((*value - min) / (max - min));
      }
      break;
    }
    case ScalarType::eCellData: {
      auto lineIter = vtk::TakeSmartPointer(mData->GetLines()->NewIterator());
      for (lineIter->GoToFirstCell(); !lineIter->IsDoneWithTraversal(); lineIter->GoToNextCell()) {
        double* value = mData->GetCellData()
                            ->GetScalars(scalars[axis].mName.c_str())
                            ->GetTuple(lineIter->GetCurrentCellId());

        vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
        lineIter->GetCurrentCell(idList);
        for (auto point = idList->begin(); point != idList->end() - 1; point++) {
          texCoords[(uint32_t)*point][axis] = (float)((*value - min) / (max - min));
        }
      }
      break;
    }
    }
  }

  return texCoords;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<uint32_t> Pathlines::getIndices(std::vector<ScalarFilter> const& filters) const {
  std::vector<std::pair<Scalar, ScalarFilter>> pointFilters;
  std::vector<std::pair<Scalar, ScalarFilter>> cellFilters;
  for (auto const& filter : filters) {
    Scalar scalar = mScalars.at(filter.mAttrIndex);
    switch (scalar.mType) {
    case ScalarType::eCellData:
      cellFilters.push_back({scalar, filter});
      break;
    case ScalarType::ePointData:
      pointFilters.push_back({scalar, filter});
      break;
    }
  }

  std::vector<uint32_t> indices;

  auto lineIter = vtk::TakeSmartPointer(mData->GetLines()->NewIterator());
  for (lineIter->GoToFirstCell(); !lineIter->IsDoneWithTraversal(); lineIter->GoToNextCell()) {
    if (!isCellValid(cellFilters, lineIter->GetCurrentCellId())) {
      continue;
    }
    vtkSmartPointer<vtkIdList> idList = vtkSmartPointer<vtkIdList>::New();
    lineIter->GetCurrentCell(idList);
    for (auto index = idList->begin(); index < idList->end() - 1; index++) {
      indices.push_back((uint32_t)*index);
    }
  }

  return indices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Pathlines::isCellValid(
    std::vector<std::pair<Scalar, ScalarFilter>> const& filters, vtkIdType cellId) const {
  for (auto const& [scalar, filter] : filters) {
    double* value = mData->GetCellData()->GetScalars(scalar.mName.c_str())->GetTuple(cellId);
    if (*value < filter.mMin || *value > filter.mMax) {
      return false;
    }
  }
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
