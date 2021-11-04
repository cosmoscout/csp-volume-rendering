////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayUtility.hpp"

#include "../logger.hpp"

#include "../../../../src/cs-utils/logger.hpp"

#include <glm/geometric.hpp>

#include <OpenImageDenoise/oidn.hpp>

#include <ospray/ospray_cpp/ext/rkcommon.h>

#include <rkcommon/math/vec.h>

#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUnsignedCharArray.h>

#include <cmath>
#include <future>
#include <thread>

namespace {
oidn::DeviceRef oidnDevice;
}

namespace csp::volumerendering::OSPRayUtility {

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& osprayLogger() {
  static auto logger = cs::utils::createLogger("OSPRay");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& oidnLogger() {
  static auto logger = cs::utils::createLogger("OIDN");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void initOSPRay() {
  int         argc = 0;
  const char* arg  = "";

  OSPError init_error = ospInit(&argc, &arg);
  if (init_error != OSP_NO_ERROR) {
    osprayLogger().error("OSPRay Initialization failed: {}", init_error);
    throw std::runtime_error("OSPRay Initialization failed.");
  }

  OSPDevice dev = ospGetCurrentDevice();
  ospDeviceSetErrorCallback(
      dev,
      [](void* userData, OSPError e, const char* errorDetails) {
        osprayLogger().error(errorDetails);
      },
      nullptr);
  ospDeviceSetStatusCallback(
      dev, [](void* userData, const char* message) { osprayLogger().info(message); }, nullptr);
  ospDeviceRelease(dev);

  ospLoadModule("volume_depth");

  oidnDevice = oidn::newDevice();
  oidnDevice.setErrorFunction([](void* userPtr, oidn::Error e, const char* errorDetails) {
    oidnLogger().error(errorDetails);
  });
  oidnDevice.commit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkUnstructuredGrid> vtkVolume, ScalarType scalarType) {
  std::vector<rkcommon::math::vec3f> vertexPositions(
      (rkcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0),
      (rkcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0) +
          vtkVolume->GetNumberOfPoints());

  std::vector<float> data;
  switch (scalarType) {
  case ScalarType::ePointData: {
    std::vector<double> vertexDataD(vtkVolume->GetNumberOfPoints());
    vtkVolume->GetPointData()->GetScalars()->ExportToVoidPointer(vertexDataD.data());
    data.reserve(vertexDataD.size());
    std::transform(vertexDataD.begin(), vertexDataD.end(), std::back_inserter(data),
        [](double d) { return (float)d; });
    break;
  }
  case ScalarType::eCellData: {
    std::vector<double> cellDataD(vtkVolume->GetNumberOfCells());
    vtkVolume->GetCellData()->GetScalars()->ExportToVoidPointer(cellDataD.data());
    data.reserve(cellDataD.size());
    std::transform(cellDataD.begin(), cellDataD.end(), std::back_inserter(data),
        [](double d) { return (float)d; });
    break;
  }
  }

  std::vector<uint64_t> vertexIndices(vtkVolume->GetCells()->GetConnectivityArray64()->Begin(),
      vtkVolume->GetCells()->GetConnectivityArray64()->End());

  // This VTK array includes an additional offset for where the next cell would be placed
  // It has to be removed for using the array with OSPRay
  std::vector<uint64_t> cellIndices(vtkVolume->GetCells()->GetOffsetsArray64()->Begin(),
      vtkVolume->GetCells()->GetOffsetsArray64()->End() - 1);

  std::vector<uint8_t> cellTypes(
      vtkVolume->GetCellTypesArray()->Begin(), vtkVolume->GetCellTypesArray()->End());

  ospray::cpp::Volume volume("unstructured");
  switch (scalarType) {
  case ScalarType::ePointData:
    volume.setParam("vertex.data", ospray::cpp::Data(data));
    break;
  case ScalarType::eCellData:
    volume.setParam("cell.data", ospray::cpp::Data(data));
    break;
  }
  volume.setParam("vertex.position", ospray::cpp::Data(vertexPositions));
  volume.setParam("index", ospray::cpp::Data(vertexIndices));
  volume.setParam("cell.index", ospray::cpp::Data(cellIndices));
  volume.setParam("cell.type", ospray::cpp::Data(cellTypes));
  volume.commit();

  return volume;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkStructuredPoints> vtkVolume, std::vector<Scalar> const& scalars) {
  double spacing[3];
  vtkVolume->GetSpacing(spacing);
  int dimensions[3];
  vtkVolume->GetDimensions(dimensions);
  double origin[3];
  for (int i = 0; i < 3; i++) {
    origin[i] = -(vtkVolume->GetBounds()[i * 2 + 1] - vtkVolume->GetBounds()[i * 2]) / 2;
  }

  std::vector<ospray::cpp::CopiedData> ospData;
  vtkSmartPointer<vtkDataArray>        vtkData;

  for (size_t i = 0; i < scalars.size(); i++) {
    if (scalars[i].mType != scalars[0].mType) {
      continue;
    }
    switch (scalars[i].mType) {
    case ScalarType::ePointData:
      vtkData = vtkVolume->GetPointData()->GetScalars(scalars[i].mName.c_str());
      break;
    case ScalarType::eCellData:
      vtkData = vtkVolume->GetCellData()->GetScalars(scalars[i].mName.c_str());
      break;
    }

    switch (vtkData->GetDataType()) {
    case VTK_FLOAT:
      ospData.emplace_back((float*)vtkData->GetVoidPointer(0), OSP_FLOAT,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_DOUBLE:
      ospData.emplace_back((double*)vtkData->GetVoidPointer(0), OSP_DOUBLE,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_UNSIGNED_CHAR:
      ospData.emplace_back((uint8_t*)vtkData->GetVoidPointer(0), OSP_UCHAR,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_CHAR:
      ospData.emplace_back((int8_t*)vtkData->GetVoidPointer(0), OSP_UCHAR,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_SHORT:
      ospData.emplace_back((int16_t*)vtkData->GetVoidPointer(0), OSP_SHORT,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_UNSIGNED_SHORT:
      ospData.emplace_back((uint16_t*)vtkData->GetVoidPointer(0), OSP_USHORT,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    }
  }

  ospray::cpp::Volume volume("structuredRegular");
  volume.setParam(
      "gridOrigin", rkcommon::math::vec3f{(float)origin[0], (float)origin[1], (float)origin[2]});
  volume.setParam("gridSpacing",
      rkcommon::math::vec3f{(float)spacing[0], (float)spacing[1], (float)spacing[2]});
  volume.setParam("data", ospray::cpp::Data(ospData.data(), OSP_DATA, ospData.size()));
  volume.setParam("dimensions", rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
  volume.commit();

  return volume;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkStructuredGrid> vtkVolume, std::vector<Scalar> const& scalars) {
  std::array<int, 6> extent;
  vtkVolume->GetExtent(extent.data());
  std::array<double, 6> bounds;
  vtkVolume->GetBounds(bounds.data());
  std::array<int, 3> dimensions;
  vtkVolume->GetDimensions(dimensions.data());
  std::array<double, 3> origin;
  vtkVolume->GetPoint(0, 0, 0, origin.data());

  std::vector<ospray::cpp::CopiedData> ospData;
  vtkSmartPointer<vtkDataArray>        vtkData;
  rkcommon::math::vec3i                dim;

  for (size_t i = 0; i < scalars.size(); i++) {
    if (scalars[i].mType != scalars[0].mType) {
      continue;
    }
    switch (scalars[i].mType) {
    case ScalarType::ePointData:
      dim     = {dimensions[1], dimensions[2], dimensions[0]};
      vtkData = vtkVolume->GetPointData()->GetScalars(scalars[i].mName.c_str());
      break;
    case ScalarType::eCellData:
      dim     = {dimensions[1] - 1, dimensions[2] - 1, dimensions[0] - 1};
      vtkData = vtkVolume->GetCellData()->GetScalars(scalars[i].mName.c_str());
      break;
    }

    int                   dataSize = vtkData->GetDataTypeSize();
    rkcommon::math::vec3i stride{dataSize * dim[2], dataSize * dim[2] * dim[0], dataSize};
    switch (vtkData->GetDataType()) {
    case VTK_FLOAT:
      ospData.emplace_back((float*)vtkData->GetVoidPointer(0), OSP_FLOAT, dim, stride);
      break;
    case VTK_DOUBLE:
      ospData.emplace_back((double*)vtkData->GetVoidPointer(0), OSP_DOUBLE, dim, stride);
      break;
    case VTK_UNSIGNED_CHAR:
      ospData.emplace_back((uint8_t*)vtkData->GetVoidPointer(0), OSP_UCHAR, dim, stride);
      break;
    case VTK_CHAR:
      ospData.emplace_back((int8_t*)vtkData->GetVoidPointer(0), OSP_UCHAR, dim, stride);
      break;
    case VTK_SHORT:
      ospData.emplace_back((int16_t*)vtkData->GetVoidPointer(0), OSP_SHORT,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    case VTK_UNSIGNED_SHORT:
      ospData.emplace_back((uint16_t*)vtkData->GetVoidPointer(0), OSP_USHORT,
          rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]});
      break;
    }
  }

  rkcommon::math::vec3f spacing = {((float)(bounds[3] - bounds[2]) / 2 - (float)origin[2]) / dim[0],
      180.f / dim[1], 360.f / dim[2]};
  rkcommon::math::vec3f gridOrigin =
      rkcommon::math::vec3f{(float)origin[2], spacing[1] / 2, spacing[2] / 2};

  ospray::cpp::Volume volume("structuredSpherical");
  volume.setParam("gridSpacing", spacing);
  volume.setParam("gridOrigin", gridOrigin);
  volume.setParam("data", ospray::cpp::Data(ospData.data(), OSP_DATA, ospData.size()));
  volume.setParam("dimensions", dim);
  volume.commit();

  return volume;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::TransferFunction createOSPRayTransferFunction() {
  std::vector<rkcommon::math::vec3f> color   = {rkcommon::math::vec3f(0.f, 0.f, 1.f),
      rkcommon::math::vec3f(0.f, 1.f, 0.f), rkcommon::math::vec3f(1.f, 0.f, 0.f)};
  std::vector<float>                 opacity = {.0f, 1.f};

  rkcommon::math::vec2f valueRange = {0.f, 1.f};

  ospray::cpp::TransferFunction transferFunction("piecewiseLinear");
  transferFunction.setParam("color", ospray::cpp::Data(color));
  transferFunction.setParam("opacity", ospray::cpp::Data(opacity));
  transferFunction.setParam("valueRange", valueRange);
  transferFunction.commit();
  return transferFunction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::TransferFunction createOSPRayTransferFunction(
    float min, float max, std::vector<glm::vec4> const& colors) {
  std::vector<rkcommon::math::vec3f> color;
  std::vector<float>                 opacity;

  for (glm::vec4 c : colors) {
    color.emplace_back(c[0], c[1], c[2]);
    opacity.push_back(std::pow(c[3], 10));
  }

  rkcommon::math::vec2f valueRange = {min, max};

  ospray::cpp::TransferFunction transferFunction("piecewiseLinear");
  transferFunction.setParam("color", ospray::cpp::Data(color));
  transferFunction.setParam("opacity", ospray::cpp::Data(opacity));
  transferFunction.setParam("valueRange", valueRange);
  transferFunction.commit();
  return transferFunction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> depthToGrayscale(float* depth, int resolution) {
  std::vector<float> grayscale;
  grayscale.reserve(resolution * resolution * 3);

  for (int i = 0; i < resolution * resolution; i++) {
    float val = depth[i];
    for (int i = 0; i < 3; i++) {
      grayscale.push_back((val + 1) / 2);
    }
  }

  return grayscale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void grayscaleToDepth(std::vector<float> const& grayscale, float* output) {
  for (size_t i = 0; i < grayscale.size() / 3; i++) {
    output[i] = grayscale[i * 3] * 2 - 1;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void denoiseImage(float* image, int channelCount, int resolution) {
  oidn::FilterRef filter = oidnDevice.newFilter("RT");
  filter.setImage("color", image, oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * channelCount, sizeof(float) * channelCount * resolution);
  filter.setImage("output", image, oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * channelCount, sizeof(float) * channelCount * resolution);
  filter.commit();

  filter.execute();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::OSPRayUtility
