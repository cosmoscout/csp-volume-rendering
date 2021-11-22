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

bool warnedLatRange = false;
bool warnedLonRange = false;
} // namespace

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

ospray::cpp::Volume createOSPRayVolume(vtkSmartPointer<vtkStructuredGrid> vtkVolume,
    std::vector<Scalar> const&                                            scalars,
    DataManager::Metadata::StructuredSpherical const&                     metadata) {
  std::array<int, 3> dimensions;
  vtkVolume->GetDimensions(dimensions.data());

  constexpr int OSP_RAD_AXIS = 0;
  constexpr int OSP_LON_AXIS = 1;
  constexpr int OSP_LAT_AXIS = 2;

  rkcommon::math::vec3i dim;
  rkcommon::math::vec3f spacing;
  rkcommon::math::vec3f gridOrigin;

  std::array<double, 2> latRange = metadata.mRanges.mLat;
  std::array<double, 2> lonRange = metadata.mRanges.mLon;

  bool   warnLat = false;
  double minLat  = std::min(latRange[0], latRange[1]);
  if (minLat < 0.) {
    latRange[0] += minLat;
    latRange[1] += minLat;
    warnLat = true;
  }
  for (int i = 0; i < 2; i++) {
    if (latRange[i] > 360.) {
      latRange[i] = 360.;
      warnLat     = true;
    }
  }
  if (warnLat && !warnedLatRange) {
    logger().warn("The latitudinal range of the dataset is [{}, {}], which is outside the maximum "
                  "range supported by OSPRay ([0, 360]). [{}, {}] will be used as a range instead.",
        metadata.mRanges.mLat[0], metadata.mRanges.mLat[1], latRange[0], latRange[1]);
    warnedLatRange = true;
  }

  double minLon  = std::min(lonRange[0], lonRange[1]);
  bool   warnLon = false;
  if (minLon < 0.) {
    lonRange[0] += minLon;
    lonRange[1] += minLon;
    warnLon = true;
  }
  for (int i = 0; i < 2; i++) {
    if (lonRange[i] > 180) {
      lonRange[i] = 180.;
      warnLon     = true;
    }
  }
  if (warnLon && !warnedLonRange) {
    logger().warn("The longitudinal range of the dataset is [{}, {}], which is outside the maximum "
                  "range supported by OSPRay ([0, 180]). [{}, {}] will be used as a range instead.",
        metadata.mRanges.mLon[0], metadata.mRanges.mLon[1], lonRange[0], lonRange[1]);
    warnedLonRange = true;
  }

  switch (scalars[0].mType) {
  case ScalarType::ePointData:
    dim[OSP_RAD_AXIS] = dimensions[metadata.mAxes.mRad];
    dim[OSP_LON_AXIS] = dimensions[metadata.mAxes.mLon];
    dim[OSP_LAT_AXIS] = dimensions[metadata.mAxes.mLat];

    spacing[OSP_RAD_AXIS] =
        static_cast<float>((metadata.mRanges.mRad[1] - metadata.mRanges.mRad[0])) /
        (dim[OSP_RAD_AXIS] - 1);
    spacing[OSP_LON_AXIS] =
        static_cast<float>((lonRange[1] - lonRange[0])) / (dim[OSP_LON_AXIS] - 1);
    spacing[OSP_LAT_AXIS] =
        static_cast<float>((latRange[1] - latRange[0])) / (dim[OSP_LAT_AXIS] - 1);

    gridOrigin[OSP_RAD_AXIS] = static_cast<float>(metadata.mRanges.mRad[0]);
    gridOrigin[OSP_LON_AXIS] = static_cast<float>(lonRange[0]);
    gridOrigin[OSP_LAT_AXIS] = static_cast<float>(latRange[0]);
    break;

  case ScalarType::eCellData:
    dim[OSP_RAD_AXIS] = dimensions[metadata.mAxes.mRad] - 1;
    dim[OSP_LON_AXIS] = dimensions[metadata.mAxes.mLon] - 1;
    dim[OSP_LAT_AXIS] = dimensions[metadata.mAxes.mLat] - 1;

    spacing[OSP_RAD_AXIS] =
        static_cast<float>((metadata.mRanges.mRad[1] - metadata.mRanges.mRad[0])) /
        dim[OSP_RAD_AXIS];
    spacing[OSP_LON_AXIS] = static_cast<float>((lonRange[1] - lonRange[0])) / dim[OSP_LON_AXIS];
    spacing[OSP_LAT_AXIS] = static_cast<float>((latRange[1] - latRange[0])) / dim[OSP_LAT_AXIS];

    gridOrigin[OSP_RAD_AXIS] =
        static_cast<float>(metadata.mRanges.mRad[0]) + spacing[OSP_RAD_AXIS] / 2;
    gridOrigin[OSP_LON_AXIS] = static_cast<float>(lonRange[0]) + spacing[OSP_LON_AXIS] / 2;
    gridOrigin[OSP_LAT_AXIS] = static_cast<float>(latRange[0]) + spacing[OSP_LAT_AXIS] / 2;
    break;
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

    int                   dataSize = vtkData->GetDataTypeSize();
    rkcommon::math::vec3i stride{dataSize, dataSize, dataSize};
    if (metadata.mAxes.mRad > metadata.mAxes.mLat) {
      stride[OSP_RAD_AXIS] *= dim[OSP_LAT_AXIS];
    }
    if (metadata.mAxes.mRad > metadata.mAxes.mLon) {
      stride[OSP_RAD_AXIS] *= dim[OSP_LON_AXIS];
    }
    if (metadata.mAxes.mLon > metadata.mAxes.mRad) {
      stride[OSP_LON_AXIS] *= dim[OSP_RAD_AXIS];
    }
    if (metadata.mAxes.mLon > metadata.mAxes.mLat) {
      stride[OSP_LON_AXIS] *= dim[OSP_LAT_AXIS];
    }
    if (metadata.mAxes.mLat > metadata.mAxes.mRad) {
      stride[OSP_LAT_AXIS] *= dim[OSP_RAD_AXIS];
    }
    if (metadata.mAxes.mLat > metadata.mAxes.mLon) {
      stride[OSP_LAT_AXIS] *= dim[OSP_LON_AXIS];
    }

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
      ospData.emplace_back((int16_t*)vtkData->GetVoidPointer(0), OSP_SHORT, dim, stride);
      break;
    case VTK_UNSIGNED_SHORT:
      ospData.emplace_back((uint16_t*)vtkData->GetVoidPointer(0), OSP_USHORT, dim, stride);
      break;
    }
  }

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
