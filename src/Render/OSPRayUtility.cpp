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

#include <vtk-8.2/vtkCellArray.h>
#include <vtk-8.2/vtkCellData.h>
#include <vtk-8.2/vtkPointData.h>
#include <vtk-8.2/vtkUnsignedCharArray.h>

#include <cmath>
#include <future>
#include <thread>

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
  ospDeviceSetErrorCallback(dev,
      [](void* userData, OSPError e, const char* errorDetails) {
        osprayLogger().error(errorDetails);
      },
      nullptr);
  ospDeviceSetStatusCallback(dev,
      [](void* userData, const char* message) { osprayLogger().info(message); }, nullptr);
  ospDeviceRelease(dev);

  ospLoadModule("volume_depth");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolumeUnstructured(vtkSmartPointer<vtkUnstructuredGrid> vtkVolume) {
  std::vector<rkcommon::math::vec3f> vertexPositions(
      (rkcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0),
      (rkcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0) +
          vtkVolume->GetNumberOfPoints());

  std::vector<double> vertexDataD(vtkVolume->GetNumberOfPoints());
  vtkVolume->GetPointData()->GetScalars()->ExportToVoidPointer(vertexDataD.data());
  std::vector<float> vertexData;
  vertexData.reserve(vertexDataD.size());
  std::transform(vertexDataD.begin(), vertexDataD.end(), std::back_inserter(vertexData),
      [](double d) { return (float)d; });

  std::vector<uint64_t> vertexIndices(vtkVolume->GetCells()->GetPointer(),
      vtkVolume->GetCells()->GetPointer() +
          vtkVolume->GetCells()->GetNumberOfConnectivityEntries());

  std::vector<uint64_t> cellIndices(vtkVolume->GetNumberOfCells());
  int                   index = -4;
  std::generate(cellIndices.begin(), cellIndices.end(), [&index] {
    index += 5;
    return index;
  });

  std::vector<uint8_t> cellTypes(
      vtkVolume->GetCellTypesArray()->Begin(), vtkVolume->GetCellTypesArray()->End());

  ospray::cpp::Volume volume("unstructured");
  volume.setParam("vertex.position", ospray::cpp::Data(vertexPositions));
  volume.setParam("vertex.data", ospray::cpp::Data(vertexData));
  volume.setParam("index", ospray::cpp::Data(vertexIndices));
  volume.setParam("indexPrefixed", false);
  volume.setParam("cell.index", ospray::cpp::Data(cellIndices));
  volume.setParam("cell.type", ospray::cpp::Data(cellTypes));
  volume.commit();

  return volume;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolumeStructured(vtkSmartPointer<vtkStructuredPoints> vtkVolume) {
  double origin[3];
  vtkVolume->GetOrigin(origin);
  double spacing[3];
  vtkVolume->GetSpacing(spacing);
  int dimensions[3];
  vtkVolume->GetDimensions(dimensions);

  for (int i = 0; i < 3; i++) {
    origin[i] = -vtkVolume->GetBounds()[i * 2 + 1] / 2;
  }
  ospray::cpp::Volume volume("structuredRegular");
  volume.setParam(
      "gridOrigin", rkcommon::math::vec3f{(float)origin[0], (float)origin[1], (float)origin[2]});
  volume.setParam("gridSpacing",
      rkcommon::math::vec3f{(float)spacing[0], (float)spacing[1], (float)spacing[2]});

  if (vtkVolume->GetScalarSize() == 4) {
    std::vector<float> data((float*)vtkVolume->GetScalarPointer(),
        (float*)vtkVolume->GetScalarPointer() + vtkVolume->GetNumberOfPoints());
    volume.setParam(
        "data", ospray::cpp::CopiedData(data.data(),
                    rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]}));
  } else if (vtkVolume->GetScalarSize() == 8) {
    std::vector<double> data((double*)vtkVolume->GetScalarPointer(),
        (double*)vtkVolume->GetScalarPointer() + vtkVolume->GetNumberOfPoints());
    volume.setParam(
        "data", ospray::cpp::CopiedData(data.data(),
                    rkcommon::math::vec3i{dimensions[0], dimensions[1], dimensions[2]}));
  }
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
    float min, float max, std::vector<glm::vec4> colors) {
  std::vector<rkcommon::math::vec3f> color;
  std::vector<float>                 opacity;

  for (glm::vec4 c : colors) {
    color.push_back(rkcommon::math::vec3f(c[0], c[1], c[2]));
    opacity.push_back(c[3]);
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

std::vector<float> depthToGrayscale(const std::vector<float>& depth) {
  std::vector<float> grayscale;
  grayscale.reserve(depth.size() * 3);

  for (const float& val : depth) {
    for (int i = 0; i < 3; i++) {
      grayscale.push_back((val + 1) / 2);
    }
  }

  return grayscale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> grayscaleToDepth(const std::vector<float>& grayscale) {
  std::vector<float> depth;
  depth.reserve(grayscale.size() / 3);

  for (int i = 0; i < grayscale.size() / 3; i++) {
    depth.push_back(grayscale[i * 3] * 2 - 1);
  }

  return depth;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<float> denoiseImage(std::vector<float>& image, int channelCount, int resolution) {
  oidn::DeviceRef device = oidn::newDevice();
  device.setErrorFunction([](void* userPtr, oidn::Error e, const char* errorDetails) {
    oidnLogger().error(errorDetails);
  });
  device.commit();

  oidn::FilterRef filter = device.newFilter("RT");
  filter.setImage("color", image.data(), oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * channelCount, sizeof(float) * channelCount * resolution);
  filter.setImage("output", image.data(), oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * channelCount, sizeof(float) * channelCount * resolution);
  filter.commit();

  filter.execute();

  return image;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::OSPRayUtility
