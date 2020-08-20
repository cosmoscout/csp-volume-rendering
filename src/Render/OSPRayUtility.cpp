////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayUtility.hpp"

#include "../logger.hpp"

#include "../../../src/cs-utils/logger.hpp"

#include <glm/geometric.hpp>

#include <OpenImageDenoise/oidn.hpp>

#include <vtk-8.2/vtkCellArray.h>
#include <vtk-8.2/vtkCellData.h>
#include <vtk-8.2/vtkCellSizeFilter.h>
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

  ospDeviceSetErrorFunc(ospGetCurrentDevice(),
      [](OSPError e, const char* errorDetails) { osprayLogger().error(errorDetails); });
  ospDeviceSetStatusFunc(
      ospGetCurrentDevice(), [](const char* message) { osprayLogger().info(message); });

  ospLoadModule("denoiser");
  ospLoadModule("volume_depth");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Camera createOSPRayCamera(float fov, float modelHeight, glm::mat4 observerTransform) {
  float fovRad = fov / 180 * (float)M_PI;

  // Create a transformation matrix for a camera placed on (0,0,0) looking along the negative z axis
  glm::mat4 cameraTransform(1);
  cameraTransform[2][2] = -1;

  cameraTransform = observerTransform * cameraTransform;

  glm::vec3 camRight(cameraTransform[0].xyz);
  camRight = glm::normalize(camRight);
  glm::vec3 camUp(cameraTransform[1].xyz);
  camUp = glm::normalize(camUp);
  glm::vec3 camDir(cameraTransform[2].xyz);
  camDir = glm::normalize(camDir);
  glm::vec3 camPos(cameraTransform[3].xyz);

  float camXLen = glm::dot(camPos, camRight);
  float camYLen = glm::dot(camPos, camUp);
  float camZLen = glm::dot(camPos, camDir);

  float cameraAngleX = atan(camXLen / camZLen);
  float cameraAngleY = atan(camYLen / camZLen);

  float modelAngleX = asin(modelHeight / sqrt(camXLen * camXLen + camZLen * camZLen));
  float modelAngleY = asin(modelHeight / sqrt(camYLen * camYLen + camZLen * camZLen));

  float leftAngle  = cameraAngleX - modelAngleX;
  float rightAngle = cameraAngleX + modelAngleX;
  float downAngle  = cameraAngleY - modelAngleY;
  float upAngle    = cameraAngleY + modelAngleY;

  float leftPercent  = 0.5f + tan(leftAngle) / (2 * tan(fovRad / 2));
  float rightPercent = 0.5f + tan(rightAngle) / (2 * tan(fovRad / 2));
  float downPercent  = 0.5f + tan(downAngle) / (2 * tan(fovRad / 2));
  float upPercent    = 0.5f + tan(upAngle) / (2 * tan(fovRad / 2));

  ospcommon::math::vec3f camPosOsp{camPos.x, camPos.y, camPos.z};
  ospcommon::math::vec3f camUpOsp{camUp.x, camUp.y, camUp.z};
  ospcommon::math::vec3f camViewOsp{camDir.x, camDir.y, camDir.z};

  ospcommon::math::vec2f camImageStartOsp{leftPercent, downPercent};
  ospcommon::math::vec2f camImageEndOsp{rightPercent, upPercent};

  ospray::cpp::Camera osprayCamera("perspective");
  osprayCamera.setParam("aspect", 1);
  osprayCamera.setParam("position", camPosOsp);
  osprayCamera.setParam("up", camUpOsp);
  osprayCamera.setParam("direction", camViewOsp);
  osprayCamera.setParam("fovy", fov);
  osprayCamera.setParam("imageStart", camImageStartOsp);
  osprayCamera.setParam("imageEnd", camImageEndOsp);
  osprayCamera.commit();

  Camera camera;
  camera.osprayCamera    = osprayCamera;
  camera.positionRotated = glm::vec3(camXLen, camYLen, -camZLen) / modelHeight;

  glm::mat4 projection =
      glm::perspective(fovRad, 1.f, -camZLen / modelHeight - 1, -camZLen / modelHeight + 1);
  glm::mat4 view =
      glm::translate(glm::mat4(1.f), -glm::vec3(camXLen, camYLen, -camZLen) / modelHeight);

  glm::mat4 fitToView(1);
  fitToView[0][0] = (projection * view)[3][3] / (projection * view)[0][0];
  fitToView[1][1] = (projection * view)[3][3] / (projection * view)[1][1];
  fitToView[3][0] = -(projection * view)[3][0] * fitToView[0][0] / (projection * view)[3][3];
  fitToView[3][1] = -(projection * view)[3][1] * fitToView[1][1] / (projection * view)[3][3];

  camera.transformationMatrix = fitToView * projection * view;
  return camera;
} // namespace csp::volumerendering::OSPRayUtility

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkUnstructuredGrid> vtkVolume, std::string scalar) {
  vtkSmartPointer<vtkCellSizeFilter> sizeFilter = vtkSmartPointer<vtkCellSizeFilter>::New();
  sizeFilter->SetComputeArea(false);
  sizeFilter->SetComputeLength(false);
  sizeFilter->SetComputeSum(false);
  sizeFilter->SetComputeVertexCount(false);
  sizeFilter->SetComputeVolume(true);
  sizeFilter->SetInputData(vtkVolume);
  sizeFilter->Update();
  vtkVolume = vtkUnstructuredGrid::SafeDownCast(sizeFilter->GetOutput());

  std::vector<ospcommon::math::vec3f> vertexPositions(
      (ospcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0),
      (ospcommon::math::vec3f*)vtkVolume->GetPoints()->GetVoidPointer(0) +
          vtkVolume->GetNumberOfPoints());

  vtkVolume->GetPointData()->SetActiveScalars(scalar.c_str());
  std::vector<double> vertexDataD(vtkVolume->GetNumberOfPoints());
  vtkVolume->GetPointData()->GetScalars()->ExportToVoidPointer(vertexDataD.data());
  std::vector<float> vertexData(vertexDataD.begin(), vertexDataD.end());

  std::vector<uint64_t> vertexIndices(vtkVolume->GetCells()->GetPointer(),
      vtkVolume->GetCells()->GetPointer() +
          vtkVolume->GetCells()->GetNumberOfConnectivityEntries());

  std::vector<uint64_t> allCellIndices(vtkVolume->GetNumberOfCells());
  int                   index = -4;
  std::generate(allCellIndices.begin(), allCellIndices.end(), [&index] {
    index += 5;
    return index;
  });

  std::vector<uint8_t> allCellTypes(
      vtkVolume->GetCellTypesArray()->Begin(), vtkVolume->GetCellTypesArray()->End());

  std::vector<uint64_t> cellIndices;
  std::vector<uint8_t>  cellTypes;
  vtkVolume->GetCellData()->SetActiveScalars("Volume");

  std::vector<std::thread>                        splitThreads;
  std::vector<std::future<std::vector<uint8_t>>>  futureCellTypes;
  std::vector<std::future<std::vector<uint64_t>>> futureCellIndices;
  int                                             threadCount = 4;

  for (int i = 0; i < threadCount; i++) {
    std::promise<std::vector<uint8_t>>  pT;
    std::promise<std::vector<uint64_t>> pI;
    futureCellTypes.push_back(pT.get_future());
    futureCellIndices.push_back(pI.get_future());
    int start = (int)((float)vtkVolume->GetNumberOfCells() / threadCount * i);
    int end   = (int)((float)vtkVolume->GetNumberOfCells() / threadCount * (i + 1));

    std::thread thread(
        [&vertexData, vtkVolume, allCellTypes, &allCellIndices](
            std::promise<std::vector<uint8_t>> pT, std::promise<std::vector<uint64_t>> pI,
            int start, int end) {
          std::cout << "Starting thread for " << start << " - " << end << std::endl;
          std::vector<uint8_t>  cellTypes;
          std::vector<uint64_t> cellIndices;

          for (int i = start; i < end; i++) {
            double volume;
            vtkVolume->GetCellData()->GetScalars()->GetTuple(i, &volume);
            if (volume > 0.000001 && volume < 2.59941e-05) {
              cellIndices.push_back(allCellIndices[i]);
              cellTypes.push_back(allCellTypes[i]);
            }
          }

          pT.set_value(cellTypes);
          pI.set_value(cellIndices);
          std::cout << "Finished thread for " << start << " - " << end << std::endl;
        },
        std::move(pT), std::move(pI), start, end);
    splitThreads.push_back(std::move(thread));
  }

  for (int i = 0; i < threadCount; i++) {
    splitThreads[i].join();
    auto cellTypesFromFuture   = futureCellTypes[i].get();
    auto cellIndicesFromFuture = futureCellIndices[i].get();
    cellTypes.insert(cellTypes.end(), cellTypesFromFuture.begin(), cellTypesFromFuture.end());
    cellIndices.insert(
        cellIndices.end(), cellIndicesFromFuture.begin(), cellIndicesFromFuture.end());
  }

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

ospray::cpp::TransferFunction createOSPRayTransferFunction() {
  std::vector<ospcommon::math::vec3f> color   = {ospcommon::math::vec3f(0.f, 0.f, 1.f),
      ospcommon::math::vec3f(0.f, 1.f, 0.f), ospcommon::math::vec3f(1.f, 0.f, 0.f)};
  std::vector<float>                  opacity = {.0f, 5.f};

  ospcommon::math::vec2f valueRange = {0.f, 1.f};

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
  std::vector<ospcommon::math::vec3f> color;
  std::vector<float>                  opacity;

  for (glm::vec4 c : colors) {
    color.push_back(ospcommon::math::vec3f(c[0], c[1], c[2]));
    opacity.push_back(c[3] * 2);
  }

  ospcommon::math::vec2f valueRange = {min, max};

  ospray::cpp::TransferFunction transferFunction("piecewiseLinear");
  transferFunction.setParam("color", ospray::cpp::Data(color));
  transferFunction.setParam("opacity", ospray::cpp::Data(opacity));
  transferFunction.setParam("valueRange", valueRange);
  transferFunction.commit();
  return transferFunction;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::World createOSPRayWorld(ospray::cpp::VolumetricModel model) {
  ospray::cpp::Group group;
  group.setParam("volume", ospray::cpp::Data(model));
  group.commit();

  return createOSPRayWorld(group);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::World createOSPRayWorld(ospray::cpp::GeometricModel model) {
  ospray::cpp::Group group;
  group.setParam("geometry", ospray::cpp::Data(model));
  group.commit();

  return createOSPRayWorld(group);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::World createOSPRayWorld(ospray::cpp::Group group) {
  ospray::cpp::Instance instance(group);
  instance.commit();

  ospray::cpp::Light light("ambient");
  light.commit();

  ospray::cpp::World world;
  world.setParam("instance", ospray::cpp::Data(instance));
  world.setParam("light", ospray::cpp::Data(light));
  world.commit();

  return world;
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

std::vector<float> denoiseImage(std::vector<float>& image, int componentCount, int resolution) {
  oidn::DeviceRef device = oidn::newDevice();
  device.setErrorFunction([](void* userPtr, oidn::Error e, const char* errorDetails) {
    oidnLogger().error(errorDetails);
  });
  device.commit();

  oidn::FilterRef filter = device.newFilter("RT");
  filter.setImage("color", image.data(), oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * componentCount, sizeof(float) * componentCount * resolution);
  filter.setImage("output", image.data(), oidn::Format::Float3, resolution, resolution, 0,
      sizeof(float) * componentCount, sizeof(float) * componentCount * resolution);
  filter.commit();

  filter.execute();

  return image;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering::OSPRayUtility
