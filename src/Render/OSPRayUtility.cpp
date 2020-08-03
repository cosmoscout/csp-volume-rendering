////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OSPRayUtility.hpp"

#include "../../../src/cs-utils/logger.hpp"

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ospray::cpp::Camera createOSPRayCamera(
    int width, int height, float fov, float distance, glm::mat4 cameraRotation) {
  glm::vec4 camPos(0, 0, distance, 1);
  camPos = cameraRotation * camPos;
  glm::vec4 camUp(0, 1, 0, 1);
  camUp = cameraRotation * camUp;
  glm::vec4 camView(0, 0, -1, 1);
  camView = cameraRotation * camView;

  ospcommon::math::vec3f camPosOsp{camPos[0], camPos[1], camPos[2]};
  ospcommon::math::vec3f camUpOsp{camUp[0], camUp[1], camUp[2]};
  ospcommon::math::vec3f camViewOsp{camView[0], camView[1], camView[2]};

  ospray::cpp::Camera camera("perspective");
  camera.setParam("aspect", width / (float)height);
  camera.setParam("position", camPosOsp);
  camera.setParam("up", camUpOsp);
  camera.setParam("direction", camViewOsp);
  camera.setParam("fovy", fov);
  camera.commit();
  return camera;
}

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

} // namespace csp::volumerendering::OSPRayUtility
