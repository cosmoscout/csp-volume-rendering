////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP
#define CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP

#include <ospray/ospray_cpp.h>

#include <vtk-8.2/vtkSmartPointer.h>
#include <vtk-8.2/vtkUnstructuredGrid.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering::OSPRayUtility {
void initOSPRay();

ospray::cpp::Camera createOSPRayCamera(
    int width, int height, float fov, float modelHeight, glm::mat4 cameraRotation);
ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkUnstructuredGrid> vtkVolume, std::string scalar);
ospray::cpp::TransferFunction createOSPRayTransferFunction();
ospray::cpp::TransferFunction createOSPRayTransferFunction(
    float min, float max, std::vector<glm::vec4> colors);

} // namespace csp::volumerendering::OSPRayUtility

#endif // CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP
