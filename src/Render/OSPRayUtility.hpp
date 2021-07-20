////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP
#define CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP

#include "../Data/DataManager.hpp"
#include "../Enums.hpp"

#include <ospray/ospray_cpp.h>

#include <vtkSmartPointer.h>
#include <vtkStructuredGrid.h>
#include <vtkStructuredPoints.h>
#include <vtkUnstructuredGrid.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering::OSPRayUtility {

/// Initializes OSPRay and registers an logger for OSPRay status and error mesages.
/// Loads the volume_depth OSPRay extension module. Also intializes an OIDN device.
void initOSPRay();

/// Creates a volume representation usable by OSPRay from different vtk data set types.
/// Only the active scalar data of the given type will be present in the resulting object.
ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkUnstructuredGrid> vtkVolume, ScalarType scalarType);
ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkStructuredPoints> vtkVolume, std::vector<Scalar> const& scalars);
ospray::cpp::Volume createOSPRayVolume(
    vtkSmartPointer<vtkStructuredGrid> vtkVolume, std::vector<Scalar> const& scalars);

/// Creates a default transfer function for OSPRay.
/// The transfer function interpolates from fully transparent blue to fully opaque red.
ospray::cpp::TransferFunction createOSPRayTransferFunction();
/// Creates a transfer function for OSPRay from an array of RGBA colors.
/// The colors are distributed evenly over the range from 'min' to 'max'.
/// Between two colors the output is linearly interpolated.
ospray::cpp::TransferFunction createOSPRayTransferFunction(
    float min, float max, std::vector<glm::vec4> const& colors);

/// Converts an image with one channel of depth values in the range [-1,1] to an RGB image (three
/// channels with ranges [0,1]) representing the depth as grayscale.
std::vector<float> depthToGrayscale(float* depth, int resolution);
/// Converts an grayscale RGB image (three channels with ranges [0,1]) to a depth image with one
/// channel (range [-1,1]). It is assumed that all pixels of the RGB image have the same value on
/// all three channels.
void grayscaleToDepth(std::vector<float> const& grayscale, float* output);
/// Denoises the given image in-place using OIDN. The image has to contain at least three channels,
/// the exact number has to be given as the channelCount parameter. However, only the first three
/// channels are used while denoising.
void denoiseImage(float* image, int channelCount, int resolution);

} // namespace csp::volumerendering::OSPRayUtility

#endif // CSP_VOLUME_RENDERING_OSPRAYUTILITY_HPP
