////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IrregularGrid.hpp"

#include "../logger.hpp"
#include "Shaders.hpp"

#include "../../../../src/cs-utils/FrameTimings.hpp"
#include "../../../../src/cs-utils/utils.hpp"

#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

#include <utility>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

IrregularGrid::IrregularGrid(
    VolumeShape shape, std::shared_ptr<cs::core::Settings> settings, std::string anchor)
    : DisplayNode(shape, settings, anchor) {
  createBuffers(16, 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void IrregularGrid::setDepthTexture(float* texture, int width, int height) {
  DisplayNode::setDepthTexture(texture, width, height);
  // TODO Generate quadtree
  // TODO Pass quadtree to createBuffers?
  createBuffers(width, height);
  //surfaceDetection(width, height, texture);
  SurfaceDetectionBuffer buffer(texture, width, height);
  //buffer.print();
  buffer.generateVertices();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool IrregularGrid::DoImpl() {
  cs::utils::FrameTimings::ScopedTimer timer("Volume Rendering");

  if (mShaderDirty) {
    mShader = VistaGLSLShader();
    mShader.InitVertexShaderFromString(BILLBOARD_VERT);
    mShader.InitFragmentShaderFromString(BILLBOARD_FRAG);
    mShader.Link();

    mVisShader = VistaGLSLShader();
    mVisShader.InitVertexShaderFromString(BILLBOARD_VERT);
    mVisShader.InitFragmentShaderFromString(VIS_FRAG);
    mVisShader.Link();

    mShaderDirty = false;
  }

  mShader.Bind();

  // Get modelview and projection matrices.
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
  auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE, glm::value_ptr(mTransform));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererProjectionInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(
      mShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mShader.SetUniform(mShader.GetUniformLocation("uTexture"), 0);
  mShader.SetUniform(mShader.GetUniformLocation("uDepthTexture"), 1);
  mShader.SetUniform(mShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mShader.SetUniform(
      mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mShader.SetUniform(mShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mShader.SetUniform(mShader.GetUniformLocation("uInside"), mInside);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  glPushAttrib(GL_ENABLE_BIT);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_FRONT);
  glEnable(GL_BLEND);
  glDisable(GL_DEPTH_TEST);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Draw.
  mVAO.Bind();
  glDrawElements(GL_TRIANGLE_STRIP, mIndexCount, GL_UNSIGNED_INT, nullptr);
  mVAO.Release();

  // Clean up.
  mTexture.Unbind(GL_TEXTURE0);
  mTexture.Unbind(GL_TEXTURE1);
  mShader.Release();

  // Visualization of grid
  mVisShader.Bind();

  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatTransform"), 1, GL_FALSE, glm::value_ptr(mTransform));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererProjection"), 1, GL_FALSE,
      glm::value_ptr(mRendererProjection));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererProjectionInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererProjection)));
  glUniformMatrix4fv(
      mVisShader.GetUniformLocation("uMatRendererMVP"), 1, GL_FALSE, glm::value_ptr(mRendererMVP));
  glUniformMatrix4fv(mVisShader.GetUniformLocation("uMatRendererMVPInv"), 1, GL_FALSE,
      glm::value_ptr(glm::inverse(mRendererMVP)));

  mVisShader.SetUniform(mVisShader.GetUniformLocation("uTexture"), 0);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uDepthTexture"), 1);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uRadii"), static_cast<float>(mRadii[0]),
      static_cast<float>(mRadii[0]), static_cast<float>(mRadii[0]));
  mVisShader.SetUniform(
      mVisShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uUseDepth"), mUseDepth);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uDrawDepth"), mDrawDepth);
  mVisShader.SetUniform(mVisShader.GetUniformLocation("uInside"), mInside);

  mTexture.Bind(GL_TEXTURE0);
  mDepthTexture.Bind(GL_TEXTURE1);

  mVisVAO.Bind();
  glDrawElements(GL_LINES, mVisIndexCount, GL_UNSIGNED_INT, nullptr);
  mVisVAO.Release();
  mVisShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define BIT_IS_SURFACE 0

#define BIT_CONTINUOUS_T 4
#define BIT_CONTINUOUS_R 5
#define BIT_CONTINUOUS_B 6
#define BIT_CONTINUOUS_L 7

#define BIT_CONTINUOUS_TR 8
#define BIT_CONTINUOUS_TL 9
#define BIT_CONTINUOUS_BR 10
#define BIT_CONTINUOUS_BL 11

#define BIT_CURRENT_LEVEL 12 // 12-15 (requires 3 bits)

#define ALL_CONTINUITY_BITS 4080
#define ALL_DATA_BITS 4095

// width and length should be width and length of current level
size_t getIndexToLast(
    size_t currentIndex, int width, int height, int offsetX = 0, int offsetY = 0) {
  // Assume line by line layout
  int curX  = static_cast<int>(currentIndex) % width;
  int curY  = static_cast<int>(currentIndex) / width;
  int lastX = std::clamp(curX * 2 + offsetX, 0, width * 2);
  int lastY = std::clamp(curY * 2 + offsetY, 0, height * 2);
  return lastY * width * 2 + lastX;
}

bool is_on_line(float a, float b, float c) {
  float threshold = 0.1f;
  return abs(a - 2 * b + c) < threshold;
}

//void IrregularGrid::surfaceDetection(uint32_t width, uint32_t height, float* texture) {
//  const int    cellSize   = 16;
//  const int    levels     = static_cast<int>(std::log2l(cellSize));
//  const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
//
//  thrust::device_vector<float> dDepth(texture, texture + pixelCount);
//
//  SurfaceDetectionBuffer surfaceDetectionBuffers;
//  for (int i = 0; i < levels; ++i) {
//    thrust::device_vector<uint16_t>        currentSurface(pixelCount >> (2 * (i + 1)));
//    thrust::device_vector<uint16_t> const& lastSurface = surfaceDetectionBuffers.back();
//
//    const int curWidth  = static_cast<int>(width >> (i + 1));
//    const int curHeight = static_cast<int>(height >> (i + 1));
//    if (i == 0) {
//      thrust::tabulate(
//          thrust::device, currentSurface.begin(), currentSurface.end(), [&](size_t index) {
//            // d0  d1   d2   d3
//            //   \  |    |  /
//            // d4--d5-- d6-- d7
//            //      |    |
//            // d8--d9--d10--d11
//            //   /  |    |  \
//            // d12 d13  d14 d15
//
//            const float d0 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 2)];
//            const float d1 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 2)];
//            const float d2 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 2)];
//            const float d3 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 2)];
//
//            const float d4 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 1)];
//            const float d5 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 1)];
//            const float d6 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 1)];
//            const float d7 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 1)];
//
//            const float d8  = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 0)];
//            const float d9  = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 0)];
//            const float d10 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 0)];
//            const float d11 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 0)];
//
//            const float d12 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, -1)];
//            const float d13 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, -1)];
//            const float d14 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, -1)];
//            const float d15 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, -1)];
//
//            // check for horizontal and vertical continuity
//            const bool t = is_on_line(d1, d5, d9) && is_on_line(d2, d6, d10);
//            const bool r = is_on_line(d5, d6, d7) && is_on_line(d9, d10, d11);
//            const bool b = is_on_line(d5, d9, d13) && is_on_line(d6, d10, d14);
//            const bool l = is_on_line(d4, d5, d6) && is_on_line(d8, d9, d10);
//
//            // check for diagonal continuity
//            const bool tl = is_on_line(d0, d5, d10);
//            const bool tr = is_on_line(d3, d6, d9);
//            const bool bl = is_on_line(d12, d9, d6);
//            const bool br = is_on_line(d5, d10, d15);
//
//            // if the patch is connected on two othogonal sides, it represents a surface
//            const uint16_t is_surface = (t & r) | (r & b) | (b & l) | (l & t);
//            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
//                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
//                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
//                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);
//
//            // store all continuities
//            return is_surface | continuous;
//          });
//    } else {
//      thrust::tabulate(
//          thrust::device, currentSurface.begin(), currentSurface.end(), [&](size_t index) {
//            // s0-s1
//            // |   |
//            // s2-s3
//
//            const uint16_t s0 = lastSurface[getIndexToLast(index, curWidth, curHeight, 0, 1)];
//            const uint16_t s1 = lastSurface[getIndexToLast(index, curWidth, curHeight, 1, 1)];
//            const uint16_t s2 = lastSurface[getIndexToLast(index, curWidth, curHeight, 0, 0)];
//            const uint16_t s3 = lastSurface[getIndexToLast(index, curWidth, curHeight, 1, 0)];
//
//            // check for internal continuity
//            const uint16_t internal_continuity =
//                (s0 >> BIT_CONTINUOUS_R) & (s0 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_T) &
//                (s3 >> BIT_CONTINUOUS_L) & 1;
//
//            // if any child is no complete surface, the parent is neither
//            const uint16_t is_surface = s0 & s1 & s2 & s3 & internal_continuity;
//
//            // check for horizontal and vertical continuity
//            const uint16_t t = (s0 >> BIT_CONTINUOUS_T) & (s1 >> BIT_CONTINUOUS_T) & 1;
//            const uint16_t r = (s1 >> BIT_CONTINUOUS_R) & (s3 >> BIT_CONTINUOUS_R) & 1;
//            const uint16_t b = (s2 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_B) & 1;
//            const uint16_t l = (s0 >> BIT_CONTINUOUS_L) & (s2 >> BIT_CONTINUOUS_L) & 1;
//
//            // check for diagonal continuity
//            const uint16_t tl = (s0 >> BIT_CONTINUOUS_TL) & 1;
//            const uint16_t tr = (s1 >> BIT_CONTINUOUS_TR) & 1;
//            const uint16_t bl = (s2 >> BIT_CONTINUOUS_BL) & 1;
//            const uint16_t br = (s3 >> BIT_CONTINUOUS_BR) & 1;
//
//            // check for external continuity
//            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
//                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
//                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
//                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);
//
//            return is_surface | continuous;
//          });
//    }
//    surfaceDetectionBuffers.push_back(currentSurface);
//  }
//  for (int level = 0; level < levels; level++) {
//    thrust::host_vector<uint16_t> output = surfaceDetectionBuffers[level];
//    for (int i = 0; i < static_cast<int>(height / std::powl(2, level + 1)); i++) {
//      thrust::host_vector<uint16_t> line(
//          output.begin() + static_cast<int>(height / std::powl(2, level + 1)) * i,
//          output.begin() + static_cast<int>(height / std::powl(2, level + 1)) * (i + 1));
//      logger().trace("{}", line);
//    }
//  }
//  mSurfaces = std::move(surfaceDetectionBuffers);
//}void IrregularGrid::surfaceDetection(uint32_t width, uint32_t height, float* texture) {
//  const int    cellSize   = 16;
//  const int    levels     = static_cast<int>(std::log2l(cellSize));
//  const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
//
//  thrust::device_vector<float> dDepth(texture, texture + pixelCount);
//
//  SurfaceDetectionBuffer surfaceDetectionBuffers;
//  for (int i = 0; i < levels; ++i) {
//    thrust::device_vector<uint16_t>        currentSurface(pixelCount >> (2 * (i + 1)));
//    thrust::device_vector<uint16_t> const& lastSurface = surfaceDetectionBuffers.back();
//
//    const int curWidth  = static_cast<int>(width >> (i + 1));
//    const int curHeight = static_cast<int>(height >> (i + 1));
//    if (i == 0) {
//      thrust::tabulate(
//          thrust::device, currentSurface.begin(), currentSurface.end(), [&](size_t index) {
//            // d0  d1   d2   d3
//            //   \  |    |  /
//            // d4--d5-- d6-- d7
//            //      |    |
//            // d8--d9--d10--d11
//            //   /  |    |  \
//            // d12 d13  d14 d15
//
//            const float d0 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 2)];
//            const float d1 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 2)];
//            const float d2 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 2)];
//            const float d3 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 2)];
//
//            const float d4 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 1)];
//            const float d5 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 1)];
//            const float d6 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 1)];
//            const float d7 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 1)];
//
//            const float d8  = dDepth[getIndexToLast(index, curWidth, curHeight, -1, 0)];
//            const float d9  = dDepth[getIndexToLast(index, curWidth, curHeight, 0, 0)];
//            const float d10 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, 0)];
//            const float d11 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, 0)];
//
//            const float d12 = dDepth[getIndexToLast(index, curWidth, curHeight, -1, -1)];
//            const float d13 = dDepth[getIndexToLast(index, curWidth, curHeight, 0, -1)];
//            const float d14 = dDepth[getIndexToLast(index, curWidth, curHeight, 1, -1)];
//            const float d15 = dDepth[getIndexToLast(index, curWidth, curHeight, 2, -1)];
//
//            // check for horizontal and vertical continuity
//            const bool t = is_on_line(d1, d5, d9) && is_on_line(d2, d6, d10);
//            const bool r = is_on_line(d5, d6, d7) && is_on_line(d9, d10, d11);
//            const bool b = is_on_line(d5, d9, d13) && is_on_line(d6, d10, d14);
//            const bool l = is_on_line(d4, d5, d6) && is_on_line(d8, d9, d10);
//
//            // check for diagonal continuity
//            const bool tl = is_on_line(d0, d5, d10);
//            const bool tr = is_on_line(d3, d6, d9);
//            const bool bl = is_on_line(d12, d9, d6);
//            const bool br = is_on_line(d5, d10, d15);
//
//            // if the patch is connected on two othogonal sides, it represents a surface
//            const uint16_t is_surface = (t & r) | (r & b) | (b & l) | (l & t);
//            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
//                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
//                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
//                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);
//
//            // store all continuities
//            return is_surface | continuous;
//          });
//    } else {
//      thrust::tabulate(
//          thrust::device, currentSurface.begin(), currentSurface.end(), [&](size_t index) {
//            // s0-s1
//            // |   |
//            // s2-s3
//
//            const uint16_t s0 = lastSurface[getIndexToLast(index, curWidth, curHeight, 0, 1)];
//            const uint16_t s1 = lastSurface[getIndexToLast(index, curWidth, curHeight, 1, 1)];
//            const uint16_t s2 = lastSurface[getIndexToLast(index, curWidth, curHeight, 0, 0)];
//            const uint16_t s3 = lastSurface[getIndexToLast(index, curWidth, curHeight, 1, 0)];
//
//            // check for internal continuity
//            const uint16_t internal_continuity =
//                (s0 >> BIT_CONTINUOUS_R) & (s0 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_T) &
//                (s3 >> BIT_CONTINUOUS_L) & 1;
//
//            // if any child is no complete surface, the parent is neither
//            const uint16_t is_surface = s0 & s1 & s2 & s3 & internal_continuity;
//
//            // check for horizontal and vertical continuity
//            const uint16_t t = (s0 >> BIT_CONTINUOUS_T) & (s1 >> BIT_CONTINUOUS_T) & 1;
//            const uint16_t r = (s1 >> BIT_CONTINUOUS_R) & (s3 >> BIT_CONTINUOUS_R) & 1;
//            const uint16_t b = (s2 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_B) & 1;
//            const uint16_t l = (s0 >> BIT_CONTINUOUS_L) & (s2 >> BIT_CONTINUOUS_L) & 1;
//
//            // check for diagonal continuity
//            const uint16_t tl = (s0 >> BIT_CONTINUOUS_TL) & 1;
//            const uint16_t tr = (s1 >> BIT_CONTINUOUS_TR) & 1;
//            const uint16_t bl = (s2 >> BIT_CONTINUOUS_BL) & 1;
//            const uint16_t br = (s3 >> BIT_CONTINUOUS_BR) & 1;
//
//            // check for external continuity
//            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
//                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
//                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
//                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);
//
//            return is_surface | continuous;
//          });
//    }
//    surfaceDetectionBuffers.push_back(currentSurface);
//  }
//  for (int level = 0; level < levels; level++) {
//    thrust::host_vector<uint16_t> output = surfaceDetectionBuffers[level];
//    for (int i = 0; i < static_cast<int>(height / std::powl(2, level + 1)); i++) {
//      thrust::host_vector<uint16_t> line(
//          output.begin() + static_cast<int>(height / std::powl(2, level + 1)) * i,
//          output.begin() + static_cast<int>(height / std::powl(2, level + 1)) * (i + 1));
//      logger().trace("{}", line);
//    }
//  }
//  mSurfaces = std::move(surfaceDetectionBuffers);
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

void emit(thrust::device_vector<glm::u16vec3> verts, glm::u16vec3 pos, glm::u16vec2 offset,
    uint16_t data, uint32_t width, uint32_t height) {
  glm::u16vec3 out(pos.xy + offset, data);
  if (out.x < width && out.y < height) {
    verts.push_back(out);
  }
}

void IrregularGrid::createBuffers(uint32_t width, uint32_t height) {
  {
    //thrust::device_vector<glm::u16vec3> dVertices;
    //thrust::device_vector<unsigned>     dIndices;

    //uint16_t levels = static_cast<uint16_t>(mSurfaces.size());

    //thrust::host_vector<glm::u16vec3> hVertices((width >> levels) * (height >> levels));
    //thrust::tabulate(thrust::host, hVertices.begin(), hVertices.end(), [&](size_t index) {
    //  glm::u16vec3 val;
    //  val.x = (index % (width >> levels)) * (1 << levels);
    //  val.y = (index / (width >> levels)) * (1 << levels);
    //  val.z = (levels - 1) << BIT_CURRENT_LEVEL;
    //  return val;
    //});
    //std::vector<uint16_t> stdVerts(
    //    (uint16_t*)hVertices.data(), (uint16_t*)hVertices.data() + hVertices.size() * 3);
    //dVertices = hVertices;

    //for (int level = levels; level >= 0; --level) {
    //  thrust::device_vector<uint16_t> const& currentSurface = mSurfaces[level];
    //  thrust::transform_reduce(
    //      thrust::device, dVertices.begin(), dVertices.end(),
    //      [&](glm::u16vec3 val) {
    //        thrust::device_vector<glm::u16vec3> out;

    //        uint16_t vertexLevel = val.z >> BIT_CURRENT_LEVEL;
    //        uint16_t surfaceData = 0; // TODO

    //        if (vertexLevel == level) {
    //          if (level == levels - 1) {
    //          } else {
    //          }
    //        } else {
    //          emit(out, val, glm::u16vec2(0, 0),
    //              (level << BIT_CURRENT_LEVEL | (ALL_CONTINUITY_BITS & surfaceData) | 1), width,
    //              height);
    //        }

    //        return out;
    //      },
    //      dVertices,
    //      [&](thrust::device_vector<glm::u16vec3> acc, thrust::device_vector<glm::u16vec3> val) {
    //        // TODO Bound checking?
    //        thrust::copy(thrust::device, val.begin(), val.end(), acc.begin());
    //        return acc;
    //      });
    //}

    /*hVertices                              = dVertices;
    thrust::host_vector<unsigned> hIndices = dIndices;

    mVAO.Bind();

    mVBO.Bind(GL_ARRAY_BUFFER);
    mVBO.BufferData(hVertices.size() * sizeof(float), hVertices.data(), GL_STATIC_DRAW);

    mIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
    mIBO.BufferData(hIndices.size() * sizeof(unsigned), hIndices.data(), GL_STATIC_DRAW);

    mVAO.EnableAttributeArray(0);
    mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

    mVAO.Release();
    mIBO.Release();
    mVBO.Release();*/
  }
  {

    mIndexCount = (height - 1) * (2 + 2 * width);
    std::vector<float>    vertices(width * height * 3);
    std::vector<unsigned> indices(mIndexCount);

    for (uint32_t x = 0; x < width; ++x) {
      for (uint32_t y = 0; y < height; ++y) {
        vertices[(x * height + y) * 3 + 0] = 2.f / (width - 1) * x - 1.f;
        vertices[(x * height + y) * 3 + 1] = 2.f / (height - 1) * y - 1.f;
        vertices[(x * height + y) * 3 + 2] = 0.f;
      }
    }

    uint32_t index = 0;

    for (uint32_t x = 0; x < width - 1; ++x) {
      indices[index++] = x * height;
      for (uint32_t y = 0; y < height; ++y) {
        indices[index++] = x * height + y;
        indices[index++] = (x + 1) * height + y;
      }
      indices[index] = indices[index - 1];
      ++index;
    }

    mVAO.Bind();

    mVBO.Bind(GL_ARRAY_BUFFER);
    mVBO.BufferData(vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    mIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
    mIBO.BufferData(indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

    mVAO.EnableAttributeArray(0);
    mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

    mVAO.Release();
    mIBO.Release();
    mVBO.Release();

    // Create visualization buffers
    mVisIndexCount = ((height - 1) * (width) + (height) * (width - 1)) * 2;
    std::vector<unsigned> visIndices(mVisIndexCount);
    index = 0;
    for (uint32_t x = 0; x < width; ++x) {
      for (uint32_t y = 0; y < height - 1; ++y) {
        visIndices[index++] = x * height + y;
        visIndices[index++] = x * height + y + 1;
      }
    }
    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width - 1; ++x) {
        visIndices[index++] = x * height + y;
        visIndices[index++] = (x + 1) * height + y;
      }
    }

    mVisVAO.Bind();

    mVBO.Bind(GL_ARRAY_BUFFER);

    mVisIBO.Bind(GL_ELEMENT_ARRAY_BUFFER);
    mVisIBO.BufferData(visIndices.size() * sizeof(unsigned), visIndices.data(), GL_STATIC_DRAW);

    mVisVAO.EnableAttributeArray(0);
    mVisVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, &mVBO);

    mVisVAO.Release();
    mVisIBO.Release();
    mVBO.Release();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
