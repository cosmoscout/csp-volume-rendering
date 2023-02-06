////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SURFACE_DETECTION_BUFFER_HPP
#define CSP_VOLUME_RENDERING_SURFACE_DETECTION_BUFFER_HPP

#include <thrust/device_vector.h>

#include <glm/fwd.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace csp::volumerendering {

class SurfaceDetectionBuffer {
 public:
  using Buffer = thrust::device_vector<uint16_t>;

  SurfaceDetectionBuffer(float* depthTexture, int width, int height, int cellSize = 16);

  void                              print() const;
  thrust::device_vector<glm::ivec3> generateVertices();

 private:
  int        to1dIndex(glm::ivec2 index, int level) const;
  glm::ivec2 to2dIndex(int index, int level) const;
  glm::ivec2 toLevel(glm::ivec2 index, int from, int to) const;
  int        toLevel(int index, int from, int to, glm::ivec2 offset) const;
  int        levelSize(int level) const;
  glm::ivec2 levelDim(int level) const;
  void emit(thrust::device_vector<glm::ivec3>& verts, glm::ivec3 pos, glm::ivec2 offset, int factor,
      int data) const;

  const int           mCellSize;
  const int           mWidth;
  const int           mHeight;
  const int           mLevels;
  const int           mLastLevel;
  std::vector<Buffer> mBuffers;

  // Bit pattern for storing connectivity
  static constexpr int BIT_IS_SURFACE = 0;

  static constexpr int BIT_CONTINUOUS_T = 4;
  static constexpr int BIT_CONTINUOUS_R = 5;
  static constexpr int BIT_CONTINUOUS_B = 6;
  static constexpr int BIT_CONTINUOUS_L = 7;

  static constexpr int BIT_CONTINUOUS_TR = 8;
  static constexpr int BIT_CONTINUOUS_TL = 9;
  static constexpr int BIT_CONTINUOUS_BR = 10;
  static constexpr int BIT_CONTINUOUS_BL = 11;

  static constexpr int BIT_CURRENT_LEVEL = 12; // 12-15 (requires 3 bits)

  static constexpr int ALL_CONTINUITY_BITS = 4080;
  static constexpr int ALL_DATA_BITS       = 4095;
};

} // namespace csp::volumerendering

#endif
