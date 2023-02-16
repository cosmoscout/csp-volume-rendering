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

class SurfaceDetectionBuffer {
 public:
  using Index2d = glm::uvec2;
  using Vertex  = glm::uvec3;
  using Buffer  = thrust::device_vector<uint16_t>;

  SurfaceDetectionBuffer(float* depthTexture, int width, int height, int cellSize = 16);

  void                          print() const;
  thrust::device_vector<Vertex> generateVertices();

 private:
  __host__ __device__ int        to1dIndex(Index2d index, int level) const;
  __host__ __device__ Index2d    to2dIndex(int index, int level) const;
  __host__ __device__ Index2d    toLevel(Index2d index, int from, int to) const;
  __host__ __device__ int        toLevel(int index, int from, int to, Index2d offset) const;
  __host__ __device__ int        levelSize(int level) const;
  __host__ __device__ glm::uvec2 levelDim(unsigned int level) const;
  __host__ __device__ Vertex     emit(Vertex pos, Index2d offset, unsigned int factor, int data) const;

  const unsigned int  mCellSize;
  const unsigned int  mWidth;
  const unsigned int  mHeight;
  const unsigned int  mLevels;
  const unsigned int  mLastLevel;
  std::vector<Buffer> mBuffers;

  friend struct DetectSurfaceInBase;
  friend struct DetectSurfaceInHigherLevel;
  friend struct GenerateHighLevelVerts;
  friend struct SplitVerts;
  friend struct GenerateStencil;
  friend struct GetVertCount;
};

} // namespace csp::volumerendering

#endif
