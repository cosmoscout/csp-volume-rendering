////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SURFACE_DETECTION_BUFFER_HPP
#define CSP_VOLUME_RENDERING_SURFACE_DETECTION_BUFFER_HPP

#include <thrust/device_vector.h>

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
  struct Vec2 {
    int x;
    int y;

    __host__ __device__ Vec2(int pX, int pY)
        : x(pX)
        , y(pY) {
    }

    __host__ __device__ inline Vec2 operator+(Vec2 const& other) const {
      Vec2 res(x + other.x, y + other.y);
      return res;
    }
  };

  struct Vertex {
    unsigned int x;
    unsigned int y;
    unsigned int data;

    Vertex() = default;

    __host__ __device__ Vertex(unsigned int pX, unsigned int pY, unsigned int pData)
        : x(pX)
        , y(pY)
        , data(pData) {
    }

    __host__ __device__ inline Vec2 xy() const {
      Vec2 xy(x, y);
      return xy;
    }
  };

  using Buffer = thrust::device_vector<uint16_t>;
  using Level  = unsigned int;

  struct GridParams {
    const unsigned int mCellSize;
    const unsigned int mWidth;
    const unsigned int mHeight;
    const unsigned int mLevels;
    const unsigned int mLastLevel;

    __host__ GridParams(
        const unsigned int cellSize, const unsigned int width, const unsigned int height)
        : mCellSize(cellSize)
        , mWidth(width)
        , mHeight(height)
        , mLevels(static_cast<int>(std::log2(mCellSize)))
        , mLastLevel(mLevels - 1) {
    }

    GridParams(GridParams const& other) = default;

    __host__ __device__ int    to1dIndex(Vec2 index, Level level) const;
    __host__ __device__ Vec2   to2dIndex(size_t index, Level level) const;
    __host__ __device__ Vec2   toLevel(Vec2 index, Level from, Level to) const;
    __host__ __device__ int    toLevel(size_t index, Level from, Level to, Vec2 offset) const;
    __host__ __device__ int    levelSize(Level level) const;
    __host__ __device__ Vec2   levelDim(Level level) const;
    __host__ __device__ Vertex emit(
        Vertex pos, Vec2 offset, unsigned int factor, unsigned int data) const;
  };

  SurfaceDetectionBuffer(float* depthTexture, int width, int height, int cellSize = 16);

  void                          print() const;
  thrust::device_vector<Vertex> generateVertices();

 private:
  std::vector<Buffer> mBuffers;
  GridParams          mGridParams;

  friend struct DetectSurfaceInBase;
  friend struct DetectSurfaceInHigherLevel;
  friend struct GenerateHighLevelVerts;
  friend struct SplitVerts;
  friend struct GenerateStencil;
};

} // namespace csp::volumerendering

#endif
