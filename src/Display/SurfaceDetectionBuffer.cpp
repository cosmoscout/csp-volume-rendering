////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SurfaceDetectionBuffer.hpp"

#include "../logger.hpp"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_on_line(float a, float b, float c) {
  float threshold = 0.1f;
  return abs(a - 2 * b + c) < threshold || (isinf(a) && isinf(b) && isinf(c));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceDetectionBuffer::SurfaceDetectionBuffer(
    float* depthTexture, int width, int height, int cellSize)
    : mCellSize(cellSize)
    , mWidth(width)
    , mHeight(height)
    , mLevels(static_cast<int>(std::log2(mCellSize)))
    , mLastLevel(mLevels - 1) {
  thrust::device_vector<float> dDepth(depthTexture, depthTexture + levelSize(0));
  const float*                 pDepth = thrust::raw_pointer_cast(dDepth.data());

  for (unsigned int i = 0; i < mLevels; ++i) {
    thrust::device_vector<uint16_t> currentSurface(levelSize(i + 1));
    const uint16_t*                 lastSurface = thrust::raw_pointer_cast(mBuffers.back().data());
    if (i == 0) {
      thrust::tabulate(thrust::host, currentSurface.begin(), currentSurface.end(),
          [=] __host__ __device__(int const& index) {
            // d0  d1   d2   d3
            //   \  |    |  /
            // d4--d5-- d6-- d7
            //      |    |
            // d8--d9--d10--d11
            //   /  |    |  \
            // d12 d13  d14 d15

            const float d0 = pDepth[toLevel(index, 1, 0, Index2d(-1, 2))];
            const float d1 = pDepth[toLevel(index, 1, 0, Index2d(0, 2))];
            const float d2 = pDepth[toLevel(index, 1, 0, Index2d(1, 2))];
            const float d3 = pDepth[toLevel(index, 1, 0, Index2d(2, 2))];

            const float d4 = pDepth[toLevel(index, 1, 0, Index2d(-1, 1))];
            const float d5 = pDepth[toLevel(index, 1, 0, Index2d(0, 1))];
            const float d6 = pDepth[toLevel(index, 1, 0, Index2d(1, 1))];
            const float d7 = pDepth[toLevel(index, 1, 0, Index2d(2, 1))];

            const float d8  = pDepth[toLevel(index, 1, 0, Index2d(-1, 0))];
            const float d9  = pDepth[toLevel(index, 1, 0, Index2d(0, 0))];
            const float d10 = pDepth[toLevel(index, 1, 0, Index2d(1, 0))];
            const float d11 = pDepth[toLevel(index, 1, 0, Index2d(2, 0))];

            const float d12 = pDepth[toLevel(index, 1, 0, Index2d(-1, -1))];
            const float d13 = pDepth[toLevel(index, 1, 0, Index2d(0, -1))];
            const float d14 = pDepth[toLevel(index, 1, 0, Index2d(1, -1))];
            const float d15 = pDepth[toLevel(index, 1, 0, Index2d(2, -1))];

            // check for horizontal and vertical continuity
            const bool t = is_on_line(d1, d5, d9) && is_on_line(d2, d6, d10);
            const bool r = is_on_line(d5, d6, d7) && is_on_line(d9, d10, d11);
            const bool b = is_on_line(d5, d9, d13) && is_on_line(d6, d10, d14);
            const bool l = is_on_line(d4, d5, d6) && is_on_line(d8, d9, d10);

            // check for diagonal continuity
            const bool tl = is_on_line(d0, d5, d10);
            const bool tr = is_on_line(d3, d6, d9);
            const bool bl = is_on_line(d12, d9, d6);
            const bool br = is_on_line(d5, d10, d15);

            // if the patch is connected on two othogonal sides, it represents a surface
            const uint16_t is_surface = (t & r) | (r & b) | (b & l) | (l & t);
            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);

            // store all continuities
            return is_surface | continuous;
          });
    } else {
      thrust::tabulate(thrust::host, currentSurface.begin(), currentSurface.end(),
          [=] __host__ __device__(int const& index) {
            // s0-s1
            // |   |
            // s2-s3

            const uint16_t s0 = lastSurface[toLevel(index, i + 1, i, Index2d(0, 1))];
            const uint16_t s1 = lastSurface[toLevel(index, i + 1, i, Index2d(1, 1))];
            const uint16_t s2 = lastSurface[toLevel(index, i + 1, i, Index2d(0, 0))];
            const uint16_t s3 = lastSurface[toLevel(index, i + 1, i, Index2d(1, 0))];

            // check for internal continuity
            const uint16_t internal_continuity =
                (s0 >> BIT_CONTINUOUS_R) & (s0 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_T) &
                (s3 >> BIT_CONTINUOUS_L) & 1;

            // if any child is no complete surface, the parent is neither
            const uint16_t is_surface = s0 & s1 & s2 & s3 & internal_continuity;

            // check for horizontal and vertical continuity
            const uint16_t t = (s0 >> BIT_CONTINUOUS_T) & (s1 >> BIT_CONTINUOUS_T) & 1;
            const uint16_t r = (s1 >> BIT_CONTINUOUS_R) & (s3 >> BIT_CONTINUOUS_R) & 1;
            const uint16_t b = (s2 >> BIT_CONTINUOUS_B) & (s3 >> BIT_CONTINUOUS_B) & 1;
            const uint16_t l = (s0 >> BIT_CONTINUOUS_L) & (s2 >> BIT_CONTINUOUS_L) & 1;

            // check for diagonal continuity
            const uint16_t tl = (s0 >> BIT_CONTINUOUS_TL) & 1;
            const uint16_t tr = (s1 >> BIT_CONTINUOUS_TR) & 1;
            const uint16_t bl = (s2 >> BIT_CONTINUOUS_BL) & 1;
            const uint16_t br = (s3 >> BIT_CONTINUOUS_BR) & 1;

            // check for external continuity
            const uint16_t continuous = (t << BIT_CONTINUOUS_T) | (r << BIT_CONTINUOUS_R) |
                                        (b << BIT_CONTINUOUS_B) | (l << BIT_CONTINUOUS_L) |
                                        (tl << BIT_CONTINUOUS_TL) | (tr << BIT_CONTINUOUS_TR) |
                                        (bl << BIT_CONTINUOUS_BL) | (br << BIT_CONTINUOUS_BR);

            return is_surface | continuous;
          });
    }
    mBuffers.push_back(currentSurface);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SurfaceDetectionBuffer::print() const {
  for (unsigned int level = 0; level < mLevels; level++) {
    thrust::host_vector<uint16_t> output = mBuffers[level];
    for (unsigned int i = 0; i < levelDim(level + 1).y; i++) {
      thrust::host_vector<uint16_t> line(output.begin() + levelDim(level + 1).y * i,
          output.begin() + levelDim(level + 1).y * (i + 1));
      logger().trace("{}", line);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

thrust::device_vector<SurfaceDetectionBuffer::Vertex> SurfaceDetectionBuffer::generateVertices() {
  thrust::device_vector<Vertex> dVertices;
  thrust::host_vector<Vertex>   hVertices(levelSize(mLastLevel));
  thrust::tabulate(thrust::host, hVertices.begin(), hVertices.end(), [&](int index) {
    Vertex val;
    val.xy = to2dIndex(index, mLastLevel) * mCellSize;
    val.z  = mLastLevel << BIT_CURRENT_LEVEL;
    return val;
  });
  dVertices = hVertices;

  constexpr Vertex NO_VERTEX = {0u, 0u, ~0u};

  for (int level = mLastLevel; level >= 0; --level) {
    const uint16_t* currentSurface = thrust::raw_pointer_cast(mBuffers[level].data());
    const uint16_t* finerSurface   = thrust::raw_pointer_cast(mBuffers[level - 1].data());

    const Vertex*                 pOldVertices = thrust::raw_pointer_cast(dVertices.data());
    thrust::device_vector<Vertex> dNewVertices(dVertices.size() * 4);
    thrust::tabulate(thrust::host, dNewVertices.begin(), dNewVertices.end(),
        [=] __host__ __device__(int const& index) {
          Vertex       oldVal      = pOldVertices[index >> 2];
          int          quadSize    = 1 << (level);
          unsigned int vertexLevel = oldVal.z >> BIT_CURRENT_LEVEL;
          unsigned int surfaceData =
              currentSurface[to1dIndex(toLevel(oldVal.xy, 0, level + 1), level + 1)];

          if (vertexLevel == level) {
            if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
              if (level == mLastLevel) {
                const unsigned int s0 =
                    finerSurface[to1dIndex(toLevel(oldVal.xy, 0, level) + Index2d(0, 1), level)];
                const unsigned int s1 =
                    finerSurface[to1dIndex(toLevel(oldVal.xy, 0, level) + Index2d(1, 1), level)];
                const unsigned int s2 =
                    finerSurface[to1dIndex(toLevel(oldVal.xy, 0, level) + Index2d(0, 0), level)];
                const unsigned int s3 =
                    finerSurface[to1dIndex(toLevel(oldVal.xy, 0, level) + Index2d(1, 0), level)];

                switch (index % 4) {
                case 0:
                  return emit(
                      oldVal, Index2d(0, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s0);
                case 1:
                  return emit(
                      oldVal, Index2d(1, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s1);
                case 2:
                  return emit(
                      oldVal, Index2d(0, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s2);
                case 3:
                  return emit(
                      oldVal, Index2d(1, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s3);
                }
              } else {
                switch (index % 4) {
                case 0:
                  return emit(oldVal, Index2d(0, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                case 1:
                  return emit(oldVal, Index2d(1, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                case 2:
                  return emit(oldVal, Index2d(0, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                case 3:
                  return emit(oldVal, Index2d(1, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                }
              }
            } else {
              return index % 4 == 0 ? emit(oldVal, glm::u16vec2(0, 0), 1,
                                          (level << BIT_CURRENT_LEVEL |
                                              (ALL_CONTINUITY_BITS & surfaceData) | 1))
                                    : NO_VERTEX;
            }
          } else {
            return index % 4 == 0 ? oldVal : NO_VERTEX;
          }
        });

    thrust::device_vector<int> dVertexCounts(dVertices.size());
    thrust::tabulate(thrust::host, dVertexCounts.begin(), dVertexCounts.end(),
        [=] __host__ __device__(int const& index) {
          Vertex       oldVal      = pOldVertices[index >> 2];
          unsigned int vertexLevel = oldVal.z >> BIT_CURRENT_LEVEL;
          unsigned int surfaceData =
              currentSurface[to1dIndex(toLevel(oldVal.xy, 0, level + 1), level + 1)];

          if (vertexLevel == level) {
            if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
              return 4;
            } else {
              return 1;
            }
          } else {
            return 1;
          }
        });

    thrust::device_vector<int> dVertexStartIndex(dVertices.size());
    thrust::exclusive_scan(dVertexCounts.begin(), dVertexCounts.end(), dVertexStartIndex.begin());
  }

  logger().trace("Got {} verts", dVertices.size());
  return dVertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceDetectionBuffer::to1dIndex(Index2d index, int level) const {
  return std::clamp(index.y, 0u, levelDim(level).y) * levelDim(level).x +
         std::clamp(index.x, 0u, levelDim(level).x);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceDetectionBuffer::Index2d SurfaceDetectionBuffer::to2dIndex(int index, int level) const {
  int width = levelDim(level).x;
  int x     = index % width;
  int y     = index / width;
  return {x, y};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceDetectionBuffer::Index2d SurfaceDetectionBuffer::toLevel(
    Index2d index, int from, int to) const {
  if (to - from > 0) {
    return index >> static_cast<unsigned int>(to - from);
  } else {
    return index << static_cast<unsigned int>(from - to);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Offset is applied after transformation to new level
int SurfaceDetectionBuffer::toLevel(int index, int from, int to, Index2d offset) const {
  return to1dIndex(toLevel(to2dIndex(index, from), from, to) + offset, to);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceDetectionBuffer::levelSize(int level) const {
  return (mWidth >> level) * (mHeight >> level);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::uvec2 SurfaceDetectionBuffer::levelDim(unsigned int level) const {
  return glm::uvec2(mWidth, mHeight) >> level;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceDetectionBuffer::Vertex SurfaceDetectionBuffer::emit(
    Vertex pos, Index2d offset, unsigned int factor, int data) const {
  Vertex out(pos.xy + offset * factor, data);
  out.x = std::min(out.x, mWidth);
  out.y = std::min(out.y, mHeight);
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
