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
  return abs(a - 2 * b + c) < threshold;
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

  for (int i = 0; i < mLevels; ++i) {
    thrust::device_vector<uint16_t>        currentSurface(levelSize(i + 1));
    thrust::device_vector<uint16_t> const& lastSurface = mBuffers.back();
    if (i == 0) {
      thrust::tabulate(
          thrust::device, currentSurface.begin(), currentSurface.end(), [&](int const& index) {
            // d0  d1   d2   d3
            //   \  |    |  /
            // d4--d5-- d6-- d7
            //      |    |
            // d8--d9--d10--d11
            //   /  |    |  \
            // d12 d13  d14 d15

            const float d0 = dDepth[toLevel(index, 1, 0, glm::ivec2(-1, 2))];
            const float d1 = dDepth[toLevel(index, 1, 0, glm::ivec2(0, 2))];
            const float d2 = dDepth[toLevel(index, 1, 0, glm::ivec2(1, 2))];
            const float d3 = dDepth[toLevel(index, 1, 0, glm::ivec2(2, 2))];

            const float d4 = dDepth[toLevel(index, 1, 0, glm::ivec2(-1, 1))];
            const float d5 = dDepth[toLevel(index, 1, 0, glm::ivec2(0, 1))];
            const float d6 = dDepth[toLevel(index, 1, 0, glm::ivec2(1, 1))];
            const float d7 = dDepth[toLevel(index, 1, 0, glm::ivec2(2, 1))];

            const float d8  = dDepth[toLevel(index, 1, 0, glm::ivec2(-1, 0))];
            const float d9  = dDepth[toLevel(index, 1, 0, glm::ivec2(0, 0))];
            const float d10 = dDepth[toLevel(index, 1, 0, glm::ivec2(1, 0))];
            const float d11 = dDepth[toLevel(index, 1, 0, glm::ivec2(2, 0))];

            const float d12 = dDepth[toLevel(index, 1, 0, glm::ivec2(-1, -1))];
            const float d13 = dDepth[toLevel(index, 1, 0, glm::ivec2(0, -1))];
            const float d14 = dDepth[toLevel(index, 1, 0, glm::ivec2(1, -1))];
            const float d15 = dDepth[toLevel(index, 1, 0, glm::ivec2(2, -1))];

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
      thrust::tabulate(
          thrust::device, currentSurface.begin(), currentSurface.end(), [&](int const& index) {
            // s0-s1
            // |   |
            // s2-s3

            const uint16_t s0 = lastSurface[toLevel(index, i + 1, i, glm::ivec2(0, 1))];
            const uint16_t s1 = lastSurface[toLevel(index, i + 1, i, glm::ivec2(1, 1))];
            const uint16_t s2 = lastSurface[toLevel(index, i + 1, i, glm::ivec2(0, 0))];
            const uint16_t s3 = lastSurface[toLevel(index, i + 1, i, glm::ivec2(1, 0))];

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
  for (int level = 0; level < mLevels; level++) {
    thrust::host_vector<uint16_t> output = mBuffers[level];
    for (int i = 0; i < levelDim(level + 1).y; i++) {
      thrust::host_vector<uint16_t> line(output.begin() + levelDim(level + 1).y * i,
          output.begin() + levelDim(level + 1).y * (i + 1));
      logger().trace("{}", line);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

thrust::device_vector<glm::ivec3> SurfaceDetectionBuffer::generateVertices() {
  thrust::device_vector<glm::ivec3> dVertices;

  thrust::host_vector<glm::ivec3> hVertices(levelSize(mLastLevel));
  thrust::tabulate(thrust::host, hVertices.begin(), hVertices.end(), [&](int index) {
    glm::ivec3 val;
    val.xy = to2dIndex(index, mLastLevel) * mCellSize;
    val.z  = mLastLevel << BIT_CURRENT_LEVEL;
    return val;
  });
  dVertices = hVertices;

  for (int level = mLastLevel; level >= 0; --level) {
    thrust::device_vector<uint16_t> const& currentSurface = mBuffers[level];
    thrust::device_vector<uint16_t> const& finerSurface   = mBuffers[level - 1];
    thrust::device_vector<glm::ivec3>      dNewVertices;
    dVertices = thrust::transform_reduce(
        thrust::device, dVertices.begin(), dVertices.end(),
        [&](glm::ivec3 const& val) {
          thrust::device_vector<glm::ivec3> out;

          int      quadSize    = 1 << (level - 1);
          uint16_t vertexLevel = val.z >> BIT_CURRENT_LEVEL;
          uint16_t surfaceData = currentSurface[to1dIndex(toLevel(val.xy, 0, level), level)];

          if (vertexLevel == level) {
            if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
              if (level == mLastLevel) {
                const uint16_t s0 = finerSurface[to1dIndex(
                    toLevel(val.xy, 0, level - 1) + glm::ivec2(0, 1), level - 1)];
                const uint16_t s1 = finerSurface[to1dIndex(
                    toLevel(val.xy, 0, level - 1) + glm::ivec2(1, 1), level - 1)];
                const uint16_t s2 = finerSurface[to1dIndex(
                    toLevel(val.xy, 0, level - 1) + glm::ivec2(0, 0), level - 1)];
                const uint16_t s3 = finerSurface[to1dIndex(
                    toLevel(val.xy, 0, level - 1) + glm::ivec2(1, 0), level - 1)];

                emit(out, val, glm::ivec2(0, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s0);
                emit(out, val, glm::ivec2(1, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s1);
                emit(out, val, glm::ivec2(0, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s2);
                emit(out, val, glm::ivec2(1, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL) | s3);
              } else {
                emit(out, val, glm::ivec2(0, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                emit(out, val, glm::ivec2(1, 1), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                emit(out, val, glm::ivec2(0, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
                emit(out, val, glm::ivec2(1, 0), quadSize, ((level - 1) << BIT_CURRENT_LEVEL));
              }
            } else {
              emit(out, val, glm::u16vec2(0, 0), 1,
                  (level << BIT_CURRENT_LEVEL | (ALL_CONTINUITY_BITS & surfaceData) | 1));
            }
          } else {
            out.push_back(val);
          }
          return out;
        },
        dNewVertices,
        [&](thrust::device_vector<glm::ivec3> acc, thrust::device_vector<glm::ivec3> val) {
          thrust::copy(thrust::device, val.begin(), val.end(), std::back_inserter(acc));
          return acc;
        });
    logger().trace("L{}: {} vertices", level, dVertices.size());
  }

  return dVertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceDetectionBuffer::to1dIndex(glm::ivec2 index, int level) const {
  return index.y * levelDim(level).x + index.x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec2 SurfaceDetectionBuffer::to2dIndex(int index, int level) const {
  int width = levelDim(level).x;
  int x     = index % width;
  int y     = index / width;
  return {x, y};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec2 SurfaceDetectionBuffer::toLevel(glm::ivec2 index, int from, int to) const {
  int shift = to - from;
  if (shift > 0) {
    return index >> shift;
  } else {
    return index << -shift;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Offset is applied after transformation to new level
int SurfaceDetectionBuffer::toLevel(int index, int from, int to, glm::ivec2 offset) const {
  return to1dIndex(toLevel(to2dIndex(index, from), from, to) + offset, to);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int SurfaceDetectionBuffer::levelSize(int level) const {
  return (mWidth >> level) * (mHeight >> level);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec2 SurfaceDetectionBuffer::levelDim(int level) const {
  return glm::ivec2(mWidth, mHeight) >> level;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SurfaceDetectionBuffer::emit(thrust::device_vector<glm::ivec3>& verts, glm::ivec3 pos,
    glm::ivec2 offset, int factor, int data) const {
  glm::ivec3 out(pos.xy + offset * factor, data);
  if (out.x < mWidth && out.y < mHeight) {
    verts.push_back(out);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering