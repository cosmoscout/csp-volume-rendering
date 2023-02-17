////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Functors.hpp"

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ bool is_on_line(float a, float b, float c) {
  float threshold = 0.1f;
  return abs(a - 2 * b + c) < threshold || (isinf(a) && isinf(b) && isinf(c));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

namespace csp::volumerendering {

#define NO_VERTEX SurfaceDetectionBuffer::Vertex(0u, 0u, ~0u)

////////////////////////////////////////////////////////////////////////////////////////////////////

DetectSurfaceInBase::DetectSurfaceInBase(SurfaceDetectionBuffer surfaces, const float* depth)
    : mSurfaces(surfaces)
    , mDepth(depth) {
}

__host__ __device__ uint16_t DetectSurfaceInBase::operator()(size_t const index) const {
  // d0  d1   d2   d3
  //   \  |    |  /
  // d4--d5-- d6-- d7
  //      |    |
  // d8--d9--d10--d11
  //   /  |    |  \
  // d12 d13  d14 d15

  const float d0 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(-1, 2))];
  const float d1 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(0, 2))];
  const float d2 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(1, 2))];
  const float d3 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(2, 2))];

  const float d4 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(-1, 1))];
  const float d5 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(0, 1))];
  const float d6 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(1, 1))];
  const float d7 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(2, 1))];

  const float d8  = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(-1, 0))];
  const float d9  = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(0, 0))];
  const float d10 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(1, 0))];
  const float d11 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(2, 0))];

  const float d12 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(-1, -1))];
  const float d13 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(0, -1))];
  const float d14 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(1, -1))];
  const float d15 = mDepth[mSurfaces.toLevel(index, 1, 0, SurfaceDetectionBuffer::Vec2(2, -1))];

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DetectSurfaceInHigherLevel::DetectSurfaceInHigherLevel(
    SurfaceDetectionBuffer surfaces, const uint16_t* surfaceBuffer, int level)
    : mSurfaces(surfaces)
    , mSurfaceBuffer(surfaceBuffer)
    , mLevel(level) {
}

__host__ __device__ uint16_t DetectSurfaceInHigherLevel::operator()(size_t const index) const {
  // s0-s1
  // |   |
  // s2-s3

  const uint16_t s0 = mSurfaceBuffer[mSurfaces.toLevel(
      index, mLevel + 1, mLevel, SurfaceDetectionBuffer::Vec2(0, 1))];
  const uint16_t s1 = mSurfaceBuffer[mSurfaces.toLevel(
      index, mLevel + 1, mLevel, SurfaceDetectionBuffer::Vec2(1, 1))];
  const uint16_t s2 = mSurfaceBuffer[mSurfaces.toLevel(
      index, mLevel + 1, mLevel, SurfaceDetectionBuffer::Vec2(0, 0))];
  const uint16_t s3 = mSurfaceBuffer[mSurfaces.toLevel(
      index, mLevel + 1, mLevel, SurfaceDetectionBuffer::Vec2(1, 0))];

  // check for internal continuity
  const uint16_t internal_continuity = (s0 >> BIT_CONTINUOUS_R) & (s0 >> BIT_CONTINUOUS_B) &
                                       (s3 >> BIT_CONTINUOUS_T) & (s3 >> BIT_CONTINUOUS_L) & 1;

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GenerateHighLevelVerts::GenerateHighLevelVerts(SurfaceDetectionBuffer surfaces)
    : mSurfaces(surfaces) {
}

__host__ __device__ SurfaceDetectionBuffer::Vertex GenerateHighLevelVerts::operator()(
    size_t const index) const {
  SurfaceDetectionBuffer::Vertex val;
  val.x    = mSurfaces.to2dIndex(index, mSurfaces.mLastLevel).x * mSurfaces.mCellSize;
  val.y    = mSurfaces.to2dIndex(index, mSurfaces.mLastLevel).y * mSurfaces.mCellSize;
  val.data = mSurfaces.mLastLevel << BIT_CURRENT_LEVEL;
  return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SplitVerts::SplitVerts(SurfaceDetectionBuffer surfaces, int level,
    const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface,
    const uint16_t* finerSurface)
    : mSurfaces(surfaces)
    , mLevel(level)
    , mOldVerts(oldVerts)
    , mCurrentSurface(currentSurface)
    , mFinerSurface(finerSurface) {
}

__host__ __device__ SurfaceDetectionBuffer::Vertex SplitVerts::operator()(
    size_t const index) const {
  SurfaceDetectionBuffer::Vertex oldVal      = mOldVerts[index >> 2];
  int                            quadSize    = 1 << (mLevel);
  unsigned int                   vertexLevel = oldVal.data >> BIT_CURRENT_LEVEL;
  unsigned int                   surfaceData = mCurrentSurface[mSurfaces.to1dIndex(
      mSurfaces.toLevel(oldVal.xy(), 0, mLevel + 1), mLevel + 1)];

  if (vertexLevel == mLevel) {
    if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
      if (mLevel == mSurfaces.mLastLevel) {
        const unsigned int s0 = mFinerSurface[mSurfaces.to1dIndex(
            mSurfaces.toLevel(oldVal.xy(), 0, mLevel) + SurfaceDetectionBuffer::Vec2(0, 1),
            mLevel)];
        const unsigned int s1 = mFinerSurface[mSurfaces.to1dIndex(
            mSurfaces.toLevel(oldVal.xy(), 0, mLevel) + SurfaceDetectionBuffer::Vec2(1, 1),
            mLevel)];
        const unsigned int s2 = mFinerSurface[mSurfaces.to1dIndex(
            mSurfaces.toLevel(oldVal.xy(), 0, mLevel) + SurfaceDetectionBuffer::Vec2(0, 0),
            mLevel)];
        const unsigned int s3 = mFinerSurface[mSurfaces.to1dIndex(
            mSurfaces.toLevel(oldVal.xy(), 0, mLevel) + SurfaceDetectionBuffer::Vec2(1, 0),
            mLevel)];

        switch (index % 4) {
        case 0:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(0, 1), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL) | s0);
        case 1:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(1, 1), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL) | s1);
        case 2:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(0, 0), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL) | s2);
        case 3:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(1, 0), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL) | s3);
        }
      } else {
        switch (index % 4) {
        case 0:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(0, 1), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL));
        case 1:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(1, 1), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL));
        case 2:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(0, 0), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL));
        case 3:
          return mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(1, 0), quadSize,
              ((mLevel - 1) << BIT_CURRENT_LEVEL));
        }
      }
    } else {
      return index % 4 == 0
                 ? mSurfaces.emit(oldVal, SurfaceDetectionBuffer::Vec2(0, 0), 1,
                       (mLevel << BIT_CURRENT_LEVEL | (ALL_CONTINUITY_BITS & surfaceData) | 1))
                 : NO_VERTEX;
    }
  } else {
    return index % 4 == 0 ? oldVal : NO_VERTEX;
  }
  return NO_VERTEX;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GetVertCount::GetVertCount(SurfaceDetectionBuffer surfaces, int level,
    const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface)
    : mSurfaces(surfaces)
    , mLevel(level)
    , mOldVerts(oldVerts)
    , mCurrentSurface(currentSurface) {
}

__host__ __device__ int GetVertCount::operator()(size_t const index) const {
  SurfaceDetectionBuffer::Vertex oldVal      = mOldVerts[index >> 2];
  unsigned int                   vertexLevel = oldVal.data >> BIT_CURRENT_LEVEL;
  unsigned int                   surfaceData = mCurrentSurface[mSurfaces.to1dIndex(
      mSurfaces.toLevel(oldVal.xy(), 0, mLevel + 1), mLevel + 1)];

  if (vertexLevel == mLevel) {
    if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
      return 4;
    } else {
      return 1;
    }
  } else {
    return 1;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GenerateStencil::GenerateStencil(SurfaceDetectionBuffer surfaces, int level,
    const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface)
    : mSurfaces(surfaces)
    , mLevel(level)
    , mOldVerts(oldVerts)
    , mCurrentSurface(currentSurface) {
}

__host__ __device__ int GenerateStencil::operator()(size_t const index) const {
  SurfaceDetectionBuffer::Vertex oldVal      = mOldVerts[index >> 2];
  unsigned int                   vertexLevel = oldVal.data >> BIT_CURRENT_LEVEL;
  unsigned int                   surfaceData = mCurrentSurface[mSurfaces.to1dIndex(
      mSurfaces.toLevel(oldVal.xy(), 0, mLevel + 1), mLevel + 1)];

  if (vertexLevel == mLevel) {
    if ((surfaceData & (1 << BIT_IS_SURFACE)) == 0) {
      return 1;
    } else {
      return index % 4 == 0 ? 1 : 0;
    }
  } else {
    return index % 4 == 0 ? 1 : 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ bool IsNoVertex::operator()(SurfaceDetectionBuffer::Vertex vertex) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
