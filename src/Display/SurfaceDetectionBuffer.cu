////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SurfaceDetectionBuffer.hpp"

#include "../logger.hpp"
#include "Functors.hpp"

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

SurfaceDetectionBuffer::SurfaceDetectionBuffer(
    float* depthTexture, int width, int height, int cellSize)
    : mGridParams(cellSize, width, height) {
  thrust::device_vector<float> dDepth(depthTexture, depthTexture + mGridParams.levelSize(0));
  const float*                 pDepth = thrust::raw_pointer_cast(dDepth.data());

  for (unsigned int i = 0; i < mGridParams.mLevels; ++i) {
    thrust::device_vector<uint16_t> dCurrentSurface(mGridParams.levelSize(i + 1));
    if (i == 0) {
      thrust::tabulate(thrust::device, dCurrentSurface.begin(), dCurrentSurface.end(),
          DetectSurfaceInBase(mGridParams, pDepth));
    } else {
      const uint16_t* pLastSurface = thrust::raw_pointer_cast(mBuffers.back().data());
      thrust::tabulate(thrust::device, dCurrentSurface.begin(), dCurrentSurface.end(),
          DetectSurfaceInHigherLevel(mGridParams, pLastSurface, i));
    }
    mBuffers.push_back(dCurrentSurface);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SurfaceDetectionBuffer::print() const {
  for (unsigned int level = 0; level < mGridParams.mLevels; level++) {
    thrust::host_vector<uint16_t> output = mBuffers[level];
    for (int i = 0; i < mGridParams.levelDim(level + 1).y; i++) {
      thrust::host_vector<uint16_t> line(output.begin() + mGridParams.levelDim(level + 1).y * i,
          output.begin() + mGridParams.levelDim(level + 1).y * (i + 1));
      logger().trace("{}", line);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

thrust::device_vector<SurfaceDetectionBuffer::Vertex> SurfaceDetectionBuffer::generateVertices() {
  thrust::device_vector<Vertex> dVertices(mGridParams.levelSize(mGridParams.mLastLevel + 1));
  thrust::tabulate(
      thrust::device, dVertices.begin(), dVertices.end(), GenerateHighLevelVerts(mGridParams));

  for (int level = mGridParams.mLastLevel; level >= 0; --level) {
    const uint16_t* pCurrentSurface = thrust::raw_pointer_cast(mBuffers[level].data());
    const uint16_t* pFinerSurface   = thrust::raw_pointer_cast(mBuffers[level - 1].data());

    const Vertex*                 pOldVertices = thrust::raw_pointer_cast(dVertices.data());
    thrust::device_vector<Vertex> dNewVertices(dVertices.size() * 4);
    thrust::tabulate(thrust::device, dNewVertices.begin(), dNewVertices.end(),
        SplitVerts(mGridParams, level, pOldVertices, pCurrentSurface, pFinerSurface));

    thrust::device_vector<int> dStencil(dNewVertices.size());
    thrust::tabulate(thrust::device, dStencil.begin(), dStencil.end(),
        GenerateStencil(mGridParams, level, pOldVertices, pCurrentSurface));

    dVertices.reserve(dNewVertices.size());
    auto iNewEnd = thrust::copy_if(thrust::device, dNewVertices.begin(), dNewVertices.end(),
        dStencil.begin(), dVertices.begin(), thrust::identity<int>());
    dVertices.assign(dVertices.begin(), iNewEnd);
  }

  return dVertices;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ int SurfaceDetectionBuffer::GridParams::to1dIndex(
    Vec2 index, Level level) const {
#ifdef __CUDA_ARCH__
  return min(max(index.y, 0), levelDim(level).y - 1) * levelDim(level).x +
         min(max(index.x, 0), levelDim(level).x - 1);
#else
  return std::clamp(index.y, 0, levelDim(level).y - 1) * levelDim(level).x +
         std::clamp(index.x, 0, levelDim(level).x - 1);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ SurfaceDetectionBuffer::Vec2 SurfaceDetectionBuffer::GridParams::to2dIndex(
    size_t index, Level level) const {
  int width = levelDim(level).x;
  int x     = static_cast<int>(index) % width;
  int y     = static_cast<int>(index) / width;
  return {x, y};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ SurfaceDetectionBuffer::Vec2 SurfaceDetectionBuffer::GridParams::toLevel(
    Vec2 index, Level from, Level to) const {
  if (to > from) {
    Level dist = to - from;
    Vec2  newIndex(index.x >> dist, index.y >> dist);
    return newIndex;
  } else {
    Level dist = from - to;
    Vec2  newIndex(index.x << dist, index.y << dist);
    return newIndex;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Offset is applied after transformation to new level
__host__ __device__ int SurfaceDetectionBuffer::GridParams::toLevel(
    size_t index, Level from, Level to, Vec2 offset) const {
  return to1dIndex(toLevel(to2dIndex(index, from), from, to) + offset, to);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ int SurfaceDetectionBuffer::GridParams::levelSize(Level level) const {
  return (mWidth >> level) * (mHeight >> level);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ SurfaceDetectionBuffer::Vec2 SurfaceDetectionBuffer::GridParams::levelDim(
    Level level) const {
  Vec2 dim(mWidth >> level, mHeight >> level);
  return dim;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__host__ __device__ SurfaceDetectionBuffer::Vertex SurfaceDetectionBuffer::GridParams::emit(
    Vertex pos, Vec2 offset, unsigned int factor, unsigned int data) const {
  Vertex out;
  out.data = data;
#ifdef __CUDA_ARCH__
  out.x = min(pos.x + offset.x * factor, mWidth);
  out.y = min(pos.y + offset.y * factor, mHeight);
#else
  out.x = std::min(pos.x + offset.x * factor, mWidth);
  out.y = std::min(pos.y + offset.y * factor, mHeight);
#endif
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
