////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2023 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_FUNCTORS_HPP
#define CSP_VOLUME_RENDERING_FUNCTORS_HPP

#include "SurfaceDetectionBuffer.hpp"

#include <thrust/device_vector.h>

namespace csp::volumerendering {

struct DetectSurfaceInBase {
 public:
  DetectSurfaceInBase(SurfaceDetectionBuffer::GridParams params, const float* depth);
  __host__ __device__ uint16_t operator()(size_t const index) const;

 private:
  SurfaceDetectionBuffer::GridParams const mParams;
  const float*                             mDepth;
};

struct DetectSurfaceInHigherLevel {
 public:
  DetectSurfaceInHigherLevel(
      SurfaceDetectionBuffer::GridParams params, const uint16_t* surfaceBuffer, int level);
  __host__ __device__ uint16_t operator()(size_t const index) const;

 private:
  SurfaceDetectionBuffer::GridParams const mParams;
  const uint16_t*                          mSurfaceBuffer;
  int                                      mLevel;
};

struct GenerateHighLevelVerts {
 public:
  GenerateHighLevelVerts(SurfaceDetectionBuffer::GridParams params);
  __host__ __device__ SurfaceDetectionBuffer::Vertex operator()(size_t const index) const;

 private:
  SurfaceDetectionBuffer::GridParams const mParams;
};

struct SplitVerts {
 public:
  SplitVerts(SurfaceDetectionBuffer::GridParams params, int level,
      const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface,
      const uint16_t* finerSurface);
  __host__ __device__ SurfaceDetectionBuffer::Vertex operator()(size_t const index) const;

 private:
  SurfaceDetectionBuffer::GridParams const mParams;
  int                                      mLevel;
  const SurfaceDetectionBuffer::Vertex*    mOldVerts;
  const uint16_t*                          mCurrentSurface;
  const uint16_t*                          mFinerSurface;
};

struct GenerateStencil {
 public:
  GenerateStencil(SurfaceDetectionBuffer::GridParams params, int level,
      const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface);
  __host__ __device__ int operator()(size_t const index) const;

 private:
  SurfaceDetectionBuffer::GridParams const mParams;
  int                                      mLevel;
  const SurfaceDetectionBuffer::Vertex*    mOldVerts;
  const uint16_t*                          mCurrentSurface;
};

} // namespace csp::volumerendering

#endif
