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
  DetectSurfaceInBase(SurfaceDetectionBuffer surfaces, const float* depth);
  __host__ __device__ uint16_t operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const mSurfaces;
  const float*                 mDepth;
};

struct DetectSurfaceInHigherLevel {
 public:
  DetectSurfaceInHigherLevel(
      SurfaceDetectionBuffer surfaces, const uint16_t* surfaceBuffer, int level);
  __host__ __device__ uint16_t operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const mSurfaces;
  const uint16_t*              mSurfaceBuffer;
  int                          mLevel;
};

struct GenerateHighLevelVerts {
 public:
  GenerateHighLevelVerts(SurfaceDetectionBuffer surfaces);
  __host__ __device__ SurfaceDetectionBuffer::Vertex operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const mSurfaces;
};

struct SplitVerts {
 public:
  SplitVerts(SurfaceDetectionBuffer surfaces, int level,
      const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface,
      const uint16_t* finerSurface);
  __host__ __device__ SurfaceDetectionBuffer::Vertex operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const          mSurfaces;
  int                                   mLevel;
  const SurfaceDetectionBuffer::Vertex* mOldVerts;
  const uint16_t*                       mCurrentSurface;
  const uint16_t*                       mFinerSurface;
};

struct GetVertCount {
 public:
  GetVertCount(SurfaceDetectionBuffer surfaces, int level,
      const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface);
  __host__ __device__ int operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const          mSurfaces;
  int                                   mLevel;
  const SurfaceDetectionBuffer::Vertex* mOldVerts;
  const uint16_t*                       mCurrentSurface;
};

struct GenerateStencil {
 public:
  GenerateStencil(SurfaceDetectionBuffer surfaces, int level,
      const SurfaceDetectionBuffer::Vertex* oldVerts, const uint16_t* currentSurface);
  __host__ __device__ int operator()(int const index) const;

 private:
  SurfaceDetectionBuffer const          mSurfaces;
  int                                   mLevel;
  const SurfaceDetectionBuffer::Vertex* mOldVerts;
  const uint16_t*                       mCurrentSurface;
};

struct IsNoVertex {
  __host__ __device__ bool operator()(SurfaceDetectionBuffer::Vertex vertex);
};

} // namespace csp::volumerendering

#endif
