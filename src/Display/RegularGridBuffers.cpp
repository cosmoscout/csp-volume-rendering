////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RegularGridBuffers.hpp"

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

RegularGridBuffers::RegularGridBuffers(uint32_t resolution)
    : mResolution(resolution) {
  std::vector<float>    vertices(mResolution * mResolution * 3);
  std::vector<unsigned> indices((mResolution - 1) * (2 + 2 * mResolution));

  for (uint32_t x = 0; x < mResolution; ++x) {
    for (uint32_t y = 0; y < mResolution; ++y) {
      vertices[(x * mResolution + y) * 3 + 0] = 2.f / (mResolution - 1) * x - 1.f;
      vertices[(x * mResolution + y) * 3 + 1] = 2.f / (mResolution - 1) * y - 1.f;
      vertices[(x * mResolution + y) * 3 + 2] = 0.f;
    }
  }

  uint32_t index = 0;

  for (uint32_t x = 0; x < mResolution - 1; ++x) {
    indices[index++] = x * mResolution;
    for (uint32_t y = 0; y < mResolution; ++y) {
      indices[index++] = x * mResolution + y;
      indices[index++] = (x + 1) * mResolution + y;
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RegularGridBuffers::Draw() {
  mVAO.Bind();
  glDrawElements(
      GL_TRIANGLE_STRIP, (mResolution - 1) * (2 + 2 * mResolution), GL_UNSIGNED_INT, nullptr);
  mVAO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
