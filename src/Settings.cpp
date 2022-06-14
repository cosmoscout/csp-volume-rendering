////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Settings.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

NLOHMANN_JSON_SERIALIZE_ENUM(VolumeFileType, {
                                                 {VolumeFileType::eInvalid, nullptr},
                                                 {VolumeFileType::eVtk, "vtk"},
                                                 {VolumeFileType::eNetCdf, "netcdf"},
                                             })

NLOHMANN_JSON_SERIALIZE_ENUM(
    VolumeStructure, {
                         {VolumeStructure::eInvalid, nullptr},
                         {VolumeStructure::eStructured, "structured"},
                         {VolumeStructure::eStructuredSpherical, "structuredSpherical"},
                         {VolumeStructure::eRectilinear, "rectilinear"},
                         {VolumeStructure::eImageSpherical, "imageSpherical"},
                         {VolumeStructure::eRectilinearSpherical, "rectilinearSpherical"},
                         {VolumeStructure::eUnstructured, "unstructured"},
                     })

NLOHMANN_JSON_SERIALIZE_ENUM(VolumeShape, {
                                              {VolumeShape::eInvalid, nullptr},
                                              {VolumeShape::eCubic, "cubic"},
                                              {VolumeShape::eSpherical, "spherical"},
                                          })

NLOHMANN_JSON_SERIALIZE_ENUM(DisplayMode, {
                                              {DisplayMode::ePoints, "points"},
                                              {DisplayMode::eMesh, "mesh"},
                                          })

NLOHMANN_JSON_SERIALIZE_ENUM(DepthMode, {
                                            {DepthMode::eNone, "none"},
                                            {DepthMode::eIsosurface, "isosurface"},
                                            {DepthMode::eFirstHit, "firstHit"},
                                            {DepthMode::eLastHit, "lastHit"},
                                            {DepthMode::eThreshold, "threshold"},
                                            {DepthMode::eMultiThreshold, "multiThreshold"},
                                        })

void from_json(nlohmann::json const& j, Settings::Data::Metadata::StructuredSpherical& o) {
  cs::core::Settings::deserialize(j, "axes", o.mAxes);
  cs::core::Settings::deserialize(j, "ranges", o.mRanges);
}

void to_json(nlohmann::json& j, Settings::Data::Metadata::StructuredSpherical const& o) {
  cs::core::Settings::serialize(j, "axes", o.mAxes);
  cs::core::Settings::serialize(j, "ranges", o.mRanges);
}

void from_json(nlohmann::json const& j, Settings::Data& o) {
  cs::core::Settings::deserialize(j, "path", o.mPath);
  cs::core::Settings::deserialize(j, "namePattern", o.mNamePattern);
  cs::core::Settings::deserialize(j, "type", o.mType);
  cs::core::Settings::deserialize(j, "structure", o.mStructure);
  cs::core::Settings::deserialize(j, "shape", o.mShape);
  cs::core::Settings::deserialize(j, "activeScalar", o.mActiveScalar);

  if (j.contains("metadata")) {
    switch (o.mStructure.get()) {
    case VolumeStructure::eStructuredSpherical:
      Settings::Data::Metadata::StructuredSpherical metadata;
      cs::core::Settings::deserialize(j, "metadata", metadata);
      o.mMetadata = {metadata};
      break;
    case VolumeStructure::eRectilinearSpherical:
      Settings::Data::Metadata::StructuredSpherical metadata1;
      cs::core::Settings::deserialize(j, "metadata", metadata1);
      o.mMetadata = {metadata1};
      break;
    default:
      // No metadata expected
      break;
    }
  }
};

void to_json(nlohmann::json& j, Settings::Data const& o) {
  cs::core::Settings::serialize(j, "path", o.mPath);
  cs::core::Settings::serialize(j, "namePattern", o.mNamePattern);
  cs::core::Settings::serialize(j, "type", o.mType);
  cs::core::Settings::serialize(j, "structure", o.mStructure);
  cs::core::Settings::serialize(j, "shape", o.mShape);
  cs::core::Settings::serialize(j, "activeScalar", o.mActiveScalar);

  if (o.mMetadata.has_value()) {
    switch (o.mStructure.get()) {
    case VolumeStructure::eStructuredSpherical:
      cs::core::Settings::serialize(j, "metadata", o.mMetadata->mStructuredSpherical);
      break;
    case VolumeStructure::eRectilinearSpherical:
      cs::core::Settings::serialize(j, "metadata", o.mMetadata->mStructuredSpherical);
      break;
    default:
      // No metadata to write
      break;
    }
  }
};

void from_json(nlohmann::json const& j, Settings::Rendering& o) {
  cs::core::Settings::deserialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::deserialize(j, "resolution", o.mResolution);
  cs::core::Settings::deserialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::deserialize(j, "aoSamples", o.mAOSamples);
  cs::core::Settings::deserialize(j, "maxPasses", o.mMaxPasses);
  cs::core::Settings::deserialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::deserialize(j, "useMaxDepth", o.mUseMaxDepth);
  cs::core::Settings::deserialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::deserialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::deserialize(j, "depthMode", o.mDepthMode);
  cs::core::Settings::deserialize(j, "transferFunction", o.mTransferFunction);
};

void to_json(nlohmann::json& j, Settings::Rendering const& o) {
  cs::core::Settings::serialize(j, "requestImages", o.mRequestImages);
  cs::core::Settings::serialize(j, "resolution", o.mResolution);
  cs::core::Settings::serialize(j, "samplingRate", o.mSamplingRate);
  cs::core::Settings::serialize(j, "aoSamples", o.mAOSamples);
  cs::core::Settings::serialize(j, "maxPasses", o.mMaxPasses);
  cs::core::Settings::serialize(j, "densityScale", o.mDensityScale);
  cs::core::Settings::serialize(j, "useMaxDepth", o.mUseMaxDepth);
  cs::core::Settings::serialize(j, "denoiseColor", o.mDenoiseColor);
  cs::core::Settings::serialize(j, "denoiseDepth", o.mDenoiseDepth);
  cs::core::Settings::serialize(j, "depthMode", o.mDepthMode);
  cs::core::Settings::serialize(j, "transferFunction", o.mTransferFunction);
};

void from_json(nlohmann::json const& j, Settings::Lighting& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::deserialize(j, "ambientStrength", o.mSunStrength);
}

void to_json(nlohmann::json& j, Settings::Lighting const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "sunStrength", o.mSunStrength);
  cs::core::Settings::serialize(j, "ambientStrength", o.mSunStrength);
}

void from_json(nlohmann::json const& j, Settings::Display& o) {
  cs::core::Settings::deserialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::deserialize(j, "reuseImages", o.mReuseImages);
  cs::core::Settings::deserialize(j, "useDepth", o.mDepthData);
  cs::core::Settings::deserialize(j, "drawDepth", o.mDrawDepth);
  cs::core::Settings::deserialize(j, "displayMode", o.mDisplayMode);
};

void to_json(nlohmann::json& j, Settings::Display const& o) {
  cs::core::Settings::serialize(j, "predictiveRendering", o.mPredictiveRendering);
  cs::core::Settings::serialize(j, "reuseImages", o.mReuseImages);
  cs::core::Settings::serialize(j, "useDepth", o.mDepthData);
  cs::core::Settings::serialize(j, "drawDepth", o.mDrawDepth);
  cs::core::Settings::serialize(j, "displayMode", o.mDisplayMode);
};

void from_json(nlohmann::json const& j, Settings::Transform& o) {
  cs::core::Settings::deserialize(j, "anchor", o.mAnchor);
  cs::core::Settings::deserialize(j, "position", o.mPosition);
  cs::core::Settings::deserialize(j, "scale", o.mScale);
  cs::core::Settings::deserialize(j, "rotation", o.mRotation);
};

void to_json(nlohmann::json& j, Settings::Transform const& o) {
  cs::core::Settings::serialize(j, "anchor", o.mAnchor);
  cs::core::Settings::serialize(j, "position", o.mPosition);
  cs::core::Settings::serialize(j, "scale", o.mScale);
  cs::core::Settings::serialize(j, "rotation", o.mRotation);
};

void from_json(nlohmann::json const& j, Settings::Core& o) {
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "scalar", o.mScalar);
  cs::core::Settings::deserialize(j, "radius", o.mRadius);
}

void to_json(nlohmann::json& j, Settings::Core const& o) {
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "scalar", o.mScalar);
  cs::core::Settings::serialize(j, "radius", o.mRadius);
}

void from_json(nlohmann::json const& j, Settings::Pathlines& o) {
  cs::core::Settings::deserialize(j, "path", o.mPath);
  cs::core::Settings::deserialize(j, "enabled", o.mEnabled);
  cs::core::Settings::deserialize(j, "size", o.mLineSize);
}

void to_json(nlohmann::json& j, Settings::Pathlines const& o) {
  cs::core::Settings::serialize(j, "path", o.mPath);
  cs::core::Settings::serialize(j, "enabled", o.mEnabled);
  cs::core::Settings::serialize(j, "size", o.mLineSize);
}

void from_json(nlohmann::json const& j, Settings& o) {
  cs::core::Settings::deserialize(j, "data", o.mData);
  if (j.contains("rendering")) {
    cs::core::Settings::deserialize(j, "rendering", o.mRendering);
  }
  if (j.contains("lighting")) {
    cs::core::Settings::deserialize(j, "lighting", o.mLighting);
  }
  if (j.contains("display")) {
    cs::core::Settings::deserialize(j, "display", o.mDisplay);
  }
  cs::core::Settings::deserialize(j, "transform", o.mTransform);
  cs::core::Settings::deserialize(j, "core", o.mCore);
  cs::core::Settings::deserialize(j, "pathlines", o.mPathlines);
}

void to_json(nlohmann::json& j, Settings const& o) {
  cs::core::Settings::serialize(j, "data", o.mData);
  cs::core::Settings::serialize(j, "rendering", o.mRendering);
  cs::core::Settings::serialize(j, "lighting", o.mLighting);
  cs::core::Settings::serialize(j, "display", o.mDisplay);
  cs::core::Settings::serialize(j, "transform", o.mTransform);
  cs::core::Settings::serialize(j, "core", o.mCore);
  cs::core::Settings::serialize(j, "pathlines", o.mPathlines);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering
