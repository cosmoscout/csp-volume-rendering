////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SETTINGS_HPP
#define CSP_VOLUME_RENDERING_SETTINGS_HPP

#include "Enums.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <optional>
#include <string>

namespace csp::volumerendering {

struct Settings {
  struct Data {
    cs::utils::Property<std::string>        mPath;
    cs::utils::Property<std::string>        mNamePattern;
    cs::utils::Property<VolumeFileType>     mType;
    cs::utils::Property<VolumeStructure>    mStructure;
    cs::utils::Property<VolumeShape>        mShape;
    cs::utils::DefaultProperty<std::string> mActiveScalar{""};
    cs::utils::DefaultProperty<bool>        mUseTimeBar{false};

    union Metadata {
      struct StructuredSpherical {
        template <typename T>
        struct Axes {
          T mRad;
          T mLat;
          T mLon;
        };
        Axes<std::array<double, 2>> mRanges;
        Axes<int>                   mAxes;
      } mStructuredSpherical;
    };
    std::optional<Metadata> mMetadata;
  } mData;

  struct Rendering {
    cs::utils::DefaultProperty<bool>        mRequestImages{true};
    cs::utils::DefaultProperty<int>         mResolution{256};
    cs::utils::DefaultProperty<float>       mSamplingRate{0.05f};
    cs::utils::DefaultProperty<int>         mAOSamples{4};
    cs::utils::DefaultProperty<int>         mMaxPasses{1};
    cs::utils::DefaultProperty<int>         mLayers{5};
    cs::utils::DefaultProperty<float>       mDensityScale{10.f};
    cs::utils::DefaultProperty<bool>        mUseMaxDepth{false};
    cs::utils::DefaultProperty<bool>        mDenoiseColor{true};
    cs::utils::DefaultProperty<bool>        mDenoiseDepth{true};
    cs::utils::DefaultProperty<DepthMode>   mDepthMode{DepthMode::eNone};
    cs::utils::DefaultProperty<std::string> mTransferFunction{"BlackBody.json"};
  } mRendering;

  struct Lighting {
    cs::utils::DefaultProperty<bool>  mEnabled{false};
    cs::utils::DefaultProperty<float> mSunStrength{1.f};
    cs::utils::DefaultProperty<float> mAmbientStrength{.5f};
  } mLighting;

  struct Display {
    cs::utils::DefaultProperty<bool>        mPredictiveRendering{false};
    cs::utils::DefaultProperty<bool>        mReuseImages{false};
    cs::utils::DefaultProperty<bool>        mDepthData{true};
    cs::utils::DefaultProperty<bool>        mDrawDepth{false};
    cs::utils::DefaultProperty<int>         mHoleFilling{0};
    cs::utils::DefaultProperty<DisplayMode> mDisplayMode{DisplayMode::eMultilayer};
  } mDisplay;

  struct Transform {
    cs::utils::Property<std::string>       mAnchor;
    cs::utils::DefaultProperty<glm::dvec3> mPosition{glm::dvec3(0, 0, 0)};
    cs::utils::DefaultProperty<double>     mScale{1.};
    cs::utils::DefaultProperty<glm::dvec3> mRotation{glm::dvec3(0, 0, 0)};
  } mTransform;

  struct Core {
    cs::utils::DefaultProperty<bool>        mEnabled{true};
    cs::utils::DefaultProperty<std::string> mScalar{""};
    cs::utils::Property<float>              mRadius;

    // Needed for deserialization.
    Core(){};
  };
  std::optional<Core> mCore;

  struct Pathlines {
    cs::utils::Property<std::string>  mPath;
    cs::utils::DefaultProperty<bool>  mEnabled{true};
    cs::utils::DefaultProperty<float> mLineSize{1.f};

    // Needed for deserialization.
    Pathlines(){};
  };
  std::optional<Pathlines> mPathlines;
};

template <typename T>
void from_json(nlohmann::json const& j, Settings::Data::Metadata::StructuredSpherical::Axes<T>& o) {
  cs::core::Settings::deserialize(j, "radial", o.mRad);
  cs::core::Settings::deserialize(j, "lon", o.mLon);
  cs::core::Settings::deserialize(j, "lat", o.mLat);
}
template <typename T>
void to_json(nlohmann::json& j, Settings::Data::Metadata::StructuredSpherical::Axes<T> const& o) {
  cs::core::Settings::serialize(j, "radial", o.mRad);
  cs::core::Settings::serialize(j, "lon", o.mLon);
  cs::core::Settings::serialize(j, "lat", o.mLat);
}
void from_json(nlohmann::json const& j, Settings::Data::Metadata::StructuredSpherical& o);
void to_json(nlohmann::json& j, Settings::Data::Metadata::StructuredSpherical const& o);
void from_json(nlohmann::json const& j, Settings::Data& o);
void to_json(nlohmann::json& j, Settings::Data const& o);
void from_json(nlohmann::json const& j, Settings::Rendering& o);
void to_json(nlohmann::json& j, Settings::Rendering const& o);
void from_json(nlohmann::json const& j, Settings::Lighting& o);
void to_json(nlohmann::json& j, Settings::Lighting const& o);
void from_json(nlohmann::json const& j, Settings::Display& o);
void to_json(nlohmann::json& j, Settings::Display const& o);
void from_json(nlohmann::json const& j, Settings::Transform& o);
void to_json(nlohmann::json& j, Settings::Transform const& o);
void from_json(nlohmann::json const& j, Settings::Core& o);
void to_json(nlohmann::json& j, Settings::Core const& o);
void from_json(nlohmann::json const& j, Settings::Pathlines& o);
void to_json(nlohmann::json& j, Settings::Pathlines const& o);
void from_json(nlohmann::json const& j, Settings& o);
void to_json(nlohmann::json& j, Settings const& o);

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_SETTINGS_HPP
