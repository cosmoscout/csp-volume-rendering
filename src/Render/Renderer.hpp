////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_RENDERER_HPP
#define CSP_VOLUME_RENDERING_RENDERER_HPP

#include "../Data/DataManager.hpp"
#include "../Enums.hpp"

#include "../../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/hash.hpp>

#include <future>
#include <string>

namespace csp::volumerendering {

class RendererException : public std::exception {
 public:
  const char* what() const noexcept override;
};

/// The abstract Renderer class provides an interface for controlling a volume renderer and
/// providing it with parameters.
class Renderer {
 public:
  /// A RenderedImage object contains all relevant information on a rendered image.
  class RenderedImage {
   public:
    virtual ~RenderedImage(){};

    /// Specifies, whether the other fields of this object contain valid information (true),
    /// or if there was an error resulting in invalid data (false).
    bool isValid() const;
    void setValid(bool value);
    /// Returns the resolution of the image.
    int getResolution() const;
    /// Returns the camera transform used by the renderer.
    glm::mat4 const& getCameraTransform() const;
    /// Returns the view matrix used by the renderer.
    glm::mat4 const& getModelView() const;
    /// Returns the projection matrix used by the renderer.
    glm::mat4 const& getProjection() const;

    /// Returns the color values of the image as RGBA values.
    virtual float* getColorData() = 0;
    /// Returns the depth values of the image as float values in the range [-1,1].
    virtual float* getDepthData() = 0;

    /// Assumes, that images from roughly the same camera perspective with the same resolution are
    /// identical.
    bool operator==(const RenderedImage& other) const;

   protected:
    RenderedImage(bool valid, int resolution, glm::mat4 cameraTransform, glm::mat4 modelView,
        glm::mat4 projection);
    RenderedImage(RenderedImage const& other) = default;
    RenderedImage& operator=(RenderedImage const& other) = default;

    RenderedImage(RenderedImage&& other) = default;
    RenderedImage& operator=(RenderedImage&& other) = default;

    bool      mValid = false;
    int       mResolution;
    glm::mat4 mCameraTransform;
    glm::mat4 mModelView;
    glm::mat4 mProjection;
  };

  class CopiedImage : public RenderedImage {
   public:
    CopiedImage() = delete;
    ~CopiedImage() override{};

    CopiedImage(RenderedImage& other);
    CopiedImage& operator=(RenderedImage& other);

    CopiedImage(CopiedImage const& other) = default;
    CopiedImage& operator=(CopiedImage const& other) = default;

    CopiedImage(CopiedImage&& other) = delete;
    CopiedImage& operator=(CopiedImage&& other) = delete;

    float* getColorData() override;
    float* getDepthData() override;

   private:
    std::vector<float> mColorData;
    std::vector<float> mDepthData;
  };

  /// Creates a Renderer for volumes of the given structure and shape.
  /// The data is provided by the given DataManager.
  Renderer(std::shared_ptr<DataManager> dataManager, VolumeStructure structure, VolumeShape shape);
  virtual ~Renderer() = default;

  /// Sets the desired resolution of the rendered images horizontally and vertically.
  void setResolution(int resolution);
  /// Sets the sampling rate used by the renderer. Higher sampling rates result in images with
  /// less noise.
  void setSamplingRate(float samplingRate);
  /// Sets a factor making the volume more or less dense.
  /// The higher the density, the more opaque the volume will appear.
  void setDensityScale(float densityScale);
  /// Sets the heuristic with which the depth image should be rendered.
  void setDepthMode(DepthMode depthMode);
  /// Sets the maximum number of render passes for constant rendering parameters.
  void setMaxRenderPasses(int maxRenderPasses);

  void setMaxLod(int maxLod);
  void clearMaxLod();

  /// Enables or disables denoising of the color image.
  void setDenoiseColor(bool denoiseColor);
  /// Enables or disables denoising of the depth image.
  void setDenoiseDepth(bool denoiseDepth);

  /// Sets the transfer function to be used in rendering.
  /// The transferFunction parameter should contain a vector of RGBA color values.
  /// The colors are evenly spaced over the domain of the function.
  void setTransferFunction(std::vector<glm::vec4> transferFunction);
  /// Sets filters to restrict the shown volume based on different scalars.
  /// For each scalar a minimum and maximum value can be given. Only parts of the volume,
  /// for which all scalars lie in the given ranges, will be visible.
  void setScalarFilters(std::vector<ScalarFilter> filters);

  /// Enables or disables shading of the volume.
  void setShading(bool shading);
  /// Sets the strength of the ambient light that should be used in shading.
  void setAmbientLight(float strength);
  /// Sets the direction from the volume towards the sun.
  void setSunDirection(glm::vec3 sunDirection);
  /// Sets the strength of the light of the sun.
  void setSunStrength(float strength);

  void setCoreEnabled(bool value);
  void setCoreScalar(std::string value);
  void setCoreRadius(float value);

  void setPathlinesEnabled(bool enable);
  void setPathlineOpacity(float value);
  void setPathlineSize(float value);
  void setPathlineScalarFilters(std::vector<ScalarFilter> const& value);
  void setPathlineLength(float value);
  void setPathlineActiveScalar(std::string const& value);

  /// Starts asynchronously rendering an image of the volume for the given camera perspective.
  /// The rendering process will use all parameters set before calling this method
  /// and will not be influenced by any later changes to the parameters.
  /// Returns a future that will eventually contain the rendered image.
  std::future<std::unique_ptr<Renderer::RenderedImage>> getFrame(glm::mat4 const& cameraTransform);
  /// Returns the current progress of the rendering processon the range [0,1].
  /// Returns 1 if no image is currently being rendered.
  virtual float getProgress() = 0;
  /// Requests the renderer to start preparing data matching the given state for rendering.
  virtual void preloadData(DataManager::State state) = 0;
  /// Requests to cancel the current rendering process.
  virtual void cancelRendering() = 0;

 protected:
  struct Parameters {
    struct World {
      DepthMode mDepthMode;

      struct Lights {
        bool      mShading;
        float     mAmbientStrength;
        glm::vec3 mSunDirection;
        float     mSunStrength;

        bool operator==(const Lights& other) const {
          // If shading is deactivated, other parameters can be ignored
          return mShading == other.mShading &&
                 (!mShading || (mAmbientStrength == other.mAmbientStrength &&
                                   mSunDirection == other.mSunDirection &&
                                   mSunStrength == other.mSunStrength));
        }
      } mLights;

      struct Volume {
        std::vector<glm::vec4> mTransferFunction;
        float                  mDensityScale = 0.f;

        bool operator==(const Volume& other) const {
          return mDensityScale == other.mDensityScale &&
                 mTransferFunction == other.mTransferFunction;
        }
      } mVolume;

      struct Core {
        bool        mEnable = false;
        std::string mScalar;
        float       mRadius = 0.0f;

        bool operator==(const Core& other) const {
          return mEnable == other.mEnable && mScalar == other.mScalar && mRadius == other.mRadius;
        }
      } mCore;

      struct Pathlines {
        bool                      mEnable = false;
        float                     mLineOpacity = 0.f;
        float                     mLineSize = 0.f;
        std::vector<ScalarFilter> mScalarFilters;
        std::string               mActiveScalar;

        bool operator==(const Pathlines& other) const {
          return mEnable == other.mEnable && mLineOpacity == other.mLineOpacity &&
                 mLineSize == other.mLineSize && mScalarFilters == other.mScalarFilters &&
                 mActiveScalar == other.mActiveScalar;
        }
      } mPathlines;

      struct PathlinesTexture {
        float mLength = 0.f;

        bool operator==(const PathlinesTexture& other) const {
          return mLength == other.mLength;
        }
      } mPathlinesTexture;

      bool operator==(const World& other) const {
        return mDepthMode == other.mDepthMode && mLights == other.mLights &&
               mVolume == other.mVolume && mCore == other.mCore && mPathlines == other.mPathlines &&
               mPathlinesTexture == other.mPathlinesTexture;
      }
    } mWorld;

    int mMaxRenderPasses;

    std::optional<int> mMaxLod;

    int   mResolution;
    float mSamplingRate;

    bool mDenoiseColor;
    bool mDenoiseDepth;

    std::vector<ScalarFilter> mScalarFilters;

    bool operator==(const Parameters& other) const {
      return mMaxRenderPasses == other.mMaxRenderPasses && mMaxLod == other.mMaxLod &&
             mResolution == other.mResolution && mSamplingRate == other.mSamplingRate &&
             mDenoiseColor == other.mDenoiseColor && mDenoiseDepth == other.mDenoiseDepth &&
             mScalarFilters == other.mScalarFilters && mWorld == other.mWorld;
    }
  };

  std::shared_ptr<DataManager> mDataManager;

  const VolumeStructure mStructure;
  const VolumeShape     mShape;

  virtual std::unique_ptr<RenderedImage> getFrameImpl(glm::mat4 const& cameraTransform,
      Parameters parameters, DataManager::State const& dataState) = 0;

 private:
  std::mutex mParameterMutex;
  Parameters mParameters;
};

} // namespace csp::volumerendering

namespace std {

template <>
struct hash<csp::volumerendering::Renderer::RenderedImage> {
  std::size_t operator()(csp::volumerendering::Renderer::RenderedImage const& image) {
    size_t hash = 0u;
    hash ^= std::hash<int>{}(image.getResolution());
    hash ^= std::hash<glm::mat4>{}(image.getCameraTransform());
    return hash;
  }
};

} // namespace std

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
