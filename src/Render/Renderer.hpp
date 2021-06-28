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
  struct RenderedImage {
    /// Contains the color values of the image as RGBA values.
    std::vector<uint8_t> mColorData;
    /// Contains the depth values of the image as float values in the range [-1,1].
    std::vector<float> mDepthData;
    /// Contains the model-view-projection matrix used by the renderer.
    glm::mat4 mMVP;
    /// Specifies, whether the other fields of this object contain valid information (true),
    /// or if there was an error resulting in invalid data (false).
    bool mValid;
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
  std::future<Renderer::RenderedImage> getFrame(glm::mat4 cameraTransform);
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
        float                  mDensityScale;

        bool operator==(const Volume& other) const {
          return mDensityScale == other.mDensityScale &&
                 mTransferFunction == other.mTransferFunction;
        }
      } mVolume;

      struct Pathlines {
        bool                      mEnable;
        float                     mLineOpacity;
        float                     mLineSize;
        std::vector<ScalarFilter> mScalarFilters;
        std::string               mActiveScalar;

        bool operator==(const Pathlines& other) const {
          return mEnable == other.mEnable && mLineOpacity == other.mLineOpacity &&
                 mLineSize == other.mLineSize && mScalarFilters == other.mScalarFilters &&
                 mActiveScalar == other.mActiveScalar;
        }
      } mPathlines;

      struct PathlinesTexture {
        float mLength;

        bool operator==(const PathlinesTexture& other) const {
          return mLength == other.mLength;
        }
      } mPathlinesTexture;

      bool operator==(const World& other) const {
        return mDepthMode == other.mDepthMode && mLights == other.mLights &&
               mVolume == other.mVolume && mPathlines == other.mPathlines &&
               mPathlinesTexture == other.mPathlinesTexture;
      }
    } mWorld;

    int mMaxRenderPasses;

    int   mResolution;
    float mSamplingRate;

    bool mDenoiseColor;
    bool mDenoiseDepth;

    std::vector<ScalarFilter> mScalarFilters;

    bool operator==(const Parameters& other) const {
      return mMaxRenderPasses == other.mMaxRenderPasses && mResolution == other.mResolution &&
             mSamplingRate == other.mSamplingRate && mDenoiseColor == other.mDenoiseColor &&
             mDenoiseDepth == other.mDenoiseDepth && mScalarFilters == other.mScalarFilters &&
             mWorld == other.mWorld;
    }
  };

  std::shared_ptr<DataManager> mDataManager;

  const VolumeStructure mStructure;
  const VolumeShape     mShape;

  virtual RenderedImage getFrameImpl(
      glm::mat4 cameraTransform, Parameters parameters, DataManager::State dataState) = 0;

 private:
  std::mutex mParameterMutex;
  Parameters mParameters;
};

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_RENDERER_HPP
