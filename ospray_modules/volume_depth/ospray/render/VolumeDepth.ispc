// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "render/Renderer.ih"
// ao renderer
#include "VolumeDepth.ih"
#include "math/random.ih"
#include "math/sampling.ih"
#include "surfaces.ih"
#include "volumes.ih"

void VolumeDepth_renderSample(Renderer* uniform _self, FrameBuffer* uniform fb,
    World* uniform world, void* uniform perFrameData, varying ScreenSample& sample) {
  uniform VolumeDepth* uniform self = (uniform VolumeDepth * uniform) _self;

  LDSampler ldSamplerObj;
  varying LDSampler* uniform ldSampler = &ldSamplerObj;
  LDSampler_init(ldSampler, sample.sampleID.x + fb->size.x * sample.sampleID.y, sample.sampleID.z);

  uniform bool firstHit        = true;
  const float  originalRayTFar = sample.ray.t;

  // Allocate memory for volume intervals
  VolumeIntervals volumeIntervals;
  allocVolumeIntervals(volumeIntervals);

  // This is our main ray
  Ray& ray = sample.ray;

  // First trace the ray across clipping scene to calculate ray intervals,
  // this step should keep ray structure unchanged
  RayIntervals rayIntervals;
  traceClippingRay(world, ray, rayIntervals);

  // Iterate over all translucent geometry till we are fully opaque
  vec3f outColor        = make_vec3f(0.f);
  vec3f outTransmission = make_vec3f(1.f);
  float distance        = inf;
  while (true) {
    // Then trace normal geometry using calculated ray intervals,
    // if hit ray.t will be updated
    traceGeometryRayIntervals(world, ray, rayIntervals);

    // Determine volume intervals by tracing ray in the volume scene
    Ray volumeRay = ray;
    traceVolumeRay(world, volumeRay, volumeIntervals);

    // Sample volumes across volume intervals (in front of geometry hit)
    if (volumeIntervals.numVolumeIntervals > 0) {
      VolumeDepthRenderContext rc;
      rc.renderer       = self;
      rc.fb             = fb;
      rc.world          = world;
      rc.sample         = sample;
      rc.ldSampler      = ldSampler;
      vec4f volumeColor = integrateVolumeIntervalsGradient(rc, volumeIntervals, rayIntervals,
          volumeRay, ldSampler, self->volumeSamplingRate, true, distance);

      // Blend volume
      outColor        = outColor + outTransmission * make_vec3f(volumeColor);
      outTransmission = outTransmission * volumeColor.w;
    }

    // Add contribution from visible lights, P is used by light
    // evaluation
    DifferentialGeometry dg;
    dg.P     = ray.org;
    outColor = outColor + outTransmission * evaluateLights(world, dg, ray);

    // If any geometry has been hit
    const bool hadHit = hadHit(ray);
    if (hadHit) {
      // Prepare differential geometry structure
      postIntersect(world, &self->super, dg, ray,
          DG_NG | DG_NS | DG_NORMALIZE | DG_FACEFORWARD | DG_COLOR | DG_TEXCOORD);

      // Shade geometry
      SSI surfaceShading;
      surfaceShading = computeShading(self, fb, world, dg, sample, ldSampler, ray.dir, ray.time);

      // Initialize other per sample data with first hit values
      if (firstHit) {
        if (ray.t < distance) {
          distance = ray.t;
        }
        sample.albedo = surfaceShading.albedo;
        sample.normal = dg.Ns;
        firstHit      = false;
      }

      // Blend with output final color
      outColor        = outColor + outTransmission * surfaceShading.shadedColor;
      outTransmission = outTransmission * surfaceShading.transmission;

      // Early exit if remaining transmission is below min contribution
      // threshold
      if (luminance(outTransmission) < self->super.minContribution) {
        outTransmission = make_vec3f(0.f);
        break;
      }

      // Prepare ray for next loop iteration,
      // start from the last geometry hit all over to initial Tfar
      setRay(ray, ray.t + dg.epsilon, originalRayTFar);
    } else {
      // Blend background with output final color
      vec4f backgroundColor = Renderer_getBackground(&self->super, sample.pos);
      outColor              = outColor + outTransmission * make_vec3f(backgroundColor);
      outTransmission       = outTransmission * (1.f - backgroundColor.w);

      // Initialize other per sample data with first hit values
      if (firstHit) {
        sample.z      = ray.t;
        sample.albedo = make_vec3f(backgroundColor);
        sample.normal = sample.ray.dir;
      }

      // No more intersections
      break;
    }
  }

  freeVolumeIntervals(volumeIntervals);
  sample.z     = distance;
  sample.rgb   = outColor;
  sample.alpha = 1.f - luminance(outTransmission);
}

// SciVis C++ interface /////////////////////////////////////////////////////

export void* uniform VolumeDepth_create(void* uniform cppE) {
  uniform VolumeDepth* uniform self = uniform new uniform VolumeDepth;
  Renderer_Constructor(&self->super, cppE);
  self->super.renderSample = VolumeDepth_renderSample;
  return self;
}

export void VolumeDepth_set(void* uniform _self, uniform int depthMode, uniform bool shadowsEnabled,
    uniform int aoSamples, uniform float aoRadius, uniform float volumeSamplingRate,
    void* uniform scalarFilters, uniform int numScalarFilters) {
  uniform VolumeDepth* uniform self = (uniform VolumeDepth * uniform) _self;

  self->depthMode          = depthMode;
  self->shadowsEnabled     = shadowsEnabled;
  self->aoSamples          = aoSamples;
  self->aoRadius           = aoRadius;
  self->volumeSamplingRate = volumeSamplingRate;
  self->scalarFilters      = (ScalarFilter*)scalarFilters;
  self->numScalarFilters   = numScalarFilters;
}

vec3f VolumeDepth_computeAO(const uniform VolumeDepth* uniform self, const FrameBuffer* uniform fb,
    const World* uniform world, ScreenSample& sample, varying LDSampler* uniform ldSampler,
    const varying DifferentialGeometry& dg, const uniform int sampleCnt,
    const uniform float aoRadius, const varying vec3i& sampleID) {
  const uniform int accumID = reduce_max(sampleID.z) * sampleCnt;

  // init TEA RNG //
  RandomTEA rng_state;
  varying RandomTEA* const uniform rng = &rng_state;
  RandomTEA__Constructor(rng, sampleID.x, sampleID.y);
  const vec2f rot = RandomTEA__getFloats(rng);

  vec3f hits = make_vec3f(0.f);

  const linear3f localToWorld = frame(dg.Ns);

  for (uniform int i = 0; i < sampleCnt; i++) {
    // Use base-5 and base-7 Halton to avoid correlation with sample position
    const vec2f halton =
        make_vec2f(PermHalton_sample5(accumID + i), PermHalton_sample7(accumID + i));
    const vec2f r            = CranleyPattersonRotation(halton, rot);
    const vec3f local_ao_dir = cosineSampleHemisphere(r);
    const vec3f ao_dir       = localToWorld * local_ao_dir;

    if (dot(ao_dir, dg.Ns) < 0.01f) { // check below surface
      hits = hits + make_vec3f(1.f);
      continue;
    }

    Ray ao_ray;
    setRay(ao_ray, dg.P, ao_dir, dg.epsilon, aoRadius);

    hits = hits + (1.f - lightAlpha(self, ao_ray, fb, world, sample, ldSampler, make_vec3f(1.f),
                             dg.epsilon, 0.1f));
  }

  // the cosTheta of cosineSampleHemispherePDF and dot(shadingNormal, ao_dir)
  // cancel
  return 1.0f - (hits / (float)sampleCnt);
}
