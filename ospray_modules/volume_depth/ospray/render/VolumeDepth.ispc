// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "math/random.ih"
#include "render/Renderer.ih"

#include "VolumeDepth.ih"
#include "volumes.ih"

unmasked void clippingIntersectionFilterV(const struct RTCFilterFunctionNArguments* uniform args) {
  // Set execution mask
  varying int* uniform valid = (varying int* uniform)args->valid;
  if (*valid == 0)
    return;

  // Get pointer to the intersection context
  uniform UserIntersectionContextForClipping* uniform context =
      (uniform UserIntersectionContextForClipping * uniform) args->context;

  // We skip this intersection to collect all remaining intersections
  *valid = 0;

  // Skip if array is full
  if (context->hitsCount >= CLIPPING_HITS_MAX_COUNT)
    return;

  // Get ray and hit
  varying Ray* uniform ray          = (varying Ray * uniform) args->ray;
  varying const RTCHit* uniform hit = (varying const RTCHit* uniform)args->hit;

  // Prepare clipping normal
  vec3f Nc;
  {
    // The postIntersect function needs the hit part of the Ray structure to be
    // initialized. Making a ray copy here is quite costly. That is why just
    // the necessary ray fields are set here.
    DifferentialGeometry dg;
    ray->Ng     = make_vec3f(hit->Ng_x, hit->Ng_y, hit->Ng_z);
    ray->u      = hit->u;
    ray->v      = hit->v;
    ray->primID = hit->primID;
    ray->geomID = hit->geomID;

    // We need to call postIntersect on clipping geometry
    foreach_unique(instID in hit->instID[0]) {
      // Call postIntersect to get shading normal
      Instance* uniform instance = *(context->world->clippersInst + instID);
      Instance_postIntersect(instance, NULL, dg, *ray, DG_NG | DG_NS, true);

      // Use geometry normal for clipping
      // but use shading normal to check if invertion is needed
      Nc = (dot(dg.Ns, dg.Ng) > 0.f) ? ray->Ng : neg(ray->Ng);
    }

    // Restore IDs initial values
    ray->primID = RTC_INVALID_GEOMETRY_ID;
    ray->geomID = RTC_INVALID_GEOMETRY_ID;
  }

  // Based on clipping normal vector decide if the ray is
  // entering clipping geometry (set positive hit value) or exitting clipping
  // geometry (set negative hit value)
  const bool  exitsClipping = (dot(ray->dir, Nc) > 0.f);
  ClippingHit cHit;
  cHit.t      = (exitsClipping) ? -ray->t : ray->t;
  cHit.primID = hit->primID;
  cHit.geomID = hit->geomID;
  cHit.instID = hit->instID[0];

  // Some geometry types (e.g. curves, subdivisions)
  // may give more than 2 intersections, we have to filter them out
  for (uint32 i = 0; i < context->hitsCount; i++) {
    if ((context->hits[i].primID == cHit.primID) && (context->hits[i].geomID == cHit.geomID) &&
        (context->hits[i].instID == cHit.instID) &&
        floatUlpCompare(context->hits[i].t, cHit.t, 512))
      return;
  }

  // Now we know that this intersection will be taken into account,
  // so we can update clipping depth correction value accordingly
  if (exitsClipping) {
    context->corrClippingDepth--;
  } else {
    context->corrClippingDepth++;
  }

  // Insert hit value into sorted array
  for (uint32 i = 0; i < context->hitsCount; i++) {
    if (abs(context->hits[i].t) > abs(cHit.t)) {
      const ClippingHit tmp = context->hits[i];
      context->hits[i]      = cHit;
      cHit                  = tmp;
    }
  }
  context->hits[context->hitsCount] = cHit;
  context->hitsCount++;
}

void VolumeDepth_renderSample(Renderer* uniform _self, FrameBuffer* uniform fb,
    World* uniform world, void* uniform perFrameData, varying ScreenSample& sample) {
  uniform VolumeDepth* uniform self = (uniform VolumeDepth * uniform) _self;

  LDSampler ldSamplerObj;
  varying LDSampler* uniform ldSampler = &ldSamplerObj;
  LDSampler_init(ldSampler, sample.sampleID.x + fb->size.x * sample.sampleID.y, sample.sampleID.z);

  const float originalRayTFar = sample.ray.t;

  // This is our main ray
  Ray& ray = sample.ray;

  // First trace the ray across clipping scene to calculate ray intervals,
  // this step should keep ray structure unchanged
  RayIntervals rayIntervals;
  traceClippingRay(world, ray, rayIntervals);

  // Iterate over all translucent geometry till we are fully opaque
  vec4f outputColor = make_vec4f(0.f);

  // Determine volume intervals by tracing ray in the volume scene
  Ray             volumeRay = ray;
  VolumeIntervals volumeIntervals;
  traceVolumeRay(world, volumeRay, volumeIntervals);

  float distance = inf;
  // Sample volumes across volume intervals (in front of geometry hit)
  if (volumeIntervals.numVolumeIntervals > 0) {
    vec4f volumeColor = integrateVolumeIntervals(volumeIntervals, rayIntervals, volumeRay,
        ldSampler, self->volumeSamplingRate, distance, self->depthMode);
    outputColor       = outputColor + (1.f - outputColor.w) * volumeColor;
  }
  if (self->depthMode == eIsosurface) {
    traceGeometryRayIntervals(world, ray, rayIntervals);
    if (hadHit(ray)) {
      distance = ray.t;
    }
  }

  sample.z     = distance;
  sample.rgb   = make_vec3f(outputColor);
  sample.alpha = outputColor.w;
}

// VolumeDepth C++ interface /////////////////////////////////////////////////////

export void* uniform VolumeDepth_create(void* uniform cppE) {
  uniform VolumeDepth* uniform self = uniform new uniform VolumeDepth;
  Renderer_Constructor(&self->super, cppE);
  self->super.renderSample = VolumeDepth_renderSample;
  return self;
}

export void VolumeDepth_set(
    void* uniform _self, uniform float volumeSamplingRate, uniform int depthMode) {
  uniform VolumeDepth* uniform self = (uniform VolumeDepth * uniform) _self;

  self->volumeSamplingRate = volumeSamplingRate;
  self->depthMode          = depthMode;
}
