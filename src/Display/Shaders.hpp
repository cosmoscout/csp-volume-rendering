////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SHADERS_HPP
#define CSP_VOLUME_RENDERING_SHADERS_HPP

namespace csp::volumerendering {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string HOLE_FILLING_FRAG = R"(
#version 330
uniform int uCurrentLevel;
uniform sampler2D uHoleFillingTexture;
uniform sampler2D uColorBuffer;
uniform sampler2D uDepthBuffer;

in vec2 vTexCoords;

layout(pixel_center_integer) in vec4 gl_FragCoord;

// write output
layout(location=0) out vec4 oColor;

void main() {
  vec4 samples[4*4];
  oColor = vec4(0);

  if (uCurrentLevel == 0) {
    ivec2 max_res = textureSize(sampler2D(uColorBuffer), 0);
    for (int x=0; x<4; ++x) {
      for (int y=0; y<4; ++y) {
        ivec2 pos = clamp(ivec2(gl_FragCoord.xy*2) + ivec2(x-4/2+1, y-4/2+1), ivec2(0), max_res-1);
        samples[x+y*4].rgb = texelFetch(uColorBuffer, pos, 0).rgb;
        samples[x+y*4].a = texelFetch(uDepthBuffer, pos, 0).r;
      }
    }
  } else {
    ivec2 max_res = textureSize(sampler2D(uHoleFillingTexture), uCurrentLevel-1);
    for (int x=0; x<4; ++x) {
      for (int y=0; y<4; ++y) {
        ivec2 pos = clamp(ivec2(gl_FragCoord.xy*2) + ivec2(x-4/2+1, y-4/2+1), ivec2(0), max_res-1);
        samples[x+y*4] = texelFetch(uHoleFillingTexture, pos, uCurrentLevel-1);
      }
    }
  }

  // count number of hole pixels
  int hole_count = 0;

  for (int i=0; i<4*4; ++i) {
    if (samples[i].a == 1.0) ++hole_count;
  }

  // calculate average depth of none hole pixels
  if (hole_count < 4*4) {
    float average_depth = 0;
    for (int i=0; i<4*4; ++i) {
      average_depth += samples[i].a;
    }

    average_depth = (average_depth-hole_count) / (4*4-hole_count);

    float max_depth = 1;
    float weight = 0;
    float weights[16] = float[16](0.4, 0.9, 0.9, 0.4,
                         0.9, 1.8, 1.8, 0.9,
                         0.9, 1.8, 1.8, 0.9,
                         0.4, 0.9, 0.9, 0.4);
    for (int i=0; i<4*4; ++i) {
      // calculate average color of all none hole pixels with a depth larger or equal to average
      if (samples[i].a != 1.0 && samples[i].a > average_depth-0.000001) {
        max_depth = min(max_depth, samples[i].a);
        oColor.rgb += samples[i].rgb * weights[i];
        weight += weights[i];
      }
    }

    // return color and average depth
    if (weight > 0) {
      oColor /= weight;
    }

    oColor.a = max_depth;
  } else {
    oColor = vec4(0, 0, 0, 0);
  }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string FULLSCREEN_QUAD_VERT = R"(
#version 330

void main() {
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string FULLSCREEN_QUAD_GEOM = R"(
#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out vec2 vTexCoords;

void main()
{
  gl_Position = vec4( 1.0, 1.0, 0.5, 1.0 );
  vTexCoords = vec2( 1.0, 1.0 );
  EmitVertex();

  gl_Position = vec4(-1.0, 1.0, 0.5, 1.0 );
  vTexCoords = vec2( 0.0, 1.0 );
  EmitVertex();

  gl_Position = vec4( 1.0,-1.0, 0.5, 1.0 );
  vTexCoords = vec2( 1.0, 0.0 );
  EmitVertex();

  gl_Position = vec4(-1.0,-1.0, 0.5, 1.0 );
  vTexCoords = vec2( 0.0, 0.0 );
  EmitVertex();

  EndPrimitive();
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string FULLSCREEN_QUAD_FRAG = R"(
#version 330

in vec2 vTexCoords;

out vec4 oColor;

uniform sampler2D uTexColor;
uniform sampler2D uTexDepth;
uniform sampler2D uTexHoleFilling;

vec2 get_epipolar_direction() {
return normalize(vec2(1,1));
  vec4 epipol = /*warp_matrix * */vec4(0, 0, -1, 0);
  vec2 epi_dir = vec2(0);

  if (abs(epipol.w) < 0.01) {
    epipol.xy = 100*epipol.xy*0.5 + 0.5;
    epi_dir = epipol.xy - vTexCoords;
  } else if (epipol.w < 0) {
    epipol /= epipol.w;
    epipol.xy = epipol.xy*0.5 + 0.5;
    epi_dir = epipol.xy - vTexCoords;
  } else {
    epipol /= epipol.w;
    epipol.xy = epipol.xy*0.5 + 0.5;
    epi_dir = vTexCoords - epipol.xy;
  }

  return normalize(epi_dir);
}

vec4 hole_filling_blur() {
  const float step_size = 0.2;
  const float max_level = 7;
  vec2  epi_dir = get_epipolar_direction();
  vec2  dirs[2] = vec2[2](
    vec2( epi_dir.x,  epi_dir.y),
    vec2(-epi_dir.x, -epi_dir.y)
  );

  float depth = 0.1;
  float level = 6;

  /*for (int i=0; i<dirs.length(); ++i) {
    for (float l=0; l<=max_level; l+=step_size) {
      vec2  p = vTexCoords - pow(2,l)*dirs[i]/uResolution;
      float d = texelFetch(sampler2D(uDepthTexture), ivec2(p*uResolution), 0).x;

      if (d < 1.0) {
        if (d > depth+0.0001 || (abs(d-depth)<0.0001 && l<level)) {
          level = l;
          depth = d;
        }
        break;
      }
    }
  }*/

  if (depth == 0) {
    return vec4(0.6, 0., 0.6, 1);
  }

  return vec4(textureLod(uTexHoleFilling, vTexCoords, level+1).rgb, 1);
}

void main() {
  oColor = texture(uTexColor, vTexCoords);
    oColor = hole_filling_blur();

  if (oColor.a == 0) {
    oColor = hole_filling_blur();
  }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string PASS_VERT = R"(
#version 330

layout(location=0) in uvec3 position;

flat out uvec3 iPos;

void main() {
  iPos = position;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string IRREGULAR_GRID_GEOM = R"(
#version 330

#define GAP 0

#define BIT_IS_SURFACE   0

#define BIT_CONTINUOUS_T    4
#define BIT_CONTINUOUS_R    5
#define BIT_CONTINUOUS_B    6
#define BIT_CONTINUOUS_L    7

#define BIT_CONTINUOUS_TR   8
#define BIT_CONTINUOUS_TL   9
#define BIT_CONTINUOUS_BR   10
#define BIT_CONTINUOUS_BL   11

#define BIT_CURRENT_LEVEL   12 // 12-15 (requires 3 bits)

#define ALL_CONTINUITY_BITS 4080
#define ALL_DATA_BITS       4095

uniform sampler2D uDepthTexture;
uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform mat4 uMatTransform;
uniform mat4 uMatRendererProjection;
uniform mat4 uMatRendererProjectionInv;
uniform mat4 uMatRendererMVP;
uniform mat4 uMatRendererMVPInv;
uniform bool uUseDepth;
uniform bool uInside;
uniform uvec2 uResolution;

layout(points) in;
layout(triangle_strip, max_vertices = 16) out;

// inputs
flat in uvec3 iPos[];

// outputs
flat out uint vCellsize;
out vec2 vCellcoords;

out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

float get_depth_raw(vec2 pos) {
  float cameraDistance = texelFetch(sampler2D(uDepthTexture), ivec2(pos), 0).x;

  if (isinf(cameraDistance)) {
    if (-uMatRendererMVP[3][2] / uMatRendererMVP[2][2] < 0.7) {
      return uInside ? 1 : 0;
    } else {
      return 1;
    }
  }

  vec4 normalizedPos = vec4(pos / uResolution, 0, 1);
  normalizedPos = uMatRendererProjectionInv * normalizedPos;
  normalizedPos /= normalizedPos.w;
  normalizedPos = vec4(normalize(normalizedPos.xyz) * cameraDistance, 1);
  normalizedPos = uMatRendererProjection * normalizedPos;
  normalizedPos /= normalizedPos.w;
  return normalizedPos.z;
}

float get_min_depth(vec2 frag_pos) {
  float depth0 = get_depth_raw(frag_pos + vec2(-0.5, -0.5));
  float depth1 = get_depth_raw(frag_pos + vec2( 0.5, -0.5));
  float depth2 = get_depth_raw(frag_pos + vec2(-0.5,  0.5));
  float depth3 = get_depth_raw(frag_pos + vec2( 0.5,  0.5));

  return min(depth0, min(depth1, min(depth2, depth3)));
}

void emit_grid_vertex(vec2 frag_pos, float depth) {
  vDepth = (depth + 1) / 2;

  vec4 pos = vec4((frag_pos / uResolution) * 2 - vec2(1, 1), 0, 1);
  if (uUseDepth) {
    pos.z = depth;
  } else {
    pos.z = 0;
  }

  pos = uMatRendererMVPInv * pos;

  vPosition    = pos.xyz / pos.w;
  vPosition    = uRadii * vPosition;
  vPosition    = (uMatTransform * vec4(vPosition, 1.0)).xyz;
  vPosition    = (uMatModelView * vec4(vPosition, 1.0)).xyz;
  gl_Position  = uMatProjection * vec4(vPosition, 1);

  if (gl_Position.w > 0) {
    gl_Position /= gl_Position.w;
    if (gl_Position.z >= 1) {
      gl_Position.z = 0.999999;
    }
  }
  EmitVertex();
}

void emit_quad(uvec2 offset, uvec2 size) {
  if (size.x > 0u && size.y > 0u) {
    float depth1, depth2, depth3, depth4;

    vec2 pos1 = vec2(iPos[0].xy)                        + vec2(offset);
    vec2 pos2 = vec2(iPos[0].xy) + vec2(size.x, 0)      + vec2(offset);
    vec2 pos3 = vec2(iPos[0].xy) + vec2(0,      size.y) + vec2(offset);
    vec2 pos4 = vec2(iPos[0].xy) + vec2(size.x, size.y) + vec2(offset);

    int cont_l = int(iPos[0].z >> BIT_CONTINUOUS_L) & 1;
    int cont_r = int(iPos[0].z >> BIT_CONTINUOUS_R) & 1;
    int cont_t = int(iPos[0].z >> BIT_CONTINUOUS_T) & 1;
    int cont_b = int(iPos[0].z >> BIT_CONTINUOUS_B) & 1;

    int cont_tl = int(iPos[0].z >> BIT_CONTINUOUS_TL) & 1;
    int cont_tr = int(iPos[0].z >> BIT_CONTINUOUS_TR) & 1;
    int cont_bl = int(iPos[0].z >> BIT_CONTINUOUS_BL) & 1;
    int cont_br = int(iPos[0].z >> BIT_CONTINUOUS_BR) & 1;

    depth1 = get_depth_raw(vec2(-cont_l, -cont_b)*cont_bl*0.5 + pos1+vec2( 0.5,  0.5));
    depth2 = get_depth_raw(vec2( cont_r, -cont_b)*cont_br*0.5 + pos2+vec2(-0.5,  0.5));
    depth3 = get_depth_raw(vec2(-cont_l,  cont_t)*cont_tl*0.5 + pos3+vec2( 0.5, -0.5));
    depth4 = get_depth_raw(vec2( cont_r,  cont_t)*cont_tr*0.5 + pos4+vec2(-0.5, -0.5));

    vCellsize = min(size.x, size.y);

    vTexCoords = pos1 / uResolution;
    vCellcoords = vec2(0, 0);
    emit_grid_vertex(pos1 + vec2(-GAP, -GAP), depth1);

    vTexCoords = pos2 / uResolution;
    vCellcoords = vec2(1, 0);
    emit_grid_vertex(pos2 + vec2( GAP, -GAP), depth2);

    vTexCoords = pos3 / uResolution;
    vCellcoords = vec2(0, 1);
    emit_grid_vertex(pos3 + vec2(-GAP,  GAP), depth3);

    vTexCoords = pos4 / uResolution;
    vCellcoords = vec2(1, 1);
    emit_grid_vertex(pos4 + vec2( GAP,  GAP), depth4);

    EndPrimitive();
  }
}

void emit_pixel(uvec2 offset) {
  vCellsize = 1u;
  vec2 position = iPos[0].xy + offset;

  // remove strange one-pixel line
  if (position.y == uResolution.y) return;

  vTexCoords = position / uResolution;
  float depth = get_depth_raw(position);

  vCellcoords = vec2(0, 0);
  emit_grid_vertex(position + vec2(0, 0) + vec2(-GAP, -GAP), depth);
  vCellcoords = vec2(1, 0);
  emit_grid_vertex(position + vec2(1, 0) + vec2( GAP, -GAP), depth);
  vCellcoords = vec2(1, 1);
  emit_grid_vertex(position + vec2(0, 1) + vec2(-GAP,  GAP), depth);
  vCellcoords = vec2(0, 1);
  emit_grid_vertex(position + vec2(1, 1) + vec2( GAP,  GAP), depth);

  EndPrimitive();
}

void main()
{
  if ((iPos[0].z & 1u) > 0u) {
    emit_quad(uvec2(0), uvec2(1 << ((iPos[0].z >> BIT_CURRENT_LEVEL) + 1u)));
  } else {
    emit_pixel(uvec2(0, 0));
    /*emit_pixel(uvec2(1, 0));
    emit_pixel(uvec2(1, 1));
    emit_pixel(uvec2(0, 1));*/
  }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string IRREGULAR_GRID_FRAG = R"(
#version 330

uniform sampler2D uTexture;
uniform sampler2D uDepthTexture;
uniform float uFarClip;
uniform bool uDrawDepth;

// inputs
flat in uint vCellsize;
in vec2 vCellcoords;

in vec2 vTexCoords;
in vec3 vPosition;
in float vDepth;

// outputs
layout(location = 0) out vec4 oColor;

vec3 heat(float v) {
  float value = 1.0-v;
  return (0.5+0.5*smoothstep(0.0, 0.1, value))*vec3(
    smoothstep(0.5, 0.3, value),
    value < 0.3 ? smoothstep(0.0, 0.3, value) : smoothstep(1.0, 0.6, value),
    smoothstep(0.4, 0.6, value)
  );
}

void main()
{
    oColor = texture(uTexture, vTexCoords);
    if(oColor.a <= 0)
    {
      discard;
    }
    if (uDrawDepth) {
      oColor = vec4(vDepth, vDepth, vDepth, 1);

      float intensity = log2(float(vCellsize)) / 7.0;
      oColor.rgb = heat(1-intensity);

      if (any(lessThan(vCellcoords, vec2(0.6/float(vCellsize)))) || any(greaterThan(vCellcoords, vec2(1.0-0.6/float(vCellsize))))) {
        oColor.rgb = mix(oColor.rgb, vec3(0), 0.7);
      }
    }

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string BILLBOARD_VERT = R"(
#version 330

uniform sampler2D uDepthTexture;
uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform mat4 uMatTransform;
uniform mat4 uMatRendererProjection;
uniform mat4 uMatRendererProjectionInv;
uniform mat4 uMatRendererMVP;
uniform mat4 uMatRendererMVPInv;
uniform bool uUseDepth;
uniform bool uInside;

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

float normalizeDepth(float cameraDistance, vec4 pos) {
    if (isinf(cameraDistance)) {
      if (-uMatRendererMVP[3][2] / uMatRendererMVP[2][2] < 0.7) {
        return uInside ? 1 : 0;
      } else {
        return 1;
      }
    }
    pos = uMatRendererProjectionInv * pos;
    pos /= pos.w;
    pos = vec4(normalize(pos.xyz) * cameraDistance, 1);
    pos = uMatRendererProjection * pos;
    pos /= pos.w;
    return pos.z;
}

void main()
{
    vec3 pos = iPos;
    vDepth = (normalizeDepth(texture(uDepthTexture, (pos.xy + vec2(1)) / 2.f).r, vec4(pos, 1)) + 1) / 2;
    if (uUseDepth) {
      pos.z = vDepth * 2 - 1;
    } else {
      pos.z = 0;
    }

    vTexCoords  = vec2((pos.x + 1) / 2, (pos.y + 1) / 2);

    vec4 objSpacePos = vec4(pos, 1);
    /*if (!uUseDepth) {
      objSpacePos.z = -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
    }*/

    objSpacePos = uMatRendererMVPInv * objSpacePos;

    vPosition   = objSpacePos.xyz / objSpacePos.w;
    vPosition   = uRadii * vPosition;
    vPosition   = (uMatTransform * vec4(vPosition, 1.0)).xyz;
    vPosition   = (uMatModelView * vec4(vPosition, 1.0)).xyz;
    gl_Position = uMatProjection * vec4(vPosition, 1);

    if (gl_Position.w > 0) {
      gl_Position /= gl_Position.w;
      if (gl_Position.z >= 1) {
        gl_Position.z = 0.999999;
      }
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string BILLBOARD_FRAG = R"(
#version 330

uniform sampler2D uTexture;
uniform sampler2D uDepthTexture;
uniform float uFarClip;
uniform bool uDrawDepth;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;
in float vDepth;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    oColor = texture(uTexture, vTexCoords);
    if(oColor.a <= 0 || vDepth < 0)
    {
      discard;
    }
    if (uDrawDepth) {
      oColor = vec4(vDepth, vDepth, vDepth, 1);
    }

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string VIS_FRAG = R"(
#version 330

uniform float uFarClip;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;
in float vDepth;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    if(vDepth < 0)
    {
      discard;
    }

    oColor = vec4(0, 0, 1, 1);
    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string POINTS_FORWARD_VERT = R"(
#version 330

uniform sampler2D uDepthTexture;
uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform mat4 uMatTransform;
uniform mat4 uMatRendererProjection;
uniform mat4 uMatRendererProjectionInv;
uniform mat4 uMatRendererMVP;
uniform mat4 uMatRendererMVPInv;
uniform bool uUseDepth;
uniform int uBasePointSize;
uniform float uBaseDepth;
uniform bool uInside;

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

float normalizeDepth(float cameraDistance, vec4 pos) {
    if (isinf(cameraDistance)) {
      return uInside ? 1 : 0;
    }
    pos = uMatRendererProjectionInv * pos;
    pos /= pos.w;
    pos = vec4(normalize(pos.xyz) * cameraDistance, 1);
    pos = uMatRendererProjection * pos;
    pos /= pos.w;
    return pos.z;
}

void main()
{
    vec3 pos = iPos;
    vDepth = (normalizeDepth(texture(uDepthTexture, (pos.xy + vec2(1)) / 2.f).r, vec4(pos, 1)) + 1) / 2;
    if (uUseDepth) {
      pos.z = vDepth * 2 - 1;
    } else {
      pos.z = 0;
    }

    vTexCoords  = vec2((pos.x + 1) / 2, (pos.y + 1) / 2);

    vec4 objSpacePos = vec4(pos, 1);
    /*if (!uUseDepth) {
      objSpacePos.z = -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
    }*/

    objSpacePos = uMatRendererMVPInv * objSpacePos;
    vPosition   = objSpacePos.xyz / objSpacePos.w;
    vPosition   = uRadii * vPosition;
    vPosition   = (uMatTransform * vec4(vPosition, 1.0)).xyz;
    vPosition   = (uMatModelView * vec4(vPosition, 1.0)).xyz;
    gl_PointSize = uBasePointSize * uBaseDepth / vPosition.z;
    gl_Position = uMatProjection * vec4(vPosition, 1);

    if (gl_Position.w > 0) {
      gl_Position /= gl_Position.w;
      if (gl_Position.z >= 1) {
        gl_Position.z = 0.999999;
      }
    }
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string POINTS_FORWARD_FRAG = R"(
#version 330

uniform sampler2D uTexture;
uniform float uFarClip;
uniform bool uDrawDepth;

// inputs
in vec2 vTexCoords;
in vec3 vPosition;
in float vDepth;

// outputs
layout(location = 0) out vec4 oColor;

void main()
{
    oColor = texture(uTexture, vTexCoords);
    if(oColor.a <= 0)
    {
      discard;
    }
    if (uDrawDepth) {
      oColor = vec4(vDepth, vDepth, vDepth, 1);
    }

    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string GET_DEPTH_COMP = R"(
#version 430

layout(local_size_x = 1, local_size_y = 1) in;

layout(rgba32f, binding = 0) readonly uniform image2D uInDepth;
layout(r32f, binding = 1) writeonly uniform image2D uOutDepth;

uniform vec2 uBottomCorner;
uniform vec2 uTopCorner;
uniform float uRadius;
uniform float uNear;
uniform float uFar;

void main() {
    vec2 pos = uBottomCorner + (vec2(gl_GlobalInvocationID.xy) + vec2(0.5f)) /
        vec2(gl_NumWorkGroups * gl_WorkGroupSize) * (uTopCorner - uBottomCorner);
    ivec2 pix = ivec2(gl_GlobalInvocationID);

    //vec4 val = texture(uInDepth, pos);
    vec4 val = imageLoad(uInDepth, ivec2(pos * imageSize(uInDepth)));
    val *= uFar / uRadius;

    imageStore(uOutDepth, pix, vec4(100000000000000.0f));
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_SHADERS_HPP
