////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_VOLUME_RENDERING_SHADERS_HPP
#define CSP_VOLUME_RENDERING_SHADERS_HPP

namespace csp::volumerendering {

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

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

float normalizeDepth(float cameraDistance, vec4 pos) {
    if (isinf(cameraDistance)) {
      return -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
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

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

float normalizeDepth(float cameraDistance, vec4 pos) {
    if (isinf(cameraDistance)) {
      return -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
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

layout(rgba32f, binding = 0) readonly uniform sampler2DRect uInDepth;
layout(r32f, binding = 1) writeonly uniform image2D uOutDepth;

uniform vec2 uBottomCorner;
uniform vec2 uTopCorner;
uniform float uNear;
uniform float uFar;

void main() {
    vec2 pos = uBottomCorner + (vec2(gl_GlobalInvocationID.xy) + vec2(0.5f)) /
        vec2(gl_NumWorkGroups * gl_WorkGroupSize) * (uTopCorner - uBottomCorner);
    ivec2 pix = ivec2(gl_GlobalInvocationID);

    vec4 val = texture(uInDepth, pos * textureSize(uInDepth));
    val *= uFar * 0.92;

    imageStore(uOutDepth, pix, val);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_SHADERS_HPP
