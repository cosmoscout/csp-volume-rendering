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

uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform mat4 uMatTransform;
uniform mat4 uMatRendererMVP;
uniform bool uUseDepth;

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

void main()
{
   vDepth = (iPos.z + 1) / 2;

   vTexCoords  = vec2((iPos.x + 1) / 2, (iPos.y + 1) / 2);

   vec4 objSpacePos = vec4(iPos, 1);
   if (!uUseDepth) {
     objSpacePos.z = -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
   }

   objSpacePos = inverse(uMatRendererMVP) * objSpacePos;
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

uniform vec3 uRadii;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;
uniform mat4 uMatTransform;
uniform mat4 uMatRendererMVP;
uniform bool uUseDepth;
uniform int uBasePointSize;
uniform float uBaseDepth;

// inputs
layout(location = 0) in vec3 iPos;

// outputs
out vec2 vTexCoords;
out vec3 vPosition;
out float vDepth;

void main()
{
    vDepth = (iPos.z + 1) / 2;

    vTexCoords  = vec2((iPos.x + 1) / 2, (iPos.y + 1) / 2);

    vec4 objSpacePos = vec4(iPos, 1);
    if (!uUseDepth) {
      objSpacePos.z = -uMatRendererMVP[3][2] / uMatRendererMVP[2][2];
    }

    objSpacePos = inverse(uMatRendererMVP) * objSpacePos;
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

} // namespace csp::volumerendering

#endif // CSP_VOLUME_RENDERING_SHADERS_HPP
