# Volume rendering for CosmoScout VR

A CosmoScout VR plugin which allows rendering of volumetric datasets.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
    ...
    "plugins": {
        ...
        "csp-volume-rendering": {
            "volumeDataPath": <path to directory containing simulation data>,
            "volumeDataPattern": <regex pattern that matches simulation file names>,
            "volumeDataType": <"vtk"/"gaia">,
            "volumeStructure": <"structured"/"unstructured">,
            "volumeShape": <"cubic"/"spherical">,
            "anchor": <Anchor name, e.g. "Earth">
        }
    }
}
```

This configuration only contains the mandatory settings.
All available settings are described in the following sections.
Mandatory settings are shown **bold**, while optional settings are shown in *italics*.

### Data settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| **volumeDataPath** | string | - | Path to the directory that contains the volumetric data files. |
| **volumeDataPattern** | string | - | Regex pattern that matches the filename of all relevant data files. The index of the simulation step that produced the file has to be marked using a capturing group. If files are named "Sim_01.vtk", "Sim_02.vtk" etc., `"Sim_([0-9]+).vtk"` would be a suitable regex. |
| **volumeDataType** | `"vtk"` / `"gaia"` | - | Data format of the specified files. Currently supports VTK data and data produced by the GAIA simulation code. |
| **volumeStructure** | `"structured"` / `"unstructured"` | - | Structure of the volumetric data. Currently supports structured regular grids and unstructured grids. |
| **volumeShape** | `"cubic"` / `"spherical"` | - | Shape of the volume. By default, spherical volumes are rendered with the same size as the planet they are bound to. Cubic volumes are rendered, so that their corners touch the planets surface. |

### Rendering settings

These settings can be dynamically changed in the CosmoScout UI.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| *requestImages* | bool | `true` | When false, no new images of the volumetric data will be rendered. |
| *resolution* | int | `256` | Horizontal and vertical resolution of the rendered images in pixels. |
| *samplingRate* | float | `0.05` | Sampling rate to be used while rendering. Higher values result in higher quality images with less noise. |
| *sunStrength* | float | `1` | Factor for the strength of the sunlight. Only used, when shading is enabled in CosmoScout. |
| *densityScale* | float | `1` | Sets the density of the volume. |
| *denoiseColor* | bool | `true` | Use the OIDN library to denoise the color image of the volume before displaying it in CosmoScout. |
| *denoiseDepth* | bool | `true` | Use the OIDN library to denoise the image containing depth information of the volume before using it for image based rendering. |
| *depthMode* | `"none"` / `"isosurface"` / `"firstHit"` / `"lastHit"` / `"threshold"` / `"multiThreshold"` | `"none"` | Heuristic for determining per pixel depth values for the rendered images. |

### Display settings

These settings can be dynamically changed in the CosmoScout UI.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| *predictiveRendering* | bool | `false` | When true, images will be rendered for a predicted observer perspective instead of the current one. This may result in smaller viewing angles, which improves the quality of the image based rendering. |
| *reuseImages* | bool | `false` | When true, previously rendered images will be cached and may be displayed again, if the viewing angle is suitable. |
| *useDepth* | bool | `true` | When false, depth information on the volume is ignored and not used in the image based rendering. |
| *drawDepth* | bool | `false` | When true, a grayscale image of the depth information will be displayed instead of the color image of the volume. |
| *displayMode* | `"mesh"` / `"points"` | `"mesh"` | Geometry used for displaying rendered images. `"mesh"` uses a continuous triangle mesh for displaying the image, while `"points"` displays the pixels of the image as points of varying size. |

### Transform settings

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| **anchor** | string | - | Name of the SPICE frame, in which the volume should be placed. |
| *position* | double[3] | `[0,0,0]` | Offset from the center of the frame in meters. |
| *scale* | double | `1` | Factor by which the volume should be scaled. |
| *rotation* | double[4] | `[0,0,0,1]` | Rotation of the volume as a quaternion. |