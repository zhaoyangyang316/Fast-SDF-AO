# Sponza architecture

This mesh depicts the Atrium of the Sponza Palace in Dubrovnik. Frank Meinl remodeled the scene at Crytek in 2010, inspired by Marko Dabrović’s 2002 model. Morgan McGuire converted and corrected the OBJ and texture distribution used here. The McGuire graphics data archive identifies this release as © 2010 Frank Meinl, Crytek, licensed under **Creative Commons Attribution 3.0**. [Source archive](https://casual-effects.com/g3d/data10/common/model/crytek_sponza/sponza.zip) · [Release metadata and attribution](https://casual-effects.com/g3d/data10/common/model/crytek_sponza/info.js) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) · [Original copyright notice](SPONZA-COPYRIGHT.txt)

Fast-SDF-AO’s adaptation uses meters and a Z-up coordinate system. The ground-floor center is at the origin and the long nave runs along X. The conversion triangulates the OBJ, samples its diffuse textures into linear vertex colors, subdivides broad masonry and floor surfaces, and reduces tessellation on dense cloth and architectural details. The opaque renderer omits the opacity-masked foliage and chain cards. Surface roughness values are assigned for the renderer. The static mesh and its derived distance field carry the source attribution.

The OBJ’s `floor` material covers two levels. There are 11,248 upward-facing triangles at ground level (`z = 0`) and 7,744 on the upper walkway (`z ≈ 4.18 m`). Both use the original vertex colors and a signed roughness of −5/127. This encodes a smooth conductor whose reflections are tinted by the floor color. Turning off reflections leaves only the direct specular lighting on this material.

The WebGPU preset passes `planarReflectionZ: 0` to `loadScene` for the courtyard mirror. A camera reflected across this plane rasterizes the static architecture. Each output pixel stores the nearest static surface’s radiance and distance from the reflected camera. This distance limits the reflection ray’s search for a closer hit on the dragon’s BVH. The upper walkway follows the existing smooth-conductor reflection path at its own height.

The previous ground-floor reflection traced Sponza’s 12.5 cm distance field for static hits. If a hit matched a surface visible in the main camera, its color came from the G-buffer. Other hits used the coarse radiance volume, and misses returned sky lighting. This produced gaps and streaks along columns and banners even when the reflected dragon silhouette was accurate.

The planar mirror gives those static surfaces triangle silhouettes and vertex colors at output resolution. It shows one reflection bounce. For lighting, it reuses AO, GI and shadow values where the reflected surface matches a surface in the main view. For offscreen surfaces it estimates visibility from the static distance field and indirect light from the radiance volume. Shadows and indirect lighting on those surfaces therefore have coarser detail than their outlines.

To regenerate the compact mesh, download the source ZIP above (SHA-256 `da005cbee0be2df2abc8513f3ceb61bcb6f69aac112babcd9c00169a27c2770c`) and run this from the repository root with Python, NumPy and Pillow installed.

```sh
python3 tools/gen_sponza_mesh.py /path/to/sponza.zip
```

The conversion produces `sponza-mesh.bin.gz` with 162,515 vertices and 243,645 triangles. Its decoder is shared with the dragon mesh and exposed by `sponza-geometry.js`.

## Static distance field

`sponza.f16.gz` stores distances to the static architecture on a 304 × 192 × 136 grid of 0.125 m cubes. It uses FP16 values clipped to ±1.5 m and occupies 2.12 MiB compressed. The courtyard floor at z = 0 passes through a layer of voxel centers. The animated dragon is traced separately against its own triangles.

To rebuild the field from the distributed mesh, run the following command with Node.js and a C++17 compiler. It calls the repository’s native baker and regenerates `sponza-sdf.js` with the world-space volume bounds. The checked-in mesh takes about six seconds on four CPU cores once the baker has compiled.

```sh
node tools/bake_sponza_sdf.mjs
```

The native baker obtains distance magnitudes from the triangles and assigns signs by counting oriented surface crossings along vertical lines. Open architectural surfaces can leave a nonzero crossing count above the roof. A flood starting outside the mesh sets known exterior voxels positive. It crosses an edge between two voxel centers when their nearest-surface distances sum to more than the edge length plus an FP16 rounding allowance. A triangle intersecting the edge would make that sum at most the edge length. Enclosed solids keep their native signs. Thin sheets and gaps narrower than one voxel can still be blurred or missed by trilinear sampling.

Each bake validates 128 random voxels against the native brute-force reference, then checks the corrected field over the entire top face, at 2,405 points along the open nave, and at six signed depths through the central floor. Run the scene-specific checks on the shipped field with the command below.

```sh
node tools/bake_sponza_sdf.mjs --verify
```
