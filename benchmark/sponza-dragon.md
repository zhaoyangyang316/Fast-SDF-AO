# A moving dragon reflected in the Sponza courtyard

![A red dragon above a polished floor, with its wings and tail visible in the reflection](sponza-dragon-preview.png)

The WebGPU preset frames the gliding dragon and its reflection from a camera 1.5 m above the nave floor, looking down toward the courtyard center. Sponza contributes 243,645 static triangles and the rigid dragon adds 43,440, for 287,085 in the scene. The dragon keeps its modeled wing pose. Rigid-mesh rays intersect its local-space BVH. The architecture uses a static contact hierarchy and baked distance field. [Sponza attribution](../assets/sponza-dragon/SPONZA.md) · [Red Dragon attribution](../assets/sponza-dragon/DRAGON-LICENSE.md)

## Geometry and motion

The architecture's contact hierarchy contains 464,024 nodes across a 38 × 24 × 17 grid. Its GPU buffers occupy 60.1 MiB in total. The dragon's traversal buffers occupy another 6.5 MiB. The static FP16 field uses 304 × 192 × 136 voxels at 12.5 cm per voxel and occupies 15.1 MiB uncompressed.

At 33 reference points in free space and on surfaces, the static field's distance magnitude differed from the nearest static triangle by 5.4 mm on average and 27.8 mm at worst. The baker also checks that the entire sky face and 2,405 samples in the nave are in free space. [Field generation and verification](../assets/sponza-dragon/SPONZA.md)

Sampling the flight for 10 seconds at 30 Hz covered a complete loop. At all 301 sampled poses, the dragon's lowest vertex was 0.36–0.60 m above the floor. Even the triangle bounding boxes stayed more than 0.10 m from the non-floor architecture.

## Floor reflections

The 18,992 triangles using Sponza’s floor material have a signed roughness of −5/127. The negative sign marks a conductor and the low magnitude selects the renderer’s full-resolution mirror trace. The original vertex colors tint the polished surface. A reflected ray that hits the dragon intersects its transformed triangles. For the static architecture, the renderer traces the 12.5 cm distance field and shades the hit from the G-buffer where visible, with a coarse radiance volume as a fallback. Narrow columns and banner edges can appear broken or smeared in the reflection. [Mirror tracing shader](../src/webgpu.js)

At 704 × 396, we compared the reflections-only view with the dragon present and hidden. We reflected the dragon’s triangles across the floor plane and projected them into the camera at 0 and 2 seconds, accounting for the shader’s ray-origin offset. The predicted silhouette and the measured red reflection had 94.2% and 93.9% intersection over union. At 2 seconds, turning **Reflections** off gave zero reflection-channel output for all 6,733 tested pixels inside the silhouette. [Measurements](sponza-dragon-results.json)

## Lighting and flicker

![The dragon and its floor reflection moving together over two seconds of flight at 15 frames per second](sponza-dragon-flight.webp)

Chromium rendered 61 poses across two seconds at 320 × 180 on High with SHaRC enabled. The dragon itself covered 1.7% of the visible surface samples initially and 2.6% at the endpoint. At that endpoint, 149 of 7,159 matched floor samples lost more than 10 percentage points of direct sun visibility when the dragon was present. Ambient visibility fell by more than 5 points at 102 floor samples.

| Display RGB comparison | Mean absolute difference | 95th percentile |
|---|---:|---:|
| Distant fixed surfaces between consecutive frames, dragon parked | 0.40% | 0.78% |
| Same surfaces between consecutive frames, dragon moving | 0.56% | 0.78% |
| Moving endpoint versus a separately accumulated frame at the same pose, all visible surfaces | 0.41% | 1.07% |
| Same comparison on surfaces recently uncovered by the dragon | 0.86% | 2.98% |

The background comparison spans 16 consecutive frame pairs. During motion, 0.62% of those pixel comparisons changed by more than 5% of the RGB range, compared with 0.0002% when parked. Movement in the reflection can also change these static-surface pixels. The same-pose reference averages frames 41–48 from the same renderer. Comparing that reference with the previous eight-frame average gives 0.08% mean drift.

After hiding the dragon, the pixels in its former silhouette differed from the settled empty view by 0.77% on the first frame. On surfaces that had received its shadow, the difference was 0.27% on frame one, 0.26% on frame three and 0.27% on frame eight.

## Frame time

Headless Chromium 156 used the SwiftShader software adapter at 320 × 180 on High. AO, GI, shadows and reflections were active. SHaRC was disabled so every measured frame traced 28,800 primary AO/GI rays. Each case had three 12-frame blocks with rotated order, six warm-up frames per block and one frame in flight. Timings include the pose update, submission and wait for rendering to finish. Asset loading and the initial BVH build occurred before sampling.

| Courtyard workload | Median completed frame | 95th percentile |
|---|---:|---:|
| Empty | 518.1 ms | 558.5 ms |
| Dragon parked | 1,069.0 ms | 1,520.8 ms |
| Dragon moving | 1,043.2 ms | 1,265.0 ms |

The parked dragon took 2.06× the empty-courtyard median frame time. The moving dragon took 0.98× the parked median, with block medians ranging from 985.8 ms to 1,077.2 ms. These software timings include pose-dependent changes to visibility and shading. Hardware GPU timings remain unmeasured.

The dragon BVH was built once. Each of the 36 measured moving frames refitted the instance hierarchy and uploaded 80 bytes of pose data and 32 bytes of bounds. The renderer recorded zero geometry uploads or acceleration-structure builds during those frames. The 95th percentile for JavaScript preparation of the rigid scene was 0.1 ms. The median WebGPU timestamp span from the G-buffer pass through final compositing was 1,040.8 ms.

The machine-readable results include individual frame timings, image comparisons, mirror checks and the clearance sweep. [Measurements](sponza-dragon-results.json)

## Reproduce

Serve the checkout with `python3 -m http.server 8000`, open [the preset](http://localhost:8000/webgpu.html?scene=sponza-dragon), and use **Pause dragon** to hold a pose. Toggle **Reflections** to inspect the floor with the dragon still in view, or toggle **Show dragon** for the empty courtyard. For the automated probes, install Python Playwright and Chromium. The clearance check also uses Node.js and NumPy.

```sh
python3 benchmark/sponza-dragon-clearance.py
python3 test/webgpu-browser.py benchmark/sponza-dragon-quality.mjs --scene sponza-dragon --width 320 --height 180
python3 test/webgpu-browser.py benchmark/sponza-dragon-perf.mjs --scene sponza-dragon --width 320 --height 180
```

Add `--software --browser /usr/bin/chromium` to the browser probes to reproduce the software setup. For hardware timings, run the performance probe on an idle WebGPU adapter at your target resolution.
