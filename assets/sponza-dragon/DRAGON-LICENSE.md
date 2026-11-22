# Red Dragon

Tomek Zamojski created **Red Dragon** and released it under the Creative Commons Attribution 3.0 Unported license. The bundled `red-dragon-mesh.bin.gz` derives from the GLB distributed by Poly Pizza. [Original model and author credit](https://poly.pizza/m/5SgYrV6nhws) · [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/) · [License text](https://creativecommons.org/licenses/by/3.0/legalcode)

The conversion keeps all 43,440 triangles and their winding. It rotates the source to Z-up with the head pointing along +X, centers the origin between the feet at floor level, and scales the wingtip height to 1.25 m. The package encodes the authored normals as signed 16-bit components and the ten linear material colors and roughness values in 8 bits per component.

To regenerate from the repository root, run the following commands with Python 3 installed. The converter checks the original file's SHA-256 digest before writing the asset.

```sh
curl -fL 'https://static.poly.pizza/0f978555-ded1-4a6d-96e9-c719d31efa8c.glb' -o /tmp/red-dragon.glb
python3 tools/pack_sponza_dragon.py /tmp/red-dragon.glb
```

## Local distance and material field

The bundled `red-dragon-field.bin.gz` is a voxel adaptation of Tomek Zamojski’s Red Dragon and carries the same CC BY 3.0 attribution. It derives directly from the bundled mesh. [Original model](https://poly.pizza/m/5SgYrV6nhws) · [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/)

The dragon-local grid has 139 × 150 × 56 cells at a pitch of 0.025 m, starting at (−2.15, −1.875, −0.075) m. Each FP16 voxel holds the distance in meters to the nearest triangle. Interior distances are negative. Some edges of the source mesh are open, so the bake assigns the sign by majority vote among three axis-aligned winding queries. Thin wings can pass between adjacent voxel centers. The field also records the nearest triangle’s material, normal, and tangent plane within 0.075 m of the mesh to support surface refinement. The representation is shared across rigid poses.

The gzip contains a little-endian `DDF1` header (four magic bytes, three uint32 dimensions, three float32 origin coordinates, one float32 pitch), followed by three x-fast arrays at voxel centers. These contain R16Float distance, RGBA8 linear albedo with signed roughness in alpha, and RGBA8 normal with a signed tangent-plane offset in alpha. Signed channels decode as `2 * unorm - 1`; multiply the plane offset by the pitch to obtain meters. The offset is positive on the triangle normal’s side and clamps at one pitch. Material and plane records outside the near-surface band are empty.

Rebuild from the repository root with Python 3, NumPy, VTK, Node.js and a C++17 compiler. The generator runs the repository’s native triangle-distance baker in three orientations and checks every near-surface distance against a separate VTK closest-point query.

```sh
python3 -m pip install numpy vtk
python3 tools/bake_dragon_field.py
```
