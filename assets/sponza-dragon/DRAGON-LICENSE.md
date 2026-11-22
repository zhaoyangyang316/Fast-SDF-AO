# Red Dragon

Tomek Zamojski created **Red Dragon** and released it under the Creative Commons Attribution 3.0 Unported license. The bundled `red-dragon-mesh.bin.gz` derives from the GLB distributed by Poly Pizza. [Original model and author credit](https://poly.pizza/m/5SgYrV6nhws) · [CC BY 3.0 license](https://creativecommons.org/licenses/by/3.0/) · [License text](https://creativecommons.org/licenses/by/3.0/legalcode)

The conversion keeps all 43,440 triangles and their winding. It rotates the source to Z-up with the head pointing along +X, centers the origin between the feet at floor level, and scales the wingtip height to 1.25 m. The package encodes the authored normals as signed 16-bit components and the ten linear material colors and roughness values in 8 bits per component.

To regenerate from the repository root, run the following commands with Python 3 installed. The converter checks the original file's SHA-256 digest before writing the asset.

```sh
curl -fL 'https://static.poly.pizza/0f978555-ded1-4a6d-96e9-c719d31efa8c.glb' -o /tmp/red-dragon.glb
python3 tools/pack_sponza_dragon.py /tmp/red-dragon.glb
```
