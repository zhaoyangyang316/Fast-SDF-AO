"""Pack the licensed Poly Pizza Red Dragon GLB in the demo's scene mesh format."""
import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path


SOURCE_SHA256 = 'b8092c588ba33a274b30fc4f457d5d15b13a79ad679c8ade7baa04749cbe703d'
OUTPUT = Path(__file__).resolve().parents[1] / 'assets/sponza-dragon/red-dragon-mesh.bin.gz'


def pack(source, height):
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('The file differs from the credited Red Dragon GLB')
    json_length = struct.unpack_from('<I', data, 12)[0]
    model = json.loads(data[20:20 + json_length])
    binary = memoryview(data)[28 + json_length:]

    def attribute(index):
        item = model['accessors'][index]
        view = model['bufferViews'][item['bufferView']]
        fmt = '<3f' if item['type'] == 'VEC3' else '<I'
        offset = view.get('byteOffset', 0) + item.get('byteOffset', 0)
        end = offset + item['count'] * struct.calcsize(fmt)
        return list(struct.iter_unpack(fmt, binary[offset:end]))

    primitives = model['meshes'][0]['primitives']
    materials = model['materials']
    positions, normals, colors, roughness, indices = [], [], [], [], []
    for primitive in primitives:
        base = len(positions)
        positions.extend(attribute(primitive['attributes']['POSITION']))
        normals.extend(attribute(primitive['attributes']['NORMAL']))
        indices.extend(base + i for i, in attribute(primitive['indices']))
        material = materials[primitive['material']]['pbrMetallicRoughness']
        count = len(positions) - base
        colors.extend([tuple(round(c * 255) for c in material['baseColorFactor'][:3])] * count)
        roughness.extend([round(material['roughnessFactor'] * 127)] * count)

    # Source Y is up and +Z points toward the eyes. Center the pivot between the feet.
    bottom = min(y for _, y, _ in positions)
    top = max(y for _, y, _ in positions)
    feet = [p for p in positions if p[1] < bottom + 0.025 * (top - bottom)]
    pivot_z = 0.5 * (min(z for _, _, z in feet) + max(z for _, _, z in feet))
    scale = height / (top - bottom)
    out = bytearray(struct.pack('<4I', 0x314d4753, len(positions), len(indices), 0))
    for x, y, z in positions:
        out.extend(struct.pack('<3f', (z - pivot_z) * scale, x * scale, (y - bottom) * scale))
    for x, y, z in normals:
        out.extend(struct.pack('<3h', round(z * 32767), round(x * 32767), round(y * 32767)))
    for color in colors:
        out.extend(color)
    out.extend(roughness)
    out.extend(bytes(-len(out) % 4))
    for i in indices:
        out.extend(struct.pack('<I', i))
    with OUTPUT.open('wb') as target:
        with gzip.GzipFile(fileobj=target, filename='', mode='wb', mtime=0) as compressed:
            compressed.write(out)
    print(f'{len(indices) // 3:,} triangles, {len(positions):,} vertices, {OUTPUT.stat().st_size:,} bytes')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Original Red Dragon GLB')
    parser.add_argument('--height', type=float, default=1.25, help='Wingtip height in meters')
    args = parser.parse_args()
    if not 0 < args.height < float('inf'):
        parser.error('height must be positive and finite')
    pack(args.source, args.height)
