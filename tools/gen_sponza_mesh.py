#!/usr/bin/env python3
"""Convert McGuire's CC BY 3.0 Sponza ZIP to vertex-colored SGM1.

    python3 tools/gen_sponza_mesh.py /path/to/sponza.zip

Dependencies: NumPy and Pillow. See assets/sponza-dragon/SPONZA.md for the
source, checksum, dimensions, and attribution. No network is needed here.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = 'da005cbee0be2df2abc8513f3ceb61bcb6f69aac112babcd9c00169a27c2770c'
# All three of these materials are opacity-masked triangle cards in the source.
SKIP = {'leaf', 'Material__57', 'chain'}


def read_obj(archive):
    pos, uv, normal = [], [], []
    faces = defaultdict(list)
    material = None
    for line in archive.open('sponza.obj'):
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == b'v':
            pos.append(tuple(map(float, fields[1:4])))
        elif tag == b'vt':
            uv.append(tuple(map(float, fields[1:3])))
        elif tag == b'vn':
            normal.append(tuple(map(float, fields[1:4])))
        elif tag == b'usemtl':
            material = fields[1].decode()
        elif tag == b'f' and material not in SKIP:
            vertices = [tuple(int(x) - 1 for x in field.split(b'/')) for field in fields[1:]]
            if min(min(v) for v in vertices) < 0:
                raise ValueError('Expected positive position/UV/normal OBJ indices')
            for j in range(1, len(vertices) - 1):
                faces[material].append((vertices[0], vertices[j], vertices[j + 1]))
    p = np.asarray(pos, dtype=np.float64)[:, [0, 2, 1]]
    p += [63.0, 35.0, 2.5062]
    p *= [.01, -.01, .01]
    n = np.asarray(normal)[:, [0, 2, 1]]
    n[:, 1] *= -1
    return p, np.asarray(uv), n, faces


def read_materials(archive):
    res = {}
    for line in archive.read('sponza.mtl').decode().splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == 'newmtl':
            material = res[tok[1]] = {'kd': [1, 1, 1]}
        elif tok[0] == 'Kd':
            material['kd'] = list(map(float, tok[1:4]))
        elif tok[0] == 'map_Kd':
            material['map'] = tok[-1].replace('\\', '/')
    return res


def refine_material_mesh(vertices, triangles, max_edge):
    """Subdivide broad flat walls/floors so their texture survives vertex coloring."""
    for _ in range(16):
        p = vertices[:, :3][triangles]
        lengths = np.sum((p - np.roll(p, -1, axis=1)) ** 2, axis=2)
        longest = np.argmax(lengths, axis=1)
        split = lengths[np.arange(len(triangles)), longest] > max_edge ** 2
        if not np.any(split):
            return vertices, triangles
        ii = np.nonzero(split)[0]
        e = longest[ii]
        a, b, c = (triangles[ii, (e + k) % 3] for k in range(3))
        edges, ids = np.unique(np.sort(np.column_stack((a, b)), axis=1), axis=0, return_inverse=True)
        mid = len(vertices) + ids
        vertices = np.vstack((vertices, vertices[edges].mean(axis=1)))
        triangles = np.vstack((triangles[~split], np.column_stack((a, mid, c)), np.column_stack((mid, b, c))))
    raise RuntimeError('Subdivision failed to converge')


def sample_diffuse(archive, material, uv):
    path = material.get('map')
    if path:
        with Image.open(io.BytesIO(archive.read(path))) as image:
            # Smooth texture details smaller than the baked vertex sample spacing.
            image.thumbnail((128, 128), Image.Resampling.LANCZOS)
            tex = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
        h, w = tex.shape[:2]
        xy = uv * [w, -h] - 0.5
        ij = np.floor(xy).astype(np.int64)
        wx, wy = (xy - ij).T[:, :, None]
        x, y = ij.T
        x0, x1, y0, y1 = x % w, (x + 1) % w, y % h, (y + 1) % h
        rgb = (1-wy)*((1-wx)*tex[y0, x0] + wx*tex[y0, x1]) + wy*((1-wx)*tex[y1, x0] + wx*tex[y1, x1])
        rgb = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    else:
        rgb = np.ones((len(uv), 3))
    return np.round(np.clip(rgb * material['kd'], 0, 1) * 255).astype(np.uint8)


def convert(source, output):
    with zipfile.ZipFile(source) as archive:
        positions, uv, normals, faces = read_obj(archive)
        materials = read_materials(archive)
        posall, norall, colors, roughness, indices = [], [], [], [], []
        base = 0
        for name, group in faces.items():
            refs, inverse = np.unique(np.asarray(group, dtype=np.int32).reshape(-1, 3), axis=0, return_inverse=True)
            v = np.column_stack((positions[refs[:, 0]], normals[refs[:, 2]], uv[refs[:, 1]]))
            tri = inverse.reshape(-1, 3)
            # Cluster only densely tessellated surfaces. Keep separate normal
            # directions to preserve fold creases and geometric silhouettes.
            if len(group) > 6000:
                cell = 0.055
                key = np.column_stack((np.rint(v[:, :3] / cell), np.rint(v[:, 3:6] * 3)))
                _, keep, remap = np.unique(key, axis=0, return_index=True, return_inverse=True)
                v, tri = v[keep], remap[tri]
                a, b, c = tri.T
                tri = tri[(a != b) & (b != c) & (c != a)]
            # Architectural relief/lion murals need denser color samples. The
            # cloth folds and carved columns have sufficiently dense source meshes.
            edge = {'floor': 0.42, 'bricks': 1.9, 'Material__25': 0.10,
                    'Material__298': 0.21, 'Material__47': 0.08, 'ceiling': 1.1}.get(name)
            if edge:
                v, tri = refine_material_mesh(v, tri, edge)
            n = v[:, 3:6]
            n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-15)
            posall.append(v[:, :3].astype('<f4'))
            norall.append(np.round(n * 32767).astype('<i2'))
            colors.append(sample_diffuse(archive, materials[name], v[:, 6:8]))
            rough = {'floor': .22, 'vase': .29, 'vase_hanging': .47, 'flagpole': .35}.get(name, .86 if name.startswith('fabric') else .69)
            roughness.append(np.full(len(v), round(rough * 127), np.int8))
            indices.append((tri + base).astype('<u4'))
            base += len(v)
    positions = np.vstack(posall)
    index = np.vstack(indices).ravel()
    payload = b''.join((struct.pack('<4I', 0x314d4753, len(positions), len(index), 0),
                        positions.tobytes(), np.vstack(norall).tobytes(), np.vstack(colors).tobytes(),
                        np.concatenate(roughness).tobytes()))
    payload += bytes(-len(payload) % 4) + index.tobytes()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as file, gzip.GzipFile(filename='', mode='wb', fileobj=file, mtime=0, compresslevel=7) as gz:
        gz.write(payload)
    print(f'{len(positions):,} vertices, {len(index)//3:,} triangles, {output.stat().st_size:,} compressed bytes')
    print('Bounds:', positions.min(axis=0), positions.max(axis=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--out', type=Path, default=BASE / 'assets/sponza-dragon/sponza-mesh.bin.gz')
    args = parser.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise SystemExit('Source ZIP checksum does not match the documented McGuire release')
    convert(args.source, args.out)
