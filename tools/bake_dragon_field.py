#!/usr/bin/env python3
"""Bake a local distance and nearest-material volume from the checked-in Red Dragon.

Requires the existing C++17/Node SDF baker and Python 3 with NumPy and VTK.
"""
import gzip
import multiprocessing
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets/sponza-dragon'
PITCH = 0.025


def read_mesh():
    data = gzip.decompress((ASSETS / 'red-dragon-mesh.bin.gz').read_bytes())
    magic, vertex_count, index_count, _ = struct.unpack_from('<4s3I', data)
    if magic != b'SGM1':
        raise ValueError('Expected the bundled SGM1 mesh')
    positions = np.frombuffer(data, '<f4', 3 * vertex_count, 16).reshape(-1, 3)
    normal = np.frombuffer(data, '<i2', 3 * vertex_count, 16 + 12 * vertex_count).reshape(-1, 3)
    color = np.frombuffer(data, 'u1', 3 * vertex_count, 16 + 18 * vertex_count).reshape(-1, 3)
    roughness = np.frombuffer(data, 'i1', vertex_count, 16 + 21 * vertex_count)
    indices = np.frombuffer(data, '<u4', index_count, (16 + 22 * vertex_count + 3) // 4 * 4).reshape(-1, 3)
    # The source dragon is flat shaded and has one material per face.
    for attribute in (normal, color, roughness):
        for j in (1, 2):
            if not np.array_equal(attribute[indices[:, 0]], attribute[indices[:, j]]):
                raise ValueError('Expected face-constant attributes in Red Dragon')
    normals = normal[indices[:, 0]].astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    material = np.column_stack((color[indices[:, 0]], biased(roughness[indices[:, 0]] / 127.0)))
    return positions, indices, normals, material


def biased(value):
    return np.rint((np.clip(value, -1, 1) * 0.5 + 0.5) * 255).astype('u1')


def signed_distance(positions, indices, minimum, dims, directory):
    # Nonzero winding along one axis can leak through an open seam. Vote using
    # the three independent cyclic projections; the exact distance magnitudes agree.
    votes = np.zeros(tuple(dims[::-1]), dtype='u1')
    magnitude = None
    for axes in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        obj = directory / 'dragon.obj'
        with obj.open('w') as file:
            np.savetxt(file, positions[:, axes], fmt='v %.9g %.9g %.9g')
            np.savetxt(file, indices + 1, fmt='f %d %d %d')
        output = directory / 'field.f32'
        def values(array):
            return [str(x) for x in array[axes]]
        subprocess.run([
            'node', str(ROOT / 'bin/bake-sdf.mjs'), '--obj', str(obj),
            '--lo', *values(minimum), '--span', *values(dims * PITCH),
            '--res-xyz', *values(dims), '--clip', '4', '--threads', str(min(8, multiprocessing.cpu_count())),
            '--block', '0', '--validate', '64', '--out', str(output),
        ], check=True)
        xyz = np.fromfile(output, '<f4').reshape(tuple(dims[axes][::-1])).T
        field = xyz.transpose(np.argsort(axes)).T
        votes += field < 0
        if magnitude is None:
            magnitude = np.abs(field)
        else:
            np.testing.assert_allclose(magnitude, np.abs(field), atol=2e-6, rtol=1e-6)
    signed = np.where(votes >= 2, -magnitude, magnitude)
    if not np.any(signed < -2 * PITCH) or any(np.any(np.take(signed, side, axis=axis) <= 0)
                                                for axis in range(3) for side in (0, -1)):
        raise ValueError('Missing solid interior or non-positive exterior padding')
    print(f'{np.sum(votes >= 2):,} interior voxels, {np.sum((votes == 1) | (votes == 2)):,} winding disagreements', flush=True)
    return np.ascontiguousarray(signed)


def near_surface_payload(positions, indices, normals, materials, minimum, signed):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(positions, deep=True))
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtkIdTypeArray(3 * np.arange(len(indices) + 1, dtype=np.int64), deep=True),
                  numpy_to_vtkIdTypeArray(indices.astype(np.int64).ravel(), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    # Three pitches cover the filtered hit band plus all neighboring texels
    # used for grazing/edge refinement. Distant texels only supply distance.
    near = np.flatnonzero(abs(signed.ravel()) <= 3 * PITCH)
    centers = minimum + (np.column_stack(np.unravel_index(near, signed.shape))[:, ::-1] + 0.5) * PITCH
    face_ids = np.empty(len(near), dtype=np.int32)
    nearest = np.empty_like(centers)
    cell, sub_id, dist2 = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    point = [0.0, 0.0, 0.0]
    for i, center in enumerate(centers):
        locator.FindClosestPoint(center, point, cell, sub_id, dist2)
        face_ids[i] = int(cell)
        nearest[i] = point
    # An independent BVH implementation must agree with the native baker on
    # distance at every voxel where we use triangle attributes.
    residual = centers - nearest
    np.testing.assert_allclose(np.linalg.norm(residual, axis=1), abs(signed.ravel()[near]), atol=2e-6, rtol=1e-6)
    normal = normals[face_ids]
    tangent_offset = np.einsum('ij,ij->i', residual, normal)
    material = np.zeros((signed.size, 4), dtype='u1')
    normal_plane = np.zeros_like(material)
    material[near] = materials[face_ids]
    normal_plane[near, :3] = biased(normal)
    normal_plane[near, 3] = biased(tangent_offset / PITCH)
    print(f'{len(near):,} narrow-band material/plane samples checked against triangle distance', flush=True)
    return material, normal_plane


def main():
    positions, indices, normals, materials = read_mesh()
    lower = np.floor(positions.min(axis=0).astype(np.float64) / PITCH).astype(int) - 3
    upper = np.ceil(positions.max(axis=0).astype(np.float64) / PITCH).astype(int) + 3
    dims = upper - lower
    minimum = lower * PITCH
    with tempfile.TemporaryDirectory() as directory:
        signed = signed_distance(positions, indices, minimum, dims, Path(directory))
    material, normal = near_surface_payload(positions, indices, normals, materials, minimum, signed)
    header = struct.pack('<4s3I4f', b'DDF1', *dims, *minimum, PITCH)
    output = ASSETS / 'red-dragon-field.bin.gz'
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        packed = Path(directory) / output.name
        with packed.open('wb') as target:
            with gzip.GzipFile(fileobj=target, filename='', mode='wb', mtime=0) as stream:
                stream.write(header)
                stream.write(signed.astype('<f2').tobytes())
                stream.write(material.tobytes())
                stream.write(normal.tobytes())
        packed.replace(output)
    print(f'{dims.tolist()} voxels, local min {minimum.tolist()}, pitch {PITCH} m, {output.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
