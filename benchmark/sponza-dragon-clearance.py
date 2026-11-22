#!/usr/bin/env python3
"""Check the shipped meshes at sampled dragon poses. Requires NumPy and Node.js."""
import gzip
import json
from pathlib import Path
import struct
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets' / 'sponza-dragon'


def triangles(name):
    data = gzip.decompress((ASSETS / name).read_bytes())
    magic, vertices, count = struct.unpack_from('<4sII', data)
    assert magic == b'SGM1'
    xyz = np.frombuffer(data, '<f4', 3 * vertices, 16).reshape(-1, 3)
    index = np.frombuffer(data, '<u4', count, (16 + 22 * vertices + 3) // 4 * 4)
    return xyz[index].reshape(-1, 3, 3).astype(np.float64)


def intersecting(static, dynamic):
    """Separating-axis test, including in-plane axes for coplanar triangles."""
    a = np.roll(static, -1, axis=0) - static
    b = np.roll(dynamic, -1, axis=1) - dynamic
    na = np.cross(a[0], a[1])
    nb = np.cross(b[:, 0], b[:, 1])
    axes = [np.broadcast_to(na, nb.shape), nb]
    axes += [np.cross(aa, b[:, k]) for aa in a for k in range(3)]
    axes += [np.broadcast_to(np.cross(na, aa), nb.shape) for aa in a]
    axes += [np.cross(nb, b[:, k]) for k in range(3)]
    overlap = np.ones(len(dynamic), dtype=bool)
    for axis in axes:
        sa = axis @ static.T
        sb = np.einsum('mi,mvi->mv', axis, dynamic)
        tolerance = np.linalg.norm(axis, axis=1) * 1e-8
        overlap &= (sa.max(axis=1) + tolerance >= sb.min(axis=1)) & (sb.max(axis=1) + tolerance >= sa.min(axis=1))
    return overlap


def main():
    dragon = triangles('red-dragon-mesh.bin.gz')
    hall = triangles('sponza-mesh.bin.gz')
    hall_lo, hall_hi = hall.min(axis=1), hall.max(axis=1)
    floor = np.max(np.abs(hall[:, :, 2]), axis=1) < 0.001
    # The animation repeats within this 10 s interval. Include both photographed poses.
    times = np.arange(301) / 30
    source = "import {dragonPose} from './assets/sponza-dragon/scene.js'; console.log(JSON.stringify(JSON.parse(process.argv[1]).map(t => Array.from(dragonPose(t)))));"
    mats = np.asarray(json.loads(subprocess.check_output(['node', '--input-type=module', '-e', source, json.dumps(times.tolist())], cwd=ROOT))).reshape(-1, 4, 4)
    vertices, vertex_map = np.unique(dragon.reshape(-1, 3), axis=0, return_inverse=True)
    faces = vertex_map.reshape(-1, 3)
    clearance = 0.10
    total_hall = len(hall)
    floor_count = int(floor.sum())
    # A box containing every sampled pose; cull distant architecture only once.
    radius = np.linalg.norm(vertices, axis=1).max() + clearance
    swept_lo, swept_hi = mats[:, 3, :3].min(axis=0) - radius, mats[:, 3, :3].max(axis=0) + radius
    relevant = np.all((hall_hi >= swept_lo) & (hall_lo <= swept_hi), axis=1)
    hall, hall_lo, hall_hi, floor = hall[relevant], hall_lo[relevant], hall_hi[relevant], floor[relevant]
    checks, collisions, poses_hit, min_floor, max_floor = 0, 0, 0, np.inf, -np.inf
    near_architecture = 0
    for m in mats:
        world = vertices @ m[:3, :3] + m[3, :3]
        lo, hi = world.min(axis=0), world.max(axis=0)
        min_floor, max_floor = min(lo[2], min_floor), max(lo[2], max_floor)
        # Individual triangle AABBs conservatively bound the true closest distance.
        nearby = np.all((hall_hi >= lo - clearance) & (hall_lo <= hi + clearance), axis=1) & ~floor
        candidates = np.all((hall_hi >= lo) & (hall_lo <= hi), axis=1)
        if not np.any(nearby | candidates):
            continue
        transformed = world[faces]
        tri_lo, tri_hi = transformed.min(axis=1), transformed.max(axis=1)
        close, hit_pose = False, False
        for j in np.flatnonzero(nearby | candidates):
            gap = np.maximum(0, np.maximum(tri_lo - hall_hi[j], hall_lo[j] - tri_hi))
            sq = np.einsum('ij,ij->i', gap, gap)
            if nearby[j] and np.any(sq <= clearance ** 2):
                close = True
            overlapping = sq == 0
            if candidates[j] and np.any(overlapping):
                checks += int(overlapping.sum())
                hits = int(intersecting(hall[j], transformed[overlapping]).sum())
                collisions += hits
                hit_pose |= hits > 0
        near_architecture += int(close)
        poses_hit += int(hit_pose)
    print(json.dumps({'sampledPoses': len(times), 'timeRangeSeconds': [times[0], times[-1]], 'stepSeconds': 1/30,
      'dragonTriangles': len(dragon), 'sponzaTriangles': total_hall, 'trianglePairsAfterBroadPhase': checks,
      'intersectingTrianglePairsAcrossPoses': collisions, 'posesWithIntersections': poses_hit,
      'lowestVertexHeightRangeAboveFloorMeters': [float(min_floor), float(max_floor)], 'nominalFloorZ': 0,
      'nonFloorClearanceThresholdMeters': clearance, 'posesWithTriangleAABBsWithinThresholdOfNonFloor': near_architecture,
      'planarFloorTriangleCount': floor_count}, indent=2))


if __name__ == '__main__':
    main()
