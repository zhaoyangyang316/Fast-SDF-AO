"""Compare a Sponza GPU capture to independent triangle rays. Requires numpy, Pillow, scipy and vtk.

python3 benchmark/sponza-reflection-reference.py capture.json --out results
"""

import argparse
import base64
import gzip
import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import reference, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellArray,
    vtkGenericCell,
    vtkPolyData,
    vtkStaticCellLocator,
)

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "sponza-dragon"
# Contiguous vertex ranges of the source OBJ material groups after gen_sponza_mesh.py's conversion.
MATERIAL_GROUPS = {
    "arches": [(31224, 40612)],
    "columns": [(48414, 51538), (63048, 71655), (72753, 86073)],
    "banners": [(93963, 140322)],
}


def mesh(name):
    data = gzip.decompress((ASSETS / name).read_bytes())
    magic, nv, ni, _ = struct.unpack_from("<4sIII", data)
    assert magic == b"SGM1"
    vertices = np.frombuffer(data, "<f4", 3 * nv, 16).reshape(-1, 3)
    colors = np.frombuffer(data, "u1", 3 * nv, 16 + 18 * nv).reshape(-1, 3) / 255
    indices = np.frombuffer(data, "<u4", ni, (16 + 22 * nv + 3) // 4 * 4).reshape(-1, 3)
    return vertices, indices, colors


class TriangleReference:
    def __init__(self, vertices, faces):
        points, cells = vtkPoints(), vtkCellArray()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
        cells.SetData(
            numpy_to_vtkIdTypeArray(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
            numpy_to_vtkIdTypeArray(np.asarray(faces.ravel(), dtype=np.int64), deep=True),
        )
        self.mesh = vtkPolyData()
        self.mesh.SetPoints(points)
        self.mesh.SetPolys(cells)
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.mesh)
        self.locator.BuildLocator()

    def cast(self, starts, ends):
        cells = np.full(len(ends), -1, dtype=np.int32)
        points = np.zeros((len(ends), 3))
        weights = np.zeros((len(ends), 3))
        t, sub_id, cell_id = reference(0.0), reference(0), reference(0)
        pos, bary, cell = [0.0] * 3, [0.0] * 3, vtkGenericCell()
        for i, (start, end) in enumerate(zip(starts, ends)):
            if self.locator.IntersectWithLine(start, end, 1e-9, t, pos, bary, sub_id, cell_id, cell):
                cells[i], points[i], weights[i] = (
                    int(cell_id),
                    pos,
                    [1 - bary[0] - bary[1], bary[0], bary[1]],
                )
        return cells, points, weights


def image(shot, key):
    return Image.open(BytesIO(base64.b64decode(shot[key]))).convert("RGB")


def measure_color(render, reference_color, mask):
    d = np.abs(render[mask] - reference_color[mask]).mean(axis=1)
    return (
        {"pixels": len(d), "RGB_MAE": float(d.mean()), "RGB_p95": float(np.percentile(d, 95))}
        if len(d)
        else {"pixels": 0}
    )


def depth_score(ref, actual, mask):
    n = int(mask.sum())
    matched = mask & (actual > 0)
    e = np.abs(actual[matched] - ref[matched])
    return (
        {
            "referenceHits": n,
            "reportedHits": int(matched.sum()),
            "fractionWithin5cmIncludingMisses": float(np.count_nonzero(e <= 0.05) / n),
            "fractionWithin25cmIncludingMisses": float(np.count_nonzero(e <= 0.25) / n),
            "medianErrorMetersOnReportedHits": float(np.median(e)) if len(e) else None,
            "p95ErrorMetersOnReportedHits": float(np.percentile(e, 95)) if len(e) else None,
        }
        if n
        else {"referenceHits": 0}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--old-depth",
        type=Path,
        help="optional diagnostic PNG, little-endian millimeters in RG, hit validity in B",
    )
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    doc = json.loads(args.capture.read_text())
    shots = {s["name"]: s for s in doc["shots"]}
    empty = shots["empty"]
    w, h = doc["width"], doc["height"]
    eye = np.array(doc["camera"])
    virtual = eye * [1, 1, -1]
    vp = np.array(doc["projection"]).reshape(4, 4).T @ np.array(doc["view"]).reshape(4, 4).T

    def mirror(s):
        m = np.asarray(image(s, "mirror").convert("L").resize((w, h), Image.Resampling.NEAREST)) > 128
        return binary_erosion(m, iterations=2)

    floor = mirror(empty)
    y, x = np.nonzero(floor)
    ndc = np.column_stack((2 * (x + 0.5) / w - 1, 1 - 2 * (y + 0.5) / h, np.zeros(len(x)), np.ones(len(x))))
    homogeneous = ndc @ np.linalg.inv(vp).T
    primary = homogeneous[:, :3] / homogeneous[:, 3:4] - eye
    origin = eye + primary * (-eye[2] / primary[:, 2:3])
    rays = origin - virtual
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    # Start just above z=0 to exclude the mirror itself. Reference rays have no renderer normal bias or distance cutoff.
    starts = origin + rays * 1e-4
    ends = starts + rays * 100
    verts, faces, colors = mesh("sponza-mesh.bin.gz")
    assert len(verts) == 162515 and len(faces) == 243645
    scene = TriangleReference(verts, faces)
    ids, hits, weights = scene.cast(starts, ends)
    valid = ids >= 0
    expected = np.where(valid, np.linalg.norm(hits - virtual, axis=1), 0)
    first_vertex = np.where(valid, faces[ids, 0], -1)
    groups = {
        "allStatic": valid,
        **{
            name: np.logical_or.reduce(
                [(first_vertex >= start) & (first_vertex < end) for start, end in ranges]
            )
            for name, ranges in MATERIAL_GROUPS.items()
        },
    }
    albedo = np.einsum("pvc,pv->pc", colors[faces[ids]], weights)
    result = {
        "size": [w, h],
        "mirrorPixels": len(x),
        "staticReferenceHits": int(valid.sum()),
        "referenceRays": "from floor z=0, reflected about +Z, double-sided triangles, 100 m extent",
    }
    # An independent primary ray identifies which reflected points are directly observable in the shaded main view.
    # Compare those reflection samples to the exact same surfaces in the unmodified primary image.
    screen = np.column_stack((hits, np.ones(len(x)))) @ vp.T
    sx = np.floor((1 + screen[:, 0] / screen[:, 3]) * w / 2).astype(int)
    sy = np.floor((1 - screen[:, 1] / screen[:, 3]) * h / 2).astype(int)
    on_screen = valid & (screen[:, 3] > 0) & (sx >= 0) & (sy >= 0) & (sx < w) & (sy < h)
    indices = np.flatnonzero(on_screen)
    targets = hits[indices]
    _, unobstructed, _ = scene.cast(np.broadcast_to(eye, targets.shape), targets + (targets - eye) * 0.001)
    visible = np.zeros(len(x), dtype=bool)
    visible[indices] = np.linalg.norm(targets - unobstructed, axis=1) < 0.03
    real = np.asarray(image(empty, "image")) / 255.0
    same_surface = real[sy.clip(0, h - 1), sx.clip(0, w - 1)]
    debug = np.asarray(image(empty, "reflections"))[y, x] / 255.0
    final = real[y, x]
    result["sameSurfaceColor"] = {
        name: {
            "reflectionsOnly": measure_color(debug, same_surface, mask & visible),
            "normallyShaded": measure_color(final, same_surface, mask & visible),
        }
        for name, mask in groups.items()
    }
    # Cloth color identification is independent of brightness, mirror-Fresnel attenuation and the GI model.
    sorted_rgb = np.sort(albedo, axis=1)
    saturated = groups["banners"] & (sorted_rgb[:, 2] > 1.6 * sorted_rgb[:, 1]) & (sorted_rgb[:, 2] > 0.015)
    expected_channel = albedo.argmax(1)
    result["bannerChannel"] = {}
    for name, render in [("reflectionsOnly", debug), ("normallyShaded", final)]:
        sort = np.sort(render, axis=1)
        # Dark or gray pixels count as misses, not as arbitrary dominant-channel agreement.
        correct = (render.argmax(1) == expected_channel) & (sort[:, 2] - sort[:, 1] > 0.025)
        result["bannerChannel"][name] = {
            "tested": int(saturated.sum()),
            "correctFraction": float(correct[saturated].mean()),
        }
    measured = None
    if "staticDepth" in doc:
        d = doc["staticDepth"]
        assert (w, h) == (d["width"], d["height"])
        measured = np.frombuffer(gzip.decompress(base64.b64decode(d["gzipFloat32"])), "<f4").reshape(h, w)[
            y, x
        ]
    elif args.old_depth:
        d = np.asarray(Image.open(args.old_depth).convert("RGB"), dtype=np.uint32)[y, x]
        measured = np.where(d[:, 2] > 127, (d[:, 0] + 256 * d[:, 1]) * 0.001, 0)
    if measured is not None:
        result["staticDepth"] = {
            name: depth_score(expected, measured, region) for name, region in groups.items()
        }
        result["reportedHitWhereReferenceMisses"] = int(np.count_nonzero((measured > 0) & ~valid))
    dv, df, _ = mesh("red-dragon-mesh.bin.gz")
    result["dragon"] = {}
    for name in ["A", "B"]:
        s = shots[name]
        transform = np.array(s["model"]).reshape(4, 4).T
        mv = dv @ transform[:3, :3].T + transform[:3, 3]
        dragon = TriangleReference(mv, df)
        object_id, obj_hit, _ = dragon.cast(starts, ends)
        object_dist = np.linalg.norm(obj_hit - virtual, axis=1)
        test = mirror(s)[y, x]
        predicted = (object_id >= 0) & (~valid | (object_dist < expected)) & test
        frame = np.asarray(image(s, "reflections"))[y, x] / 255.0

        def red(im):
            return im[:, 0] - np.maximum(im[:, 1], im[:, 2])

        observed = (red(frame) - red(debug) > 0.10) & test
        result["dragon"][name] = {
            "referenceFootprintPixels": int(predicted.sum()),
            "observedRedPixels": int(observed.sum()),
            "IoU": float(np.count_nonzero(observed & predicted) / np.count_nonzero(observed | predicted)),
        }
    illustration = np.zeros((h, w, 3), dtype=np.uint8)
    illustration[y[valid], x[valid]] = np.round(255 * np.clip(albedo[valid], 0, 1) ** (1 / 2.2)).astype(
        np.uint8
    )
    Image.fromarray(illustration).save(args.out / "triangle-albedo.png")
    palette = {
        "allStatic": [89, 90, 91],
        "arches": [255, 177, 26],
        "columns": [72, 170, 255],
        "banners": [228, 76, 166],
    }
    for name, mask in groups.items():
        illustration[y[mask], x[mask]] = palette[name]
    Image.fromarray(illustration).save(args.out / "triangle-groups.png")
    for s in shots.values():
        image(s, "image").save(args.out / f"shaded-{s['name']}.png")
        image(s, "reflections").save(args.out / f"reflections-{s['name']}.png")
    (args.out / "reference-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
