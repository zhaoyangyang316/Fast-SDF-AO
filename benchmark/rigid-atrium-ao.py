#!/usr/bin/env python3
"""Score half-resolution production AO against independent triangle visibility.

Run after benchmark/rigid-atrium-quality.mjs. Pass the JSON containing its two
contactAO records, and an output directory. No GPU or network is needed here.
"""
import argparse
import base64
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RADIUS = 1.12
BIAS = 0.0005  # WebGPU contact-ray offset, in scene units
SIDE = 32     # 32 x 32 stratified cosine-weighted directions, shared between cases
MAX_RECEIVERS = 1536
SEED = 5017

# Reuse the benchmark's actual registered local mesh data and transform matrices.
# Only the fixture's scene-registration calls are intercepted for this CPU export.
EXPORT_RIGID = """
import fs from 'node:fs/promises';
import {createAtriumWorkload, rigidAtriumPose} from './benchmark/rigid-atrium-scene.mjs';
globalThis.fetch = async url => new Response(await fs.readFile(url));
const noop = () => {};
let handle = 0;
const fixture = await createAtriumWorkload({
  setRigidMover:noop, setMovers:noop, setEmissive:noop, selectScene:noop,
  camera:{position:{set:noop}}, controls:{target:{set:noop},update:noop},
  gpu:{rigidMeshes:{registerAsset:()=>++handle,createInstance:()=>++handle}},
}, {count:8,time:1.2});
const bounds = [], soups = [];
for (const [i,{assetIndex}] of fixture.instances.entries()) {
  const d = fixture.assets[assetIndex].data, m = rigidAtriumPose(i,1.2);
  const positions = d.positions, soup = new Float32Array(3*d.indices.length);
  const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
  for (const [j,id] of d.indices.entries()) {
    const x=positions[id*3], y=positions[id*3+1], z=positions[id*3+2];
    for (let k=0; k<3; ++k) {
      soup[3*j+k] = m[k]*x + m[4+k]*y + m[8+k]*z + m[12+k];
      lo[k] = Math.min(lo[k],soup[3*j+k]); hi[k] = Math.max(hi[k],soup[3*j+k]);
    }
  }
  soups.push(Buffer.from(soup.buffer)); bounds.push([lo,hi]);
}
const out = process.argv[1];
await fs.writeFile(out+'/props.f32', Buffer.concat(soups));
process.stdout.write(JSON.stringify(bounds));
"""


def decode(record):
    return np.frombuffer(base64.b64decode(record["base64"]), dtype="<f4").reshape(
        record["height"], record["width"], 9
    )


def cosine_rays(pos, normal, rng):
    n = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    axis = np.where(np.abs(n[:, :1]) < .9, [1, 0, 0], [0, 1, 0])
    tangent = np.cross(n, axis)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(n, tangent)
    cells = np.arange(SIDE * SIDE)
    shape = (len(pos), len(cells))
    u = (cells // SIDE + rng.random(shape)) / SIDE
    phi = 2 * np.pi * (cells % SIDE + rng.random(shape)) / SIDE
    direction = np.sqrt(u)[:, :, None] * (
        np.cos(phi)[:, :, None] * tangent[:, None] + np.sin(phi)[:, :, None] * bitangent[:, None]
    ) + np.sqrt(1-u)[:, :, None] * n[:, None]
    rays = np.empty((*shape, 7), np.float32)
    rays[:, :, :3] = (pos + BIAS * n)[:, None]
    rays[:, :, 3:6] = direction
    rays[:, :, 6] = RADIUS
    return rays


def m(error):
    return dict(bias=float(np.mean(error)), mae=float(np.mean(np.abs(error))),
                rms=float(np.sqrt(np.mean(error**2))), p95_abs=float(np.percentile(np.abs(error), 95)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quality_json", type=Path)
    ap.add_argument("output_dir", type=Path)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.quality_json.read_text())["contactAO"]
    rigid, empty = (decode(records[k]) for k in ("rigid", "empty"))
    assert rigid.shape == empty.shape and np.all(np.isfinite(rigid)) and np.all(np.isfinite(empty))
    pos = rigid[:, :, :3]
    normal = rigid[:, :, 4:7]
    # Matched visible static receivers only; exclude new surfaces and uncovered disocclusions.
    visible = ((rigid[:, :, 3] > 0) & np.isin(rigid[:, :, 7], [1, 2])
               & (rigid[:, :, 7] == empty[:, :, 7])
               & (np.max(np.abs(pos-empty[:, :, :3]), axis=2) < 1e-4)
               & (np.max(np.abs(normal-empty[:, :, 4:7]), axis=2) < 1e-4))
    with tempfile.TemporaryDirectory() as work:
        tmp = Path(work)
        bounds = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", EXPORT_RIGID, work], cwd=ROOT))
        dist = np.min([np.linalg.norm(np.maximum(np.maximum(lo-pos, pos-hi), 0), axis=2) for lo,hi in bounds], axis=0)
        near = visible & (dist <= RADIUS)
        y,x = np.nonzero(near)
        if len(y) == 0:
            raise ValueError("No matching static receivers near prop bounds")
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(y), min(MAX_RECEIVERS,len(y)), replace=False))
        y,x = y[idx], x[idx]
        subprocess.run(["node","tools/gen_atrium.mjs","--tris-only","--export-tris",tmp/"static.f32"], check=True, cwd=ROOT)
        static = (tmp/"static.f32").read_bytes()
        props = (tmp/"props.f32").read_bytes()
        (tmp/"combined.f32").write_bytes(static+props)
        subprocess.run(["c++","-O3","-std=c++17","-pthread",ROOT/"tools/trace_tris.cpp","-o",tmp/"trace"],check=True)
        ref = np.empty((len(y),2))
        for start in range(0,len(y),256):
            sl = slice(start,start+256)
            rays = cosine_rays(pos[y[sl],x[sl]],normal[y[sl],x[sl]],rng)
            rays.tofile(tmp/"rays.f32")
            for k,file in enumerate(["static","combined"]):
                subprocess.run([tmp/"trace",tmp/(file+".f32"),tmp/"rays.f32",tmp/"hits.u8"],check=True,stdout=subprocess.DEVNULL)
                ref[sl,k] = 1 - np.fromfile(tmp/"hits.u8",dtype=np.uint8).reshape(rays.shape[:2]).mean(axis=1)
    gpu0,gpu8 = (record[y,x,8].astype(float) for record in (empty,rigid))
    delta, reference = gpu0-gpu8, ref[:,0]-ref[:,1]
    # Paired rays hit extra geometry in the combined case only. The physically
    # expected visibility loss is thus nonnegative on every reference receiver.
    assert np.all(reference >= 0)
    material = reference >= .02
    def stats(mask):
        if not mask.any():
            return {"receivers":0}
        return {"receivers":int(mask.sum()),"reference_mean_darkening":float(np.mean(reference[mask])),
                "production_mean_darkening":float(np.mean(delta[mask])),"darkening_error":m(delta[mask]-reference[mask]),
                "positive_production_darkening":int((delta[mask]>0).sum()),
                "production_darkening_above_1pp":int((delta[mask]>.01).sum()),
                "production_lightening_above_1pp":int((delta[mask]<-.01).sum())}
    summary = {
        "pose_seconds":1.2,"full_resolution":[rigid.shape[1]*2,rigid.shape[0]*2],
        "half_resolution":[rigid.shape[1],rigid.shape[0]],"frames":[records[k]["frame"] for k in ("empty","rigid")],
        "texture":[records[k]["target"] for k in ("empty","rigid")],
        "static_triangles":len(static)//36,"instance_triangles":len(props)//36,
        "instances":len(bounds),"ray_length_m":RADIUS,"normal_bias_m":BIAS,"cosine_rays_per_receiver":SIDE**2,
        "rays_per_scene":int(len(y)*SIDE**2),"matching_static_pixels":int(visible.sum()),
        "matching_pixels_within_radius_of_prop_bounds":int(near.sum()),"sampled_receivers":len(y),
        "near_selection":"Uniform without replacement over matched pixels within 1.12 m of a transformed prop AABB; independent of either AO output",
        "empty_visibility_error":m(gpu0-ref[:,0]),"eight_visibility_error":m(gpu8-ref[:,1]),
        "all":stats(np.ones(len(y),dtype=bool)), "reference_darkening_at_least_2pp":stats(material),
        "reference_darkening_at_least_5pp":stats(reference >= .05),
        "reference_darkening_below_1pp":stats(reference < .01),
        "within_15cm_of_prop_bounds":stats(dist[y,x]<=.15),
        "near_horizontal_receivers":stats(normal[y,x,2]>.9),
        "notes":"Visibility equals one minus the fraction of sampled rays intersecting a triangle. Darkening is empty minus eight. Positive error means more production darkening for darkening metrics, brighter production visibility for visibility metrics. Production includes finite-precision G-buffer normals, soft SDF visibility for static geometry, caching, temporal and spatial filtering; CPU reference uses hard two-sided world-space intersections for all geometry. Bounds distance is a broad-phase measure, not triangle-surface distance."
    }
    (args.output_dir/"contact-ao.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(args.output_dir/"contact-ao-receivers.npz", y=y,x=x,pos=pos[y,x],normal=normal[y,x],
                        bbox_distance=dist[y,x],gpu_empty=gpu0,gpu_eight=gpu8,ref_empty=ref[:,0],ref_eight=ref[:,1])
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
