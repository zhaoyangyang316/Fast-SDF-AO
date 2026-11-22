// python3 test/webgpu-browser.py benchmark/rigid-atrium-queries.mjs --browser /usr/bin/chromium --software
import * as THREE from 'three';
import { rigidQueryPipeline, queryRigidBatch, referenceRigidHit } from '../test/rigid-queries.mjs';
import { createAtriumWorkload, rigidAtriumPose, RIGID_ATRIUM_CAMERA } from './rigid-atrium-scene.mjs';

function makeWorldMesh(asset, transform, material, id) {
  const {positions, normals, colors, roughness, indices} = asset.data;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(Float64Array.from(positions), 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(Float64Array.from(normals), 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('roughness', new THREE.BufferAttribute(roughness, 1));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.applyMatrix4(new THREE.Matrix4().fromArray(transform));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.instance = id;
  return mesh;
}

// Sample triangle interiors, avoiding shared-edge tie ambiguity. Each shadow
// direction is tested with light endpoints on both sides of the first blocker.
function sampleRays(meshes, sunDir) {
  const rays = [];
  const camera = new THREE.Vector3(...RIGID_ATRIUM_CAMERA.position);
  const add = (kind, origin, dir, max) => rays.push({kind, values: [...origin.toArray(), 0.0005, ...dir.toArray(), max]});
  for (const [i, mesh] of meshes.entries()) {
    const g = mesh.geometry, tris = g.index.count / 3;
    for (let j = 0; j < 24; j++) {
      const tri = (i * 193 + j * 397 + 83) % tris;
      const p = new THREE.Vector3();
      for (const [corner, w] of [0.23, 0.36, 0.41].entries()) {
        const v = new THREE.Vector3().fromBufferAttribute(g.attributes.position, g.index.getX(tri * 3 + corner));
        // Target the same float32 raster vertices regardless of CPU reference precision.
        p.addScaledVector(new THREE.Vector3(Math.fround(v.x), Math.fround(v.y), Math.fround(v.z)), w);
      }
      add('camera', camera, p.clone().sub(camera).normalize(), 30);
      const origin = p.clone().addScaledVector(sunDir, -2.0);
      const far = [ ...origin.toArray(), 0.0005, ...sunDir.toArray(), 4 ];
      const blocker = referenceRigidHit(meshes, Float32Array.from(far));
      if (!blocker) throw Error('Directed shadow reference missed its target');
      const t = blocker.values[0];
      add('light-before-blocker', origin, sunDir, Math.max(0.0005, t - 0.015));
      add('light-behind-blocker', origin, sunDir, t + 0.015);
    }
  }
  // Cover the unoccupied space between instances as viewed from the fixed camera.
  const forward = new THREE.Vector3(...RIGID_ATRIUM_CAMERA.target).sub(camera).normalize();
  const right = new THREE.Vector3().crossVectors(forward, new THREE.Vector3(0,0,1)).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward);
  for (let y = -4; y <= 4; y++) for (let x = -10; x <= 10; x++) {
    const dir = forward.clone().addScaledVector(right, (x + 0.13) * 0.05)
      .addScaledVector(up, (y + 0.31) * 0.06).normalize();
    add('camera-grid', camera, dir, 30);
  }
  return rays;
}

function planeDiagnostic(meshes, ray, ref, gpuT) {
  const g = meshes.find(mesh => mesh.userData.instance === ref.id).geometry;
  const vertices = [0,1,2].map(c => new THREE.Vector3().fromBufferAttribute(g.attributes.position, g.index.getX(ref.primitive * 3 + c)));
  const quantized = vertices.map(v => new THREE.Vector3(Math.fround(v.x), Math.fround(v.y), Math.fround(v.z)));
  const r = new THREE.Ray(new THREE.Vector3(...ray.slice(0,3)), new THREE.Vector3(...ray.slice(4,7)));
  const normal = THREE.Triangle.getNormal(...vertices, new THREE.Vector3());
  const p32 = r.intersectTriangle(...quantized, false, new THREE.Vector3());
  const cpu32Distance = p32 ? r.origin.distanceTo(p32) : null;
  return {cpuWorldF64Distance:ref.values[0], cpuWorldF32Distance:cpu32Distance,
    gpuMinusCpuF64:gpuT-ref.values[0], gpuMinusCpuF32:cpu32Distance === null ? null : gpuT-cpu32Distance,
    grazingDegrees:Math.asin(Math.min(1,Math.abs(normal.dot(r.direction) / r.direction.length()))) * 180 / Math.PI};
}

function compare(meshes, samples, input, ids, floats) {
  const byKind = {};
  const mismatches = [];
  let maxDistanceError = 0, maxBaryError = 0, minDistance = Infinity, maxDistance = 0;
  for (const [i, {kind}] of samples.entries()) {
    byKind[kind] ??= {rays:0, referenceHits:0, nearestMismatches:0, identityMismatches:0, distanceMismatches:0, shadowMismatches:0};
    const row = byKind[kind];
    row.rays++;
    const ref = referenceRigidHit(meshes, input.subarray(i*8,i*8+8));
    const k = i * 16;
    let identityError = ids[k] !== Number(!!ref);
    let distanceError = false;
    const shadowError = ids[k+3] !== Number(!!ref);
    if (ref) {
      row.referenceHits++;
      const t = ref.values[0];
      minDistance = Math.min(minDistance, t);
      maxDistance = Math.max(maxDistance, t);
      const dt = Math.abs(t - floats[k+4]);
      const duv = Math.max(Math.abs(floats[k+5] - ref.values[1]), Math.abs(floats[k+6] - ref.values[2]));
      maxDistanceError = Math.max(maxDistanceError, dt);
      maxBaryError = Math.max(maxBaryError, duv);
      identityError ||= ids[k+1] !== ref.id || ids[k+2] !== ref.primitive;
      distanceError = !(dt <= 2e-4);
    }
    const nearestError = identityError || distanceError;
    if (nearestError) row.nearestMismatches++;
    if (identityError) row.identityMismatches++;
    if (distanceError) row.distanceMismatches++;
    if (shadowError) row.shadowMismatches++;
    if ((nearestError || shadowError) && mismatches.length < 12) {
      mismatches.push({i, kind, ray: Array.from(input.slice(i*8,i*8+8)),
        plane: ref && !identityError ? planeDiagnostic(meshes, input.slice(i*8,i*8+8), ref, floats[k+4]) : null,
        cpu: ref ? {instance:ref.id, triangle:ref.primitive, t:ref.values[0]} : null,
        gpu: {valid:ids[k], instance:ids[k+1], triangle:ids[k+2], t:floats[k+4], blocked:ids[k+3]}});
    }
  }
  return {byKind, maxDistanceError, maxBaryError, minDistance, maxDistance, mismatches};
}

export async function run(demo) {
  const device = demo.gpu.device;
  const pipeline = await rigidQueryPipeline(device);
  const workload = await createAtriumWorkload(demo, {count:8, time:0});
  const material = new THREE.MeshBasicMaterial({side:THREE.DoubleSide});
  const poses = [];
  let grazingStress;
  try {
    for (const time of [0, 0.4, 0.8, 1.2]) {
      workload.setTime(time);
      const meshes = workload.instances.map(({handle,assetIndex}, i) =>
        makeWorldMesh(workload.assets[assetIndex], rigidAtriumPose(i,time), material, handle));
      try {
        const sun = new THREE.Vector3(...demo.gpu.sunDir).normalize();
        const samples = sampleRays(meshes, sun);
        const input = Float32Array.from(samples.flatMap(s => s.values));
        const {ids,floats} = await queryRigidBatch(device, demo.gpu.rigidMeshes, pipeline, input);
        poses.push({time, ...compare(meshes, samples, input, ids, floats)});
        // Keep this recorded near-tangent ray from the exploratory camera separate
        // from the fixed final-camera score. It grazes a triangle on the jade knot.
        if (time === 0.4) {
          const ray = Float32Array.from([0.1,-7.15,2.52,0.0005,
            0.3752371370792389,0.8684917688369751,-0.3239121735095978,30]);
          const obs = await queryRigidBatch(device, demo.gpu.rigidMeshes, pipeline, ray);
          const ref = referenceRigidHit(meshes, ray);
          if (!ref) throw Error('Recorded grazing reference missed the jade knot');
          const inst = workload.instances.find(i => i.handle === ref.id);
          grazingStress = {time, ray:Array.from(ray), asset:workload.assets[inst.assetIndex].name,
            cpu:{instance:ref.id,triangle:ref.primitive,blocked:true},
            gpu:{valid:!!obs.ids[0],instance:obs.ids[1],triangle:obs.ids[2],blocked:!!obs.ids[3],distance:obs.floats[4]},
            ...planeDiagnostic(meshes, ray, ref, obs.floats[4])};
          grazingStress.exceedsDistanceTolerance = Math.abs(grazingStress.gpuMinusCpuF64) > 0.0002;
        }
      } finally {
        for (const mesh of meshes) mesh.geometry.dispose();
      }
    }
    const stats = poses.flatMap(p => Object.values(p.byKind));
    const sum = key => stats.reduce((n,row) => n + row[key], 0);
    return {adapter:{vendor:demo.gpu.adapterInfo.vendor,architecture:demo.gpu.adapterInfo.architecture},
      scope:'Dynamic rigid triangles only', reference:'Double-sided Three.js CPU raycast on Float64 world vertices from Float32 local assets and poses',
      camera:RIGID_ATRIUM_CAMERA, distanceTolerance:0.0002,
      instanceCount:workload.count, assets:workload.assets.map(({name,triangles})=>({name,triangles})),
      rays:sum('rays'), hits:sum('referenceHits'), nearestMismatches:sum('nearestMismatches'),
      identityMismatches:sum('identityMismatches'), distanceMismatches:sum('distanceMismatches'),
      shadowMismatches:sum('shadowMismatches'), poses, grazingStress};
  } finally {
    workload.dispose();
    material.dispose();
  }
}
