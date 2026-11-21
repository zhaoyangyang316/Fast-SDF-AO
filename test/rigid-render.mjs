// Run with: python3 test/webgpu-browser.py test/rigid-render.mjs
import * as THREE from 'three';
import { buildContactBVH } from '../contact-bvh.js';

function check(condition, message) {
  if (!condition) throw new Error(message);
}

// Read the production textures through their existing texture bindings, including
// the float32 world positions which do not require float32 filtering support.
export function textureReader(device) {
  const module = device.createShaderModule({code: `
    @group(0) @binding(0) var src: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> out: array<vec4f>;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) i: vec3u) {
      let dims = textureDimensions(src);
      if (all(i.xy < dims)) { out[i.y * dims.x + i.x] = textureLoad(src, i.xy, 0); }
    }
  ` });
  const pipeline = device.createComputePipeline({layout: 'auto', compute: {module, entryPoint: 'main'}});
  return async texture => {
    const size = texture.width * texture.height * 16;
    const output = device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC});
    const read = device.createBuffer({size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
    const bind = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
      {binding: 0, resource: texture.createView()}, {binding: 1, resource: {buffer: output}},
    ]});
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(texture.width / 8), Math.ceil(texture.height / 8));
    pass.end();
    enc.copyBufferToBuffer(output, 0, read, 0, size);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(read.getMappedRange().slice(0));
    read.destroy();
    output.destroy();
    return values;
  };
}

function rgbDelta(a, b, i) {
  return Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
}

function geometryChanges(a, b) {
  let n = 0;
  for (let i = 3; i < a.length; i += 4) {
    if (Math.abs(a[i] - b[i]) > 1e-4) n++;
  }
  return n;
}

async function gallery(demo, capture, read) {
  const gpu = demo.gpu;
  demo.setRigidMover(false);
  demo.setEmissive(true, 0.7);
  const base = await capture();
  const stats = [];
  for (const t of [0, 2]) {
    demo.setRigidMover(true, t);
    const pose = await capture();
    const pixels = geometryChanges(pose.pos, base.pos);
    let changedAO = 0, changedGI = 0, shadowed = 0, litMesh = 0;
    for (let i = 0; i < pose.pos.length; i += 4) {
      if (pose.pos[i + 3] !== base.pos[i + 3]) {
        const pixel = i / 4;
        const x = 2 * (pixel % gpu.aoW), y = 2 * Math.floor(pixel / gpu.aoW);
        if (rgbDelta(pose.color, base.color, 4 * (y * gpu.width + x)) > 0.08) litMesh++;
      } else if (pose.pos[i + 3] > 0 && Math.abs(pose.pos[i + 2]) < 0.04) {
        if (base.meta[i] - pose.meta[i] > 0.1 || base.meta[i + 1] - pose.meta[i + 1] > 0.1) shadowed++;
        if (base.gi[i + 3] - pose.gi[i + 3] > 0.05) changedAO++;
        if (rgbDelta(base.gi, pose.gi, i) > 0.1) changedGI++;
      }
    }
    check(pixels > 12 && litMesh > 6, 'Moving chair missing from G-buffer or full-resolution composite');
    check(changedAO > 0 && changedGI > 0, 'Moving chair did not affect lighting on the static floor');
    stats.push({t, geometryPixels: pixels, litMesh, changedAO, changedGI, shadowed});
  }

  check(stats.some(s => s.shadowed > 0), 'Moving chair did not cast a shadow on the static floor');
  const {asset, matrix} = demo.rigidMover;
  const single = await capture();
  const buildCount = gpu.rigidMeshes.stats.assetBuilds;
  const secondPose = matrix.slice();
  secondPose[12] -= 2;
  const second = gpu.rigidMeshes.createInstance(asset, secondPose);
  const both = await capture();
  check(geometryChanges(both.pos, single.pos) > 12, 'Second instance missing');
  check(gpu.rigidMeshes.stats.assetBuilds === buildCount, 'Second instance rebuilt the asset');
  gpu.rigidMeshes.removeInstance(second);
  const one = await capture();
  check(!geometryChanges(one.pos, single.pos), 'Instance removal left stale raster geometry');

  // First build a real cast shadow, then remove the chair with history intact.
  // Filtered visibility should follow the raw visibility at the old footprint.
  for (let i = 0; i < 5; i++) demo.renderOnce();
  const shadow = await capture(false);
  demo.setRigidMover(false);
  const firstClear = await capture(false);
  demo.renderOnce();
  const thirdClear = await capture(false);
  let hardShadowSamples = 0, recovered = 0;
  for (let i = 0; i < base.pos.length; i += 4) {
    if (base.pos[i + 3] <= 0 || shadow.pos[i + 3] !== base.pos[i + 3] || Math.abs(base.pos[i + 2]) > 0.04) continue;
    for (const light of [0, 1]) {
      const j = i + light;
      if (shadow.meta[j] > 0.1 || shadow.filtered[j] > 0.35 ||
          base.meta[j] < 0.85 || firstClear.meta[j] < 0.85 || thirdClear.meta[j] < 0.85) continue;
      hardShadowSamples++;
      if (thirdClear.filtered[j] > 0.65) recovered++;
    }
  }
  check(hardShadowSamples > 3 && recovered >= hardShadowSamples * 0.7,
    'Old chair shadow lingered on static floor: ' + JSON.stringify({hardShadowSamples, recovered}));

  demo.setRigidMover(true, 2);
  const hidden = matrix.slice();
  hidden[14] -= 2;
  gpu.rigidMeshes.setTransform(demo.rigidMover.instance, hidden);
  const belowFloor = await capture();
  check(!geometryChanges(belowFloor.pos, base.pos), 'Hidden chair appeared through the floor');
  demo.setRigidMover(false);
  await capture(false); // First frame without the chair.
  const restored = await capture(false); // Leave sampling and temporal state untouched after removal.
  const restoredFrame = gpu.frame - 1;
  check(!geometryChanges(restored.pos, base.pos), 'Removing all movers did not restore static geometry');
  // Start an independent static control after measuring the transition. Raw sun
  // and orb shadow rays are jittered by sample index, so compare the same rays.
  gpu.clearHash();
  demo.renderOnce();
  gpu.frame = restoredFrame;
  const control = await capture(false);
  let restoredShadowErrors = 0;
  for (let i = 0; i < base.pos.length; i += 4) {
    if (Math.abs(control.meta[i] - restored.meta[i]) > 0.01 ||
        Math.abs(control.meta[i + 1] - restored.meta[i + 1]) > 0.01) restoredShadowErrors++;
  }
  check(restoredShadowErrors < 3, 'Shadow remained after removing the instance');

  // Prime a settled frame deterministically instead of waiting for 72+ frames on a software GPU.
  demo.setEmissive(false);
  demo.setRigidMover(true, 0);
  await capture(false);
  const still = await capture(false);
  gpu.stillFrames = 72;
  gpu.lastSkipPct = 100;
  const slot = gpu.pingIdx;
  demo.renderOnce();
  const settled = await read(gpu.context.getCurrentTexture());
  check(slot === gpu.pingIdx, 'Stationary instance never entered the settled path');
  let missing = 0, visible = 0;
  for (let i = 3; i < still.pos.length; i += 4) {
    if (Math.abs(still.pos[i] - base.pos[i]) < 1e-4) continue;
    const pixel = (i - 3) / 4;
    const j = 4 * (2 * Math.floor(pixel / gpu.aoW) * gpu.width + 2 * (pixel % gpu.aoW));
    visible++;
    if (rgbDelta(still.color, settled, j) > 0.08) missing++;
  }
  check(visible > 12 && missing < visible / 5, 'Stationary instance disappeared from settled composite');
  demo.setRigidMover(true, 2);
  const moved = await capture(false);
  check(gpu.pingIdx !== slot && geometryChanges(still.pos, moved.pos) > 12, 'Moved instance stayed frozen in the settled frame');
  demo.setRigidMover(false);
  return {poses: stats, hardShadowSamples, recovered};
}

// A split metallic floor viewed obliquely. A red panel above the floor lies
// outside the frustum, along the reflected center ray from both floor halves.
async function offscreenPanel(demo, capture) {
  const {gpu, camera, controls} = demo;
  const positions = [], normals = [], colors = [], roughness = [], indices = [];
  for (const [left, right, material] of [[-4, 0, -0.04], [0, 4, -0.20]]) {
    const first = positions.length / 3;
    for (const [x, y] of [[left, -1], [right, -1], [right, 3], [left, 3]]) {
      positions.push(x, y, 0);
      normals.push(0, 0, 1);
      colors.push(0.8, 0.8, 0.8);
      roughness.push(material);
    }
    indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  const bvh = buildContactBVH([new THREE.Mesh(geometry)], THREE, {gridSize: 1, range: 0.2405});
  const sdfDims = [16, 16, 16], sdfMin = [-4, -4, -2], sdfSpan = [8, 8, 8];
  const sdfHalf = new Uint16Array(16 ** 3);
  for (let z = 0; z < 16; z++) {
    const d = sdfMin[2] + (z + 0.5) * sdfSpan[2] / 16;
    sdfHalf.fill(THREE.DataUtils.toHalfFloat(d), z * 256, (z + 1) * 256);
  }
  gpu.loadScene({positions: new Float32Array(positions), normals: new Float32Array(normals),
    colors: new Float32Array(colors), roughness: new Float32Array(roughness), indices: new Uint32Array(indices),
    bvh, sdfMin, sdfSpan, sdfDims, sdfHalf, contactRange: 0.24, useGI: true});
  camera.position.set(0, -2, 2);
  controls.target.set(0, 0, 0);
  demo.setSun(0);
  gpu.viewMode = 4;
  const base = await capture();
  const asset = gpu.rigidMeshes.registerAsset({
    positions: new Float32Array([-.9, 0, -.22, .9, 0, -.22, .9, 0, .22, -.9, 0, .22]),
    normals: new Float32Array([0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0]),
    colors: new Float32Array([1, .02, .02, 1, .02, .02, 1, .02, .02, 1, .02, .02]),
    roughness: new Float32Array([.7, .7, .7, .7]), indices: new Uint32Array([0, 1, 2, 0, 2, 3])
  });
  const instance = gpu.rigidMeshes.createInstance(asset, new THREE.Matrix4().makeTranslation(0, 1.5, 1.5).elements);
  const panel = await capture();
  check(!geometryChanges(base.pos, panel.pos), 'Panel must be outside the camera view');
  let mirror = 0, glossy = 0;
  for (let y = 0; y < gpu.height; y++) {
    for (let x = 0; x < gpu.width; x++) {
      const surface = 4 * (Math.floor(y / 2) * gpu.aoW + Math.floor(x / 2));
      if (base.pos[surface + 3] <= 0 || Math.abs(base.pos[surface]) < 0.05) continue;
      if (rgbDelta(panel.color, base.color, 4 * (y * gpu.width + x)) < 0.1) continue;
      if (base.pos[surface] < 0) mirror++;
      else glossy++;
    }
  }
  check(mirror > 4 && glossy > 4, 'Offscreen rigid panel missing from mirror or rough reflections');
  gpu.rigidMeshes.removeInstance(instance);
  gpu.rigidMeshes.releaseAsset(asset);
  return {mirrorPixels: mirror, glossyPixels: glossy};
}

export async function run(demo) {
  const {gpu} = demo;
  gpu.useReflections = true;
  gpu.useHash = false;
  demo.setEmissive(false);
  const read = textureReader(gpu.device);
  async function capture(reset = true) {
    if (reset) gpu.clearHash(); // Compare poses with the same camera and frame-zero ray sequence.
    demo.renderOnce();
    const slot = 1 - gpu.pingIdx;
    const [pos, gi, meta, filtered, color] = await Promise.all([
      read(gpu.gPosDepth[slot]), read(gpu.rawAOGI), read(gpu.metaTex),
      read(gpu.histMeta[slot]), read(gpu.context.getCurrentTexture()),
    ]);
    return {pos, gi, meta, filtered, color};
  }
  const galleryResult = await gallery(demo, capture, read);
  const reflections = await offscreenPanel(demo, capture);
  return {...galleryResult, reflections};
}
