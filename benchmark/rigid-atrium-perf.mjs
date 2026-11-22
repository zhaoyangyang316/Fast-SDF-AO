// From the repo root, with an idle WebGPU adapter:
// python3 test/webgpu-browser.py benchmark/rigid-atrium-perf.mjs --width 1280 --height 720
// Select a subset with --probe-args '{"cases":["eight-moving"],"rounds":[0,1,2]}'.
// The optional "eight-parked-default" case enables the production hash and warms 128 frames.
// To compare a different renderer revision, run this file, its fixture and the same browser
// harness in both checkouts with matching arguments. For SwiftShader, add --software.
import {createAtriumWorkload} from './rigid-atrium-scene.mjs';

const atriumWorkloads = {
  empty: {count: 0},
  'one-parked': {count: 1},
  'one-moving': {count: 1, moving: true},
  'eight-parked': {count: 8},
  'eight-moving': {count: 8, moving: true},
  'eight-parked-default': {count: 8, defaultHash: true},
};
const activeSamples = 12, activeWarmup = 6;

function distribution(values) {
  const v = values.filter(Number.isFinite).sort((a,b) => a-b);
  if (!v.length) return null;
  return {
    median: (v[Math.floor((v.length-1)/2)] + v[Math.floor(v.length/2)]) / 2,
    p95: v[Math.ceil(0.95*v.length)-1],
    mean: v.reduce((a,b) => a+b) / v.length,
  };
}

function delta(stats, before) {
  return Object.fromEntries(Object.keys(before).map(key => [key, stats[key]-before[key]]));
}

// Read the production shader's surface-pixel counters and pass timestamps after
// the timed queue fence. statsBuf is stale on frames using only the composite pass.
function frameCounters(gpu) {
  const device = gpu.device;
  const read = device.createBuffer({size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const resolve = gpu.hasTimestamp && device.createBuffer({
    size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  return {
    async get(compositeOnly, rays) {
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpu.statsBuf, 0, read, 0, 8);
      if (resolve) {
        encoder.resolveQuerySet(gpu.tsQuery, 0, 2, resolve, 0);
        encoder.copyBufferToBuffer(resolve, 0, read, 16, 16);
      }
      device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);
      const data = read.getMappedRange();
      const [traced, skipped] = new Uint32Array(data, 0, 2);
      const [begin, end] = new BigUint64Array(data, 16, 2);
      const passMs = resolve && end > begin ? Number(end-begin) / 1e6 : null;
      read.unmap();
      return {passMs, tracedPixels: compositeOnly ? 0 : traced, cachedPixels: compositeOnly ? 0 : skipped,
        primaryAoGiRays: compositeOnly ? 0 : traced*rays};
    },
    dispose() { read.destroy(); if (resolve) resolve.destroy(); },
  };
}

export function run(demo, options) {
  return measureRigidWorkloads(demo, createAtriumWorkload, atriumWorkloads, options);
}

export async function measureRigidWorkloads(demo, createWorkload, workloads, {
  cases = Object.keys(workloads).filter(key => !workloads[key].defaultHash), rounds = [0, 1, 2],
} = {}) {
  if (!cases.length || cases.some(key => !Object.hasOwn(workloads,key)) ||
      !rounds.length || rounds.some(i => !Number.isInteger(i) || i < 0)) throw Error('Invalid cases or rounds');
  demo.stopLoop();
  const gpu = demo.gpu, scene = gpu.rigidMeshes, queue = gpu.device.queue;
  const {vendor, architecture, description, device: adapterDevice} = gpu.adapterInfo;
  const counters = frameCounters(gpu);
  const originalPrepare = scene.prepare;
  let prepareMs;
  scene.prepare = function() {
    const begin = performance.now();
    const result = originalPrepare.call(this);
    prepareMs = performance.now() - begin;
    return result;
  };
  const oldHash = gpu.useHash;
  const blocks = [];
  let camera, target, staticTriangles;
  try {
    for (const round of rounds) {
      // Rotate the case order across rounds to distribute startup and thermal effects.
      const offset = (round*2) % cases.length;
      for (const name of [...cases.slice(offset), ...cases.slice(0,offset)]) {
        const {count, moving = false, defaultHash = false} = workloads[name];
        const warmup = defaultHash ? 128 : activeWarmup;
        const samples = defaultHash ? 36 : activeSamples;
        const startTime = round*activeSamples/30;
        const setupBefore = {...scene.stats};
        const fixture = await createWorkload(demo, {count, time: startTime});
        try {
          gpu.setQuality('high');
          gpu.useHash = defaultHash;
          gpu.viewMode = 0;
          gpu.clearHash();
          if (scene.count !== count || !gpu.useGI || !gpu.useReflections || gpu.dynObjects.length)
            throw Error('Unexpected scene geometry or renderer settings');
          const currentCamera = demo.camera.position.toArray(), currentTarget = demo.controls.target.toArray();
          if (camera && (camera.some((v,i) => Math.abs(v-currentCamera[i]) > 1e-6) ||
              target.some((v,i) => Math.abs(v-currentTarget[i]) > 1e-6)))
            throw Error('Camera or target changed between workloads');
          camera = currentCamera; target = currentTarget;
          staticTriangles = gpu.sceneRes.indexCount/3;
          demo.renderOnce();
          await queue.onSubmittedWorkDone();
          const setupUpdates = delta(scene.stats, setupBefore);
          const frameTime = f => startTime + (f + 1)/30;
          for (let f = -warmup; f < 0; f++) {
            if (moving) fixture.setTime(frameTime(f));
            demo.renderOnce();
            await queue.onSubmittedWorkDone();
          }
          const before = {...scene.stats}, frames = [];
          for (let f = 0; f < samples; f++) {
            const ping = gpu.pingIdx;
            const begin = performance.now();
            if (moving) fixture.setTime(frameTime(f));
            const posed = performance.now();
            demo.renderOnce();
            const submitted = performance.now();
            await queue.onSubmittedWorkDone();
            const completedMs = performance.now() - begin;
            const compositeOnly = gpu.pingIdx === ping;
            const countersForFrame = await counters.get(compositeOnly, demo.WEBGPU_QUALITY_PRESETS.high.rays);
            if (gpu._rigidMotion !== moving ||
                (!defaultHash && (gpu.stillFrames >= 28 || compositeOnly ||
                  !countersForFrame.tracedPixels || countersForFrame.cachedPixels)))
              throw Error('Unexpected pose update, pass skip, or SHaRC value in ' + name);
            frames.push({completedMs, poseMs: posed-begin, prepareMs, renderSubmitMs: submitted-posed,
              stillFrames: gpu.stillFrames, compositeOnly, ...countersForFrame});
          }
          const updates = delta(scene.stats, before);
          if (updates.assetBuilds || updates.tlasBuilds || updates.geometryBytesUploaded ||
              updates.tlasRefits !== (moving ? samples : 0) ||
              !!updates.poseBytesUploaded !== moving || !!updates.tlasBytesUploaded !== moving)
            throw Error('Unexpected rigid acceleration updates in ' + name);
          const used = [...new Set(fixture.instances.map(i => i.assetIndex))];
          const assets = used.map(i => ({name: fixture.assets[i].name, triangles: fixture.assets[i].triangles}));
          const instancedTriangles = fixture.instances.reduce((n, i) => n + fixture.assets[i.assetIndex].triangles, 0);
          blocks.push({name, round, count, moving, hash: defaultHash, warmup, startTime, assets,
            instancedTriangles, setupUpdates, updates, frames});
        } finally {
          fixture.dispose?.();
        }
      }
    }
  } finally {
    delete scene.prepare;
    gpu.useHash = oldHash;
    counters.dispose();
  }
  const summaries = cases.map(name => {
    const rows = blocks.filter(b => b.name === name).flatMap(b => b.frames);
    const summary = {};
    for (const key of ['completedMs','passMs','poseMs','prepareMs','renderSubmitMs','primaryAoGiRays','cachedPixels'])
      summary[key] = distribution(rows.map(f => f[key]));
    return {name, samples: rows.length, compositeOnlyFrames: rows.filter(f => f.compositeOnly).length, ...summary};
  });
  return {adapter: {vendor, architecture, description, device: adapterDevice}, browser: navigator.userAgent,
    resolution: [gpu.width,gpu.height], quality: gpu.quality, camera, target, staticTriangles,
    sampling: {activeWarmup, activeFramesPerBlock: activeSamples, rounds, timeStepSeconds: 1/30,
      completedMs: 'Pose update, CPU frame submission and queue completion; one frame in flight',
      passMs: 'WebGPU timestamps from G-buffer start (or composite on skipped frames) to final composite end',
      primaryAoGiRays: 'Traced half-resolution surface pixels times the High primary rays per pixel; excludes secondary rays and shadow/reflection rays'},
    summaries, blocks};
}
