// Hardware run from the repo root:
// python3 test/webgpu-browser.py benchmark/rigid-mesh-bench.mjs --width 1280 --height 720
// Add --software --browser /usr/bin/chromium --width 160 --height 90 for a short diagnostic.
import { Matrix4, TorusGeometry } from 'three';
import { RIGID_MESH_WGSL } from '../src/webgpu-rigid-wgsl.js';

function summary(samples) {
  const sorted = samples.slice().sort((a, b) => a - b);
  const percentile = p => Number(sorted[Math.ceil(p * sorted.length) - 1].toFixed(3));
  return { median: percentile(0.5), p95: percentile(0.95) };
}

function fixtureMesh(variant) {
  const geometry = new TorusGeometry(0.19, 0.048 + variant * 0.0005, 8, 24);
  const positions = geometry.attributes.position.array;
  return {
    positions, normals: geometry.attributes.normal.array, indices: geometry.index.array,
    colors: new Float32Array(positions.length).fill(0.65),
    roughness: new Float32Array(positions.length / 3).fill(0.45),
  };
}

function pose(index, count, clustered, t) {
  const columns = Math.ceil(Math.sqrt(count));
  const rows = Math.ceil(count / columns);
  const spacing = clustered ? 0.38 : 0.85;
  const x = (index % columns - (columns - 1) / 2) * spacing + 0.10 * Math.sin(index + t);
  const y = (Math.floor(index / columns) - (rows - 1) / 2) * spacing - 1.1;
  return new Matrix4().makeRotationY(t + index * 0.17).setPosition(x, y, 0.42);
}

function difference(after, before) {
  return Object.fromEntries(Object.keys(before).map(key => [key, after[key] - before[key]]));
}

// Keep the production BLAS and leaf boxes. The alternative top level has one
// root and a consecutive array of all leaves, forcing a linear scan of boxes.
function flatTopLevel(scene) {
  const source = scene.tlasData;
  if (scene.count === 1) return source.slice();
  const nodes = new Uint32Array((scene.count + 1) * 8);
  nodes.set(source.subarray(0, 8));
  nodes[3] = scene.count + 1;
  let next = 1;
  for (let offset = 0; offset < source.length; offset += 8) {
    if (source[offset + 7] === 0xffffffff) continue;
    nodes.set(source.subarray(offset, offset + 8), next * 8);
    nodes[next * 8 + 3] = next + 1;
    next++;
  }
  return nodes;
}

async function queryBenchmark(device, scene, samples, raySide) {
  const rayCount = raySide * raySide;
  const code = RIGID_MESH_WGSL + `
  @group(0) @binding(0) var<storage,read_write> distances: array<f32>;
  @compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ${rayCount}u) { return; }
    let uv = (vec2f(f32(id.x % ${raySide}u), f32(id.x / ${raySide}u)) + 0.5) / ${raySide}.0;
    let origin = vec3f(0., -6., 2.4);
    let rayEnd = vec3f((uv.x - 0.5)*3.5, -1.1, mix(-0.25,1.05,uv.y));
    let hit = rigidClosestHit(origin, normalize(rayEnd-origin), 0.001, 20.);
    distances[id.x] = hit.t;
  }`;
  const module = device.createShaderModule({ code });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter(m => m.type === 'error');
  if (errors.length) throw Error(errors.map(m => m.message).join('\n'));
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto', compute: { module, entryPoint: 'probe' },
  });
  const create = (size, usage) => device.createBuffer({ size, usage });
  const flat = flatTopLevel(scene);
  const flatBuffer = create(flat.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = create(rayCount * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const clock = device.features.has('timestamp-query') ? device.createQuerySet({type:'timestamp', count:2}) : null;
  const resolve = clock && create(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
  const readback = create(rayCount * 4 + 16, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  try {
    device.queue.writeBuffer(flatBuffer, 0, flat);
    const group = (n, buffers) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(n),
      entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}})),
    });
    const results = group(0, [output]);
    const groups = [scene.tlasBuffer, flatBuffer].map(top => group(1, [scene.instanceBuffer, top, scene.geometryBuffer]));
    const times = [[], []];
    let expected;
    for (let i = 0; i <= samples; i++) {
      for (let j = 0; j < 2; j++) {
        const mode = (i + j) % 2;
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass(clock ? {timestampWrites: {
          querySet: clock, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
        }} : {});
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, results);
        pass.setBindGroup(1, groups[mode]);
        pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
        pass.end();
        if (clock) {
          enc.resolveQuerySet(clock, 0, 2, resolve, 0);
          enc.copyBufferToBuffer(resolve, 0, readback, 0, 16);
        }
        if (i === 0) enc.copyBufferToBuffer(output, 0, readback, 16, rayCount * 4);
        const command = enc.finish();
        const start = performance.now();
        device.queue.submit([command]);
        await device.queue.onSubmittedWorkDone();
        let ms = performance.now() - start;
        if (clock || i === 0) {
          await readback.mapAsync(GPUMapMode.READ);
          const data = readback.getMappedRange();
          if (clock) {
            const ticks = new BigUint64Array(data, 0, 2);
            ms = Number(ticks[1] - ticks[0]) / 1e6;
          }
          if (i === 0) {
            const values = new Float32Array(data, 16);
            if (expected && values.some((t, k) => t !== expected[k]))
              throw Error('TLAS and flat scan returned different hit distances');
            expected = values.slice();
          }
          readback.unmap();
        }
        if (i > 0) times[mode].push(ms);
      }
    }
    const hits = expected.filter(t => t < 20).length;
    if (!hits) throw Error('Ray workload missed every instance');
    return {rays: rayCount, hits, clock: clock ? 'GPU timestamp' : 'completed dispatch wall',
      tlasMs: summary(times[0]), flatMs: summary(times[1])};
  } finally {
    for (const resource of [flatBuffer, output, clock, resolve, readback]) resource?.destroy();
  }
}

export async function run(demo, {samples, warmup, many} = {}) {
  const gpu = demo.gpu;
  const scene = gpu.rigidMeshes;
  const quick = gpu.width <= 256;
  samples ??= quick ? 4 : 32;
  warmup ??= quick ? 3 : 96;
  many ??= quick ? 4 : 16;
  demo.stopLoop();
  demo.setRigidMover(false);
  demo.setMovers(false);
  demo.setEmissive(false);
  await demo.selectScene('gallery');
  gpu.setQuality('high');
  const device = gpu.device;
  const cases = [];
  const originalPrepare = scene.prepare;
  let prepareDuration = 0;
  scene.prepare = function measuredPrepare() {
    const start = performance.now();
    const updates = originalPrepare.call(this);
    prepareDuration = performance.now() - start;
    return updates;
  };
  try {
    for (const [count, unique, clustered] of [
      [0,false,false], [1,false,false], [many,false,false], [many,false,true], [many,true,false], [many,true,true],
    ]) {
      const assets = [], instances = [];
      try {
        const beforeSetup = {...scene.stats};
        for (let i = 0; i < count; i++) {
          if (unique || i === 0) assets.push(scene.registerAsset(fixtureMesh(i)));
          instances.push(scene.createInstance(assets.at(-1), pose(i, count, clustered, 0)));
        }
        gpu.clearHash();
        demo.renderOnce();
        await device.queue.onSubmittedWorkDone();
        const setup = difference(scene.stats, beforeSetup);
        const storageBytes = scene.geometryBuffer.size + scene.instanceBuffer.size + scene.tlasBuffer.size;
        const phases = [];
        let t = 0;
        for (const name of count ? ['stationary', 'moving', 'stopped'] : ['stationary']) {
          const poseMs = [], prepareMs = [], submitMs = [], frameMs = [];
          let before, firstCompletedMs;
          for (let f = -warmup; f < samples; f++) {
            if (f === 0) before = {...scene.stats};
            const begin = performance.now();
            if (name === 'moving') {
              t += 0.05;
              instances.forEach((id, i) => scene.setTransform(id, pose(i, count, clustered, t)));
            }
            const posed = performance.now();
            demo.renderOnce();
            const submitted = performance.now();
            await device.queue.onSubmittedWorkDone();
            const completedMs = performance.now() - begin;
            if (f === -warmup) firstCompletedMs = completedMs;
            if (f >= 0) {
              poseMs.push(posed - begin);
              prepareMs.push(prepareDuration);
              submitMs.push(submitted - posed);
              frameMs.push(completedMs);
            }
          }
          const updates = difference(scene.stats, before);
          if (updates.assetBuilds || updates.tlasBuilds || updates.geometryBytesUploaded)
            throw Error('Rendering existing instances rebuilt or uploaded geometry');
          phases.push({name, firstCompletedMs, poseMs: summary(poseMs), rigidPrepareMs: summary(prepareMs),
            renderSubmitMs: summary(submitMs),
            completedFrameMs: summary(frameMs), updates, stillFrames: gpu.stillFrames});
        }
        const queries = count ? await queryBenchmark(device, scene, samples, quick ? 48 : 256) : null;
        cases.push({count, assets: assets.length, clustered, setup, traversalStorageBytes: storageBytes, phases, queries});
      } finally {
        for (const id of instances) scene.removeInstance(id);
        for (const asset of assets) scene.releaseAsset(asset);
        demo.renderOnce();
        await device.queue.onSubmittedWorkDone();
      }
    }
  } finally {
    delete scene.prepare;
  }
  const {vendor, architecture, description} = gpu.adapterInfo;
  return {adapter: {vendor,architecture,description}, resolution:[gpu.width,gpu.height],
    quality:gpu.quality, samples, warmup, trianglesPerAsset:384, cases};
}
