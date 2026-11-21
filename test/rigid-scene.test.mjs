import assert from 'node:assert/strict';
import { test } from 'node:test';
import { RigidMeshScene } from '../src/webgpu-rigid.js';

globalThis.GPUBufferUsage ??= { STORAGE: 128, COPY_DST: 8, VERTEX: 32, INDEX: 16 };
function mockDevice() {
  return {
    limits: { maxStorageBufferBindingSize: 16 << 20, maxBufferSize: 16 << 20 },
    queue: {writeBuffer(dst, offset, data) {
      new Uint8Array(dst.bytes, offset, data.byteLength).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }},
    createBuffer({ size }) {
      return { size, bytes: new ArrayBuffer(size), destroy() { this.destroyed = true; } };
    },
  };
}

function mesh(count) {
  // Distinct triangles ensure SAH changes their order, with material per vertex.
  const positions = [], normals = [], colors = [], indices = [];
  for (let i = 0; i < count; i++) {
    const x = i & 1 ? i : -i;
    positions.push(x, 0, 0, x + 0.25, 0, 0, x, 1, 0);
    normals.push(0, 0, 1, 0, 0, 1, 0, 0, 1);
    colors.push(1, 0, 0, 0, 1, 0, 0, 0, 1);
    indices.push(3 * i, 3 * i + 1, 3 * i + 2);
  }
  return { positions, normals, colors, indices };
}

const atX = x => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 0, 0, 1];

test('asset BLAS escape links, source triangle IDs and per-instance offsets survive repacking', () => {
  const scene = new RigidMeshScene(mockDevice());
  const asset = scene.registerAsset(mesh(47));
  const second = scene.registerAsset(mesh(2));
  scene.createInstance(second, atX(20));
  scene.createInstance(asset, atX(-20));
  scene.prepare();
  const instances = new Uint32Array(scene.instanceBuffer.bytes);
  const arena = new Uint32Array(scene.geometryBuffer.bytes);
  for (const [slot, count] of [[0, 2], [1, 47]]) {
    const [root, end, triBase] = instances.subarray(slot * 20 + 16, slot * 20 + 19);
    assert.equal(arena[root * 8 + 3], end);
    const found = [];
    for (let i = root; i < end; i++) {
      const escape = arena[i * 8 + 3], payload = arena[i * 8 + 7];
      assert.ok(escape > i && escape <= end);
      if (payload === 0) continue;
      assert.equal(escape, i + 1);
      for (let j = 0; j < (payload & 7); j++) {
        found.push(arena[triBase * 4 + ((payload >>> 3) + j) * 36 + 7]);
      }
    }
    assert.deepEqual(found.sort((a,b) => a-b), Array.from({length: count}, (_,i) => i));
  }
});

test('pose updates refit transformed bounds and upload only changed instances', () => {
  const scene = new RigidMeshScene(mockDevice());
  const asset = scene.registerAsset(mesh(1));
  const first = scene.createInstance(asset);
  const second = scene.createInstance(asset, atX(12));
  assert.equal(scene.getInstanceCountForAsset(asset), 2);
  scene.prepare();
  const geometry = scene.geometryBuffer;
  const old = {...scene.stats};
  scene.setTransform(first, [0,1,0,0, -1,0,0,0, 0,0,1,0, 20,0,0,1]);
  assert.deepEqual(scene.prepare(), {changed:true, bindingsChanged:false});
  assert.deepEqual(scene.prepare(), {changed:false, bindingsChanged:false});
  assert.equal(scene.geometryBuffer, geometry);
  assert.equal(scene.stats.geometryBytesUploaded, old.geometryBytesUploaded);
  assert.equal(scene.stats.poseBytesUploaded, old.poseBytesUploaded + 80);
  assert.equal(scene.stats.tlasBuilds, old.tlasBuilds);
  assert.equal(scene.stats.tlasRefits, old.tlasRefits + 1);
  const bounds = new Float32Array(scene.tlasBuffer.bytes);
  assert.ok(Math.abs(bounds[0] - 12) < 1e-3 && Math.abs(bounds[4] - 20) < 1e-3);
  assert.throws(() => scene.releaseAsset(asset), /instances/);
  assert.throws(() => scene.setTransform(second, [2,...atX(0).slice(1)]), /rotation/);
  scene.removeInstance(first);
  assert.equal(scene.getInstanceCountForAsset(asset), 1);
  scene.prepare();
  assert.equal(new Uint32Array(scene.instanceBuffer.bytes)[19], second);
  scene.clear();
  assert.equal(scene.count, 0);
  assert.equal(scene.getInstanceCountForAsset(asset), 0);
  scene.prepare();
  assert.equal(new Uint32Array(scene.tlasBuffer.bytes)[3], 0);
  assert.ok(geometry.destroyed);
  assert.throws(() => scene.setTransform(second, atX(0)), /Unknown/);
});

test('releasing an asset compacts mesh addresses even when poses stay fixed', () => {
  const scene = new RigidMeshScene(mockDevice());
  const retired = scene.registerAsset(mesh(29));
  const live = scene.registerAsset(mesh(1));
  const handle = scene.createInstance(live);
  scene.prepare();
  const originalArena = scene.geometryBuffer;
  assert.ok(new Uint32Array(scene.instanceBuffer.bytes)[16] > 0);
  scene.releaseAsset(retired);
  assert.deepEqual(scene.prepare(), {changed:true, bindingsChanged:true});
  assert.ok(originalArena.destroyed);
  const words = new Uint32Array(scene.instanceBuffer.bytes);
  assert.deepEqual(Array.from(words.subarray(16, 20)), [0, 1, 2, handle]);
  assert.equal(scene.geometryBuffer.size, 32 + 144);
  scene.removeInstance(handle);
  scene.releaseAsset(live);
  scene.prepare();
  assert.equal(scene.geometryBuffer.size, 32);
  assert.equal(new Uint32Array(scene.tlasBuffer.bytes)[3], 0);
  const previous = {...scene.stats};
  scene.clear();
  assert.deepEqual(scene.prepare(), {changed:false, bindingsChanged:false});
  assert.deepEqual(scene.stats, previous);
});

test('rejects oversized vertex storage before accessing sparse attributes', () => {
  const device = mockDevice();
  device.limits.maxBufferSize = 1024;
  const scene = new RigidMeshScene(device);
  assert.throws(() => scene.registerAsset({ positions: {length: 3e8}, indices: [0, 1, 2] }), /buffer limit/);
  assert.equal(scene.stats.assetBuilds, 0);
});

test('retries retain the binding-change notification from a partial upload', () => {
  const device = mockDevice(), scene = new RigidMeshScene(device);
  const asset = scene.registerAsset(mesh(1));
  scene.createInstance(asset);
  const write = device.queue.writeBuffer;
  device.queue.writeBuffer = () => { throw new Error('injected failure'); };
  assert.throws(() => scene.prepare(), /injected/);
  device.queue.writeBuffer = write;
  assert.deepEqual(scene.prepare(), {changed:true, bindingsChanged:true});
  assert.equal(new Uint32Array(scene.tlasBuffer.bytes)[3], 1);
});
