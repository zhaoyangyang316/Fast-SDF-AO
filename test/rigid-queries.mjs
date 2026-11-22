// Run in the demo's WebGPU browser test harness. Three.js raycasts provide an
// independent triangle reference, including primitive order and interpolants.
import * as THREE from 'three';
import { RigidMeshScene } from '../src/webgpu-rigid.js';
import { RIGID_MESH_WGSL } from '../src/webgpu-rigid-wgsl.js';

const QUERY = /* wgsl */`
struct RayInput { originMin: vec4f, directionMax: vec4f };
struct QueryOutput { ids: vec4u, distanceBary: vec4f, normal: vec4f, colorRoughness: vec4f };
@group(0) @binding(0) var<storage, read> rays: array<RayInput>;
@group(0) @binding(1) var<storage, read_write> results: array<QueryOutput>;
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&rays)) { return; }
  let ray = rays[id.x];
  let o = ray.originMin; let d = ray.directionMax;
  let h = rigidClosestHit(o.xyz, d.xyz, o.w, d.w);
  let blocked = rigidAnyHit(o.xyz, d.xyz, o.w, d.w);
  results[id.x] = QueryOutput(vec4u(h.valid, h.instanceId, h.primitiveId, u32(blocked)),
    vec4f(h.t, h.bary, 0.0), vec4f(h.normal, 0.0), vec4f(h.color, h.roughness));
}`;

function meshData(geometry) {
  const { position, normal } = geometry.attributes;
  const colors = new Float32Array(position.count * 3);
  const roughness = new Float32Array(position.count);
  for (let i = 0; i < position.count; i++) {
    colors.set([(i % 5) / 5, (i % 7) / 7, (i % 3) / 3], 3 * i);
    roughness[i] = (i % 11) / 11;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('roughness', new THREE.BufferAttribute(roughness, 1));
  return { positions: position.array, normals: normal.array, colors, roughness, indices: geometry.index.array };
}

export function referenceRigidHit(meshes, ray) {
  const origin = new THREE.Vector3(...ray.slice(0, 3));
  const dir = new THREE.Vector3(...ray.slice(4, 7));
  const caster = new THREE.Raycaster(origin, dir, ray[3], ray[7]);
  const hit = caster.intersectObjects(meshes, false).find(h => h.distance > ray[3] && h.distance < ray[7]);
  if (!hit) return null;
  const mesh = hit.object, { position, normal, color, roughness } = mesh.geometry.attributes;
  const indices = [hit.face.a, hit.face.b, hit.face.c];
  const vertices = indices.map(i => new THREE.Vector3().fromBufferAttribute(position, i));
  const w = THREE.Triangle.getBarycoord(mesh.worldToLocal(hit.point.clone()), ...vertices, new THREE.Vector3());
  function interpolate(attribute) {
    return indices.reduce((v, ix, corner) =>
      v.addScaledVector(new THREE.Vector3().fromBufferAttribute(attribute, ix), w.getComponent(corner)), new THREE.Vector3());
  }
  const r = indices.reduce((sum, ix, corner) => sum + roughness.getX(ix) * w.getComponent(corner), 0);
  return { id: mesh.userData.instance, primitive: hit.faceIndex,
    values: [hit.distance, w.y, w.z, ...interpolate(normal).transformDirection(mesh.matrixWorld).toArray(),
      ...interpolate(color).toArray(), r] };
}

export async function rigidQueryPipeline(device, shader = RIGID_MESH_WGSL, sceneLayout = null) {
  const module = device.createShaderModule({code: shader + QUERY});
  let layout = 'auto';
  if (sceneLayout) {
    const ioLayout = device.createBindGroupLayout({entries: [
      {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'read-only-storage'}},
      {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: 'storage'}},
    ]});
    layout = device.createPipelineLayout({bindGroupLayouts: [ioLayout, sceneLayout]});
  }
  return device.createComputePipelineAsync({layout, compute: {module, entryPoint: 'main'}});
}

export async function queryRigidBatch(device, scene, pipeline, input, sceneGroup = null) {
  const bytes = input.length / 8 * 64;
  function buffer(size, usage) { return device.createBuffer({size, usage}); }
  const src = buffer(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const dst = buffer(bytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const read = buffer(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
  device.queue.writeBuffer(src, 0, input);
  function group(index, buffers) {
    return device.createBindGroup({layout: pipeline.getBindGroupLayout(index),
      entries: buffers.map((b, binding) => ({binding, resource: {buffer: b}}))});
  }
  const io = group(0, [src, dst]);
  try {
    scene.prepare();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, io);
    pass.setBindGroup(1, sceneGroup ?? group(1, [scene.instanceBuffer, scene.tlasBuffer, scene.geometryBuffer]));
    pass.dispatchWorkgroups(Math.ceil(input.length / 8 / 32));
    pass.end();
    enc.copyBufferToBuffer(dst, 0, read, 0, bytes);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const copy = read.getMappedRange().slice(0);
    read.unmap();
    return { ids: new Uint32Array(copy), floats: new Float32Array(copy) };
  } finally {
    src.destroy(); dst.destroy(); read.destroy();
  }
}

export async function testRigidQueries(device) {
  const scene = new RigidMeshScene(device);
  const pipeline = await rigidQueryPipeline(device);
  const rays = [];
  for (let j = -4; j <= 4; j++) for (let i = -6; i <= 6; i++) {
    const ray = [i * 0.29 + 0.041, j * 0.28 + 0.019, 3, 0.001, 0, 0, -1, 7];
    rays.push(ray, [...ray.slice(0, 7), 2.91]);
  }
  rays.push([0.01, 0.03, 0.08, 0.001, 1, 0, 0, 7],  // ray starts inside the torus bounds
            [0.11, 0.05, 0.04, 0.001, 1, 1e-10, 0, 7],
            [0.1, -0.1, 3, 3, 0, 0, -1, 3]);         // empty ray interval
  const input = Float32Array.from(rays.flat());
  let queries = 0, hits = 0;
  async function check(label, meshes) {
    const { ids, floats } = await queryRigidBatch(device, scene, pipeline, input);
    for (let i = 0; i < rays.length; i++) {
      const ref = referenceRigidHit(meshes, input.subarray(i*8, i*8+8));
      const k = i*16;
      if (ids[k] !== Number(!!ref) || ids[k+3] !== Number(!!ref)
          || (ref && (ids[k+1] !== ref.id || ids[k+2] !== ref.primitive))) {
        throw Error(`${label} ray ${i} hit identity: got ${ids.slice(k,k+4)}, expected ${JSON.stringify(ref)}`);
      }
      const found = [4,5,6,8,9,10,12,13,14,15].map(offset => floats[k+offset]);
      if (ref && !ref.values.every((value, c) => Math.abs(found[c]-value) < 2e-4))
        throw Error(`${label} ray ${i} interpolants: got ${found}, expected ${ref.values}`);
      if (ref) hits++;
      queries++;
    }
  }

  const torus = new THREE.TorusGeometry(0.7, 0.24, 9, 29);
  const triangle = new THREE.BufferGeometry();
  triangle.setAttribute('position', new THREE.Float32BufferAttribute([-1,-0.6,0, 0.8,-0.6,0, -0.1,0.9,0], 3));
  triangle.setIndex([0,1,2]);
  triangle.computeVertexNormals();
  const material = new THREE.MeshBasicMaterial({side: THREE.DoubleSide});
  try {
    await check('empty', []);
    const torusAsset = scene.registerAsset(meshData(torus));
    const triangleAsset = scene.registerAsset(meshData(triangle));
    function place(geometry, asset, x, z, angle) {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(x, 0, z);
      mesh.rotation.set(0.17, angle, 0.06);
      mesh.updateMatrixWorld();
      mesh.userData.instance = scene.createInstance(asset, mesh.matrixWorld);
      return mesh;
    }
    const a = place(torus, torusAsset, 0.31, -0.05, -0.36);
    const b = place(torus, torusAsset, -0.18, 0.03, 0.64);
    const c = place(triangle, triangleAsset, -0.23, 0.25, 0.4);
    await check('overlapping bounds', [a,b,c]);
    const packed = scene.geometryBuffer;
    a.position.x += 0.53;
    a.rotation.y += 0.76;
    a.updateMatrixWorld();
    scene.setTransform(a.userData.instance, a.matrixWorld);
    await check('refit', [a,b,c]);
    if (scene.geometryBuffer !== packed) throw Error('Pose-only update replaced geometry');
    scene.removeInstance(b.userData.instance);
    await check('removed instance', [a,c]);
    if (!hits) throw Error('No triangle hits exercised');
    return {queries, hits};
  } finally {
    scene.destroy();
    torus.dispose(); triangle.dispose(); material.dispose();
  }
}

export async function run(demo) { return testRigidQueries(demo.gpu.device); }
