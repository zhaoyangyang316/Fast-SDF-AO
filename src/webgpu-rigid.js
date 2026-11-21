import { buildLocalTriangleBVH } from '../contact-bvh.js';

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
const INTERNAL = 0xffffffff;
const NODE_WORDS = 8;
const INSTANCE_WORDS = 20;
const TRIANGLE_WORDS = 36;

function emptyBounds() {
  return [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
}

function include(bounds, other) {
  for (let axis = 0; axis < 3; axis++) {
    bounds[axis] = Math.min(bounds[axis], other[axis]);
    bounds[axis + 3] = Math.max(bounds[axis + 3], other[axis + 3]);
  }
  return bounds;
}

function writeNode(words, floats, index, bounds, escape, payload) {
  const at = index * NODE_WORDS;
  for (let axis = 0; axis < 3; axis++) {
    floats[at + axis] = bounds[axis];
    floats[at + 4 + axis] = bounds[axis + 3];
  }
  words[at + 3] = escape;
  words[at + 7] = payload;
}

function rigidMatrix(value) {
  const source = value?.elements ?? value;
  if (source?.length !== 16) throw new Error('Expected a column-major 4x4 rigid transform');
  const m = Float32Array.from(source);
  const epsilon = 1e-4;
  if (!m.every(Number.isFinite) || m[3] !== 0 || m[7] !== 0 || m[11] !== 0 || m[15] !== 1) {
    throw new Error('Expected a finite affine rigid transform');
  }
  for (let col = 0; col < 3; col++) {
    for (let other = col; other < 3; other++) {
      let dot = 0;
      for (let row = 0; row < 3; row++) dot += m[col * 4 + row] * m[other * 4 + row];
      if (Math.abs(dot - Number(col === other)) > epsilon) throw new Error('Rigid transforms support rotation and translation only');
    }
  }
  const det = m[0] * (m[5] * m[10] - m[6] * m[9])
            + m[1] * (m[6] * m[8] - m[4] * m[10])
            + m[2] * (m[4] * m[9] - m[5] * m[8]);
  if (det < 0) throw new Error('Reflected rigid transforms are unsupported');
  return m;
}

function worldBounds(local, m) {
  const bounds = emptyBounds();
  for (let row = 0; row < 3; row++) {
    let lo = m[12 + row], hi = lo, magnitude = Math.abs(lo);
    for (let axis = 0; axis < 3; axis++) {
      const a = m[axis * 4 + row] * local[axis], b = m[axis * 4 + row] * local[axis + 3];
      lo += Math.min(a, b);
      hi += Math.max(a, b);
      magnitude += Math.max(Math.abs(a), Math.abs(b));
    }
    // Conservatively cover rounding in the float32 vertex transform, including cancellation.
    const pad = Math.max(1, magnitude) * 5e-7;
    bounds[row] = lo - pad;
    bounds[row + 3] = hi + pad;
  }
  return bounds;
}

function asAttribute(values, size, name) {
  if (values?.length !== size) throw new Error('Invalid rigid mesh ' + name + ' length');
  const a = values instanceof Float32Array ? values : Float32Array.from(values);
  if (!a.every(Number.isFinite)) throw new Error('Non-finite rigid mesh ' + name);
  return a;
}

function buildAsset(data, remainingBytes, bufferLimit) {
  const count = data.positions?.length / 3;
  if (!Number.isInteger(count) || count < 1) throw new Error('Expected rigid mesh positions');
  const indexCount = data.indices?.length;
  if (!Number.isInteger(indexCount) || indexCount < 3 || indexCount % 3) throw new Error('Expected indexed rigid mesh triangles');
  // Check the upload sizes before copying attributes, even on meshes with
  // many unreferenced vertices and a tiny index buffer.
  if (count * 40 > bufferLimit || indexCount * 4 > bufferLimit) throw new RangeError('Rigid mesh exceeds device buffer limit');
  const triBytes = indexCount / 3 * TRIANGLE_WORDS * 4;
  if (triBytes + NODE_WORDS * 4 > remainingBytes) throw new RangeError('Rigid scene exceeds device storage limit');
  const positions = asAttribute(data.positions, count * 3, 'positions');
  const normals = asAttribute(data.normals, count * 3, 'normals');
  const colors = asAttribute(data.colors, count * 3, 'colors');
  const rough = data.roughness == null ? null : asAttribute(data.roughness, count, 'roughness');
  for (const id of data.indices) {
    if (!Number.isInteger(id) || id < 0 || id >= count) throw new RangeError('Rigid mesh vertex index out of range');
  }
  const indices = Uint32Array.from(data.indices);
  const tree = buildLocalTriangleBVH(positions, indices);
  const contactNodeCount = tree.nodes.length / 4;
  const nodeCount = contactNodeCount === 1 ? 1 : contactNodeCount + 1;
  if (triBytes + nodeCount * NODE_WORDS * 4 > remainingBytes) throw new RangeError('Rigid scene exceeds device storage limit');
  const vertices = new Float32Array(count * 10);
  for (let i = 0; i < count; i++) {
    vertices.set(positions.subarray(i * 3, i * 3 + 3), i * 10);
    vertices.set(normals.subarray(i * 3, i * 3 + 3), i * 10 + 3);
    vertices.set(colors.subarray(i * 3, i * 3 + 3), i * 10 + 6);
    vertices[i * 10 + 9] = rough ? rough[i] : 0.5;
  }
  const triangles = new Uint32Array(tree.order.length * TRIANGLE_WORDS);
  const triFloats = new Float32Array(triangles.buffer);
  for (let i = 0; i < tree.order.length; i++) {
    const source = tree.order[i];
    for (let j = 0; j < 3; j++) {
      const v = indices[source * 3 + j] * 10, offset = i * TRIANGLE_WORDS + j * 12;
      triFloats.set(vertices.subarray(v, v + 3), offset);
      triFloats[offset + 3] = vertices[v + 9];
      triFloats.set(vertices.subarray(v + 3, v + 6), offset + 4);
      triFloats.set(vertices.subarray(v + 6, v + 9), offset + 8);
    }
    triangles[i * TRIANGLE_WORDS + 7] = source;
  }

  // The contact builder emits adjacent children with an implicit root. Repack
  // that SAH topology in preorder with one escape link per node for long rays.
  const nodes = new Uint32Array(nodeCount * NODE_WORDS);
  const floats = new Float32Array(nodes.buffer);
  let cursor = 0;
  function emit(index) {
    const node = cursor++, meta = index < 0 ? 0 : tree.nodes[index * 4 + 3];
    let bounds = emptyBounds(), payload = 0;
    if (meta & 0x80000000) {
      payload = meta & 0x7fffffff;
      const first = payload >>> 3, end = first + (payload & 7);
      for (let i = first; i < end; i++) {
        for (let j = 0; j < 3; j++) {
          const at = i * TRIANGLE_WORDS + j * 12;
          for (let a = 0; a < 3; a++) {
            bounds[a] = Math.min(bounds[a], triFloats[at + a]);
            bounds[a + 3] = Math.max(bounds[a + 3], triFloats[at + a]);
          }
        }
      }
    } else {
      bounds = include(emit(meta), emit(meta + 1));
    }
    writeNode(nodes, floats, node, bounds, cursor, payload);
    return bounds;
  }
  const bounds = emit(contactNodeCount === 1 ? 0 : -1);
  return { vertices, indices, nodes, triangles, bounds };
}

function upload(device, label, data, usage) {
  const buf = device.createBuffer({ label, size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  } catch (err) {
    buf.destroy();
    throw err;
  }
}

// Assets own immutable mesh data. Instances own a rigid pose and a stable ID.
// prepare() commits pending changes once before drawing or submitting ray queries.
export class RigidMeshScene {
  constructor(device) {
    this.device = device;
    this.maxBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.stats = { assetBuilds: 0, tlasBuilds: 0, tlasRefits: 0, geometryBytesUploaded: 0, poseBytesUploaded: 0, tlasBytesUploaded: 0 };
    this.assets = new Map();
    this.instances = new Map();
    this.nextAsset = 1;
    this.nextInstance = 1;
    this.geometryBytes = 0;
    this.bindingsPending = false;
    this.trimBuffers = false;
    this.geometryBuffer = this._storage('geometry', 32);
    this.instanceBuffer = this._storage('instances', INSTANCE_WORDS * 4);
    this.tlasBuffer = this._storage('TLAS', NODE_WORDS * 4);
    this.dirtyPoses = new Set();
    this.geometryDirty = false;
    this.topologyDirty = false;
    this.slots = [];
    this.tree = [];
  }

  get count() { return this.instances.size; }

  getInstanceCountForAsset(id) {
    return this.assets.get(id)?.users ?? 0;
  }

  _storage(label, size) {
    if (size > this.maxBytes) throw new RangeError('Rigid ' + label + ' exceeds device storage limit');
    return this.device.createBuffer({ label: 'Rigid ' + label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  }

  _resize(field, size) {
    if (this[field].size === size || (this[field].size > size && !this.trimBuffers)) return;
    const next = this._storage(field, size);
    this[field].destroy();
    this[field] = next;
    // Keep the notification until all uploads commit, including across a retry.
    this.bindingsPending = true;
  }

  registerAsset(data) {
    const asset = buildAsset(data, this.maxBytes - this.geometryBytes, this.device.limits.maxBufferSize);
    const { vertices, indices, ...stored } = asset;
    const vb = upload(this.device, 'Rigid vertices', vertices, GPUBufferUsage.VERTEX);
    let ib;
    try { ib = upload(this.device, 'Rigid indices', indices, GPUBufferUsage.INDEX); }
    catch (err) { vb.destroy(); throw err; }
    const id = this.nextAsset++;
    this.assets.set(id, { ...stored, vb, ib, indexCount: indices.length, users: 0 });
    this.geometryBytes += stored.nodes.byteLength + stored.triangles.byteLength;
    this.stats.assetBuilds++;
    this.geometryDirty = true;
    return id;
  }

  releaseAsset(id) {
    const asset = this.assets.get(id);
    if (!asset) throw new Error('Unknown rigid asset');
    if (asset.users) throw new Error('Remove rigid instances before releasing their asset');
    asset.vb.destroy();
    asset.ib.destroy();
    this.assets.delete(id);
    this.geometryBytes -= asset.nodes.byteLength + asset.triangles.byteLength;
    this.geometryDirty = true;
    // Recover substantial released capacity; leave room for minor membership churn.
    if (this.geometryBytes <= this.geometryBuffer.size / 2) this.trimBuffers = true;
    if (!this.assets.size) this.clear();
  }

  createInstance(assetId, matrix = IDENTITY) {
    const asset = this.assets.get(assetId);
    if (!asset) throw new Error('Unknown rigid asset');
    if ((this.count + 1) * INSTANCE_WORDS * 4 > this.maxBytes || this.nextInstance > 0xffffffff) {
      throw new RangeError('Rigid instance limit exceeded');
    }
    const transform = rigidMatrix(matrix);
    const id = this.nextInstance++;
    this.instances.set(id, { asset, id, transform, bounds: worldBounds(asset.bounds, transform) });
    asset.users++;
    this.topologyDirty = true;
    return id;
  }

  setTransform(id, matrix) {
    const instance = this.instances.get(id);
    if (!instance) throw new Error('Unknown rigid instance');
    const transform = rigidMatrix(matrix);
    if (transform.every((v, i) => v === instance.transform[i])) return;
    instance.transform = transform;
    instance.bounds = worldBounds(instance.asset.bounds, transform);
    this.dirtyPoses.add(instance);
  }

  removeInstance(id) {
    const instance = this.instances.get(id);
    if (!instance) throw new Error('Unknown rigid instance');
    instance.asset.users--;
    this.instances.delete(id);
    this.dirtyPoses.delete(instance);
    this.topologyDirty = true;
  }

  clear() {
    for (const asset of this.assets.values()) { asset.vb.destroy(); asset.ib.destroy(); }
    const hadAssets = this.assets.size > 0;
    const hadInstances = this.instances.size > 0;
    this.assets.clear();
    this.geometryBytes = 0;
    this.instances.clear();
    this.slots = [];
    this.tree = [];
    this.trimBuffers = true;
    this.dirtyPoses.clear();
    this.geometryDirty ||= hadAssets || this.geometryBuffer.size > 32;
    this.topologyDirty ||= hadInstances || this.instanceBuffer.size > INSTANCE_WORDS * 4 || this.tlasBuffer.size > NODE_WORDS * 4;
  }

  destroy() {
    this.clear();
    this.geometryBuffer.destroy();
    this.instanceBuffer.destroy();
    this.tlasBuffer.destroy();
  }

  _uploadGeometry() {
    let nodeWords = 0;
    for (const a of this.assets.values()) nodeWords += a.nodes.length;
    this._resize('geometryBuffer', Math.max(32, this.geometryBytes));
    let node = 0, tri = nodeWords;
    for (const a of this.assets.values()) {
      a.root = node / NODE_WORDS;
      a.end = a.root + a.nodes.length / NODE_WORDS;
      a.triBase = tri / 4;
      // Write large immutable triangle blocks directly, avoiding a second CPU
      // copy of the entire scene during membership changes.
      const nodes = a.root === 0 ? a.nodes : a.nodes.slice();
      if (a.root) {
        for (let i = 3; i < nodes.length; i += NODE_WORDS) nodes[i] += a.root;
      }
      this.device.queue.writeBuffer(this.geometryBuffer, node * 4, nodes);
      this.device.queue.writeBuffer(this.geometryBuffer, tri * 4, a.triangles);
      this.stats.geometryBytesUploaded += nodes.byteLength + a.triangles.byteLength;
      node += nodes.length;
      tri += a.triangles.length;
    }
  }

  _buildTLAS() {
    const tree = this.tree = [];
    this.slots = Array.from(this.instances.values());
    this.slots.forEach((instance, slot) => { instance.slot = slot; });
    function build(group, parent) {
      const index = tree.length;
      const node = { parent, payload: INTERNAL, bounds: emptyBounds() };
      tree.push(node);
      if (group.length === 1) {
        const i = group[0];
        i.leaf = index;
        node.payload = i.slot;
        node.bounds = i.bounds;
      } else {
        for (const i of group) include(node.bounds, i.bounds);
        let axis = 0;
        for (let a = 1; a < 3; a++) {
          if (node.bounds[a + 3] - node.bounds[a] > node.bounds[axis + 3] - node.bounds[axis]) axis = a;
        }
        group.sort((a,b) => (a.bounds[axis] + a.bounds[axis + 3]) - (b.bounds[axis] + b.bounds[axis + 3]));
        const mid = Math.floor(group.length / 2);
        node.left = build(group.slice(0, mid), index);
        node.right = build(group.slice(mid), index);
      }
      node.escape = tree.length;
      return index;
    }
    if (this.slots.length) build(this.slots.slice(), -1);
    this.tlasData = new Uint32Array(Math.max(1, tree.length) * NODE_WORDS);
    this.tlasFloats = new Float32Array(this.tlasData.buffer);
    for (let n = 0; n < tree.length; n++) this._encodeNode(n);
  }

  _encodeNode(index) {
    const n = this.tree[index];
    writeNode(this.tlasData, this.tlasFloats, index, n.bounds, n.escape, n.payload);
  }

  _refitTLAS() {
    const changed = new Set();
    for (const i of this.dirtyPoses) {
      this.tree[i.leaf].bounds = i.bounds;
      for (let n = i.leaf; n >= 0 && !changed.has(n); n = this.tree[n].parent) changed.add(n);
    }
    const ordered = [...changed].sort((a,b) => b - a);
    for (const index of ordered) {
      const node = this.tree[index];
      if (node.payload === INTERNAL) node.bounds = include(this.tree[node.left].bounds.slice(), this.tree[node.right].bounds);
      this._encodeNode(index);
    }
    return ordered.reverse();
  }

  prepare() {
    if (!this.geometryDirty && !this.topologyDirty && !this.dirtyPoses.size) return { changed: false, bindingsChanged: false };
    if (this.geometryDirty) this._uploadGeometry();
    if (this.topologyDirty) {
      this._buildTLAS();
      this._resize('tlasBuffer', this.tlasData.byteLength);
      this.device.queue.writeBuffer(this.tlasBuffer, 0, this.tlasData);
      this.stats.tlasBuilds++;
      this.stats.tlasBytesUploaded += this.tlasData.byteLength;
    } else if (this.dirtyPoses.size) {
      this.stats.tlasRefits++;
      this.stats.tlasBytesUploaded += this._uploadRanges(this.tlasBuffer, this.tlasData, NODE_WORDS, this._refitTLAS());
    }
    if (this.topologyDirty) {
      this.instanceData = new Uint32Array(Math.max(1, this.count) * INSTANCE_WORDS);
      this.instanceFloats = new Float32Array(this.instanceData.buffer);
      this._resize('instanceBuffer', this.instanceData.byteLength);
    }
    const all = this.geometryDirty || this.topologyDirty;
    const movers = all ? this.slots : this.dirtyPoses;
    for (const i of movers) {
      const at = i.slot * INSTANCE_WORDS;
      this.instanceFloats.set(i.transform, at);
      this.instanceData.set([i.asset.root, i.asset.end, i.asset.triBase, i.id], at + 16);
    }
    if (this.count && all) {
      this.device.queue.writeBuffer(this.instanceBuffer, 0, this.instanceData);
      this.stats.poseBytesUploaded += this.instanceData.byteLength;
    } else if (this.dirtyPoses.size) {
      this.stats.poseBytesUploaded += this._uploadRanges(this.instanceBuffer, this.instanceData, INSTANCE_WORDS, [...this.dirtyPoses].map(i => i.slot).sort((a,b) => a-b));
    }
    this.dirtyPoses.clear();
    this.topologyDirty = this.geometryDirty = false;
    this.trimBuffers = false;
    const bindingsChanged = this.bindingsPending;
    this.bindingsPending = false;
    return { changed: true, bindingsChanged };
  }

  _uploadRanges(buffer, data, stride, ordered) {
    let bytes = 0;
    for (let i = 0; i < ordered.length;) {
      const first = ordered[i++];
      let last = first;
      while (i < ordered.length && ordered[i] === last + 1) last = ordered[i++];
      const slice = data.subarray(first * stride, (last + 1) * stride);
      this.device.queue.writeBuffer(buffer, first * stride * 4, slice);
      bytes += slice.byteLength;
    }
    return bytes;
  }

  draw(pass) {
    for (let first = 0; first < this.slots.length;) {
      const a = this.slots[first].asset;
      let end = first + 1;
      while (end < this.slots.length && this.slots[end].asset === a) end++;
      pass.setVertexBuffer(0, a.vb);
      pass.setIndexBuffer(a.ib, 'uint32');
      pass.drawIndexed(a.indexCount, end - first, 0, 0, first);
      first = end;
    }
  }
}
