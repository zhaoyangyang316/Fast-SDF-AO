import { LOCAL_RIGID_SDF_WGSL } from './webgpu-rigid-sdf-wgsl.js';

// The CPU scene packs both hierarchies in preorder. A missed box's escape
// index jumps past its subtree, so traversal needs no stack or visit limit.
const rigidMeshWGSL = localSdf => /* wgsl */`
struct RigidInstance {
  world: mat4x4<f32>,
  asset: vec4<u32>, // first BLAS node, exclusive end, triangle base, stable instance ID
};
struct RigidNode {
  lo: vec3<f32>,
  escape: u32,
  hi: vec3<f32>,
  payload: u32,
};
@group(1) @binding(0) var<storage, read> rigidInstances: array<RigidInstance>;
@group(1) @binding(1) var<storage, read> rigidTopNodes: array<RigidNode>;
@group(1) @binding(2) var<storage, read> rigidGeometry: array<vec4<u32>>;

fn rigidWorldPosition(instanceIndex: u32, p: vec3<f32>) -> vec3<f32> {
  return (rigidInstances[instanceIndex].world * vec4<f32>(p, 1.0)).xyz;
}

fn rigidWorldNormal(instanceIndex: u32, n: vec3<f32>) -> vec3<f32> {
  return (rigidInstances[instanceIndex].world * vec4<f32>(n, 0.0)).xyz;
}

struct RigidRay {
  origin: vec3<f32>,
  direction: vec3<f32>,
  inv: vec3<f32>,
  parallel: vec3<bool>,
};

fn rigidRay(origin: vec3<f32>, direction: vec3<f32>) -> RigidRay {
  let parallel = direction == vec3<f32>(0.0);
  return RigidRay(origin, direction, 1.0 / select(direction, vec3<f32>(1.0), parallel), parallel);
}

fn rigidBoxInterval(ray: RigidRay, lo: vec3<f32>, hi: vec3<f32>, tMin: f32, tMax: f32) -> vec2<f32> {
  if (any(ray.parallel & ((ray.origin < lo) | (ray.origin > hi)))) { return vec2<f32>(1.0, -1.0); }
  let a = (lo - ray.origin) * ray.inv;
  let b = (hi - ray.origin) * ray.inv;
  let near = select(min(a, b), vec3<f32>(tMin), ray.parallel);
  let far = select(max(a, b), vec3<f32>(tMax), ray.parallel);
  let entry = max(tMin, max(near.x, max(near.y, near.z)));
  let exitT = min(tMax, min(far.x, min(far.y, far.z)));
  return vec2<f32>(entry, exitT);
}

fn rigidOverlapsBox(ray: RigidRay, lo: vec3<f32>, hi: vec3<f32>, tMin: f32, tMax: f32) -> bool {
  let range = rigidBoxInterval(ray, lo, hi, tMin, tMax);
  return range.x <= range.y;
}

// Return (t, weight of vertex 1, weight of vertex 2). A miss returns tMax.
fn rigidTriangle(ray: RigidRay, address: u32, tMin: f32, tMax: f32) -> vec3<f32> {
  let a = bitcast<vec3<f32>>(rigidGeometry[address].xyz);
  let ab = bitcast<vec3<f32>>(rigidGeometry[address + 3u].xyz) - a;
  let ac = bitcast<vec3<f32>>(rigidGeometry[address + 6u].xyz) - a;
  let p = cross(ray.direction, ac);
  let det = dot(ab, p);
  let miss = vec3<f32>(tMax, 0.0, 0.0);
  if (det == 0.0) { return miss; }
  let s = ray.origin - a;
  let q = cross(s, ab);
  let uv = vec2<f32>(dot(s, p), dot(ray.direction, q)) * sign(det);
  let denom = abs(det);
  if (any(uv < vec2<f32>(0.0)) || uv.x + uv.y > denom) { return miss; }
  let t = dot(ac, q) / det;
  if (t <= tMin || t >= tMax) { return miss; }
  return vec3<f32>(t, uv / denom);
}

struct RigidIntersection {
  instance: u32,
  address: u32,
  t: f32,
  bary: vec2<f32>,
};

${localSdf ? LOCAL_RIGID_SDF_WGSL : ''}

fn rigidIntersect(origin: vec3<f32>, direction: vec3<f32>, tMin: f32, tMax: f32,
                  firstHit: bool) -> RigidIntersection {
  var hit = RigidIntersection(0xffffffffu, 0u, tMax, vec2<f32>(0.0));
  if (tMin >= tMax) { return hit; }
  let worldRay = rigidRay(origin, direction);
  let topEnd = rigidTopNodes[0].escape;
  var top = 0u;
  while (top < topEnd) {
    let node = rigidTopNodes[top];
    if (!rigidOverlapsBox(worldRay, node.lo, node.hi, tMin, hit.t)) {
      top = node.escape;
      continue;
    }
    top++;
    if (node.payload == 0xffffffffu) { continue; }
    let instance = rigidInstances[node.payload];
    let m = instance.world;
    let invRotation = transpose(mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz));
    let ray = rigidRay(invRotation * (origin - m[3].xyz), invRotation * direction);
    ${localSdf ? `if (instance.asset.x == rigidSdfInfo.selected.x) {
      let fieldHit = rigidSdfTrace(node.payload, ray, tMin, hit.t);
      if (fieldHit.instance != 0xffffffffu) {
        if (firstHit) { return fieldHit; }
        hit = fieldHit;
      }
      continue;
    }` : ''}
    var index = instance.asset.x;
    while (index < instance.asset.y) {
      let lower = rigidGeometry[2u * index];
      let upper = rigidGeometry[2u * index + 1u];
      if (!rigidOverlapsBox(ray, bitcast<vec3<f32>>(lower.xyz), bitcast<vec3<f32>>(upper.xyz), tMin, hit.t)) {
        index = lower.w;
        continue;
      }
      index++;
      let count = upper.w & 7u;
      let first = upper.w >> 3u;
      for (var i = 0u; i < count; i++) {
        let address = instance.asset.z + (first + i) * 9u;
        let result = rigidTriangle(ray, address, tMin, hit.t);
        if (result.x < hit.t) {
          hit = RigidIntersection(node.payload, address, result.x, result.yz);
          if (firstHit) { return hit; }
        }
      }
    }
  }
  return hit;
}

fn rigidAnyHit(origin: vec3<f32>, direction: vec3<f32>, tMin: f32, tMax: f32) -> bool {
  return rigidIntersect(origin, direction, tMin, tMax, true).instance != 0xffffffffu;
}

struct RigidHit {
  valid: u32,
  instanceId: u32,
  primitiveId: u32,
  t: f32,
  bary: vec2<f32>,
  geometricNormal: vec3<f32>,
  normal: vec3<f32>,
  color: vec3<f32>,
  roughness: f32,
};

fn rigidClosestHit(origin: vec3<f32>, direction: vec3<f32>, tMin: f32, tMax: f32) -> RigidHit {
  let intersection = rigidIntersect(origin, direction, tMin, tMax, false);
  var hit: RigidHit;
  hit.t = intersection.t;
  if (intersection.instance == 0xffffffffu) { return hit; }
  ${localSdf ? `if (rigidInstances[intersection.instance].asset.x == rigidSdfInfo.selected.x) {
    let cell = rigidSdfCell(intersection.address);
    let normal = textureLoad(rigidSdfNormal, cell, 0).xyz * 2.0 - 1.0;
    let material = textureLoad(rigidSdfMaterial, cell, 0);
    hit.valid = 1u;
    hit.instanceId = rigidInstances[intersection.instance].asset.w;
    hit.primitiveId = 0xffffffffu;
    hit.normal = normalize(rigidWorldNormal(intersection.instance, normal));
    hit.geometricNormal = hit.normal;
    hit.color = material.rgb;
    hit.roughness = material.a * 2.0 - 1.0;
    return hit;
  }` : ''}
  let a = intersection.address;
  let v0 = bitcast<vec4<f32>>(rigidGeometry[a]);
  let v1 = bitcast<vec4<f32>>(rigidGeometry[a + 3u]);
  let v2 = bitcast<vec4<f32>>(rigidGeometry[a + 6u]);
  let w = vec3<f32>(1.0 - intersection.bary.x - intersection.bary.y, intersection.bary);
  let face = normalize(cross(v1.xyz - v0.xyz, v2.xyz - v0.xyz));
  let localNormal = w.x * bitcast<vec3<f32>>(rigidGeometry[a + 1u].xyz)
                  + w.y * bitcast<vec3<f32>>(rigidGeometry[a + 4u].xyz)
                  + w.z * bitcast<vec3<f32>>(rigidGeometry[a + 7u].xyz);
  hit.valid = 1u;
  hit.instanceId = rigidInstances[intersection.instance].asset.w;
  hit.primitiveId = rigidGeometry[a + 1u].w;
  hit.bary = intersection.bary;
  hit.geometricNormal = rigidWorldNormal(intersection.instance, face);
  hit.normal = normalize(rigidWorldNormal(intersection.instance,
                         select(face, localNormal, dot(localNormal, localNormal) > 1e-16)));
  hit.color = w.x * bitcast<vec3<f32>>(rigidGeometry[a + 2u].xyz)
            + w.y * bitcast<vec3<f32>>(rigidGeometry[a + 5u].xyz)
            + w.z * bitcast<vec3<f32>>(rigidGeometry[a + 8u].xyz);
  hit.roughness = dot(w, vec3<f32>(v0.w, v1.w, v2.w));
  return hit;
}
`;

export const RIGID_MESH_WGSL = rigidMeshWGSL(false);
export const RIGID_SDF_WGSL = rigidMeshWGSL(true);
