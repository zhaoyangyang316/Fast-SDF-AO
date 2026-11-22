// One object-space field is shared by every instance of the selected rigid asset.
// The existing TLAS and inverse rigid pose bound the same world-ray parameter t.
export const LOCAL_RIGID_SDF_WGSL = /* wgsl */`
struct RigidSdfInfo {
  minPitch: vec4<f32>,
  selected: vec4<u32>, // x = selected asset's current root in the packed BLAS arena
};
@group(1) @binding(3) var rigidSdfDistance: texture_3d<f32>;
@group(1) @binding(4) var rigidSdfMaterial: texture_3d<f32>;
@group(1) @binding(5) var rigidSdfNormal: texture_3d<f32>;
@group(1) @binding(6) var rigidSdfSampler: sampler;
@group(1) @binding(7) var<uniform> rigidSdfInfo: RigidSdfInfo;

fn rigidSdfSample(local: vec3<f32>, invSpan: vec3<f32>) -> f32 {
  let uv = (local - rigidSdfInfo.minPitch.xyz) * invSpan;
  return textureSampleLevel(rigidSdfDistance, rigidSdfSampler, uv, 0.0).r;
}

fn rigidSdfCell(address: u32) -> vec3<i32> {
  let size = textureDimensions(rigidSdfDistance);
  return vec3<i32>(vec3<u32>(address % size.x, (address / size.x) % size.y, address / (size.x * size.y)));
}

// Test the Hermite plane stored at one voxel without extending it past the nearby surface.
fn rigidSdfPlaneHit(ray: RigidRay, cell: vec3<i32>, n: vec3<f32>, offset: f32,
                    t: f32, end: f32, tMin: f32, tMax: f32, invSpan: vec3<f32>, speed: f32) -> f32 {
  let pitch = rigidSdfInfo.minPitch.w;
  let center = rigidSdfInfo.minPitch.xyz + (vec3<f32>(cell) + 0.5) * pitch;
  let denom = dot(n, ray.direction);
  if (abs(denom) <= 1e-6 * speed || abs(offset) >= 0.999) { return tMax; }
  let hitT = (dot(center - ray.origin, n) - offset * pitch) / denom;
  if (hitT < t || hitT > end || hitT <= tMin || hitT >= tMax) { return tMax; }
  let centerD = abs(textureLoad(rigidSdfDistance, cell, 0).r);
  let hitD = abs(rigidSdfSample(ray.origin + ray.direction * hitT, invSpan));
  if (centerD > (abs(offset) + 0.45) * pitch || hitD > 0.64 * pitch) { return tMax; }
  return hitT;
}

fn rigidSdfTrace(instanceIndex: u32, ray: RigidRay, tMin: f32, tMax: f32) -> RigidIntersection {
  let miss = RigidIntersection(0xffffffffu, 0u, tMax, vec2<f32>(0.0));
  let size = textureDimensions(rigidSdfDistance);
  let lo = rigidSdfInfo.minPitch.xyz;
  let pitch = rigidSdfInfo.minPitch.w;
  let span = pitch * vec3<f32>(size);
  let range = rigidBoxInterval(ray, lo, lo + span, tMin, tMax);
  let speed = length(ray.direction);
  if (range.x >= range.y || speed == 0.0) { return miss; }
  let voxelT = pitch / speed;
  let invSpan = 1.0 / span;
  var t = range.x;
  // A near-field step crosses a voxel boundary. Every far step advances >= 0.6 voxel.
  let maxSteps = 2u * (size.x + size.y + size.z) + 8u;
  for (var step = 0u; step < maxSteps && t < range.y; step++) {
    let p = ray.origin + ray.direction * t;
    let d = rigidSdfSample(p, invSpan);
    // Trilinear interpolation over a 1-Lipschitz field can overestimate distance
    // by sqrt(3)/2 voxel. Leave that margin plus room for FP16 rounding.
    if (abs(d) > 1.5 * pitch) {
      t += (0.99 * abs(d) - 0.87 * pitch) / speed;
      continue;
    }

    // In the narrow band, refine against the local normal and signed plane offset
    // baked at the nearest voxel center. Thin two-sided sheets can have positive
    // distance on both sides; requiring a sampled sign change would erase them.
    let ahead = p + ray.direction * (0.0001 * voxelT);
    let cell = clamp(vec3<i32>(floor((ahead - lo) / pitch)), vec3<i32>(0), vec3<i32>(size) - 1);
    let center = lo + (vec3<f32>(cell) + 0.5) * pitch;
    let nextFace = center + select(vec3<f32>(-0.5), vec3<f32>(0.5), ray.direction > vec3<f32>(0.0)) * pitch;
    let faceT = select((nextFace - ray.origin) * ray.inv, vec3<f32>(1e30), ray.parallel);
    let end = min(range.y, max(t + 0.0001 * voxelT, min(faceT.x, min(faceT.y, faceT.z))));
    let plane = textureLoad(rigidSdfNormal, cell, 0) * 2.0 - 1.0;
    let n = normalize(plane.xyz);
    var hitCell = cell;
    var candidate = rigidSdfPlaneHit(ray, cell, n, plane.w, t, end, tMin, tMax, invSpan, speed);
    // Both sides of a thin sheet can lie inside this voxel. An exit-facing
    // nearest plane may hide the entrance. Test the opposed plane held by the
    // neighboring sample on the incoming side and keep the closer crossing.
    if (dot(n, ray.direction) > 0.0) {
      let axis = abs(n);
      var offset = vec3<i32>(0);
      if (axis.x >= axis.y && axis.x >= axis.z) { offset.x = -i32(sign(n.x)); }
      else if (axis.y >= axis.z) { offset.y = -i32(sign(n.y)); }
      else { offset.z = -i32(sign(n.z)); }
      let other = cell + offset;
      if (all(other >= vec3<i32>(0)) && all(other < vec3<i32>(size))) {
        let otherPlane = textureLoad(rigidSdfNormal, other, 0) * 2.0 - 1.0;
        let otherN = normalize(otherPlane.xyz);
        if (dot(otherN, n) < -0.75 && dot(otherN, ray.direction) < 0.0) {
          let front = rigidSdfPlaneHit(ray, other, otherN, otherPlane.w, t, end, tMin, candidate, invSpan, speed);
          if (front < candidate) { candidate = front; hitCell = other; }
        }
      }
    }
    if (candidate == tMax) {
      // A crease or silhouette may change the nearest face within one cell.
      // If a signed solid changes side, refine its zero along the ray instead.
      let dEnd = rigidSdfSample(ray.origin + ray.direction * end, invSpan);
      if (d * dEnd < 0.0) {
        var a = t;
        var b = end;
        for (var j = 0; j < 5; j++) {
          let mid = 0.5 * (a + b);
          let dm = rigidSdfSample(ray.origin + ray.direction * mid, invSpan);
          if ((dm < 0.0) == (d < 0.0)) { a = mid; } else { b = mid; }
        }
        candidate = 0.5 * (a + b);
      }
    }
    if (candidate > tMin && candidate < tMax) {
      let c = vec3<u32>(hitCell);
      return RigidIntersection(instanceIndex, c.x + size.x * (c.y + size.y * c.z), candidate, vec2<f32>(0.0));
    }
    t = end;
  }
  return miss;
}
`;
