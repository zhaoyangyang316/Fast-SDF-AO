// Packed analytic movers use a positive sphere radius or a negative box half-extent.
export const DYNAMIC_SHAPES_WGSL = /* wgsl */`
fn moverDistance(p: vec3<f32>, shape: vec4<f32>) -> f32 {
  let local = p - shape.xyz;
  if (shape.w > 0.0) { return length(local) - shape.w; }
  let q = abs(local) + vec3<f32>(shape.w);
  return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn moverNormal(p: vec3<f32>, shape: vec4<f32>) -> vec3<f32> {
  let local = p - shape.xyz;
  if (shape.w > 0.0) { return local / max(length(local), 1e-6); }
  let a = abs(local);
  if (a.x >= a.y && a.x >= a.z) { return vec3<f32>(sign(local.x), 0.0, 0.0); }
  if (a.y >= a.z) { return vec3<f32>(0.0, sign(local.y), 0.0); }
  return vec3<f32>(0.0, 0.0, sign(local.z));
}

// Return the first entry ahead of the ray. An origin inside its own rasterized
// mover must not see that mover's exit as an occluding surface.
fn moverRayEntry(origin: vec3<f32>, dir: vec3<f32>, shape: vec4<f32>) -> f32 {
  let local = origin - shape.xyz;
  if (shape.w > 0.0) {
    let b = dot(local, dir);
    let disc = b * b - dot(local, local) + shape.w * shape.w;
    if (disc < 0.0) { return -1.0; }
    return -b - sqrt(disc);
  }
  let halfExtent = -shape.w;
  let parallel = abs(dir) < vec3<f32>(1e-8);
  if (any(parallel & (abs(local) > vec3<f32>(halfExtent)))) { return -1.0; }
  let safeDir = select(dir, vec3<f32>(1.0), parallel);
  let a = (-vec3<f32>(halfExtent) - local) / safeDir;
  let b = (vec3<f32>(halfExtent) - local) / safeDir;
  let lo = select(min(a, b), vec3<f32>(-1e20), parallel);
  let hi = select(max(a, b), vec3<f32>(1e20), parallel);
  let entry = max(lo.x, max(lo.y, lo.z));
  return select(-1.0, entry, entry <= min(hi.x, min(hi.y, hi.z)));
}

fn closestMoverContact(origin: vec3<f32>, dir: vec3<f32>, limit: f32) -> vec4<f32> {
  var hit = vec4<f32>(0.0, 0.0, 0.0, limit);
  for (var i = 0u; i < 4u; i++) {
    let shape = u.dynPosRad[i];
    if (shape.w == 0.0) { continue; }
    let t = moverRayEntry(origin, dir, shape);
    if (t > 0.0 && t < hit.w) { hit = vec4<f32>(moverNormal(origin + dir * t, shape), t); }
  }
  return hit;
}
`;
