// Real-Time Glossy / Rough SDF Specular Reflections for WebGPU Compute:
// - 1 GGX-importance-sampled reflection ray per half-res pixel (roughness < 0.72)
// - Hybrid trace: Storage-buffer Dual-Child SAH BVH2 (near contact) -> FP16 SDF (far) -> radVol + direct Emissive Orb hit
// - Temporal reprojection EMA + roughness-guided 3x3 bilateral spatial filter

import { DYNAMIC_SHAPES_WGSL } from './webgpu-dynamic.js';
import { RIGID_MESH_WGSL } from './webgpu-rigid-wgsl.js';

// At an offscreen moving hit the BVH supplies the material directly. Evaluate
// its outgoing radiance against the current lights and geometry in world space.
export const rigidRadianceWGSL = (sampleDistance) => /* wgsl */`
fn rigidHitVisibility(p: vec3<f32>, n: vec3<f32>, l: vec3<f32>, limit: f32) -> f32 {
  let origin = p + n * 0.001;
  if (rigidAnyHit(origin, l, 0.0005, limit)) { return 0.0; }
  var t = 0.045;
  var visibility = 1.0;
  for (var i = 0u; i < 24u && t < limit; i++) {
    let d = ${sampleDistance}(origin + l * t);
    if (d < 0.008) { return 0.0; }
    visibility = min(visibility, 8.0 * d / t);
    t += max(0.018, 0.8 * d);
  }
  return clamp(visibility, 0.0, 1.0);
}

fn rigidDirectLight(hit: RigidHit, p: vec3<f32>, n: vec3<f32>, v: vec3<f32>,
                    l: vec3<f32>, limit: f32, intensity: vec3<f32>) -> vec3<f32> {
  let nl = dot(n, l);
  if (nl <= 0.0) { return vec3<f32>(0.0); }
  let metal = hit.roughness < 0.0;
  let rough = clamp(abs(hit.roughness), 0.05, 1.0);
  let nv = max(0.001, dot(n, v));
  let halfVector = (l + v) / max(length(l + v), 1e-6);
  let nh = max(0.0, dot(n, halfVector));
  let vh = max(0.0, dot(v, halfVector));
  let a2 = pow(rough, 4.0);
  let denom = nh * nh * (a2 - 1.0) + 1.0;
  let distribution = a2 / max(3.141593 * denom * denom, 1e-6);
  let k = (rough + 1.0) * (rough + 1.0) * 0.125;
  let smith = 0.25 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
  let f0 = select(vec3<f32>(0.04), hit.color, metal);
  let fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
  let diffuse = select(hit.color * (1.0 - fresnel), vec3<f32>(0.0), metal);
  let visibility = rigidHitVisibility(p, n, l, limit);
  return intensity * nl * visibility * (diffuse + fresnel * distribution * smith);
}

fn rigidRadiance(hit: RigidHit, p: vec3<f32>, dir: vec3<f32>, sunDir: vec3<f32>,
                 orb: vec4<f32>, orbColor: vec3<f32>) -> vec3<f32> {
  let n = faceForward(hit.normal, dir, hit.geometricNormal);
  let sky = mix(vec3<f32>(0.22, 0.25, 0.30), vec3<f32>(0.55, 0.65, 0.78),
                clamp(0.5 + 0.5 * n.z, 0.0, 1.0));
  var rad = hit.color * sky * select(0.58, 0.20, hit.roughness < 0.0);
  rad += rigidDirectLight(hit, p, n, -dir, normalize(sunDir), 10.5, vec3<f32>(3.10, 2.90, 2.55));
  let toOrb = orb.xyz - p;
  let d = length(toOrb);
  if (dot(orbColor, orbColor) > 1e-4 && d > 0.001 && d < 7.5) {
    rad += rigidDirectLight(hit, p, n, -dir, toOrb / d, max(0.0, d - orb.w),
                            orbColor * (2.8 / (0.32 + d * d)));
  }
  return rad;
}
`;

export const reflectionTraceWGSL = (rigid) => /* wgsl */`
${rigid ? RIGID_MESH_WGSL + rigidRadianceWGSL('sampleSDF') : ''}
${DYNAMIC_SHAPES_WGSL}
struct ReflU {
  curViewProj: mat4x4<f32>,
  sdfMin: vec4<f32>,
  sdfInvSpan: vec4<f32>,
  gridMinSize: vec4<f32>,   // xyz=gridMin, w=gridSize
  gridDimsRange: vec4<f32>, // xyz=gridDims, w=contactRange
  eyeFrame: vec4<f32>,      // xyz=eye, w=frame
  sunDir: vec4<f32>,        // xyz=sunDir, w=0
  emissivePos: vec4<f32>,   // xyz=pos, w=radius
  emissiveCol: vec4<f32>,   // rgb=col, w=radScale
  params: vec4<f32>,        // x=W, y=H, z=reflRoughCut, w=maxDist
  rootMin: vec4<f32>,
  rootSpan: vec4<f32>,
  extra: vec4<u32>,         // x=nodeCount, y=enabled(0/1), z=flags(bit0=checker,bit1=useBvh), w=maxSteps
  dynPosRad: array<vec4<f32>, 4>,
  dynColRough: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> u: ReflU;
@group(0) @binding(1) var gPosDepth: texture_2d<f32>;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var gAlbedoRough: texture_2d<f32>;
@group(0) @binding(4) var sdfTex: texture_3d<f32>;
@group(0) @binding(5) var radVol: texture_3d<f32>;
@group(0) @binding(6) var linSamp: sampler;
@group(0) @binding(7) var<storage, read> bvhNodes: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> bvhTris: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> bvhGrid: array<vec2<u32>>;
@group(0) @binding(10) var histAOGI: texture_2d<f32>;
@group(0) @binding(11) var metaTex: texture_2d<f32>;
@group(0) @binding(12) var outRefl: texture_storage_2d<rgba16float, write>;

fn ign(q: vec2<f32>) -> f32 {
  return fract(52.9829189 * fract(dot(q, vec2<f32>(0.06711056, 0.00583715))));
}

fn vdc2(bits_in: u32) -> f32 {
  var b = bits_in;
  b = (b << 16u) | (b >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return f32(b) * 2.3283064365386963e-10;
}

fn vdc3(n_in: u32) -> f32 {
  var n = n_in;
  var f = 1.0;
  var r = 0.0;
  for (var j = 0u; j < 6u; j++) {
    f /= 3.0;
    r += f * f32(n % 3u);
    n /= 3u;
  }
  return r;
}

fn sampleSDF(p: vec3<f32>) -> f32 {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  var d = 1e4;
  if (all(uv >= vec3<f32>(0.0)) && all(uv <= vec3<f32>(1.0))) {
    d = textureSampleLevel(sdfTex, linSamp, uv, 0.0).r;
  }
  if ((u.extra.z & 4u) != 0u) {
    for (var i = 0u; i < 4u; i++) {
      let pr = u.dynPosRad[i];
      if (pr.w != 0.0) {
        d = min(d, moverDistance(p, pr));
      }
    }
  }
  return d;
}

fn sampleRad(p: vec3<f32>) -> vec3<f32> {
  let uv = (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz;
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return vec3<f32>(0.0); }
  return textureSampleLevel(radVol, linSamp, uv, 0.0).rgb * u.emissiveCol.w;
}

// Project exact 3D hit point into the current G-buffer: if visible on screen and facing the ray,
// shade with crisp pixel-resolution albedo + direct sun/orb + GI (eliminating 12.5 cm radVol blockiness);
// otherwise seamlessly fall back to 360-degree 2-bounce radVol!
fn evalHitRadiance(hitP: vec3<f32>, rayDir: vec3<f32>) -> vec3<f32> {
  if ((u.extra.z & 4u) != 0u) {
    for (var i = 0u; i < 4u; i++) {
      let pr = u.dynPosRad[i];
      if (pr.w != 0.0 && abs(moverDistance(hitP, pr)) < 0.06) {
        let hn = moverNormal(hitP, pr);
        let sDir = normalize(u.sunDir.xyz);
        let sun = max(0.0, dot(hn, sDir));
        let skyT = clamp(hn.z * 0.5 + 0.5, 0.0, 1.0);
        return u.dynColRough[i].rgb * (vec3<f32>(2.25, 2.10, 1.85) * sun + mix(vec3<f32>(0.25, 0.23, 0.21), vec3<f32>(0.68, 0.79, 0.96), skyT) * 0.45);
      }
    }
  }
  let clip = u.curViewProj * vec4<f32>(hitP, 1.0);
  if (clip.w > 0.1) {
    let ndc = clip.xy / clip.w;
    if (abs(ndc.x) < 0.995 && abs(ndc.y) < 0.995) {
      let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
      let iq = clamp(vec2<i32>(uv * u.params.xy), vec2<i32>(0), vec2<i32>(u.params.xy) - 1);
      let gpd = textureLoad(gPosDepth, iq, 0);
      let gnr = textureLoad(gNormal, iq, 0);
      if (gpd.w > 0.0 && gnr.w > 0.5 && length(gpd.xyz - hitP) < 0.14) {
        let hn = normalize(gnr.xyz);
        if (dot(hn, -rayDir) > 0.02) {
          let halb = textureLoad(gAlbedoRough, iq, 0).rgb;
          let ag = textureLoad(histAOGI, iq, 0);
          let mt = textureLoad(metaTex, iq, 0);
          let sDir = normalize(u.sunDir.xyz);
          let sun = max(0.0, dot(hn, sDir)) * clamp(mt.x, 0.0, 1.0);
          var col = halb * (vec3<f32>(2.25, 2.10, 1.85) * sun + max(ag.rgb, vec3<f32>(0.0)) + vec3<f32>(0.08, 0.09, 0.11) * clamp(ag.a, 0.0, 1.0));
          if (dot(u.emissiveCol.rgb, u.emissiveCol.rgb) > 1e-4) {
            let toO = u.emissivePos.xyz - hitP;
            let d2 = dot(toO, toO);
            let dist = sqrt(d2);
            let ndlO = max(0.0, dot(hn, toO / max(dist, 1e-4)));
            if (ndlO > 0.001 && dist < 7.5) {
              col += halb * u.emissiveCol.rgb * (2.6 * ndlO * clamp(mt.y, 0.0, 1.0) / (0.28 + d2));
            }
          }
          return max(col, vec3<f32>(0.0));
        }
      }
    }
  }
  return sampleRad(hitP);
}

fn boxHit(nd: vec4<u32>, qInv: vec3<f32>, qOff: vec3<f32>, tMax: f32) -> f32 {
  let lo = vec3<f32>(f32(nd.x & 0xFFFFu), f32(nd.x >> 16u), f32(nd.y & 0xFFFFu));
  let hi = vec3<f32>(f32(nd.y >> 16u), f32(nd.z & 0xFFFFu), f32(nd.z >> 16u));
  let a = lo * qInv + qOff;
  let b = hi * qInv + qOff;
  let nearT = min(a, b);
  let farT = max(a, b);
  let t0 = max(0.0, max(nearT.x, max(nearT.y, nearT.z)));
  let t1 = min(tMax, min(farT.x, min(farT.y, farT.z)));
  return select(-1.0, t0, t1 >= t0);
}

fn testLeafHit(leafMeta: u32, origin: vec3<f32>, w: vec3<f32>, bestT: ptr<function, f32>) -> bool {
  let first = (leafMeta & 0x7FFFFFFFu) >> 3u;
  let count = leafMeta & 7u;
  var hit = false;
  for (var j = 0u; j < 7u; j++) {
    if (j >= count) { break; }
    let tri = (first + j) * 3u;
    let v0 = bvhTris[tri].xyz;
    let e1 = bvhTris[tri + 1u].xyz;
    let e2 = bvhTris[tri + 2u].xyz;
    let h = cross(w, e2);
    let det = dot(e1, h);
    if (abs(det) < 1e-10) { continue; }
    let invDet = 1.0 / det;
    let s = origin - v0;
    let uu = dot(s, h) * invDet;
    if (uu < 0.0 || uu > 1.0) { continue; }
    let q = cross(s, e1);
    let vv = dot(w, q) * invDet;
    if (vv < 0.0 || uu + vv > 1.0) { continue; }
    let dist = dot(e2, q) * invDet;
    if (dist > 0.0 && dist < *bestT) {
      *bestT = dist;
      hit = true;
    }
  }
  return hit;
}

fn bvhContactHit(origin: vec3<f32>, w: vec3<f32>, root: u32, cnt: u32, qMin: vec3<f32>, qScale: vec3<f32>, bestT: ptr<function, f32>) -> bool {
  *bestT = u.gridDimsRange.w;
  if (cnt == 0u) { return false; }
  let inv = vec3<f32>(1.0) / vec3<f32>(
    select(w.x, 1e-12, abs(w.x) < 1e-12),
    select(w.y, 1e-12, abs(w.y) < 1e-12),
    select(w.z, 1e-12, abs(w.z) < 1e-12)
  );
  let qInv = qScale * inv;
  let qOff = (qMin - origin) * inv;
  if (cnt == 1u) {
    let nd = bvhNodes[root];
    if (boxHit(nd, qInv, qOff, *bestT) < 0.0) { return false; }
    return testLeafHit(nd.w, origin, w, bestT);
  }
  var anyHit = false;
  var stack: array<u32, 16>;
  var sp = 0u;
  var pair = root;
  for (var visit = 0u; visit < 256u; visit++) {
    let ndL = bvhNodes[pair];
    let ndR = bvhNodes[pair + 1u];
    let tL = boxHit(ndL, qInv, qOff, *bestT);
    let tR = boxHit(ndR, qInv, qOff, *bestT);
    var hitL = tL >= 0.0 && tL < *bestT;
    var hitR = tR >= 0.0 && tR < *bestT;
    if (hitL && (ndL.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndL.w, origin, w, bestT)) { anyHit = true; }
      hitL = false;
    }
    if (hitR && (ndR.w & 0x80000000u) != 0u) {
      if (testLeafHit(ndR.w, origin, w, bestT)) { anyHit = true; }
      hitR = false;
    }
    if (hitL && hitR) {
      if (tL <= tR) {
        if (sp < 16u) { stack[sp] = ndR.w; sp += 1u; }
        pair = ndL.w;
      } else {
        if (sp < 16u) { stack[sp] = ndL.w; sp += 1u; }
        pair = ndR.w;
      }
      continue;
    }
    if (hitL) { pair = ndL.w; continue; }
    if (hitR) { pair = ndR.w; continue; }
    if (sp == 0u) { break; }
    sp -= 1u;
    pair = stack[sp];
  }
  return anyHit;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = u32(u.params.x);
  let H = u32(u.params.y);
  if (gid.x >= W || gid.y >= H) { return; }
  let q = vec2<i32>(gid.xy);

  if (u.extra.y == 0u) {
    textureStore(outRefl, q, vec4<f32>(0.0));
    return;
  }

  let pd = textureLoad(gPosDepth, q, 0);
  let gn = textureLoad(gNormal, q, 0);
  let ar = textureLoad(gAlbedoRough, q, 0);
  let isMetal = gn.w > 1.5;
  let rough = clamp(ar.a, 0.03, 1.0);
  let roughCut = u.params.z;
  if (pd.w <= 0.0 || gn.w < 0.5 || (!isMetal && rough > roughCut)) {
    textureStore(outRefl, q, vec4<f32>(0.0, 0.0, 0.0, 1.0));
    return;
  }

  let frame = u32(u.eyeFrame.w);
  // Checkerboard skip for rough surfaces (rough > 0.12): 50% fewer reflection rays with identical denoised look,
  // while mirror chrome (rough <= 0.12) always traces every pixel at full rate!
  let useChecker = (u.extra.z & 1u) != 0u;
  if (useChecker && rough > 0.12 && (((gid.x ^ gid.y) & 1u) != (frame & 1u))) {
    // Negative .w signals REFL_FILTER_WGSL to reuse temporal/spatial neighbors directly
    textureStore(outRefl, q, vec4<f32>(0.0, 0.0, 0.0, -1.0));
    return;
  }

  let p = pd.xyz;
  let n = normalize(gn.xyz);
  let V = normalize(u.eyeFrame.xyz - p);
  let px = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);

  // GGX importance-sampled half vector (exact mirror ray when rough <= 0.05 for razor-sharp chrome)
  var R = reflect(-V, n);
  if (rough > 0.05) {
    let a = rough * rough;
    let u1 = clamp(fract(ign(px) + vdc2(frame)), 1e-4, 0.92);
    let u2 = fract(ign(px + vec2<f32>(29.0, 53.0)) + vdc3(frame));
    let phi = 6.2831853 * u2;
    let cosTh = sqrt((1.0 - u1) / max(1.0 + (a * a - 1.0) * u1, 1e-5));
    let sinTh = sqrt(max(0.0, 1.0 - cosTh * cosTh));

    let ax = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.x) < 0.9);
    let tVec = normalize(cross(n, ax));
    let bVec = cross(n, tVec);
    let Hvec = normalize(tVec * (sinTh * cos(phi)) + bVec * (sinTh * sin(phi)) + n * cosTh);
    R = reflect(-V, Hvec);
    if (dot(R, n) < 0.03) {
      R = normalize(reflect(-V, n) + n * 0.05);
    }
  }

  let origin = p + n * 0.008;
  let cRange = u.gridDimsRange.w;
  var hitDist = u.params.w;
  var rad = vec3<f32>(0.0);
  var hitAny = false;

  // Check direct emissive orb sphere intersection first!
  var orbT = 1e9;
  if (dot(u.emissiveCol.rgb, u.emissiveCol.rgb) > 1e-4) {
    let oc = origin - u.emissivePos.xyz;
    let b = dot(oc, R);
    let c = dot(oc, oc) - u.emissivePos.w * u.emissivePos.w;
    let disc = b * b - c;
    if (disc >= 0.0) {
      let t0 = -b - sqrt(disc);
      if (t0 > 0.01 && t0 < hitDist) { orbT = t0; }
    }
  }

  // 1) Fast 1-Lipschitz midpoint empty-ball skip before BVH2 (disabled on Low/Very Low tiers)
  var nearHit = false;
  var nearHitT = cRange;
  let useBvh = (u.extra.z & 2u) != 0u;
  if (useBvh) {
    let dMid = sampleSDF(origin + R * (0.5 * cRange));
    if (dMid < 0.5 * cRange + 0.09) {
      var cellRoot = 0u;
      var cellCnt = 0u;
      var cellQMin = u.rootMin.xyz;
      var cellQScale = u.rootSpan.xyz * (1.0 / 65535.0);
      let gSize = u.gridMinSize.w;
      if (gSize > 0.0) {
        let gDims = vec3<i32>(u.gridDimsRange.xyz);
        let cell = vec3<i32>(floor((p - u.gridMinSize.xyz) / gSize));
        if (all(cell >= vec3<i32>(0)) && all(cell < gDims)) {
          let cid = u32((cell.z * gDims.y + cell.y) * gDims.x + cell.x);
          let bnd = bvhGrid[cid];
          cellRoot = bnd.x;
          cellCnt = bnd.y;
          cellQMin = u.gridMinSize.xyz + vec3<f32>(cell) * gSize - vec3<f32>(cRange + 0.002);
          cellQScale = vec3<f32>((gSize + 2.0 * (cRange + 0.002)) * (1.0 / 65535.0));
        }
      } else {
        cellCnt = u.extra.x;
      }
      nearHit = bvhContactHit(origin, R, cellRoot, cellCnt, cellQMin, cellQScale, &nearHitT);
    }
  }

  let marchStart = select(0.06, cRange, useBvh);
  if ((u.extra.z & 4u) != 0u) {
    let limit = min(nearHitT, marchStart);
    let contact = closestMoverContact(origin, R, limit);
    if (contact.w < limit) {
      nearHit = true;
      nearHitT = contact.w;
    }
  }

  ${rigid ? `let rigidLimit = min(orbT, select(u.params.w, nearHitT, nearHit));
  let rigidSurface = rigidClosestHit(origin, R, 0.0005, rigidLimit);` : ''}
  if (nearHit && nearHitT < orbT) {
    hitAny = true;
    hitDist = nearHitT;
    rad = evalHitRadiance(origin + R * nearHitT, R);
  } else {
    // 2) Relaxed far SDF sphere trace + 1-step hit refinement (tier-controlled step budget)
    var s = marchStart;
    let maxSteps = max(u.extra.w, 12u);
    for (var k = 0u; k < maxSteps; k++) {
      if (s >= orbT) { break; }
      ${rigid ? 'if (rigidSurface.valid != 0u && s >= rigidSurface.t) { break; }' : ''}
      let d = sampleSDF(origin + R * s);
      if (d > 100.0) { break; }
      if (d < 0.025) {
        // Refine hit point onto exact surface
        let sRef = max(marchStart, s + d * 0.5);
        hitAny = true;
        hitDist = sRef;
        rad = evalHitRadiance(origin + R * sRef, R);
        break;
      }
      ${rigid ? 'if (rigidSurface.valid != 0u && s >= rigidSurface.t - 0.0005) { break; }' : ''}
      s += select(max(d * 0.85, 0.035), d * 1.15, d > 0.28);
      ${rigid ? 'if (rigidSurface.valid != 0u) { s = min(s, rigidSurface.t - 0.0005); }' : ''}
      if (s > u.params.w) { break; }
    }
  }

  ${rigid ? `if (rigidSurface.valid != 0u && rigidSurface.t < hitDist) {
    hitAny = true;
    hitDist = rigidSurface.t;
    rad = rigidRadiance(rigidSurface, origin + R * hitDist, R,
                        u.sunDir.xyz, u.emissivePos, u.emissiveCol.rgb);
  }` : ''}
  if (orbT < hitDist) {
    hitAny = true;
    hitDist = orbT;
    rad = u.emissiveCol.rgb * 1.8;
  }
  if (!hitAny) {
    // Clean studio sky gradient + 3-point softbox/sun specular lobes on open ceiling / exterior miss
    let skyT = clamp(R.z * 0.5 + 0.5, 0.0, 1.0);
    let sDir = normalize(u.sunDir.xyz);
    let sunDisc = pow(max(dot(R, sDir), 0.0), mix(36.0, 5.0, rough)) * vec3<f32>(2.8, 2.5, 2.1);
    let keyBox = pow(max(dot(R, normalize(vec3<f32>(0.55, -0.62, 0.56))), 0.0), mix(10.0, 3.0, rough)) * vec3<f32>(1.15, 1.08, 0.98);
    let rimBox = pow(max(dot(R, normalize(vec3<f32>(-0.72, -0.35, 0.58))), 0.0), mix(8.0, 3.0, rough)) * vec3<f32>(0.65, 0.82, 1.05);
    let softBox = smoothstep(0.35, 0.92, R.z) * vec3<f32>(0.48, 0.54, 0.64);
    rad = mix(vec3<f32>(0.22, 0.24, 0.28), vec3<f32>(0.62, 0.72, 0.88), skyT) + sunDisc + keyBox + rimBox + softBox;
  }

  let normDist = clamp(hitDist / 6.0, 0.05, 1.0);
  textureStore(outRefl, q, vec4<f32>(rad, normDist));
}
`;

export const REFL_TRACE_WGSL = reflectionTraceWGSL(false);

export const REFL_FILTER_WGSL = /* wgsl */`
struct ReflFiltU {
  prevViewProj: mat4x4<f32>,
  params: vec4<f32>, // x=W, y=H, z=alpha, w=validPrev(0/1)
  eyePos: vec4<f32>, // xyz=eye
};

@group(0) @binding(0) var<uniform> u: ReflFiltU;
@group(0) @binding(1) var curRefl: texture_2d<f32>;
@group(0) @binding(2) var curPosDepth: texture_2d<f32>;
@group(0) @binding(3) var curNormal: texture_2d<f32>;
@group(0) @binding(4) var gAlbedoRough: texture_2d<f32>;
@group(0) @binding(5) var prevRefl: texture_2d<f32>;
@group(0) @binding(6) var prevPosDepth: texture_2d<f32>;
@group(0) @binding(7) var prevNormal: texture_2d<f32>;
@group(0) @binding(8) var outRefl: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(u.params.x);
  let H = i32(u.params.y);
  let q = vec2<i32>(gid.xy);
  if (q.x >= W || q.y >= H) { return; }

  let pd = textureLoad(curPosDepth, q, 0);
  let gn = textureLoad(curNormal, q, 0);
  let ar = textureLoad(gAlbedoRough, q, 0);
  let isMetal = gn.w > 1.5;
  let rough = clamp(ar.a, 0.03, 1.0);
  let c0 = textureLoad(curRefl, q, 0);
  if (pd.w <= 0.0 || gn.w < 0.5 || (!isMetal && rough > 0.45)) {
    textureStore(outRefl, q, c0);
    return;
  }

  let p = pd.xyz;
  let n = normalize(gn.xyz);
  let planeTol = 0.0025 * pd.w + 0.012;

  // REBLUR Specular-lite: hit-distance-guided & roughness-guided spatial filter + local moments
  // Close contact reflections (small c0.w) stay razor sharp; distant rough reflections filter smoothly!
  let isSkipped = c0.w < -0.5;
  var spSum = select(c0, vec4<f32>(0.0), isSkipped);
  var spW = select(1.0, 0.0, isSkipped);
  var m1 = select(c0.rgb, vec3<f32>(0.0), isSkipped);
  var m2 = select(c0.rgb * c0.rgb, vec3<f32>(0.0), isSkipped);
  var mCnt = select(1.0, 0.0, isSkipped);
  let normHit = select(clamp(c0.w, 0.05, 1.0), 0.35, isSkipped);
  let spGain = smoothstep(0.06, 0.30, rough) * clamp(sqrt(normHit) * 1.35, 0.45, 1.0);
  if (spGain > 0.02 || isSkipped) {
    let stepS = select(1, 2, rough > 0.24 && normHit > 0.18 && !isSkipped);
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        if (dx == 0 && dy == 0) { continue; }
        let nq = q + vec2<i32>(dx, dy) * stepS;
        if (nq.x < 0 || nq.y < 0 || nq.x >= W || nq.y >= H) { continue; }
        let npd = textureLoad(curPosDepth, nq, 0);
        let ngn = textureLoad(curNormal, nq, 0);
        if (npd.w <= 0.0 || ngn.w < 0.5) { continue; }
        if (dot(normalize(ngn.xyz), n) < 0.85) { continue; }
        if (abs(dot(n, npd.xyz - p)) > planeTol * f32(stepS) * 1.5) { continue; }
        let nar = textureLoad(gAlbedoRough, nq, 0);
        if (abs(nar.a - rough) > 0.12) { continue; }
        let nr = textureLoad(curRefl, nq, 0);
        if (nr.w < -0.5) { continue; }
        let w = select(0.45, 0.75, dx == 0 || dy == 0) * max(spGain, select(0.0, 1.0, isSkipped));
        spSum += nr * w;
        spW += w;
        m1 += nr.rgb;
        m2 += nr.rgb * nr.rgb;
        mCnt += 1.0;
      }
    }
  }
  let filtered = select(c0, spSum / spW, spW > 1e-4);
  let rMu = select(filtered.rgb, m1 / mCnt, mCnt > 0.5);
  let rSigma = select(vec3<f32>(0.15), sqrt(max(m2 / max(mCnt, 1.0) - rMu * rMu, vec3<f32>(0.0))), mCnt > 1.5);

  // Mirror chrome uses faster temporal alpha (0.35); rough chrome uses tighter EMA (0.07);
  // checkerboard-skipped pixels keep 95% temporal history for rock-solid zero-flicker stability!
  let effAlpha = select(mix(max(u.params.z, 0.35), min(u.params.z, 0.07), spGain), 0.05, isSkipped);

  if (u.params.w < 0.5) {
    textureStore(outRefl, q, filtered);
    return;
  }

  // REBLUR Virtual-Motion (Parallax) Reprojection for glossy reflections:
  // Reproject virtual reflected point p_virt = p - V * hitDist on smooth/glossy planar surfaces,
  // falling back to surface reprojection p when virtual reprojection misses geometry!
  let V = normalize(u.eyePos.xyz - p);
  let hitDist = clamp(filtered.w, 0.05, 1.0) * 6.0;
  let virtWeight = (1.0 - smoothstep(0.08, 0.34, rough)) * smoothstep(0.25, 1.2, hitDist);
  let reprojP = mix(p, p - V * hitDist, virtWeight * 0.65);

  let clip = u.prevViewProj * vec4<f32>(reprojP, 1.0);
  if (clip.w <= 0.0) {
    textureStore(outRefl, q, filtered);
    return;
  }
  let ndc = clip.xy / clip.w;
  let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  var f = uv * vec2<f32>(u.params.xy) - 0.5;
  f = select(f, round(f), abs(f - round(f)) < vec2<f32>(0.02));
  let i0 = vec2<i32>(floor(f));
  let fw = f - floor(f);

  var acc = vec4<f32>(0.0);
  var wSum = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2<i32>(k & 1, k >> 1);
    let pq = i0 + o;
    let bw = select(1.0 - fw.x, fw.x, o.x == 1) * select(1.0 - fw.y, fw.y, o.y == 1);
    if (bw < 1e-4 || pq.x < 0 || pq.y < 0 || pq.x >= W || pq.y >= H) { continue; }
    let ppd = textureLoad(prevPosDepth, pq, 0);
    let pgn = textureLoad(prevNormal, pq, 0);
    if (ppd.w <= 0.0 || pgn.w < 0.5) { continue; }
    if (dot(normalize(pgn.xyz), n) < 0.90) { continue; }
    if (abs(dot(n, ppd.xyz - p)) > planeTol * (1.0 + virtWeight * 4.0)) { continue; }
    let par = textureLoad(gAlbedoRough, pq, 0);
    if (abs(par.a - rough) > 0.14) { continue; }
    acc += textureLoad(prevRefl, pq, 0) * bw;
    wSum += bw;
  }
  // Fallback to surface-motion reprojection if virtual point fell outside matching plane
  if (wSum <= 0.01 && virtWeight > 0.05) {
    let clipS = u.prevViewProj * vec4<f32>(p, 1.0);
    if (clipS.w > 0.0) {
      let ndcS = clipS.xy / clipS.w;
      let uvS = vec2<f32>(ndcS.x * 0.5 + 0.5, 0.5 - ndcS.y * 0.5);
      var fS = uvS * vec2<f32>(u.params.xy) - 0.5;
      fS = select(fS, round(fS), abs(fS - round(fS)) < vec2<f32>(0.02));
      let iS = vec2<i32>(floor(fS));
      let fwS = fS - floor(fS);
      for (var k = 0; k < 4; k++) {
        let o = vec2<i32>(k & 1, k >> 1);
        let pq = iS + o;
        let bw = select(1.0 - fwS.x, fwS.x, o.x == 1) * select(1.0 - fwS.y, fwS.y, o.y == 1);
        if (bw < 1e-4 || pq.x < 0 || pq.y < 0 || pq.x >= W || pq.y >= H) { continue; }
        let ppd = textureLoad(prevPosDepth, pq, 0);
        let pgn = textureLoad(prevNormal, pq, 0);
        if (ppd.w <= 0.0 || pgn.w < 0.5) { continue; }
        if (dot(normalize(pgn.xyz), n) < 0.90) { continue; }
        if (abs(dot(n, ppd.xyz - p)) > planeTol) { continue; }
        let par = textureLoad(gAlbedoRough, pq, 0);
        if (abs(par.a - rough) > 0.14) { continue; }
        acc += textureLoad(prevRefl, pq, 0) * bw;
        wSum += bw;
      }
    }
  }
  if (wSum > 0.01) {
    var prevR = acc / wSum;
    // REBLUR variance-clamped reflection anti-lag (prevents moving orb / orbit ghosting)
    let rMin = rMu - rSigma * 2.2 - vec3<f32>(0.04);
    let rMax = rMu + rSigma * 2.2 + vec3<f32>(0.04);
    prevR = vec4<f32>(mix(prevR.rgb, clamp(prevR.rgb, rMin, rMax), select(0.55, 0.25, isSkipped)), prevR.w);
    textureStore(outRefl, q, mix(prevR, filtered, effAlpha));
  } else {
    textureStore(outRefl, q, filtered);
  }
}
`;
