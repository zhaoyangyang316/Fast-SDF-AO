// A single-bounce reflection of the static mesh in a horizontal plane. Projection
// reflects vertices, while lighting, reprojection and clipping use their true world positions.
const shader = /* wgsl */`
override planeZ: f32 = 0.0;

// Binary-compatible prefix of the composite uniform.
struct LightingUniform {
  viewProj: mat4x4<f32>, eye: vec4<f32>, sunDir: vec4<f32>,
  orbPos: vec4<f32>, orbColor: vec4<f32>, screen: vec4<f32>,
  sdfMin: vec4<f32>, sdfInvSpan: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: LightingUniform;
@group(0) @binding(1) var gPosDepth: texture_2d<f32>;
@group(0) @binding(2) var gNormal: texture_2d<f32>;
@group(0) @binding(3) var aoGI: texture_2d<f32>;
@group(0) @binding(4) var shadowMeta: texture_2d<f32>;
@group(0) @binding(5) var sdf: texture_3d<f32>;
@group(0) @binding(6) var radiance: texture_3d<f32>;
@group(0) @binding(7) var linearSampler: sampler;

struct Vertex {
  @builtin(position) clip: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) colorRough: vec4<f32>,
};
fn reflected(p: vec3<f32>) -> vec3<f32> { return vec3<f32>(p.xy, 2.0 * planeZ - p.z); }
@vertex fn vs(@location(0) p: vec3<f32>, @location(1) n: vec3<f32>,
              @location(2) cr: vec4<f32>) -> Vertex {
  return Vertex(u.viewProj * vec4<f32>(reflected(p), 1.0), p, n, cr);
}

struct Lighting {
  giAO: vec4<f32>, shadows: vec2<f32>, weight: f32,
};
fn lookupLighting(p: vec3<f32>, n: vec3<f32>, metal: f32) -> Lighting {
  var result = Lighting(vec4<f32>(0.0), vec2<f32>(0.0), 0.0);
  let clip = u.viewProj * vec4<f32>(p, 1.0);
  if (clip.w <= 0.0 || clip.z < 0.0 || clip.z > clip.w) { return result; }
  let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
  if (any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0))) { return result; }
  let dims = textureDimensions(gPosDepth);
  let coord = uv * vec2<f32>(dims) - 0.5;
  let base = vec2<i32>(floor(coord));
  let subpixel = fract(coord);
  let eyeDist = length(u.eye.xyz - p);
  let primaryView = (u.eye.xyz - p) / max(0.001, eyeDist);
  let maxPlaneDistance = 0.01 + 0.0025 * eyeDist;
  // Adjacent texels on a grazing surface have a larger footprint in world space.
  let maxDistance = 0.025 + 2.5 * eyeDist / (f32(dims.y) * max(0.12, abs(dot(n, primaryView))));
  for (var i = 0; i < 4; i++) {
    let offset = vec2<i32>(i & 1, i >> 1);
    let coordI = clamp(base + offset, vec2<i32>(0), vec2<i32>(dims) - 1);
    let pos = textureLoad(gPosDepth, coordI, 0);
    let normalTag = textureLoad(gNormal, coordI, 0);
    let delta = pos.xyz - p;
    if (pos.w <= 0.0 || abs(normalTag.w - (1.0 + metal)) > 0.05 ||
        dot(normalTag.xyz, n) < 0.88 || abs(dot(n, delta)) > maxPlaneDistance ||
        dot(delta, delta) > maxDistance * maxDistance) { continue; }
    let w = select(1.0 - subpixel, subpixel, offset == vec2<i32>(1));
    result.giAO += (w.x * w.y) * textureLoad(aoGI, coordI, 0);
    result.shadows += (w.x * w.y) * textureLoad(shadowMeta, coordI, 0).xy;
    result.weight += w.x * w.y;
  }
  if (result.weight > 0.0) {
    result.giAO /= result.weight;
    result.shadows /= result.weight;
  }
  return result;
}

fn volumeUV(p: vec3<f32>) -> vec3<f32> { return (p - u.sdfMin.xyz) * u.sdfInvSpan.xyz; }
fn distanceToScene(p: vec3<f32>) -> f32 {
  let uv = volumeUV(p);
  if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return 1e4; }
  return textureSampleLevel(sdf, linearSampler, uv, 0.0).r;
}
fn shadow(p: vec3<f32>, n: vec3<f32>, dir: vec3<f32>, limit: f32) -> f32 {
  let start = p + n * 0.035;
  var t = 0.06;
  var visibility = 1.0;
  for (var i = 0u; i < 28u; i++) {
    if (t >= limit) { break; }
    let d = distanceToScene(start + t * dir);
    if (d < 0.012) { return 0.0; }
    visibility = min(visibility, 10.0 * d / t);
    t += max(d, 0.04);
  }
  return clamp(visibility, 0.0, 1.0);
}

// Estimate incident diffuse light with two fixed hemisphere directions. The
// radiance volume is sampled at the other end of a bounce, never as this
// triangle's color. Fixed samples avoid temporal noise on unseen surfaces.
fn incoming(p: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
  var t = 0.16;
  for (var i = 0u; i < 13u; i++) {
    let q = p + t * dir;
    let d = distanceToScene(q);
    if (d < 0.045) {
      return textureSampleLevel(radiance, linearSampler, volumeUV(q), 0.0).rgb * u.sdfMin.w;
    }
    t += max(d, 0.08);
    if (t > 7.0) { break; }
  }
  return vec3<f32>(0.18, 0.22, 0.28) * max(0.0, dir.z);
}
fn estimateLighting(p: vec3<f32>, n: vec3<f32>, sun: vec3<f32>) -> Lighting {
  let start = p + 0.035 * n;
  let axis = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
  let tangent = normalize(cross(axis, n));
  let oblique = tangent * 0.8 + cross(n, tangent) * 0.6;
  let gi = 0.5 * (incoming(start, 0.8 * n + 0.6 * oblique) +
                  incoming(start, 0.8 * n - 0.6 * oblique));
  let nearAO = clamp(distanceToScene(p + 0.18 * n) / 0.18, 0.0, 1.0);
  let farAO = clamp(distanceToScene(p + 0.55 * n) / 0.55, 0.0, 1.0);
  var sunShadow = 1.0;
  if (dot(n, sun) > 0.0) { sunShadow = shadow(p, n, sun, 15.0); }
  let toOrb = u.orbPos.xyz - p;
  let orbDist = length(toOrb);
  var orbShadow = 1.0;
  if (dot(u.orbColor.rgb, u.orbColor.rgb) > 1e-4 && dot(n, toOrb) > 0.0 && orbDist < 7.5) {
    orbShadow = shadow(p, n, toOrb / max(orbDist, 1e-4), orbDist - u.orbPos.w);
  }
  return Lighting(vec4<f32>(gi, mix(nearAO, farAO, 0.35)), vec2<f32>(sunShadow, orbShadow), 1.0);
}

@fragment fn fs(v: Vertex) -> @location(0) vec4<f32> {
  let p = v.world;
  // Discard the receiver itself and clip triangles that straddle the mirror.
  if (p.z <= planeZ + 0.0005) { discard; }
  let n = normalize(v.normal);
  let albedo = v.colorRough.rgb;
  let metal = select(0.0, 1.0, v.colorRough.a < 0.0);
  let rough = clamp(abs(v.colorRough.a), 0.03, 1.0);
  let sun = normalize(u.sunDir.xyz);
  var lighting = lookupLighting(p, n, metal);
  if (lighting.weight < 0.5) {
    let fallback = estimateLighting(p, n, sun);
    let reuse = smoothstep(0.0, 0.5, lighting.weight);
    lighting.giAO = mix(fallback.giAO, lighting.giAO, reuse);
    lighting.shadows = mix(fallback.shadows, lighting.shadows, reuse);
  }
  let virtualToSurface = p - reflected(u.eye.xyz);
  let rayDistance = length(virtualToSurface);
  let view = -virtualToSurface / max(1e-4, rayDistance);
  let ao = clamp(lighting.giAO.a, 0.0, 1.0);
  let sunNoL = max(0.0, dot(n, sun));
  if (u.sunDir.w < 0.5) {
    return vec4<f32>(albedo * (0.26 * sunNoL + 0.74 * ao), rayDistance);
  }
  let ground = vec3<f32>(0.30, 0.27, 0.24);
  let sky = vec3<f32>(0.68, 0.79, 0.96);
  let ambient = mix(ground, sky, n.z * 0.5 + 0.5);
  let a = 2.0404 * albedo - 0.3324;
  let b = -4.7951 * albedo + 0.6417;
  let c = 2.7552 * albedo + 0.6903;
  let multiBounceAO = max(vec3<f32>(ao), ((ao * a + b) * ao + c) * ao);
  let sunColor = vec3<f32>(2.25, 2.10, 1.85);
  let diffuse = albedo * (1.0 - metal);
  var color = diffuse * (sunColor * sunNoL * lighting.shadows.x + lighting.giAO.rgb + ambient * multiBounceAO * 0.16);
  if (rough < 0.72 || metal > 0.5) {
    let noV = clamp(dot(n, view), 0.001, 1.0);
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - noV, 5.0);
    let specOcc = clamp(pow(noV + ao, exp2(-16.0 * rough - 1.0)) - 1.0 + ao, 0.0, 1.0);
    // Secondary glossy surfaces have a low-frequency environment, avoiding recursive mirrors.
    let env = mix(ground, sky, reflect(-view, n).z * 0.5 + 0.5) * 0.16;
    let fade = mix((1.0 - smoothstep(0.12, 0.72, rough)) * 1.35, 1.05, metal);
    color += fresnel * env * specOcc * fade;
    if (sunNoL > 0.001) {
      let noH = max(dot(n, normalize(view + sun)), 0.0);
      let alpha = max(rough * rough, 0.004);
      let a2 = alpha * alpha;
      let denom = noH * noH * (a2 - 1.0) + 1.0;
      let d = a2 / (3.14159265 * denom * denom + 1e-5);
      color += sunColor * fresnel * (min(d * 0.22, 6.0) * sunNoL * lighting.shadows.x * specOcc);
    }
  }
  let toOrb = u.orbPos.xyz - p;
  let dist2 = dot(toOrb, toOrb);
  if (dist2 < 7.5 * 7.5) {
    let noL = max(dot(n, toOrb / max(sqrt(dist2), 1e-4)), 0.0);
    color += diffuse * u.orbColor.rgb * (2.6 * noL * lighting.shadows.y / (0.28 + dist2));
  }
  return vec4<f32>(color, rayDistance);
}
`;

export class PlanarReflectionPass {
  constructor(device, width, height, planeZ = 0) {
    this.device = device;
    this.groups = [];
    const size = [width, height];
    this.texture = device.createTexture({
      label: 'Planar static radiance and virtual-eye distance', size, format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.view = this.texture.createView();
    this.depth = device.createTexture({
      label: 'Planar static depth', size, format: 'depth24plus', usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.depthView = this.depth.createView();
    const fragment = GPUShaderStage.FRAGMENT;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | fragment, buffer: { type: 'uniform' } },
        ...[1, 2, 3, 4].map(binding => ({
          binding, visibility: fragment, texture: { sampleType: 'unfilterable-float' },
        })),
        ...[5, 6].map(binding => ({
          binding, visibility: fragment, texture: { sampleType: 'float', viewDimension: '3d' },
        })),
        { binding: 7, visibility: fragment, sampler: { type: 'filtering' } },
      ],
    });
    const module = device.createShaderModule({ label: 'Static planar reflection', code: shader });
    this.pipeline = device.createRenderPipeline({
      label: 'Static planar reflection', layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: {
        module, entryPoint: 'vs', constants: { planeZ },
        buffers: [{ arrayStride: 40, attributes: [
          { shaderLocation: 0, offset: 0, format: 'float32x3' },
          { shaderLocation: 1, offset: 12, format: 'float32x3' },
          { shaderLocation: 2, offset: 24, format: 'float32x4' },
        ] }],
      },
      fragment: { module, entryPoint: 'fs', constants: { planeZ }, targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list', frontFace: 'cw', cullMode: 'back' },
      depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
    });
  }

  encode(encoder, scene, { uniform, sampler, gPosDepth, gNormal, aoGI, meta, radVol }) {
    if (this.scene !== scene) {
      this.scene = scene;
      this.groups = [];
    }
    const resources = [uniform, gPosDepth, gNormal, aoGI, meta, scene.sdfTex, radVol, sampler];
    let cached = this.groups.find(group => resources.every((resource, i) => resource === group.resources[i]));
    if (!cached) {
      const entries = [
        { binding: 0, resource: { buffer: uniform } },
        ...resources.slice(1, 7).map((texture, i) => ({ binding: i + 1, resource: texture.createView() })),
        { binding: 7, resource: sampler },
      ];
      const bindGroup = this.device.createBindGroup({ label: 'Planar lighting inputs', layout: this.layout, entries });
      cached = { resources, bindGroup };
      this.groups.push(cached);
    }
    const pass = encoder.beginRenderPass({
      label: 'Static planar reflection',
      colorAttachments: [{
        view: this.view, loadOp: 'clear', storeOp: 'store',
        clearValue: { r: 0.01564376, g: 0.01985087, b: 0.02900187, a: 0 },
      }],
      depthStencilAttachment: { view: this.depthView, depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'discard' },
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, cached.bindGroup);
    pass.setVertexBuffer(0, scene.vb);
    pass.setIndexBuffer(scene.ib, 'uint32');
    pass.drawIndexed(scene.indexCount);
    pass.end();
  }

  destroy() {
    this.texture.destroy();
    this.depth.destroy();
    this.groups = [];
    this.scene = null;
  }
}
