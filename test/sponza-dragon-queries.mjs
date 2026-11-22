// python3 test/webgpu-browser.py test/sponza-dragon-queries.mjs --scene sponza-dragon --software
import * as THREE from 'three';
import {queryRigidBatch, referenceRigidHit, rigidQueryPipeline} from './rigid-queries.mjs';
import {RIGID_MESH_WGSL, RIGID_SDF_WGSL} from '../src/webgpu-rigid-wgsl.js';

const v = (x, y, z) => new THREE.Vector3(x, y, z);
const noPrimitive = 0xffffffff;

function referenceMesh(data, material, instance) {
  const geometry = new THREE.BufferGeometry();
  for (const [attr, array, size] of [['position', data.positions, 3], ['normal', data.normals, 3],
    ['color', data.colors, 3], ['roughness', data.roughness, 1]]) {
    geometry.setAttribute(attr, new THREE.BufferAttribute(array, size));
  }
  geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.instance = instance;
  mesh.matrixAutoUpdate = false;
  return mesh;
}

function samplePose(dragon, meshes) {
  const samples = [], world = new THREE.Matrix4().fromArray(dragon.matrix);
  function add(kind, origin, direction, min = 0.0005, max = 6) {
    const o = origin.clone().applyMatrix4(world), d = direction.clone().transformDirection(world);
    const ray = Float32Array.of(...o.toArray(), min, ...d.toArray(), max);
    const ref = referenceRigidHit(meshes, ray);
    samples.push({kind, ray, ref});
    return ref;
  }

  // Vertical orthographic rays through the asset bounds capture general silhouettes.
  for (let y = 0; y < 16; y++) for (let x = 0; x < 17; x++) {
    add('bounds grid', v(-2.045 + (x + .37) * 3.265 / 17, -1.79 + (y + .41) * 3.58 / 16, 2), v(0,0,-1));
  }
  // Each vertical ray enters the local bounds over an empty gap far to either side of the thin tail.
  for (const x of [-1.85, -1.56, -1.21, -.96]) for (const y of [-1.4, -.91, .68, 1.32]) {
    add('clear box entry', v(x, y, 1.8), v(0,0,-1));
  }

  const g = meshes[0].geometry, positions = g.attributes.position;
  const a = v(0,0,0), b = a.clone(), c = a.clone();
  let left = 0, right = 0;
  // Each ray crosses the interior of a wing triangle, from both sheet sides.
  // Traverse primitive IDs in a fixed interleaved order to cover different patches.
  for (let j = 0; j < g.index.count / 3 && (left < 18 || right < 18); j++) {
    const i = (j * 773 + 31) % (g.index.count / 3);
    a.fromBufferAttribute(positions, g.index.getX(3*i));
    b.fromBufferAttribute(positions, g.index.getX(3*i+1));
    c.fromBufferAttribute(positions, g.index.getX(3*i+2));
    const center = a.clone().multiplyScalar(.27).addScaledVector(b,.34).addScaledVector(c,.39);
    if (Math.abs(center.y) < .72 || (center.y < 0 ? left : right) >= 18) continue;
    const tri = new THREE.Triangle(a,b,c);
    if (tri.getArea() < 3e-5) continue;
    const normal = tri.getNormal(v(0,0,0));
    if (center.y < 0) left++; else right++;
    for (const side of [-1, 1]) {
      const dir = normal.clone().multiplyScalar(-side);
      const origin = center.clone().addScaledVector(dir, -.3);
      const hit = add('wing normal', origin, dir, .0005, .7);
      if (!hit) throw Error('Wing triangle target missed by triangle reference');
      // A 6 cm endpoint margin is larger than two voxels of the shipped asset.
      if (hit.values[0] > .07 && side === 1) {
        add('clipped before wing', origin, dir, .0005, hit.values[0] - .06);
        add('clipped after wing', origin, dir, .0005, hit.values[0] + .06);
      }
    }
    // A glancing ray stresses zero crossings in the sub-voxel wing membrane.
    const grazing = b.clone().sub(a).normalize().addScaledVector(normal,.12).normalize();
    add('grazing wing', center.clone().addScaledVector(grazing,-.31), grazing, .0005, .65);
  }
  if (left < 18 || right < 18) throw Error('Insufficient wing targets');

  // These points lie between the upper and lower triangle crossings of the torso or tail.
  for (const p of [v(.06,.045,.55), v(.02,.03,.50), v(-.8,.01,.21)]) {
    for (let k = 0; k < 11; k++) {
      const z = 1 - (k + .5) * 2 / 11, r = Math.sqrt(1 - z*z), angle = 2.39996323 * k + .37;
      const dir = v(r*Math.cos(angle), r*Math.sin(angle), z);
      const hit = add('from inside', p, dir);
      if (!hit) throw Error('Open exit from a reference interior point');
      add('clipped before exit', p, dir, .0005, hit.values[0] * .32);
    }
  }
  add('empty interval', v(.02,.03,1.6), v(0,0,-1), .7, .7);
  // For the first rays the companion sits in front of the dragon. Reversing
  // direction from within the dragon bounds makes its geometry nearer.
  for (const dy of [-.06, -.025, .018, .042]) {
    const companionHit = add('companion first', v(2.2,dy,.68), v(-1,0,0));
    const dragonHit = add('dragon first', v(-.06,dy,.68), v(1,0,0));
    if (companionHit?.id !== meshes[1].userData.instance || dragonHit?.id !== dragon.instance)
      throw Error('Mixed asset test rays missed their intended nearest object');
  }
  return samples;
}

function percentile(xs, fraction) {
  if (!xs.length) return null;
  const ordered = xs.toSorted((a,b) => a-b);
  return ordered[Math.floor((ordered.length - 1) * fraction)];
}

function compare(mode, samples, output, dragonId) {
  const groups = {}, errors = [], sdf = mode === 'sdf';
  for (const [i, {kind, ray, ref}] of samples.entries()) {
    const s = groups[kind] ??= {rays:0, referenceHits:0, falsePositive:0, falseNegative:0, anyErrors:0,
      wrongInstance:0, wrongPrimitive:0, invalidHit:0, distance:[], normalDot:[], colorMAE:[], roughness:[]};
    const k = i*16, {ids, floats} = output;
    const hit = Boolean(ids[k]), blocked = Boolean(ids[k+3]), t = floats[k+4];
    s.rays++; s.referenceHits += Number(!!ref);
    s.falsePositive += Number(hit && !ref); s.falseNegative += Number(!hit && !!ref);
    s.anyErrors += Number(blocked !== !!ref);
    if (hit) {
      const n = [...floats.subarray(k+8,k+11)], m = [...floats.subarray(k+12,k+16)];
      s.invalidHit += Number(!(t > ray[3] && t < ray[7] && Math.abs(Math.hypot(...n)-1) < .02 &&
        m.slice(0,3).every(c => c >= -1e-5 && c <= 1.00001) && Math.abs(m[3]) <= 1.00001));
      if (ref) {
        const approximate = sdf && ref.id === dragonId;
        s.wrongInstance += Number(ids[k+1] !== ref.id);
        s.wrongPrimitive += Number(ids[k+2] !== (approximate ? noPrimitive : ref.primitive));
        const actual = ref.values;
        s.distance.push(Math.abs(t-actual[0]));
        s.normalDot.push(n.reduce((sum,x,j)=>sum+x*actual[j+3],0));
        s.colorMAE.push(m.slice(0,3).reduce((sum,x,j)=>sum+Math.abs(x-actual[j+6]),0)/3);
        s.roughness.push(Math.abs(m[3]-actual[9]));
      }
    }
  }
  for (const [kind,s] of Object.entries(groups)) {
    s.distanceP50 = percentile(s.distance,.5); s.distanceP95 = percentile(s.distance,.95); s.distanceMax = percentile(s.distance,1);
    s.normalDotP10 = percentile(s.normalDot,.1); s.normalDotMedian = percentile(s.normalDot,.5);
    s.colorMAEP95 = percentile(s.colorMAE,.95); s.roughnessErrorP95 = percentile(s.roughness,.95);
    delete s.distance; delete s.normalDot; delete s.colorMAE; delete s.roughness;
    if (s.wrongPrimitive || s.wrongInstance || s.invalidHit) errors.push(kind+' returned invalid hit attributes or ID');
    // BVH and the additional independent asset must produce triangle-level identity and distance.
    if (!sdf || kind === 'companion first') {
      if (s.falsePositive || s.falseNegative || s.anyErrors || (s.distanceMax ?? 0) > .0005 ||
          (s.normalDotP10 ?? 1) < .999 || (s.colorMAEP95 ?? 0) > .0005 || (s.roughnessErrorP95 ?? 0) > .0005)
        errors.push(kind+' differs from the exact triangle reference');
    }
    // The field is approximate at silhouettes and grazing incidence. Clear
    // exterior/empty intervals must miss, and wing-interior normals should usually hit.
    if (sdf && ['clear box entry','empty interval','clipped before wing','clipped before exit'].includes(kind) &&
        (s.falsePositive || s.anyErrors)) errors.push(kind+' blocked before a surface');
    if (sdf && ['wing normal','clipped after wing','from inside'].includes(kind) &&
        (s.falseNegative > .1*s.referenceHits || s.anyErrors > .1*s.rays || s.distanceP95 > .025))
      errors.push(kind+' exceeded 10% missed hits or one voxel of P95 distance error');
    // Rays from both sides of the wing must recover the entering membrane.
    // Choosing the exit can still register a hit while reversing its shading normal.
    if (sdf && kind === 'wing normal' && (s.normalDotP10 < .75 || s.distanceP95 > .005))
      errors.push('Wing entrance depth or oriented normal disagrees with the triangle reference');
  }
  return {groups, errors};
}

export async function run(demo, options = {}) {
  const {gpu, dragon} = demo;
  if (!dragon) throw Error('Open the Sponza + Dragon scene');
  dragon.setPlaying(false);
  const scene = gpu.rigidMeshes, mat = new THREE.MeshBasicMaterial({side: THREE.DoubleSide});
  const companion = {
    positions: Float32Array.of(0,-.17,-.19, 0,.19,-.17, 0,.01,.21),
    normals: Float32Array.of(1,0,0, 1,0,0, 1,0,0),
    colors: Float32Array.of(.15,.22,.81, .35,.85,.19, .88,.12,.32),
    roughness: Float32Array.of(-.31,-.45,-.54), indices: Uint32Array.of(0,1,2),
  };
  const asset = scene.registerAsset(companion), handle = scene.createInstance(asset);
  const meshes = [referenceMesh(dragon.data, mat, dragon.instance), referenceMesh(companion, mat, handle)];
  const poses = [], errors = [], pipelines = {};
  try {
    for (const time of [0, 2.7]) {
      dragon.setTime(time);
      meshes[0].matrix.fromArray(dragon.matrix);
      meshes[1].matrix.copy(meshes[0].matrix).multiply(new THREE.Matrix4().makeTranslation(1.65,0,.68));
      for (const mesh of meshes) mesh.updateMatrixWorld(true);
      scene.setTransform(handle, meshes[1].matrixWorld);
      const samples = samplePose(dragon,meshes), input = Float32Array.from(samples.flatMap(s=>Array.from(s.ray)));
      const pose = {time, rayCount:samples.length};
      for (const mode of ['bvh','sdf']) {
        await dragon.setTraceMode(mode);
        const pipes = gpu._prepareRigidMeshes();
        const shader = mode === 'sdf' ? RIGID_SDF_WGSL : RIGID_MESH_WGSL;
        pipelines[mode] ??= await rigidQueryPipeline(gpu.device, shader, pipes.trace.getBindGroupLayout(1));
        const output = await queryRigidBatch(gpu.device, scene, pipelines[mode], input, gpu.rigidBG);
        const result = compare(mode,samples,output,dragon.instance);
        pose[mode] = result.groups;
        errors.push(...result.errors.map(e=>mode+', pose '+time+': '+e));
      }
      poses.push(pose);
    }
    const report = {adapter:gpu.adapterInfo, units:'meters, linear RGB', dragonTriangles:dragon.data.indices.length/3,
      reference:'Three.js double-sided mesh raycasts on Float32 local triangles at the rendered transforms',
      perModeRayCount:poses.reduce((n,p)=>n+p.rayCount,0), poses, errors, passed:!errors.length};
    if (errors.length && options.strict !== false) throw Error(errors.join('; '));
    return report;
  } finally {
    await dragon.setTraceMode('bvh');
    scene.removeInstance(handle); scene.releaseAsset(asset);
    for (const mesh of meshes) mesh.geometry.dispose();
    mat.dispose();
  }
}
