// Common scene, geometry, camera and motion for render, query and quality comparisons.
// The .f32 meshes are the same triangle soups used by reference_movers.py.
export const RIGID_ATRIUM_CAMERA = {
  position: [0.04, -8.05, 2.25],
  target: [0, -1.65, 0.68],
};

const props = [
  // Asset, orbit center (m), horizontal travel, vertical travel, starting yaw.
  { asset: 0, center: [-0.10, -4.02, 0.42], travel: 0.53, bob: 0.04, yaw: -0.32 },
  { asset: 1, center: [-1.35, -4.38, 0.87], travel: 0.13, bob: 0.10, yaw: 0.62 },
  { asset: 3, center: [ 1.42, -4.28, 1.08], travel: 0.14, bob: 0.13, yaw: -0.36 },
  { asset: 2, center: [-1.02, -2.54, 1.48], travel: 0.22, bob: 0.10, yaw: 0.48 },
  { asset: 1, center: [ 0.97, -2.33, 1.61], travel: 0.25, bob: 0.12, yaw: 1.14 },
  { asset: 0, center: [ 0.08, -1.57, 0.98], travel: 0.22, bob: 0.08, yaw: -0.52 },
  { asset: 3, center: [-2.75, -1.55, 1.91], travel: 0.22, bob: 0.16, yaw: -1.13 },
  { asset: 2, center: [ 2.79, -1.35, 1.96], travel: 0.24, bob: 0.12, yaw: 0.25 },
];

const materials = [
  { name: 'Vermilion glazed teapot', mesh: 'teapot', scale: 1.40, color: [0.74, 0.091, 0.035], roughness: 0.21 },
  { name: 'Brushed gold knot', mesh: 'torus_knot', scale: 1.05, color: [0.89, 0.54, 0.16], roughness: -0.23 },
  { name: 'Cobalt glazed teapot', mesh: 'teapot', scale: 1.16, color: [0.046, 0.25, 0.71], roughness: 0.18 },
  { name: 'Jade glazed knot', mesh: 'torus_knot', scale: 1.15, color: [0.035, 0.58, 0.36], roughness: 0.25 },
];

async function readMesh(name) {
  const response = await fetch(new URL('./meshes/' + name + '.f32', import.meta.url));
  if (!response.ok) throw new Error('Could not load ' + name + ' mesh, HTTP ' + response.status);
  const bytes = await response.arrayBuffer();
  const triangles = name === 'teapot' ? 4032 : 6400;
  if (bytes.byteLength !== triangles * 9 * 4) throw new Error('Unexpected ' + name + ' geometry size');
  const soup = new Float32Array(bytes), ids = new Map(), pos = [], smooth = [], indices = [];
  for (let i = 0; i < soup.length; i += 3) {
    const x = soup[i], y = soup[i+1], z = soup[i+2];
    // Weld coincident sampled vertices (including independently exported patch seams).
    const key = [x,y,z].map(v => Math.round(v * 1e6)).join(',');
    let id = ids.get(key);
    if (id === undefined) {
      id = pos.length / 3;
      ids.set(key, id);
      pos.push(x,y,z);
      smooth.push(0,0,0);
    }
    indices.push(id);
  }
  for (let i = 0; i < indices.length; i += 3) {
    const a = indices[i]*3, b = indices[i+1]*3, c = indices[i+2]*3;
    const ux = pos[b]-pos[a], uy = pos[b+1]-pos[a+1], uz = pos[b+2]-pos[a+2];
    const vx = pos[c]-pos[a], vy = pos[c+1]-pos[a+1], vz = pos[c+2]-pos[a+2];
    const nx = uy*vz-uz*vy, ny = uz*vx-ux*vz, nz = ux*vy-uy*vx;
    for (const v of [a,b,c]) { smooth[v] += nx; smooth[v+1] += ny; smooth[v+2] += nz; }
  }
  for (let i = 0; i < smooth.length; i += 3) {
    const length = Math.hypot(smooth[i], smooth[i+1], smooth[i+2]) || 1;
    smooth[i] /= length; smooth[i+1] /= length; smooth[i+2] /= length;
  }
  return { positions: Float32Array.from(pos), normals: Float32Array.from(smooth), indices: Uint32Array.from(indices) };
}

let cachedAssets;
export async function loadAtriumPropAssets() {
  if (!cachedAssets) {
    cachedAssets = Promise.all([readMesh('teapot'), readMesh('torus_knot')]).then(([pot, knot]) =>
      materials.map(material => {
        const base = material.mesh === 'teapot' ? pot : knot;
        const positions = base.positions.map(v => v * material.scale);
        const colors = new Float32Array(positions.length), roughness = new Float32Array(positions.length / 3);
        for (let i = 0; i < roughness.length; i++) {
          const z = base.positions[i*3+2];
          const gildedLip = material.mesh === 'teapot' && z > 0.12 && z < 0.145;
          colors.set(gildedLip ? [0.91,0.58,0.21] : material.color, i*3);
          roughness[i] = gildedLip ? -0.17 : material.roughness;
        }
        return { name: material.name, triangles: base.indices.length / 3,
          data: { positions, normals: base.normals, colors, roughness, indices: base.indices } };
      })
    ).catch(err => { cachedAssets = undefined; throw err; });
  }
  return cachedAssets;
}

export function rigidAtriumPose(index, seconds) {
  const prop = props[index];
  if (!prop || !Number.isFinite(seconds)) throw new Error('Invalid Atrium prop index or animation time');
  const phase = index * 1.19;
  const travel = 1.34 * seconds + phase - 0.68;
  const x = prop.center[0] + prop.travel * Math.sin(travel);
  const y = prop.center[1] + 0.14 * Math.cos(1.17 * seconds + phase);
  const z = prop.center[2] + prop.bob * Math.sin(1.53 * seconds + phase);
  const knot = materials[prop.asset].mesh === 'torus_knot';
  const yaw = prop.yaw + (index % 2 ? -1 : 1) * seconds * (knot ? 0.97 : 0.72);
  const roll = knot ? 1.06 + 0.28 * Math.sin(seconds + phase) : 0;
  const c = Math.cos(yaw), s = Math.sin(yaw), a = Math.cos(roll), b = Math.sin(roll);
  return new Float32Array([c,s,0,0, -s*a,c*a,b,0, s*b,-c*b,a,0, x,y,z,1]);
}

export async function createAtriumWorkload(demo, { count = 8, time = 0 } = {}) {
  if (!Number.isInteger(count) || count < 0 || count > props.length || !Number.isFinite(time)) {
    throw new Error('Expected an Atrium workload with 0 to 8 props and a finite start time');
  }
  demo.setRigidMover(false);
  demo.setMovers(false);
  demo.setEmissive(false);
  await demo.selectScene('atrium');
  demo.camera.position.set(...RIGID_ATRIUM_CAMERA.position);
  demo.controls.target.set(...RIGID_ATRIUM_CAMERA.target);
  demo.controls.update();
  const scene = demo.gpu.rigidMeshes;
  const assets = count ? await loadAtriumPropAssets() : [];
  const handles = new Map(), instances = [];
  function dispose() {
    for (const {handle} of instances.splice(0)) scene.removeInstance(handle);
    for (const handle of handles.values()) scene.releaseAsset(handle);
    handles.clear();
  }
  try {
    for (let i = 0; i < count; i++) {
      const assetIndex = props[i].asset;
      if (!handles.has(assetIndex)) handles.set(assetIndex, scene.registerAsset(assets[assetIndex].data));
      const handle = scene.createInstance(handles.get(assetIndex), rigidAtriumPose(i, time));
      instances.push({ handle, assetIndex });
    }
  } catch (err) {
    dispose();
    throw err;
  }
  return { count, assets, instances, dispose,
    setTime(seconds) {
      instances.forEach(({handle}, i) => scene.setTransform(handle, rigidAtriumPose(i, seconds)));
    },
  };
}
