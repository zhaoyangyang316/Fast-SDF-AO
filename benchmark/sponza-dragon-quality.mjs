// See test/webgpu-browser.py. Use --scene sponza-dragon; {"mode":"preview"} returns three PNGs.
import {textureReader} from '../test/rigid-render.mjs';
import {metric, finish, compare, meanImage, makeMask} from './rigid-image-metrics.mjs';

const POSE_A = 0;
const POSE_B = 2;
const FPS = 30;
const MOTION_PAIRS = 16;

function localBounds(positions) {
  const bounds = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i++) {
    const a = i % 3;
    bounds[a] = Math.min(positions[i], bounds[a]);
    bounds[a + 3] = Math.max(positions[i], bounds[a + 3]);
  }
  return bounds;
}

function includePose(swept, local, matrix) {
  for (let corner = 0; corner < 8; corner++) {
    const x = local[corner & 1 ? 3 : 0], y = local[corner & 2 ? 4 : 1], z = local[corner & 4 ? 5 : 2];
    for (let axis = 0; axis < 3; axis++) {
      const v = matrix[axis]*x + matrix[axis + 4]*y + matrix[axis + 8]*z + matrix[axis + 12];
      swept[axis] = Math.min(v, swept[axis]);
      swept[axis + 3] = Math.max(v, swept[axis + 3]);
    }
  }
}

function beyondContact(position, i, swept) {
  let squared = 0;
  for (let a = 0; a < 3; a++) {
    const d = Math.max(0, swept[a] - position[i+a], position[i+a] - swept[a+3]);
    squared += d*d;
  }
  return squared > 1.12**2;
}

async function encodePng(width, height, values) {
  const canvas = new OffscreenCanvas(width, height);
  const bytes = Uint8ClampedArray.from(values, v => 255*v);
  canvas.getContext('2d').putImageData(new ImageData(bytes, width, height), 0, 0);
  const blob = await canvas.convertToBlob({type: 'image/png'});
  const png = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let i = 0; i < png.length; i += 32768) binary += String.fromCharCode(...png.subarray(i, i+32768));
  return btoa(binary);
}

export async function run(demo, options = {}) {
  const {gpu, dragon} = demo;
  if (!dragon) throw Error('Open the Sponza + Dragon preset for this benchmark');
  demo.stopLoop();
  dragon.setPlaying(false);
  gpu.setQuality('high');
  gpu.useHash = true;
  gpu.viewMode = 0;
  const read = textureReader(gpu.device);
  const image = () => read(gpu.context.getCurrentTexture());
  const size = [gpu.width, gpu.height];
  const camera = {eye: demo.camera.position.toArray(), target: demo.controls.target.toArray()};
  async function draw() {
    demo.renderOnce();
    await gpu.device.queue.onSubmittedWorkDone();
  }
  function pose(enabled, time) {
    dragon.setEnabled(enabled);
    dragon.setTime(time);
  }

  if (options.mode === 'preview') {
    const captures = {};
    for (const [id, enabled, t] of [['poseA', true, POSE_A], ['poseB', true, POSE_B], ['empty', false, POSE_B]]) {
      pose(enabled, t);
      gpu.clearHash();
      for (let f = 0; f < (options.frames ?? 16); f++) await draw();
      captures[id] = await encodePng(...size, await image());
    }
    return {size, camera, poseSeconds: [POSE_A, POSE_B], framesPerView: options.frames ?? 16, pngBase64: captures};
  }

  async function snapshot(lighting = false) {
    const slot = 1 - gpu.pingIdx;
    const [rgb, pos, normal] = await Promise.all([
      image(), read(gpu.gPosDepth[slot]), read(gpu.gNormal[slot]),
    ]);
    const visible = new Uint8Array(pos.length/4), rigid = new Uint8Array(visible.length);
    for (let j = 0; j < visible.length; j++) {
      visible[j] = Number(pos[4*j+3] > 0);
      const tag = normal[4*j+3];
      rigid[j] = Number(tag === 1.25 || tag === 2.25);
    }
    const result = {rgb, pos, visible, rigid};
    if (lighting) {
      [result.aoGI, result.shadows] = await Promise.all([read(gpu.histAOGI[slot]), read(gpu.histMeta[slot])]);
    }
    return result;
  }

  // Temporal reference at the identical frozen pose, including a previous late
  // window to measure its residual change. RGB is display-referred in [0, 1].
  async function frozen(recordMotionControl = false) {
    gpu.clearHash();
    const early = [], late = [], control = [];
    for (let f = 1; f <= 48; f++) {
      await draw();
      if (recordMotionControl && f >= 17 && f <= 33) control.push(await image());
      if (f >= 33) (f <= 40 ? early : late).push(await image());
    }
    const rgb = meanImage(late);
    const final = await snapshot(true);
    return {rgb, final, control, drift: finish(compare(gpu, meanImage(early), rgb, final.visible))};
  }

  pose(true, POSE_B);
  const occupied = await frozen(true);
  pose(true, POSE_A);
  gpu.clearHash();
  const local = localBounds(dragon.data.positions);
  const swept = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  const colors = [], stable = occupied.final.visible.slice();
  const recentlyRigid = new Uint8Array(stable.length);
  const motionPngs = [];
  let first, last;
  const LAST = Math.round((POSE_B-POSE_A)*FPS);
  for (let f = 0; f <= LAST; f++) {
    dragon.setTime(POSE_A + f/FPS);
    includePose(swept, local, dragon.matrix);
    await draw();
    if (f === 0) first = await snapshot();
    if (options.captureMotion && f % 2 === 0) motionPngs.push(await encodePng(...size, await image()));
    if (f < LAST - MOTION_PAIRS) continue;
    last = await snapshot();
    colors.push(last.rgb);
    for (let j = 0; j < stable.length; j++) {
      if (last.rigid[j] || Math.abs(last.pos[4*j+3] - occupied.final.pos[4*j+3]) > 1e-5) stable[j] = 0;
      if (f < LAST) recentlyRigid[j] |= last.rigid[j];
    }
  }
  let matchedPointCount = 0, dragonPixels = 0, surfacePixels = 0;
  for (let j = 0; j < stable.length; j++) {
    if (!beyondContact(occupied.final.pos, 4*j, swept)) stable[j] = 0;
    if (occupied.final.visible[j]) surfacePixels++;
    if (occupied.final.rigid[j] === last.rigid[j] &&
        Math.abs(occupied.final.pos[4*j+3] - last.pos[4*j+3]) < 1e-5) matchedPointCount++;
    dragonPixels += last.rigid[j];
  }
  if (matchedPointCount !== stable.length || !dragonPixels) throw Error('Moving endpoint disagrees with the frozen geometry');
  const moving = metric(), parked = metric();
  for (let i = 1; i < colors.length; i++) {
    compare(gpu, colors[i-1], colors[i], stable, moving);
    compare(gpu, occupied.control[i-1], occupied.control[i], stable, parked);
  }
  const vacated = makeMask(stable, j => recentlyRigid[j] && !last.rigid[j] && last.visible[j]);
  function errorAtPose(rgb) {
    return {visibleSurfaces: finish(compare(gpu, rgb, occupied.rgb, occupied.final.visible)),
      dragon: finish(compare(gpu, rgb, occupied.rgb, last.rigid)),
      recentlyVacated: finish(compare(gpu, rgb, occupied.rgb, vacated))};
  }
  const duringMotion = errorAtPose(last.rgb);
  const afterStop = {};
  for (let f = 1; f <= 8; f++) {
    await draw();
    if ([1,3,8].includes(f)) afterStop[f] = errorAtPose(await image());
  }
  pose(false, POSE_B);
  const removed = {};
  for (let f = 1; f <= 8; f++) {
    await draw();
    if ([1,3,8].includes(f)) removed[f] = await image();
  }
  const empty = await frozen();
  const staticGeometry = makeMask(stable, j => occupied.final.visible[j] && !last.rigid[j] &&
    empty.final.visible[j] && Math.abs(occupied.final.pos[4*j+3] - empty.final.pos[4*j+3]) <= 1e-5);
  const former = makeMask(stable, j => last.rigid[j] && empty.final.visible[j]);
  const shadowReceivers = new Uint8Array(stable.length);
  let surfaceSamples = 0, darkerAO = 0, darkerSun = 0, visibleGI = 0;
  const floor = {samples: 0, aoDarkerByFivePercent: 0, sunDarkerByTenPercent: 0, shadowWorldBounds: null};
  for (let j = 0; j < stable.length; j++) {
    if (!staticGeometry[j]) continue;
    surfaceSamples++;
    const i = 4*j;
    const isFloor = Math.abs(occupied.final.pos[i+2]) < 0.03;
    if (isFloor) floor.samples++;
    if (empty.final.aoGI[i+3] - occupied.final.aoGI[i+3] > 0.05) {
      darkerAO++;
      if (isFloor) floor.aoDarkerByFivePercent++;
    }
    if (empty.final.shadows[i] - occupied.final.shadows[i] > 0.10) {
      if (isFloor) {
        floor.sunDarkerByTenPercent++;
        const x = occupied.final.pos[i], y = occupied.final.pos[i+1];
        const b = floor.shadowWorldBounds ??= [x, y, x, y];
        b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y);
        b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y);
      }
      darkerSun++;
      shadowReceivers[j] = 1;
    }
    const gi = empty.final.aoGI, withDragon = occupied.final.aoGI;
    if (Math.abs(gi[i]-withDragon[i]) + Math.abs(gi[i+1]-withDragon[i+1]) + Math.abs(gi[i+2]-withDragon[i+2]) > 0.1) visibleGI++;
  }
  function errorAfterRemoval(rgb) {
    return {oldDragonSilhouette: finish(compare(gpu,rgb,empty.rgb,former)),
      castShadowReceivers: finish(compare(gpu,rgb,empty.rgb,shadowReceivers))};
  }
  return {size, camera, ...(options.captureMotion ? {motionPngs, motionFps: FPS/2} : {}), preset: gpu.quality, adapter: gpu.adapterInfo.architecture, assetTriangles: dragon.data.indices.length / 3,
    firstPoseSeconds: POSE_A, lastPoseSeconds: POSE_B, timestep: 1/FPS, measuredFramePairs: MOTION_PAIRS,
    gbufferSize: [gpu.aoW, gpu.aoH], movingPoseAPixels: first.rigid.reduce((a,b) => a+b, 0),
    poseBDragonPixels: dragonPixels, visibleSurfacePixels: surfacePixels, dragonSurfaceCoverage: dragonPixels/surfacePixels,
    poseChangeSilhouetteSamples: first.rigid.reduce((n,r,j) => n + Number(r !== last.rigid[j]), 0),
    distantBackgroundTemporal: {parked: finish(parked), moving: finish(moving)},
    matchedPoseDuringMotion: duringMotion, stoppedFrameError: afterStop,
    frozenWindowDrift: {occupied: occupied.drift, empty: empty.drift, earlierFrames: [33,40], referenceFrames: [41,48]},
    sameSurfaceLighting: {staticSamples: surfaceSamples, aoDarkerByFivePercent: darkerAO,
      sunVisibilityDarkerByTenPercent: darkerSun, linearGIRgbL1ChangeAbovePointOne: visibleGI, floor},
    ifDragonImageWereHeld: errorAfterRemoval(occupied.rgb),
    removedFrameError: Object.fromEntries(Object.entries(removed).map(([f,rgb]) => [f,errorAfterRemoval(rgb)]))};
}
