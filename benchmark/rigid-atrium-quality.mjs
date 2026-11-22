// Run with python3 test/webgpu-browser.py benchmark/rigid-atrium-quality.mjs --width 320 --height 180
import {createAtriumWorkload, rigidAtriumPose} from './rigid-atrium-scene.mjs';
import {textureReader} from '../test/rigid-render.mjs';
import {collectContactAO} from './rigid-atrium-contact-data.mjs';
import {metric, finish, compare, meanImage, makeMask} from './rigid-image-metrics.mjs';

const LAST = 36;                 // 1.2 s of transforms sampled at 30 Hz
const CONTACT_RADIUS = 1.12;     // Exclude the vicinity of moving occluders for background flicker
function xyzBounds(data, transform) {
  const bounds = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (let i = 0; i < data.length; i += 3) {
    for (let axis = 0; axis < 3; axis++) {
      const v = transform[axis] * data[i] + transform[4 + axis] * data[i + 1] +
        transform[8 + axis] * data[i + 2] + transform[12 + axis];
      bounds[axis] = Math.min(bounds[axis], v);
      bounds[axis + 3] = Math.max(bounds[axis + 3], v);
    }
  }
  return bounds;
}

function frameBounds(fixture, time) {
  return fixture.instances.map(({assetIndex}, index) =>
    xyzBounds(fixture.assets[assetIndex].data.positions, rigidAtriumPose(index, time)));
}

function distanceFrom(p, i, bounds) {
  let squared = 0;
  for (let a = 0; a < 3; a++) {
    const distance = Math.max(0, bounds[a] - p[i + a], p[i + a] - bounds[a + 3]);
    squared += distance * distance;
  }
  return Math.sqrt(squared);
}

function coverage(frame, bounds) {
  let total = 0, primary = 0, ambiguous = 0;
  const perInstance = bounds.map(() => 0);
  for (let j = 0; j < frame.rigid.length; j++) {
    if (frame.visible[j]) primary++;
    if (!frame.rigid[j]) continue;
    total++;
    const candidates = bounds.map(b => distanceFrom(frame.position, 4 * j, b) < 0.001);
    if (candidates.filter(Boolean).length === 1) perInstance[candidates.indexOf(true)]++;
    else ambiguous++;
  }
  return {gbufferRigidPixels: total, gbufferSurfacePixels: primary, visibleSurfaceFraction: total / primary,
    instancesWithUniquePixels: perInstance.filter(n => n > 0).length, uniquePixelsByInstance: perInstance,
    ambiguousOrUnassigned: ambiguous};
}

export async function run(demo) {
  const {gpu} = demo;
  const fixture = await createAtriumWorkload(demo, {count: 8, time: LAST / 30});
  gpu.setQuality('high');
  gpu.useHash = true;
  gpu.viewMode = 0;
  demo.setEmissive(false);
  const read = textureReader(gpu.device);
  const image = () => read(gpu.context.getCurrentTexture());
  const size = [gpu.width, gpu.height];
  const adapter = Object.fromEntries(['vendor', 'architecture', 'device', 'description', 'isFallbackAdapter']
    .map(key => [key, gpu.adapterInfo[key]]));
  demo.stopLoop();
  const atEnd = frameBounds(fixture, LAST / 30);
  const swept = atEnd.map(b => b.slice());
  for (let f = 0; f < LAST; f++) {
    frameBounds(fixture, f / 30).forEach((b, i) => {
      for (let a = 0; a < 3; a++) {
        swept[i][a] = Math.min(swept[i][a], b[a]);
        swept[i][a + 3] = Math.max(swept[i][a + 3], b[a + 3]);
      }
    });
  }

  async function capture() {
    const slot = 1 - gpu.pingIdx;
    const [color, position, normal] = await Promise.all([
      image(), read(gpu.gPosDepth[slot]), read(gpu.gNormal[slot])
    ]);
    const visible = new Uint8Array(position.length / 4), rigid = new Uint8Array(visible.length);
    for (let j = 0; j < rigid.length; j++) {
      visible[j] = Number(position[4 * j + 3] > 0);
      const tag = normal[4 * j + 3];
      rigid[j] = Number(tag === 1.25 || tag === 2.25);
    }
    return {color, position, normal, rigid, visible};
  }

  async function draw() {
    demo.renderOnce();
    // Bound queued work on the software adapter during reference warm-up.
    await gpu.device.queue.onSubmittedWorkDone();
  }

  // Frames 49–56 and 57–64 form consecutive temporal windows within one parked run.
  // The second window is the finite, same-renderer reference for shaded RGB.
  async function reference(parkedFrames = null) {
    gpu.clearHash();
    const early = [], late = [];
    for (let f = 1; f <= 64; f++) {
      await draw();
      if (parkedFrames && f >= 21 && f <= 37) parkedFrames.push(await image());
      if (f >= 49) (f <= 56 ? early : late).push(await image());
    }
    const color = meanImage(late);
    const frame = await capture();
    const earlier = meanImage(early);
    return {image: color, earlier, frame, skipPercentAtEnd: gpu.lastSkipPct,
      ao: await collectContactAO(gpu, read, frame.position, frame.normal),
      drift: finish(compare(gpu, earlier, color, frame.visible))};
  }

  let disposed = false;
  try {
    await draw();
    const endpoint = await capture();
    const background = makeMask(endpoint.visible, j => endpoint.visible[j] && !endpoint.rigid[j] &&
      swept.every(b => distanceFrom(endpoint.position, 4 * j, b) > CONTACT_RADIUS));
    const parked = metric(), active = metric();
    const parkedColors = [], movingColors = [];
    const parkedReference = await reference(parkedColors);
    const coverageAtEndpoint = coverage(parkedReference.frame, atEnd);
    fixture.setTime(0);
    gpu.clearHash();
    const recentlyRigid = new Uint8Array(background.length);
    let last, previous, motionPixels = 0;
    for (let f = 0; f <= LAST; f++) {
      fixture.setTime(f / 30);
      await draw();
      if (f < LAST - 16) continue;
      last = await capture();
      movingColors.push(last.color);
      for (let j = 0; j < background.length; j++) {
        if (last.rigid[j] || Math.abs(last.position[4 * j + 3] - endpoint.position[4 * j + 3]) > 1e-5) background[j] = 0;
      }
      if (previous) {
        for (let j = 0; j < background.length; j++) motionPixels += Number(last.rigid[j] !== previous.rigid[j]);
      }
      if (f < LAST) {
        for (let j = 0; j < background.length; j++) recentlyRigid[j] |= last.rigid[j];
      }
      previous = last;
    }
    for (let j = 0; j < background.length; j++) {
      if ((last.rigid[j] !== endpoint.rigid[j]) || Math.abs(last.position[4 * j + 3] - endpoint.position[4 * j + 3]) > 1e-5) {
        throw Error('Reference and moving endpoint do not project to the same primary surfaces');
      }
    }
    for (let i = 1; i < movingColors.length; i++) {
      compare(gpu, movingColors[i - 1], movingColors[i], background, active);
      compare(gpu, parkedColors[i - 1], parkedColors[i], background, parked);
    }
    const vacated = makeMask(background, j => recentlyRigid[j] && !last.rigid[j] && last.visible[j]);
    const onReference = parkedReference.image;
    function poseError(color) {
      return {allSurfaces: finish(compare(gpu, color, onReference, endpoint.visible)),
        rigidSurfaces: finish(compare(gpu, color, onReference, endpoint.rigid)),
        stableBackground: finish(compare(gpu, color, onReference, background)),
        vacatedSilhouette: finish(compare(gpu, color, onReference, vacated))};
    }
    const stopped = {};
    for (let f = 1; f <= 8; f++) {
      await draw();
      if ([1, 3, 8].includes(f)) stopped[f] = poseError(await image());
    }
    fixture.dispose();
    disposed = true;
    const removed = {};
    for (let f = 1; f <= 8; f++) {
      await draw();
      if ([1, 3, 8].includes(f)) removed[f] = await image();
    }
    const emptyReference = await reference();
    const empty = emptyReference.frame;
    const formerRigid = makeMask(background, j => endpoint.rigid[j] && empty.visible[j]);
    // Select unchanged geometry with a material lighting difference between the
    // two parked references, independently from the transient frames being scored.
    const receivers = makeMask(background, j => {
      if (!endpoint.visible[j] || endpoint.rigid[j] || !empty.visible[j] ||
          Math.abs(endpoint.position[4 * j + 3] - empty.position[4 * j + 3]) > 1e-5) return false;
      const x = j % gpu.aoW, y = Math.floor(j / gpu.aoW);
      const px = Math.floor((x + 0.5) * gpu.width / gpu.aoW);
      const py = Math.floor((y + 0.5) * gpu.height / gpu.aoH);
      const i = 4 * (py * gpu.width + px);
      let delta = 0;
      for (let c = 0; c < 3; c++) delta += Math.abs(onReference[i + c] - emptyReference.image[i + c]) / 3;
      return delta > 0.05;
    });
    function removalError(color) {
      return {formerSilhouette: finish(compare(gpu, color, emptyReference.image, formerRigid)),
        affectedReceivers: finish(compare(gpu, color, emptyReference.image, receivers))};
    }
    if (!motionPixels || !coverageAtEndpoint.gbufferRigidPixels || !active.count) throw Error('Empty motion or visibility measurement');
    return {size, adapter, camera: {position: demo.camera.position.toArray(), target: demo.controls.target.toArray()}, preset: gpu.quality, movingInstances: fixture.count,
      pairs: 16, timeStep: 1 / 30, endpointSeconds: LAST / 30, coverageAtEndpoint, changedSilhouetteSamples: motionPixels,
      backgroundFlicker: {parked: finish(parked), moving: finish(active)},
      frozenReference: {warmupFrames: 48, averagedFrames: 8, firstWindowVsSecond: parkedReference.drift,
        zeroPropsFirstWindowVsSecond: emptyReference.drift, frozenSkipPercent: parkedReference.skipPercentAtEnd},
      equalPoseErrorDuringMotion: poseError(last.color), stoppedFrameError: stopped,
      removalErrorIfOldImageWereHeld: removalError(onReference),
      emptyReferenceDriftOnFormerSilhouette: finish(compare(gpu, emptyReference.earlier, emptyReference.image, formerRigid)),
      emptyReferenceDriftOnAffectedReceivers: finish(compare(gpu, emptyReference.earlier, emptyReference.image, receivers)),
      contactAO: {rigid: parkedReference.ao, empty: emptyReference.ao},
      removedFrameError: Object.fromEntries(Object.entries(removed).map(([f, color]) => [f, removalError(color)]))};
  } finally {
    if (!disposed) fixture.dispose();
  }
}
