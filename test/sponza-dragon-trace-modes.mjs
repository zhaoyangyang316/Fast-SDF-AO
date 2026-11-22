// Run the UI flow including the real requestAnimationFrame loop:
// python3 test/webgpu-browser.py test/sponza-dragon-trace-modes.mjs --scene sponza-dragon --width 512 --height 288 --probe-args '{"liveLoop":true}'
// To produce inputs for benchmark/sponza-reflection-reference.py, also set "includeImages":true.
import {textureReader} from './rigid-render.mjs';
import {png} from '../benchmark/sponza-reflection-capture.mjs';

function check(ok, message) {
  if (!ok) throw Error(message);
}

async function until(predicate, message) {
  const deadline = performance.now() + 120000;
  while (!predicate()) {
    if (performance.now() > deadline) throw Error(`${message}: ${document.querySelector('#status').value}`);
    await new Promise(resolve => setTimeout(resolve, 15));
  }
}

function select(element, value) {
  element.value = value;
  element.dispatchEvent(new Event('change', {bubbles:true}));
}

export async function run(demo, options = {}) {
  const {gpu, dragon, camera} = demo;
  const selector = document.querySelector('#dragon-trace');
  const visible = document.querySelector('#dragon-visible');
  const motion = document.querySelector('#dragon-motion');
  check(dragon && selector && selector.checkVisibility(), 'Sponza trace selector missing');
  const mode = async value => {
    select(selector, value);
    await until(() => {
      if (dragon.traceError) throw Error(dragon.traceError);
      return dragon.traceMode === value && selector.value === value;
    }, `Failed to select ${value}`);
  };
  const show = value => {
    if (visible.checked !== value) visible.click();
    check(dragon.enabled === value, 'Visibility control disagrees with the dragon');
  };
  check(dragon.traceMode === 'bvh' && selector.value === 'bvh', 'Expected default BVH');
  if (dragon.playing) motion.click();
  check(!dragon.playing, 'Pause button failed');

  // Supersede the initial lazy download before it finishes, then exercise the UI after it settles.
  const initialRequest = dragon.setTraceMode('sdf');
  select(selector, 'bvh');
  await initialRequest;
  check(selector.value === 'bvh' && dragon.traceMode === 'bvh', 'Stale SDF request overrode the newer selection');
  const animation = {};
  if (options.liveLoop) {
    for (const value of ['sdf','bvh']) {
      await mode(value);
      const start = dragon.time;
      motion.click();
      await until(() => dragon.time > start, 'Play button did not advance the real animation loop');
      motion.click();
      const paused = dragon.time;
      await new Promise(requestAnimationFrame);
      await new Promise(requestAnimationFrame);
      check(!dragon.playing && dragon.time === paused, 'Dragon continued moving after pause');
      animation[value] = {start, paused};
    }
  }
  demo.stopLoop();
  await gpu.device.queue.onSubmittedWorkDone();
  gpu.setQuality('high');
  const read = textureReader(gpu.device);
  const cameraState = () => JSON.stringify([camera.position.toArray(), demo.controls.target.toArray()]);
  const originalCamera = cameraState();
  async function draw() {
    demo.renderOnce();
    await gpu.device.queue.onSubmittedWorkDone();
  }
  const present = () => read(gpu.context.getCurrentTexture());
  const includeImages = options.includeImages === true;
  const captures = {bvh: [], sdf: []};
  const stored = {};

  async function capture(key, traceMode, name, seconds, enabled = true) {
    const previousTime = dragon.time;
    await mode(traceMode);
    check(dragon.time === previousTime && !dragon.playing, 'Trace selection changed the paused pose');
    dragon.setTime(seconds);
    show(enabled);
    const matrix = Array.from(dragon.matrix);
    gpu.clearHash();
    gpu.viewMode = 0;
    for (let frame = 0; frame < 3; frame++) await draw();
    const rgb = await present();
    gpu.viewMode = 4;
    await draw();
    const slot = 1 - gpu.pingIdx;
    const [reflection, pos, normal, gi, shadow] = await Promise.all([
      present(), read(gpu.gPosDepth[slot]), read(gpu.gNormal[slot]), read(gpu.histAOGI[slot]), read(gpu.histMeta[slot]),
    ]);
    const mirror = new Uint8ClampedArray(pos.length);
    let rigidPixels = 0;
    for (let i = 0; i < pos.length; i += 4) {
      if (normal[i+3] === 1.25 || normal[i+3] === 2.25) rigidPixels++;
      const floor = pos[i+3] > 0 && Math.abs(pos[i+2]) < .001 && normal[i+2] > .99 && normal[i+3] === 2;
      mirror[i] = mirror[i+1] = mirror[i+2] = floor ? 255 : 0;
      mirror[i+3] = 255;
    }
    check(cameraState() === originalCamera && dragon.time === seconds, 'Camera or pose moved during capture');
    check(enabled ? rigidPixels > 0 : rigidPixels === 0, `Unexpected primary dragon coverage (${key})`);
    stored[key] = {rgb, reflection, pos, normal, gi, shadow, mirror, rigidPixels};
    if (includeImages && name) {
      const encoded = data => png(gpu.width, gpu.height, Uint8ClampedArray.from(data, v => v*255));
      captures[traceMode].push({name, model: matrix, image: await encoded(rgb), reflections: await encoded(reflection),
        mirror: await png(gpu.aoW, gpu.aoH, mirror)});
    }
  }

  await capture('bvhA', 'bvh', 'A', 0);
  await capture('sdfA', 'sdf', 'A', 0);
  await capture('sdfB', 'sdf', 'B', 2);
  await capture('bvhB', 'bvh', 'B', 2);
  await capture('emptyBvh', 'bvh', 'empty', 2, false);
  // Select the SDF while the raster dragon is visible (the selector can be disabled while hidden).
  show(true);
  await capture('emptySdf', 'sdf', 'empty', 2, false);
  show(true);
  await capture('bvhAgain', 'bvh', null, 2);

  function compare(a, b) {
    let geometryMismatches = 0, floorPixels = 0, reflectionDelta = 0, changed = 0, aoDelta = 0, sunDelta = 0;
    for (let j = 0; j < a.pos.length; j++) geometryMismatches += Number(a.pos[j] !== b.pos[j] || a.normal[j] !== b.normal[j]);
    for (let y = 0; y < gpu.height; y++) for (let x = 0; x < gpu.width; x++) {
      const g = 4 * (Math.floor(y * gpu.aoH / gpu.height) * gpu.aoW + Math.floor(x * gpu.aoW / gpu.width));
      if (!a.mirror[g] || !b.mirror[g]) continue;
      const p = 4 * (y * gpu.width + x);
      let d = 0;
      for (let c = 0; c < 3; c++) d += Math.abs(a.reflection[p+c]-b.reflection[p+c])/3;
      reflectionDelta += d;
      if (d > .02) changed++;
      aoDelta += Math.abs(a.gi[g+3]-b.gi[g+3]);
      sunDelta += Math.abs(a.shadow[g]-b.shadow[g]);
      floorPixels++;
    }
    return {primaryComponentMismatches: geometryMismatches, rigidGbufferPixels: a.rigidPixels, floorPixels,
      reflectionRGBMAE: reflectionDelta/floorPixels, reflectionPixelsChangingAboveTwoPercent: changed,
      floorAOVisibilityMAE: aoDelta/floorPixels, floorSunVisibilityMAE: sunDelta/floorPixels};
  }
  const comparison = {
    poseA:compare(stored.bvhA, stored.sdfA), poseB:compare(stored.bvhB, stored.sdfB),
    bvhAfterRoundTrip:compare(stored.bvhB,stored.bvhAgain), hiddenModes:compare(stored.emptyBvh,stored.emptySdf),
  };
  for (const result of Object.values(comparison)) check(result.primaryComponentMismatches === 0, 'Trace mode changed raster geometry');
  check(stored.sdfA.rigidPixels !== stored.sdfB.rigidPixels, 'The two fixture poses rasterized the same coverage');
  const changesAtB = comparison.poseB.reflectionPixelsChangingAboveTwoPercent;
  const hiddenNoise = comparison.hiddenModes.reflectionPixelsChangingAboveTwoPercent;
  const repeatNoise = comparison.bvhAfterRoundTrip.reflectionPixelsChangingAboveTwoPercent;
  check(comparison.poseA.reflectionPixelsChangingAboveTwoPercent > hiddenNoise + 8 &&
    changesAtB > Math.max(hiddenNoise, repeatNoise) + 8,
    'Switching the visible dragon did not change reflections beyond the hidden and repeated-BVH controls');
  await mode('sdf');
  document.querySelector('button[data-scene="archive"]').click();
  await until(() => !demo.dragon && document.querySelector('#status').value.startsWith('ARCHIVE ·'), 'Failed to open Archive');
  check(!selector.checkVisibility(), 'Sponza-only control is visible in Archive');
  await draw();
  document.querySelector('button[data-scene="sponza-dragon"]').click();
  await until(() => demo.dragon && document.querySelector('#status').value.startsWith('SPONZA-DRAGON ·'), 'Failed to return to Sponza');
  check(selector.checkVisibility() && selector.value === 'bvh' && demo.dragon.traceMode === 'bvh', 'Reopened Sponza did not default to BVH');
  await draw();
  return {width:gpu.width,height:gpu.height,view:camera.matrixWorldInverse.elements,projection:camera.projectionMatrix.elements,
    camera:camera.position.toArray(),target:demo.controls.target.toArray(),quality:gpu.quality,adapter:gpu.adapterInfo.architecture,
    defaultMode:'bvh', supersededInitialSdfRequest:true, animation, archiveControlsHidden:true, reenteredMode:demo.dragon.traceMode,
    comparison, ...(includeImages ? {captures} : {})};
}
