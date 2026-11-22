// python3 test/webgpu-browser.py benchmark/sponza-reflection-capture.mjs --scene sponza-dragon --width 704 --height 396 > capture.json
import {textureReader} from '../test/rigid-render.mjs';

function base64(bytes) {
  let text = '';
  for (let i = 0; i < bytes.length; i += 16384) {
    text += String.fromCharCode(...bytes.subarray(i, i + 16384));
  }
  return btoa(text);
}

async function png(width, height, rgba) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
  const blob = await canvas.convertToBlob({type: 'image/png'});
  return base64(new Uint8Array(await blob.arrayBuffer()));
}

export async function run(demo) {
  demo.stopLoop();
  const {gpu, dragon, camera} = demo;
  if (!dragon) throw Error('Select the Sponza + Dragon preset');
  dragon.setPlaying(false);
  gpu.setQuality('high');
  const read = textureReader(gpu.device);
  async function draw() {
    demo.renderOnce();
    await gpu.device.queue.onSubmittedWorkDone();
  }
  async function present() {
    const rgb = await read(gpu.context.getCurrentTexture());
    return png(gpu.width, gpu.height, Uint8ClampedArray.from(rgb, x => x * 255));
  }

  const shots = [];
  for (const [name, enabled, time] of [['A', true, 0], ['B', true, 2], ['empty', false, 2]]) {
    dragon.setEnabled(enabled);
    dragon.setTime(time);
    gpu.clearHash();
    gpu.viewMode = 0;
    for (let i = 0; i < 8; i++) await draw();
    const image = await present();
    gpu.viewMode = 4;
    await draw();
    const reflections = await present();
    const slot = 1 - gpu.pingIdx;
    const [pos, normal, material] = await Promise.all([
      read(gpu.gPosDepth[slot]), read(gpu.gNormal[slot]), read(gpu.gAlbedo),
    ]);
    const floor = new Uint8ClampedArray(pos.length);
    for (let i = 0; i < floor.length; i += 4) {
      const mirror = pos[i+3] > 0 && Math.abs(pos[i+2]) < 0.001 && normal[i+2] > .99 &&
        normal[i+3] === 2 && material[i+3] <= .08;
      floor[i] = floor[i+1] = floor[i+2] = mirror ? 255 : 0;
      floor[i+3] = 255;
    }
    const mirror = await png(gpu.aoW, gpu.aoH, floor);
    shots.push({name, model: Array.from(dragon.matrix), image, reflections, mirror});
  }

  let staticDepth;
  const plate = gpu.sceneRes.planarReflection?.texture;
  if (plate) {
    const rgba = await read(plate);
    const depth = new Float32Array(plate.width * plate.height);
    for (let i = 0; i < depth.length; i++) depth[i] = rgba[4 * i + 3];
    const stream = new Blob([depth]).stream().pipeThrough(new CompressionStream('gzip'));
    const bytes = new Uint8Array(await new Response(stream).arrayBuffer());
    staticDepth = {width: plate.width, height: plate.height, gzipFloat32: base64(bytes)};
  }
  return {
    width: gpu.width, height: gpu.height, view: camera.matrixWorldInverse.elements,
    projection: camera.projectionMatrix.elements, camera: camera.position.toArray(),
    target: demo.controls.target.toArray(), warmup: 8, quality: gpu.quality,
    adapter: gpu.adapterInfo.architecture, staticDepth, shots,
  };
}
