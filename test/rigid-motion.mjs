// Run with python3 test/webgpu-browser.py test/rigid-motion.mjs --scene rigid-mesh
import {textureReader} from './rigid-render.mjs';

function backgroundChanges(gpu, previous, current) {
  let sum = 0, squares = 0, large = 0, count = 0, movingPixels = 0;
  for (let y = 0; y < gpu.aoH; y++) {
    for (let x = 0; x < gpu.aoW; x++) {
      const i = 4 * (y * gpu.aoW + x);
      if (Math.abs(current.position[i + 3] - previous.position[i + 3]) > 0.0001) {
        movingPixels++;
        continue;
      }
      if (current.position[i + 3] <= 0) continue;
      // Keep distant surfaces with an unchanged primary hit. Nearby floor
      // patches receive legitimate changes in occlusion as the chair passes.
      const p = current.position;
      if (Math.hypot(p[i], p[i + 1] + 2.44, p[i + 2] - 0.55) < 3) continue;
      const j = 4 * ((2 * y) * gpu.width + 2 * x);
      let difference = 0;
      for (let c = 0; c < 3; c++) {
        difference += Math.abs(current.color[j + c] - previous.color[j + c]) / 3;
      }
      sum += difference;
      squares += difference * difference;
      if (difference > 0.05) large++;
      count++;
    }
  }
  if (count < 100) throw Error('Insufficient stationary background pixels');
  return {sum, squares, large, count, movingPixels};
}

async function sequence(demo, read, moving) {
  const gpu = demo.gpu;
  demo.setRigidMover(true, 0);
  gpu.clearHash();
  async function snapshot() {
    const [position, color] = await Promise.all([
      read(gpu.gPosDepth[1 - gpu.pingIdx]), read(gpu.context.getCurrentTexture()),
    ]);
    return {position, color};
  }

  let previous;
  let sum = 0, squares = 0, large = 0, measured = 0, movingPixels = 0;
  for (let frame = 0; frame < 32; frame++) {
    if (moving) demo.setRigidMover(true, frame / 30);
    demo.renderOnce();
    if (frame < 15) continue; // Warm up the real shader history.
    const current = await snapshot();
    if (previous) {
      const delta = backgroundChanges(gpu, previous, current);
      sum += delta.sum;
      squares += delta.squares;
      large += delta.large;
      measured += delta.count;
      movingPixels += delta.movingPixels;
    }
    previous = current;
  }
  // The first stationary frame exercises the transition out of motion, including
  // hash invalidation. Its output should remain stable on the same background.
  demo.renderOnce();
  const stopped = backgroundChanges(gpu, previous, await snapshot());
  return {
    mean: sum / measured,
    rms: Math.sqrt(squares / measured),
    fractionAbove5pct: large / measured,
    stopMean: stopped.sum / stopped.count,
    stopRms: Math.sqrt(stopped.squares / stopped.count),
    measured, movingPixels,
  };
}

export async function run(demo) {
  demo.setEmissive(false);
  const read = textureReader(demo.gpu.device);
  const stationary = await sequence(demo, read, false);
  const moving = await sequence(demo, read, true);
  if (moving.movingPixels < 30) throw Error('The chair failed to move during the sequence');
  // Compare motion against the same device's stationary noise floor. The
  // absolute tolerances allow small differences between rasterizers/GPUs.
  const motionLimit = Math.max(0.012, stationary.rms * 2.5);
  const stopLimit = Math.max(0.014, stationary.stopRms * 3);
  if (moving.rms > motionLimit || moving.fractionAbove5pct > 0.025 || moving.stopRms > stopLimit) {
    throw Error('Distant surfaces flickered while moving or stopping the chair: ' +
      JSON.stringify({stationary, moving, motionLimit, stopLimit}));
  }
  return {stationary, moving};
}
