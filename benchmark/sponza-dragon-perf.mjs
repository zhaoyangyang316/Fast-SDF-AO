// Use the Sponza preset so all blocks share the viewer's camera, static SDF and contact BVH.
// python3 test/webgpu-browser.py benchmark/sponza-dragon-perf.mjs --scene sponza-dragon --width 1280 --height 720
// Add --software --browser /usr/bin/chromium for SwiftShader. One 12-frame block per case can be
// selected with --probe-args '{"rounds":[0]}'. The default rotates three rounds (36 frames per case).
import {measureRigidWorkloads} from './rigid-atrium-perf.mjs';

const cases = {
  empty: {count: 0},
  'dragon-parked': {count: 1},
  'dragon-moving': {count: 1, moving: true},
};

export async function run(demo, options) {
  demo.stopLoop();
  if (!demo.dragon) await demo.selectScene('sponza-dragon');
  const dragon = demo.dragon;
  const scene = demo.gpu.rigidMeshes;
  dragon.setPlaying(false);
  dragon.setEnabled(true);
  dragon.setTime(0);
  demo.renderOnce();
  await demo.gpu.device.queue.onSubmittedWorkDone();
  const asset = dragon.asset;
  const triangles = dragon.data.indices.length / 3;
  if (asset == null || !triangles || scene.count !== 1 || scene.assets.size !== 1)
    throw Error('Expected one resident dragon mesh and instance');

  // Keep the single dragon BLAS resident while toggling TLAS membership. All timing
  // starts after registration, rebuilds caused by membership changes and warm-up.
  const initialStats = {...scene.stats};
  function prepareCase(_demo, {count, time}) {
    dragon.setEnabled(count > 0);
    dragon.setTime(time);
    if (dragon.asset !== asset || dragon.enabled !== (count > 0))
      throw Error('Unexpected change in dragon asset or visibility');
    return {
      assets: [{name: 'Dragon', triangles}],
      instances: count ? [{assetIndex: 0}] : [],
      setTime: seconds => dragon.setTime(seconds),
    };
  }
  const result = await measureRigidWorkloads(demo, prepareCase, cases, options);
  if (scene.stats.assetBuilds !== initialStats.assetBuilds ||
      scene.stats.geometryBytesUploaded !== initialStats.geometryBytesUploaded)
    throw Error('Benchmark rebuilt or uploaded the dragon geometry');
  return {...result, scene: 'sponza-dragon', dragonTriangles: triangles, startupRigidStats: initialStats};
}
