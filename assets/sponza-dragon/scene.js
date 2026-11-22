// Sponza is Z-up, in meters, with its central aisle along X.
export const SPONZA_CAMERA = {
  position: [-5.9, -1.12, 1.65],
  target: [1.1, 0.12, 1.36],
};

export function dragonPose(seconds, out = new Float32Array(16)) {
  if (!Number.isFinite(seconds)) throw new TypeError('Expected finite dragon animation time');
  const phase = seconds * 0.65 - 0.75;
  const x = -1.65 * Math.sin(phase);
  const y = 0.18 + 0.10 * Math.cos(phase);
  const z = 0.48 + 0.12 * Math.cos(2 * phase);
  const yaw = 2.18 + 0.12 * Math.sin(phase);
  const c = Math.cos(yaw), s = Math.sin(yaw);
  out.set([c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
  return out;
}

export function createDragon(rigidMeshes, data, onStateChange) {
  const asset = rigidMeshes.registerAsset(data);
  const matrix = dragonPose(0);
  let instance = null, time = 0, playing = true;
  try {
    instance = rigidMeshes.createInstance(asset, matrix);
  } catch (error) {
    rigidMeshes.releaseAsset(asset);
    throw error;
  }
  return {
    asset, data, matrix,
    get instance() { return instance; },
    get time() { return time; },
    get enabled() { return instance !== null; },
    get playing() { return playing; },
    setTime(seconds) {
      if (time === seconds) return;
      dragonPose(seconds, matrix);
      if (instance !== null) rigidMeshes.setTransform(instance, matrix);
      time = seconds;
    },
    setPlaying(value) {
      playing = Boolean(value);
      onStateChange();
    },
    setEnabled(value) {
      if (Boolean(value) === (instance !== null)) return;
      if (value) instance = rigidMeshes.createInstance(asset, matrix);
      else {
        rigidMeshes.removeInstance(instance);
        instance = null;
      }
      onStateChange();
    },
    dispose() {
      if (instance !== null) rigidMeshes.removeInstance(instance);
      instance = null;
      if (!rigidMeshes.getInstanceCountForAsset(asset)) rigidMeshes.releaseAsset(asset);
    },
  };
}
