import { loadSceneGeometry } from './sponza-geometry.js';

// Red Dragon, facing +X with its feet centered over the local origin.
export function loadDragonGeometry() {
  return loadSceneGeometry(new URL('./red-dragon-mesh.bin.gz', import.meta.url));
}
