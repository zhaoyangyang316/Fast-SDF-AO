// Rebuild the checked-in Sponza field from the exact static mesh used by WebGPU:
// node tools/bake_sponza_sdf.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { gzipSync, gunzipSync } from 'node:zlib';
import { decodeSponzaGeometry } from '../assets/sponza-dragon/sponza-geometry.js';

const halfMagnitude = new Float32Array(32768);
for (let h = 0; h < halfMagnitude.length; h++) {
  const exp = h >> 10, mantissa = h & 1023;
  halfMagnitude[h] = exp === 0 ? mantissa * 2 ** -24 : (1024 + mantissa) * 2 ** (exp - 25);
}

// Open architectural sheets leave nonzero column winding in parts of the sky.
// Keep the native distance magnitude, and restore the exterior sign wherever
// centers connect to the padded outside without crossing geometry.
function repairExteriorSign(words, dims, pitch) {
  const [nx, ny] = dims;
  const plane = nx * ny;
  const free = new Uint8Array(words.length), queue = new Int32Array(words.length);
  let head = 0, tail = 0;
  // Full volume padding puts the corner in the exterior, away from geometry.
  free[0] = 1;
  queue[tail++] = 0;
  let corrected = 0;
  while (head < tail) {
    const i = queue[head++], x = i % nx, y = Math.floor(i / nx) % ny;
    if (words[i] & 0x8000) { words[i] &= 0x7fff; corrected++; }
    const d = halfMagnitude[words[i]];
    // If a triangle crossed this center-to-center edge, the two distances to
    // the mesh could sum to at most one pitch. Include a rounding margin for
    // the two FP16 reads. This preserves the native signs in enclosed solids.
    const threshold = pitch * 1.002 - d;
    if (x > 0) visit(i - 1, threshold);
    if (x + 1 < nx) visit(i + 1, threshold);
    if (y > 0) visit(i - nx, threshold);
    if (y + 1 < ny) visit(i + nx, threshold);
    if (i >= plane) visit(i - plane, threshold);
    if (i + plane < words.length) visit(i + plane, threshold);
  }
  function visit(j, threshold) {
    if (!free[j] && halfMagnitude[words[j] & 0x7fff] > threshold) {
      free[j] = 1;
      queue[tail++] = j;
    }
  }
  console.log(`Exterior flood reached ${tail} voxel centers and corrected ${corrected} winding signs`);
}

function verifySponzaField(words, dims, min, pitch) {
  const [nx, ny, nz] = dims;
  if (words.length !== nx * ny * nz) throw Error('Incomplete FP16 volume');
  // The top face is completely above the highest static triangle.
  for (const h of words.subarray(nx * ny * (nz - 1))) {
    if ((h & 0x8000) || h === 0) throw Error('Expected positive distance in the sky padding');
  }
  function atCenter(x, y, z) {
    const [i, j, k] = [x, y, z].map((p, axis) => Math.round((p - min[axis]) / pitch - 0.5));
    const word = words[(k * ny + j) * nx + i];
    return halfMagnitude[word & 0x7fff] * (word & 0x8000 ? -1 : 1);
  }
  for (let x = -9; x <= 9; x += 0.5) {
    for (let y = -1.5; y <= 1.5; y += 0.25) {
      for (const z of [0.125, 0.375, 0.875, 1.625, 2.5]) {
        if (atCenter(x, y, z) <= 0) throw Error(`Expected free space at ${x}, ${y}, ${z}`);
      }
    }
  }
  for (const z of [-0.25, -0.125, 0, 0.125, 0.25, 1]) {
    if (Math.abs(atCenter(0, 0, z) - z) > 0.002) throw Error(`Incorrect floor distance at z=${z}`);
  }
  console.log('Sky padding, 2,405 courtyard points and six signed floor depths passed');
}

const root = fileURLToPath(new URL('../', import.meta.url));
const assets = path.join(root, 'assets', 'sponza-dragon');
const packed = gunzipSync(fs.readFileSync(path.join(assets, 'sponza-mesh.bin.gz')));
const { positions, indices } = decodeSponzaGeometry(
  packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength)
);
const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < positions.length; i++) {
  const axis = i % 3;
  lo[axis] = Math.min(lo[axis], positions[i]);
  hi[axis] = Math.max(hi[axis], positions[i]);
}
// Leave two voxels of clear padding per face. Put z=0 at a voxel center so
// trilinear samples above the courtyard floor recover its distance accurately.
const pitch = 0.125;
const min = lo.map(x => (Math.floor(x / pitch) - 2.5) * pitch);
const dims = hi.map((x, axis) => 8 * Math.ceil(((x - min[axis]) / pitch + 2) / 8));
const span = dims.map(n => n * pitch);
const output = path.join(assets, 'sponza.f16.gz');
if (process.argv.includes('--verify')) {
  const field = gunzipSync(fs.readFileSync(output));
  verifySponzaField(new Uint16Array(field.buffer, field.byteOffset, field.length / 2), dims, min, pitch);
  process.exit(0);
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'bake-sponza-'));
try {
  const obj = path.join(tmp, 'sponza.obj');
  const fd = fs.openSync(obj, 'w');
  try {
    for (const [prefix, data, offset] of [['v', positions, 0], ['f', indices, 1]]) {
      for (let begin = 0; begin < data.length; begin += 3 * 8192) {
        const lines = [];
        for (let i = begin; i < Math.min(data.length, begin + 3 * 8192); i += 3) {
          lines.push(`${prefix} ${data[i]+offset} ${data[i+1]+offset} ${data[i+2]+offset}\n`);
        }
        fs.writeFileSync(fd, lines.join(''));
      }
    }
  } finally { fs.closeSync(fd); }
  const fieldPath = path.join(tmp, 'sponza.f16');
  execFileSync(process.execPath, [
    path.join(root, 'bin', 'bake-sdf.mjs'), '--obj', obj,
    '--lo', ...min.map(String), '--span', ...span.map(String), '--res-xyz', ...dims.map(String),
    '--clip', '1.5', '--threads', String(Math.min(8, os.availableParallelism())),
    '--validate', '128', '--f16', '--out', fieldPath,
  ], { stdio: 'inherit' });
  const field = fs.readFileSync(fieldPath);
  const words = new Uint16Array(field.buffer, field.byteOffset, field.length / 2);
  repairExteriorSign(words, dims, pitch);
  verifySponzaField(words, dims, min, pitch);
  fs.writeFileSync(output, gzipSync(field, { level: 9 }));
  const constants = [['MIN',min],['SPAN',span],['DIMS',dims]].map(([name,v]) =>
    `export const SPONZA_SDF_${name} = [${v.join(', ')}];`
  );
  fs.writeFileSync(path.join(assets, 'sponza-sdf.js'),
    ['// Generated by tools/bake_sponza_sdf.mjs, coordinates in world-space meters.', ...constants, ''].join('\n'));
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
