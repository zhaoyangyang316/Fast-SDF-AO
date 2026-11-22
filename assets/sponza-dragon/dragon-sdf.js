// DDF1, little endian. X varies fastest. Each voxel center is min + (i + 0.5) * pitch.
export function decodeDragonSDF(buffer) {
  if (buffer.byteLength < 32) throw new Error('Truncated dragon field');
  const header = new DataView(buffer);
  const dims = [4, 8, 12].map(offset => header.getUint32(offset, true));
  const min = [16, 20, 24].map(offset => header.getFloat32(offset, true));
  const pitch = header.getFloat32(28, true);
  const count = dims.reduce((product, n) => product * n, 1);
  if (header.getUint32(0, true) !== 0x31464444 || dims.some(n => n < 2) ||
      !min.every(Number.isFinite) || !Number.isFinite(pitch) || pitch <= 0 ||
      32 + 10 * count !== buffer.byteLength) {
    throw new Error('Invalid DDF1 dragon field');
  }
  return {
    dims, min, pitch, span: dims.map(n => n * pitch),
    distance: new Uint16Array(buffer, 32, count),
    material: new Uint8Array(buffer, 32 + 2 * count, 4 * count),
    normal: new Uint8Array(buffer, 32 + 6 * count, 4 * count),
  };
}

export async function loadDragonSDF() {
  const response = await fetch(new URL('./red-dragon-field.bin.gz', import.meta.url));
  if (!response.ok) throw new Error(`Dragon field download failed (${response.status})`);
  const bytes = await response.arrayBuffer();
  if (new Uint8Array(bytes)[0] !== 0x1f) return decodeDragonSDF(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return decodeDragonSDF(await new Response(stream).arrayBuffer());
}
