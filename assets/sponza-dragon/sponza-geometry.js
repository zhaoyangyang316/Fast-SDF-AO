// SGM1 stores little-endian attribute blocks shared by the architectural mesh and dragon.
export function decodeSceneGeometry(buffer) {
  if (buffer.byteLength < 16) throw new Error('Truncated SGM1 mesh header');
  const header = new DataView(buffer);
  const vertices = header.getUint32(4, true);
  const indexCount = header.getUint32(8, true);
  const indexOffset = Math.ceil((16 + vertices * 22) / 4) * 4;
  if (header.getUint32(0, true) !== 0x314d4753 || !vertices || !indexCount || indexCount % 3 ||
      indexOffset + indexCount * 4 !== buffer.byteLength) {
    throw new Error('Invalid SGM1 mesh');
  }
  const positions = new Float32Array(buffer, 16, vertices * 3);
  let offset = 16 + positions.byteLength;
  const normalData = new Int16Array(buffer, offset, vertices * 3);
  offset += normalData.byteLength;
  const colorData = new Uint8Array(buffer, offset, vertices * 3);
  offset += colorData.byteLength;
  const roughData = new Int8Array(buffer, offset, vertices);
  return {
    positions,
    normals: Float32Array.from(normalData, v => v / 32767),
    colors: Float32Array.from(colorData, v => v / 255),
    roughness: Float32Array.from(roughData, v => v / 127),
    indices: new Uint32Array(buffer, indexOffset, indexCount),
  };
}

export async function loadSceneGeometry(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Mesh download failed (${response.status}): ${url}`);
  const bytes = await response.arrayBuffer();
  // Some static hosts automatically decode .gz assets through Content-Encoding.
  if (new Uint8Array(bytes)[0] !== 0x1f) return decodeSceneGeometry(bytes);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return decodeSceneGeometry(await new Response(stream).arrayBuffer());
}

export const decodeSponzaGeometry = decodeSceneGeometry;

export function loadSponzaGeometry() {
  return loadSceneGeometry(new URL('./sponza-mesh.bin.gz', import.meta.url));
}
