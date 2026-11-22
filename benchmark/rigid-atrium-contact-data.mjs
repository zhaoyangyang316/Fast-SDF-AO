// Half-resolution inputs to the final spatial upscale of the High-quality AO pass.
export async function collectContactAO(gpu, read, position, normal) {
  const idx = 1 - gpu.pingIdx;
  const input = gpu._lastCompBG === gpu.compBGD1[idx] ? gpu.denoiseA :
    gpu._lastCompBG === gpu.compBGD2[idx] ? gpu.denoiseB : gpu.histAOGI[idx];
  const shaded = await read(input);
  const pixels = gpu.aoW * gpu.aoH;
  const data = new Float32Array(9 * pixels);
  for (let i = 0; i < pixels; ++i) {
    data.set(position.subarray(4*i, 4*i+4), 9*i);
    data.set(normal.subarray(4*i, 4*i+4), 9*i+4);
    data[9*i+8] = shaded[4*i+3];
  }
  const bytes = new Uint8Array(data.buffer);
  let packed = '';
  for (let i = 0; i < bytes.length; i += 32768) {
    packed += String.fromCharCode(...bytes.subarray(i, i+32768));
  }
  return {
    width: gpu.aoW, height: gpu.aoH, frame: gpu.frame,
    target: input === gpu.denoiseA ? 'denoiseA' : input === gpu.denoiseB ? 'denoiseB' : 'histAOGI',
    layout: 'row-major little-endian float32 position.xyz depth normal.xyz surfaceTag visibility',
    base64: btoa(packed),
  };
}
