const BINS = 2048;
const metric = () => ({count: 0, sum: 0, squares: 0, outliers: 0, histogram: new Uint32Array(BINS + 1)});

function finish(m) {
  if (!m.count) return {pixels: 0};
  let cumulative = 0, q = 0;
  while (q <= BINS) {
    cumulative += m.histogram[q];
    if (cumulative >= 0.95 * m.count) break;
    q++;
  }
  return {pixels: m.count, mae: m.sum / m.count, rms: Math.sqrt(m.squares / m.count),
    p95: q / BINS, fractionAbove5pct: m.outliers / m.count};
}

// Every final-resolution RGB pixel participates. The mask uses the corresponding
// half-resolution geometry sample; edge pixels consequently include raster coverage changes.
function compare(gpu, a, b, mask, m = metric()) {
  for (let y = 0; y < gpu.height; y++) {
    for (let x = 0; x < gpu.width; x++) {
      const cell = Math.floor(y * gpu.aoH / gpu.height) * gpu.aoW + Math.floor(x * gpu.aoW / gpu.width);
      if (!mask[cell]) continue;
      const i = 4 * (y * gpu.width + x);
      const delta = (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2])) / 3;
      m.count++;
      m.sum += delta;
      m.squares += delta * delta;
      m.histogram[Math.min(BINS, Math.round(delta * BINS))]++;
      if (delta > 0.05) m.outliers++;
    }
  }
  return m;
}

function meanImage(frames) {
  const result = new Float32Array(frames[0].length);
  for (const frame of frames) {
    for (let i = 0; i < result.length; i++) result[i] += frame[i] / frames.length;
  }
  return result;
}

const makeMask = (like, predicate) => Uint8Array.from(like, (_, j) => Number(predicate(j)));

export {metric, finish, compare, meanImage, makeMask};
