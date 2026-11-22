// Voxel centers are min + (index + 0.5) * pitch in the mesh's local coordinates.
export class RigidDistanceField {
  constructor(device, { dims, min, pitch, distance, material, normal }) {
    const limit = device.limits.maxTextureDimension3D;
    if (dims?.length !== 3 || !dims.every(n => Number.isInteger(n) && n >= 2 && n <= limit) ||
        min?.length !== 3 || !min.every(Number.isFinite) || !Number.isFinite(pitch) || pitch <= 0) {
      throw new RangeError('Invalid rigid distance field dimensions or voxel grid');
    }
    const count = dims[0] * dims[1] * dims[2];
    if (!(distance instanceof Uint16Array) || distance.length !== count ||
        !(material instanceof Uint8Array) || material.length !== 4 * count ||
        !(normal instanceof Uint8Array) || normal.length !== 4 * count) {
      throw new RangeError('Invalid rigid distance field voxel data');
    }
    this.device = device;
    this.textures = [];
    this.views = [];
    this.params = new ArrayBuffer(32);
    const grid = new Float32Array(this.params);
    grid.set(min);
    grid[3] = pitch;
    try {
      for (const [label, format, bytesPerVoxel, data] of [
        ['distance', 'r16float', 2, distance],
        ['material', 'rgba8unorm', 4, material],
        ['normal', 'rgba8unorm', 4, normal],
      ]) {
        const texture = device.createTexture({
          label: 'Rigid field ' + label, size: dims, dimension: '3d', format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.textures.push(texture);
        device.queue.writeTexture({ texture }, data, {
          bytesPerRow: dims[0] * bytesPerVoxel, rowsPerImage: dims[1],
        }, dims);
        this.views.push(texture.createView());
      }
      this.uniform = device.createBuffer({
        label: 'Rigid field grid and asset root', size: this.params.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    } catch (error) {
      this.destroy();
      throw error;
    }
  }

  setRoot(root) {
    // The geometry arena can repack when another asset is registered or released.
    if (this.root === root) return;
    new Uint32Array(this.params)[4] = root;
    this.device.queue.writeBuffer(this.uniform, 0, this.params);
    this.root = root;
  }

  destroy() {
    for (const texture of this.textures) texture.destroy();
    this.uniform?.destroy();
  }
}
