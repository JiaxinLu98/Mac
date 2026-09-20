import { GPUSetDifferenceByKey } from '../set_difference_by_key/set_difference';
import { WebGPUMerger } from './merge/webgpu-merge';
import TimestampQueryManager from '../TimestampQueryManager';

export class GPUSetUnionByKey {
  device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }
  public async computeUnion(
    setA: Uint32Array,
    setB: Uint32Array,
    iters: number
  ) {
    const device = this.device;

    const QUERIES_PER_ITER_DIFF  = 6; 
    const QUERIES_PER_ITER_MERGE = 4;

    const diffTimestampManager  = new TimestampQueryManager(device, iters * QUERIES_PER_ITER_DIFF);
    const mergeTimestampManager = new TimestampQueryManager(device, iters * QUERIES_PER_ITER_MERGE);

    const set_difference = new GPUSetDifferenceByKey(this.device, diffTimestampManager);
    const merger         = new WebGPUMerger(this.device, mergeTimestampManager);

    const wallStart = performance.now();
    
    const set_difference_result = await set_difference.computeDifference(setB, setA, iters);    
    const merge_result = await merger.merge(setA, set_difference_result, iters);

    const wallEnd = performance.now();
    const wallTotalMs = wallEnd - wallStart;
    const wallAvgMs   = wallTotalMs / iters;

    console.log(
      `SetUnion x${iters}:`,
      `wall total = ${(wallTotalMs / 1000).toFixed(6)} s,`,
      `wall avg = ${(wallAvgMs   / 1000).toFixed(6)} s/iter`
    );

    const unionGpuAvgSeconds =
      set_difference.lastAvgSecondsTotal + merger.lastAvgSecondsTotal;

    console.log(
      `SetUnion GPU avg (difference + merge) = ${unionGpuAvgSeconds.toFixed(9)} s/iter`
    );

    const set_union_result = merge_result;
    return set_union_result;
  }
}


