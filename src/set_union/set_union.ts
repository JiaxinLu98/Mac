import { GPUSetDifference } from '../set_difference/set_difference';
import { GPUMerger } from '../merge/merge';
import TimestampQueryManager from '../TimestampQueryManager';

export class GPUSetUnion {
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

    const set_difference = new GPUSetDifference(this.device, diffTimestampManager);
    const merger         = new GPUMerger(this.device, mergeTimestampManager);

    const wallStart = performance.now();

    const set_difference_result = await set_difference.computeDifference(setA, setB, iters);

    const merge_result = await merger.merge(set_difference_result, set_difference_result.length, setB, setB.length, iters);

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
    console.log(merge_result.length);
    return merge_result;
  }
}
