import { GPUSetDifferenceByKey } from '../set_difference_by_key/set_difference';
import { WebGPUMerger } from './merge/webgpu-merge';
import TimestampQueryManager from '../TimestampQueryManager';

export class GPUSetSymDifferenceByKey {
  device: GPUDevice;

  constructor(device: GPUDevice) {
    this.device = device;
  }
  public async computeSymDiffernece(
    setA: Uint32Array,
    setB: Uint32Array,
    iters: number
  ) {
    const device = this.device;
    const QUERIES_PER_ITER_DIFF  = 6;
    const QUERIES_PER_ITER_MERGE = 4;

    const diffABTimestampManager = new TimestampQueryManager(device, iters * QUERIES_PER_ITER_DIFF);
    const diffBATimestampManager = new TimestampQueryManager(device, iters * QUERIES_PER_ITER_DIFF);
    const mergeTimestampManager = new TimestampQueryManager(device, iters * QUERIES_PER_ITER_MERGE);
    
    const wallStart = performance.now();

    const setDifferenceAB = new GPUSetDifferenceByKey(device, diffABTimestampManager);
    const setDifferenceBA = new GPUSetDifferenceByKey(device, diffBATimestampManager);
    const merger          = new WebGPUMerger(device, mergeTimestampManager);

    // Perform Set Difference
    const set_difference_A_B = await setDifferenceAB.computeDifference(setA, setB, iters);
    const set_difference_B_A = await setDifferenceBA.computeDifference(setB, setA, iters);
    // Perform Merge
    const merge_result = await merger.merge(set_difference_A_B, set_difference_B_A, iters);

    const wallEnd    = performance.now();
    const wallTotalMs = wallEnd - wallStart;
    const wallAvgMs   = wallTotalMs / iters;

    console.log(
      `SymmetricDifference x${iters}:`,
      `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
      `wall avg = ${(wallAvgMs   / 1000).toFixed(9)} s/iter`
    );

    const avgDiffAB = setDifferenceAB.lastAvgSecondsTotal;
    const avgDiffBA = setDifferenceBA.lastAvgSecondsTotal;
    const avgMerge  = merger.lastAvgSecondsTotal;

    const symGpuAvg = avgDiffAB + avgDiffBA + avgMerge;

    console.log(
      `SymDiff GPU avg = ${symGpuAvg.toFixed(9)} s/iter`,
      ` (A\\B=${avgDiffAB.toFixed(9)}, B\\A=${avgDiffBA.toFixed(9)}, merge=${avgMerge.toFixed(9)})`
    );


    const set_symmetric_difference = merge_result;
    return set_symmetric_difference;
  }
}