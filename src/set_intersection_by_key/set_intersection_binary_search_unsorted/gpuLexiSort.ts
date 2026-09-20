import { GPUSplit } from './split';
import { GPUSorter } from './radix_sort/sort';
import { GPUPack } from './pack_pairs';

const SORT_QUERIES = 8;

export async function GPULexiSort(
  device: GPUDevice,
  sorter: GPUSorter,
  split: GPUSplit,
  pack: GPUPack,
  pairsBuffer: GPUBuffer,
  pairCount: number,
  baseIndex: number = 0,
) {
  if (pairCount === 0) return;

  // key=y, val=x
  // Radix Sort newT.y
  const sortBuffers = sorter.createSortBuffers(pairCount);

  // Split join result (newT): y->keys.
  await split.splitPairs(pairsBuffer, pairCount, sortBuffers.keys, sortBuffers.values, 1, baseIndex);

  // Call the radix sort pipeline to complete a stable sorting (in ascending order of newT.y).
  await sorter.sort(device.queue, sortBuffers, pairCount, baseIndex + 2);

  // Pack keys and values
  await pack.packPairs(sortBuffers.keys, sortBuffers.values, pairsBuffer, pairCount, baseIndex + 2 + SORT_QUERIES);

  // Release second-stage sort buffers now that packing is done
  sortBuffers.destroy();
}
