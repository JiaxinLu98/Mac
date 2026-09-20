import mark from './mark.wgsl';
import scatter from './scatter.wgsl';
import { GPUSorter } from './radix_sort/sort';
import { GPUSplit } from './split';
import { GPUPack } from './pack_pairs';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import { readSize } from '../../utils';
import TimestampQueryManager from '../../TimestampQueryManager';

const WORKGROUP_SIZE = 256;
const MAXTUPLES = 128000000;
const MAXWORKGROUP = 65535;

const QUERIES_PER_RADIX_SORT = 8;
const QUERIES_SPLIT = 2; // splitPairs(2)
const QUERIES_PACK = 2;  // packPairs(2)

// --- A: split + sortKeys + pack---
const SORT_A_SLOTS = QUERIES_SPLIT + QUERIES_PER_RADIX_SORT + QUERIES_PACK; // 2 + 8 + 2 = 12

// --- B: radix sort---
const SORT_B_SLOTS = QUERIES_PER_RADIX_SORT; // 8

// --- intersection (mark/scan/scatter)： mark(2) + scan(2) + scatter(2) ---
const INTERSECTION_SLOTS = 6;

// --- offsets ---
const SORT_A_OFFSET = 0;
const SORT_B_OFFSET = SORT_A_OFFSET + SORT_A_SLOTS; // 12
const INTERSECTION_OFFSET = SORT_B_OFFSET + SORT_B_SLOTS; // 20

const QUERIES_PER_ITER_GLOBAL =
  INTERSECTION_OFFSET + INTERSECTION_SLOTS; // 20 + 6 = 26

export class GPUSetIntersectionByKeyUnsorted {
  markPipeline: GPUComputePipeline;
  scatterPipeline: GPUComputePipeline;
  device: GPUDevice;
  timestampQueryManager: TimestampQueryManager;
  bindGroupLayoutMark: GPUBindGroupLayout;
  bindGroupLayoutScatter: GPUBindGroupLayout;

  private iterationIndex: number = 0;
  private queriesPerIter: number = 0;
  private numScanChunks: number = 0;

  constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
    this.device = device;
    this.timestampQueryManager = timestampQueryManager;

    // ---- mark ----
    this.bindGroupLayoutMark = this.device.createBindGroupLayout({
      label: 'mark bind group layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // setA (pairs)
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // setB (keys)
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // flags
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
      ],
    });

    const pipelineLayoutMark = this.device.createPipelineLayout({
      label: 'mark pipeline layout',
      bindGroupLayouts: [this.bindGroupLayoutMark],
    });

    const shader_mark = this.device.createShaderModule({
      label: 'mark shader',
      code: `${mark}`,
    });

    this.markPipeline = this.device.createComputePipeline({
      label: 'mark pipeline',
      layout: pipelineLayoutMark,
      compute: {
        module: shader_mark,
        entryPoint: 'main',
      },
    });

    // ---- scatter ----
    this.bindGroupLayoutScatter = device.createBindGroupLayout({
      label: 'scatter bind group layout',
      entries: [
        {
          // data 
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          // data_size
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'uniform' },
        },
        {
          // valid_flags 
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          // new_data
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          // new_data_size
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
        {
          // is_valid 
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'read-only-storage' },
        },
      ],
    });

    const pipelineLayoutScatter = device.createPipelineLayout({
      label: 'scatter pipeline layout',
      bindGroupLayouts: [this.bindGroupLayoutScatter],
    });

    const shader_scatter = this.device.createShaderModule({
      label: 'scatter shader',
      code: `${scatter}`,
    });

    this.scatterPipeline = device.createComputePipeline({
      label: 'scatter pipeline',
      layout: pipelineLayoutScatter,
      compute: {
        module: shader_scatter,
        entryPoint: 'main',
      },
    });
  }

  public computeQueries(lenA: number) {
    const scan = new ExclusiveScanPipeline(this.device);
    const aligned = scan.getAlignedSize(lenA);
    const maxScanSize = scan.maxScanSize;
    const numChunks = Math.ceil(aligned / maxScanSize);
    this.numScanChunks = numChunks;

    this.queriesPerIter = QUERIES_PER_ITER_GLOBAL;
    return this.queriesPerIter;
  }

  public setIterationIndex(i: number) {
    this.iterationIndex = i;
  }

  private getQueryBaseOffset(): number {
    if (this.queriesPerIter === 0) {
      throw new Error(
        'computeQueries(lenA) must be called before using timestamps.',
      );
    }
    return this.iterationIndex * this.queriesPerIter + INTERSECTION_OFFSET;
  }

  public async computeIntersection(
    setA: Uint32Array, // flattened (key, value) pairs
    setB: Uint32Array, // keys only
    iters: number,
  ) {
    const device = this.device;
    const sorter = new GPUSorter(device, 16, this.timestampQueryManager);
    const split = new GPUSplit(device, this.timestampQueryManager);
    const pack = new GPUPack(device, this.timestampQueryManager);

    if (setA.length % 2 !== 0) {
      throw new Error(
        'setA must contain (key,value) pairs, so its length must be even.',
      );
    }

    const lenA = setA.length / 2;
    const lenB = setB.length;

    this.computeQueries(lenA);
    if (this.queriesPerIter === 0) {
      throw new Error(
        'queriesPerIter is 0; make sure computeQueries(lenA) was called.',
      );
    }

    // Create GPU buffers.
    const bufferA = device.createBuffer({
      label: 'A buffer',
      size: setA.length * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

    const bufferB = device.createBuffer({
      label: 'B buffer',
      size: setB.length * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

    const bufferFlags = this.device.createBuffer({
      label: 'Flags buffer',
      size: MAXTUPLES * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferFlags, 0, new Uint32Array(lenA));

    let lastResult: Uint32Array = new Uint32Array(0);
    let wallTotalMs = 0;
    let intersectionResult: Uint32Array = new Uint32Array(0);

    for (let i = 0; i < iters; i++) {
      this.setIterationIndex(i);
      const t0 = performance.now();

      device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));
      device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

      const baseIter = i * QUERIES_PER_ITER_GLOBAL;

      //
      // 1) setA key （split + radix sort + pack）
      //
      await this.runSortPairsByKeyPhase(
        sorter,
        split,
        pack,
        bufferA,
        lenA,
        baseIter + SORT_A_OFFSET,
      );

      //
      // 2) setB radix sort
      //
      await this.runSortPhase(
        sorter,
        bufferB,
        lenB,
        baseIter + SORT_B_OFFSET,
      );

      //
      // 3) Intersection: mark + scan + scatter
      //
      const recordedFlags = await this.runMarkPhase(
        bufferA,
        bufferB,
        bufferFlags,
        lenA,
        lenB,
      );
      await this.runExclusivePhase(bufferFlags, lenA);
      intersectionResult = await this.runScatterPhase(
        bufferA,
        lenA,
        bufferFlags,
        recordedFlags,
      );

      const t1 = performance.now();
      wallTotalMs += t1 - t0;

      lastResult = intersectionResult;
      recordedFlags.destroy();
    }

    const wallAvgMs = wallTotalMs / iters;
    console.log(
      `SetIntersection x${iters}:`,
      `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
      `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`,
    );

    const timestamps =
      await this.timestampQueryManager.downloadTimestampResult();

    let sumSortAKeys = 0;
    let sumSortB = 0;
    let sumMark = 0;
    let sumScan = 0;
    let sumScatter = 0;

    const radixSortTicks = (start: number) => {
      const zeroTicks = timestamps[start + 1] - timestamps[start + 0];
      const histTicks = timestamps[start + 3] - timestamps[start + 2];
      const prefixTicks = timestamps[start + 5] - timestamps[start + 4];
      const scatterTicks = timestamps[start + 7] - timestamps[start + 6];
      return zeroTicks + histTicks + prefixTicks + scatterTicks;
    };

    const sortPairsByKeyTicks = (base: number) => {
      // splitA : [base+0, base+1]
      const splitTicks = timestamps[base + 1] - timestamps[base + 0];

      // sortA_keys : [base+2 .. base+9]
      const sortStart = base + QUERIES_SPLIT; // base+2
      const sortTicks = radixSortTicks(sortStart);

      // packA : [base+10..base+11]
      const packStart = sortStart + QUERIES_PER_RADIX_SORT; // base+10
      const packTicks = timestamps[packStart + 1] - timestamps[packStart + 0];

      return {
        splitTicks,
        sortTicks,
        packTicks,
        total: splitTicks + sortTicks + packTicks,
      };
    };

    for (let i = 0; i < iters; ++i) {
      const base = i * QUERIES_PER_ITER_GLOBAL;

      // sortA by key
      const aTicks = sortPairsByKeyTicks(base + SORT_A_OFFSET);
      sumSortAKeys += aTicks.total;

      // sortB
      sumSortB += radixSortTicks(base + SORT_B_OFFSET);

      // Intersection: mark + scan + scatter
      const baseIntersection = base + INTERSECTION_OFFSET;

      const markTicks =
        timestamps[baseIntersection + 1] - timestamps[baseIntersection + 0];

      const scanFirstChunkTicks =
        timestamps[baseIntersection + 3] - timestamps[baseIntersection + 2];
      const scanTicksApprox = scanFirstChunkTicks * this.numScanChunks;

      const scatterTicks =
        timestamps[baseIntersection + 5] - timestamps[baseIntersection + 4];

      sumMark += markTicks;
      sumScan += scanTicksApprox;
      sumScatter += scatterTicks;
    }

    const avgTicks = {
      sortAKeys: sumSortAKeys / iters,
      sortB: sumSortB / iters,
      mark: sumMark / iters,
      scan: sumScan / iters,
      scatter: sumScatter / iters,
    };

    const timestampPeriod = 1e-9;

    const avgSeconds = {
      sortAKeys: avgTicks.sortAKeys * timestampPeriod,
      sortB: avgTicks.sortB * timestampPeriod,
      mark: avgTicks.mark * timestampPeriod,
      scan: avgTicks.scan * timestampPeriod,
      scatter: avgTicks.scatter * timestampPeriod,
    };

    console.log(
      `GPU timestamps avg over ${iters} iterations:`,
      `sortAKeys = ${avgSeconds.sortAKeys.toFixed(9)} s,`,
      `sortB     = ${avgSeconds.sortB.toFixed(9)} s,`,
      `mark      = ${avgSeconds.mark.toFixed(9)} s,`,
      `scan      = ${avgSeconds.scan.toFixed(9)} s,`,
      `scatter   = ${avgSeconds.scatter.toFixed(9)} s,`,
      `total     = ${(
        avgSeconds.sortAKeys +
        avgSeconds.sortB +
        avgSeconds.mark +
        avgSeconds.scan +
        avgSeconds.scatter
      ).toFixed(9)} s`,
    );

    bufferA.destroy();
    bufferB.destroy();
    bufferFlags.destroy();

    console.log(lastResult.length / 2);

    return lastResult;
  }

  // === helper phases ===
  private async runSortPairsByKeyPhase(
    sorter: GPUSorter,
    split: GPUSplit,
    pack: GPUPack,
    pairsBuffer: GPUBuffer,
    pairCount: number,
    baseIndex: number,
  ) {
    if (pairCount === 0) return;

    // 1) split pairs -> keys / values
    const sortBuffers = sorter.createSortBuffers(pairCount);
    await split.splitPairs(
      pairsBuffer,
      pairCount,
      sortBuffers.keys,
      sortBuffers.values,
      0,             
      baseIndex,   
    );

    // 2) radix sort keys
    await sorter.sort(
      this.device.queue,
      sortBuffers,
      pairCount,
      baseIndex + QUERIES_SPLIT, // [baseIndex+2 .. baseIndex+9]
    );

    // 3) pack 
    await pack.packPairs(
      sortBuffers.keys,   
      sortBuffers.values,
      pairsBuffer,
      pairCount,
      baseIndex + QUERIES_SPLIT + QUERIES_PER_RADIX_SORT, // [baseIndex+10 .. baseIndex+11]
    );

    sortBuffers.destroy();
  }

  private async runSortPhase(
    sorter: GPUSorter,
    dataBuffer: GPUBuffer,
    length: number,
    baseIndex: number,
  ) {
    if (length === 0) return;

    const device = this.device;
    const sortBuffer = sorter.createSortBuffers(length);

    const initEncoder = device.createCommandEncoder({
      label: 'SortB init copy',
    });
    initEncoder.copyBufferToBuffer(
      dataBuffer,
      0,
      sortBuffer.keysA,
      0,
      length * Uint32Array.BYTES_PER_ELEMENT,
    );

    const zeroPayload = device.createBuffer({
      label: 'radix sort dummy payload upload',
      size: length * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: true,
    });
    new Uint32Array(zeroPayload.getMappedRange()).fill(0);
    zeroPayload.unmap();
    initEncoder.copyBufferToBuffer(
      zeroPayload,
      0,
      sortBuffer.payloadA,
      0,
      length * Uint32Array.BYTES_PER_ELEMENT,
    );
    device.queue.submit([initEncoder.finish()]);

    await sorter.sort(device.queue, sortBuffer, length, baseIndex);

    const outEncoder = device.createCommandEncoder({
      label: 'SortB output copy',
    });
    outEncoder.copyBufferToBuffer(
      sortBuffer.keysA,
      0,
      dataBuffer,
      0,
      length * Uint32Array.BYTES_PER_ELEMENT,
    );
    device.queue.submit([outEncoder.finish()]);

    sortBuffer.destroy();
    zeroPayload.destroy();
  }

  private async runMarkPhase(
    setABuffer: GPUBuffer,
    setBBuffer: GPUBuffer,
    flagsBuffer: GPUBuffer,
    lenA: number,
    lenB: number,
  ) {
    const bufferRecordFlags = this.device.createBuffer({
      label: 'Buffer Record Flags',
      size: lenA * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    const params = this.device.createBuffer({
      label: 'params buffer',
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(params.getMappedRange()).set([lenA, lenB]);
    params.unmap();

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayoutMark,
      entries: [
        { binding: 0, resource: { buffer: setABuffer } },
        { binding: 1, resource: { buffer: setBBuffer } },
        { binding: 2, resource: { buffer: flagsBuffer } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    const workGroup = Math.ceil(lenA / WORKGROUP_SIZE);
    const groupX = Math.min(workGroup, MAXWORKGROUP);
    const groupY = Math.ceil(workGroup / groupX);

    const commandEncoder = this.device.createCommandEncoder({ label: 'Mark' });
    const base = this.getQueryBaseOffset();
    const pass = commandEncoder.beginComputePass(
      this.timestampQueryManager.createComputePassDescriptor(
        base + 0,
        base + 1,
      ),
    );
    pass.setPipeline(this.markPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupX, groupY);
    pass.end();
    this.timestampQueryManager.resolve(commandEncoder);

    commandEncoder.copyBufferToBuffer(
      flagsBuffer,
      0,
      bufferRecordFlags,
      0,
      lenA * Uint32Array.BYTES_PER_ELEMENT,
    );

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    params.destroy();

    return bufferRecordFlags;
  }

  private async runExclusivePhase(flagsBuffer: GPUBuffer, lenA: number) {
    const scan = new ExclusiveScanPipeline(this.device);
    const aligned = scan.getAlignedSize(lenA);
    if (aligned > lenA) {
      const padCount = aligned - lenA;
      this.device.queue.writeBuffer(
        flagsBuffer,
        lenA * Uint32Array.BYTES_PER_ELEMENT,
        new Uint32Array(padCount),
      );
    }

    const numChunks = Math.ceil(aligned / scan.maxScanSize);
    this.numScanChunks = numChunks;

    const base = this.getQueryBaseOffset();
    const scanBaseIndex = base + 2;

    const scanner = scan.prepareGPUInput(flagsBuffer, aligned);
    await scanner.scan(lenA, this.timestampQueryManager, scanBaseIndex);
  }

  public async runScatterPhase(
    bufferASet: GPUBuffer,
    lenA: number,
    bufferFlags: GPUBuffer,
    recordedFlags: GPUBuffer,
  ) {
    const bufferDataSize = this.device.createBuffer({
      label: 'data size buffer',
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(bufferDataSize.getMappedRange()).set([lenA]);
    bufferDataSize.unmap();

    const bufferNewDataSize = this.device.createBuffer({
      label: 'new data size buffer',
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    });

    const bufferResult = this.device.createBuffer({
      label: 'Buffer Result',
      size: lenA * 2 * Uint32Array.BYTES_PER_ELEMENT,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    const bindGroup = this.device.createBindGroup({
      layout: this.bindGroupLayoutScatter,
      entries: [
        { binding: 0, resource: { buffer: bufferASet } },
        { binding: 1, resource: { buffer: bufferDataSize } },
        { binding: 2, resource: { buffer: bufferFlags } },
        { binding: 3, resource: { buffer: bufferResult } },
        { binding: 4, resource: { buffer: bufferNewDataSize } },
        { binding: 5, resource: { buffer: recordedFlags } },
      ],
    });

    const workGroup = Math.ceil(lenA / WORKGROUP_SIZE);
    const groupX = Math.min(workGroup, MAXWORKGROUP);
    const groupY = Math.ceil(workGroup / groupX);

    const commandEncoder = this.device.createCommandEncoder({
      label: 'Scatter',
    });

    const base = this.getQueryBaseOffset();
    const scatterBase = base + 4;

    const pass = commandEncoder.beginComputePass(
      this.timestampQueryManager.createComputePassDescriptor(
        scatterBase,
        scatterBase + 1,
      ),
    );
    pass.setPipeline(this.scatterPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupX, groupY);
    pass.end();

    this.timestampQueryManager.resolve(commandEncoder);

    this.device.queue.submit([commandEncoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();

    // Read Back Set Intersection Result
    const result_size = await readSize(this.device, bufferNewDataSize);
    const commandEncoder1 = this.device.createCommandEncoder();
    const readbackBuffer = this.device.createBuffer({
      label: 'read back set intersection result',
      size: result_size * 2 * Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    commandEncoder1.copyBufferToBuffer(
      bufferResult,
      0,
      readbackBuffer,
      0,
      result_size * 2 * Uint32Array.BYTES_PER_ELEMENT,
    );
    this.device.queue.submit([commandEncoder1.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();
    readbackBuffer.destroy();

    bufferDataSize.destroy();
    bufferNewDataSize.destroy();
    bufferResult.destroy();

    return result;
  }
}
