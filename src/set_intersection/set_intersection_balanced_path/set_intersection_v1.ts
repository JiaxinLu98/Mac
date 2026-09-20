/**
 * Set Intersection using ModernGPU-style Balanced Path (Strategy A / v1)
 *
 * Four-phase algorithm:
 * 1. Balanced Path (Global) - Compute DPI with merge path partition points
 * 2. Count Phase - Execute SerialSetIntersection to count matches per workgroup
 * 3. Prefix Sum - Exclusive scan on counts to get write offsets
 * 4. Write Phase - Re-execute SerialSetIntersection and scatter results
 *
 * This is the two-pass approach where SerialSetIntersection is executed twice.
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_intersection_count_v1.wgsl';
import writeShader from './set_availability_intersection_write_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const MAXWORKGROUP = 65535;

export class GPUSetIntersectionV1 {
    computeDiagonalsPipeline: GPUComputePipeline;
    countPipeline: GPUComputePipeline;
    writePipeline: GPUComputePipeline;
    device: GPUDevice;
    timestampQueryManager: TimestampQueryManager;
    bindGroupLayoutDiagonals: GPUBindGroupLayout;
    bindGroupLayoutCount: GPUBindGroupLayout;
    bindGroupLayoutWrite: GPUBindGroupLayout;

    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // ========================================================================
        // Phase 1: Balanced Path Pipeline (compute DPI)
        // ========================================================================
        this.bindGroupLayoutDiagonals = device.createBindGroupLayout({
            label: 'compute diagonals bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_uniform
            ]
        });

        const pipelineLayoutDiagonals = device.createPipelineLayout({
            label: 'compute diagonals pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutDiagonals]
        });

        const shaderDiagonals = device.createShaderModule({
            label: 'compute diagonals shader',
            code: computeDiagonalsShader
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals pipeline',
            layout: pipelineLayoutDiagonals,
            compute: {
                module: shaderDiagonals,
                entryPoint: 'compute_diagonals'
            }
        });

        // ========================================================================
        // Phase 2: Count Pipeline (using v1 shader)
        // ========================================================================
        this.bindGroupLayoutCount = device.createBindGroupLayout({
            label: 'count intersections bind group layout (v1)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // counts
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const pipelineLayoutCount = device.createPipelineLayout({
            label: 'count intersections pipeline layout (v1)',
            bindGroupLayouts: [this.bindGroupLayoutCount]
        });

        const shaderCount = device.createShaderModule({
            label: 'count intersections shader (v1)',
            code: countShader
        });

        this.countPipeline = device.createComputePipeline({
            label: 'count intersections pipeline (v1)',
            layout: pipelineLayoutCount,
            compute: {
                module: shaderCount,
                entryPoint: 'count_availability'  // v1 entry point
            }
        });

        // ========================================================================
        // Phase 4: Write Pipeline (using v1 shader)
        // ========================================================================
        this.bindGroupLayoutWrite = device.createBindGroupLayout({
            label: 'write intersections bind group layout (v1)',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // offsets
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const pipelineLayoutWrite = device.createPipelineLayout({
            label: 'write intersections pipeline layout (v1)',
            bindGroupLayouts: [this.bindGroupLayoutWrite]
        });

        const shaderWrite = device.createShaderModule({
            label: 'write intersections shader (v1)',
            code: writeShader
        });

        this.writePipeline = device.createComputePipeline({
            label: 'write intersections pipeline (v1)',
            layout: pipelineLayoutWrite,
            compute: {
                module: shaderWrite,
                entryPoint: 'write_availability'  // v1 entry point
            }
        });
    }

    public setIterationIndex(i: number) {
        this.iterationIndex = i;
    }

    public computeQueries(lenA: number, lenB: number) {
        // compute diagonals: 2 + count: 2 + prefix sum: 2 + write: 2 = 8
        return this.queriesPerIter = 8;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries(lenA, lenB) must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Automatically select optimal partition size based on target workgroup count.
     */
    public static getOptimalPartitionSize(totalElements: number, valueRange?: number): number {
        const BASE_WG = 4096;
        const MIN_WG = 256;
        const MAX_WG = 8000;
        const MAX_PARTITION = 65536;

        const density = valueRange ? totalElements / valueRange : 100;
        const logDensity = Math.log10(density + 1);
        const isDense = density > 1000;

        let minPartition: number;
        if (isDense) {
            if (totalElements <= 16_000_000) {
                minPartition = 4096;
            } else if (totalElements <= 64_000_000) {
                minPartition = 8192;
            } else {
                minPartition = 16384;
            }
        } else {
            if (totalElements <= 16_000_000) {
                minPartition = 16384;
            } else {
                minPartition = 8192;
            }
        }

        const sizeFactor = Math.sqrt(totalElements / 2_000_000);
        const densityFactor = 1.0 + logDensity * 0.1;

        let targetWg = BASE_WG * sizeFactor * densityFactor;
        targetWg = Math.max(MIN_WG, Math.min(MAX_WG, targetWg));

        const rawPartition = totalElements / targetWg;
        const log2Partition = Math.log2(rawPartition);
        let partitionSize = Math.pow(2, Math.round(log2Partition));

        partitionSize = Math.max(minPartition, Math.min(MAX_PARTITION, partitionSize));

        return partitionSize;
    }

    /**
     * Main entry point: Compute set intersection using 4-phase balanced path algorithm
     */
    public async computeIntersection(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number,
        partitionSize?: number,
        valueRange?: number
    ) {
        if (partitionSize === undefined) {
            partitionSize = GPUSetIntersectionV1.getOptimalPartitionSize(setA.length + setB.length, valueRange);
        }

        if (setA.length === 0 || setB.length === 0) {
            return {
                aIndices: new Uint32Array(0),
                bIndices: new Uint32Array(0),
                stars: [] as boolean[],
                counts: new Uint32Array(0),
                offsets: new Uint32Array(0),
                result: new Uint32Array(0),
                totalCount: 0,
                partitionSize,
                numWg: 0,
                avgDiagonalsMs: 0,
                avgCountMs: 0,
                avgPrefixSumMs: 0,
                avgWriteMs: 0,
                avgTotalMs: 0
            };
        }

        const device = this.device;

        this.computeQueries(setA.length, setB.length);
        const QUERIES_PER_ITER = this.queriesPerIter;

        // Create GPU buffers for input arrays
        const bufferA = device.createBuffer({
            label: 'A buffer',
            size: setA.length * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
            label: 'B buffer',
            size: setB.length * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        // Calculate number of workgroups
        const numWg = Math.ceil((setA.length + setB.length) / partitionSize);

        let lastAIndices: Uint32Array;
        let lastBIndices: Uint32Array;
        let lastStars: boolean[];
        let lastCounts: Uint32Array;
        let lastOffsets: Uint32Array;
        let lastResult: Uint32Array;
        let lastTotalCount: number = 0;
        let wallTotalMs = 0;

        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const t0 = performance.now();

            // Phase 1: BALANCED PATH - Compute DPI
            const diagResult = await this.runComputeDiagonalsPhase(
                bufferA, bufferB, setA.length, setB.length, numWg
            );

            // Phase 2: COUNT - Execute SerialSetIntersection to count matches
            const countResult = await this.runCountPhase(
                bufferA, bufferB, diagResult.bufferDPI,
                setA.length, setB.length, numWg
            );

            // Phase 3: PREFIX SUM - Exclusive scan on counts
            const prefixSumResult = await this.runPrefixSumPhase(
                countResult.bufferCounts, numWg
            );

            // Phase 4: WRITE - Re-execute SerialSetIntersection and scatter
            const writeResult = await this.runWritePhase(
                bufferA, bufferB, diagResult.bufferDPI, prefixSumResult.bufferOffsets,
                setA.length, setB.length, numWg, countResult.totalCount
            );

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);

            lastAIndices = diagResult.aIndices;
            lastBIndices = diagResult.bIndices;
            lastStars = diagResult.stars;
            lastCounts = countResult.counts;
            lastOffsets = prefixSumResult.offsets;
            lastResult = writeResult.result;
            lastTotalCount = countResult.totalCount;

            diagResult.bufferDPI.destroy();
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `[V1] ComputeIntersection x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        // Process GPU timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumDiagonals = 0;
        let sumCount = 0;
        let sumPrefixSum = 0;
        let sumWrite = 0;

        for (let i = 0; i < iters; ++i) {
            const base = i * QUERIES_PER_ITER;
            sumDiagonals += timestamps[base + 1] - timestamps[base + 0];
            sumCount += timestamps[base + 3] - timestamps[base + 2];
            sumPrefixSum += timestamps[base + 5] - timestamps[base + 4];
            sumWrite += timestamps[base + 7] - timestamps[base + 6];
        }

        const timestampPeriod = 1e-9;
        const avgDiagonalsMs = (sumDiagonals / iters) * timestampPeriod * 1000;
        const avgCountMs = (sumCount / iters) * timestampPeriod * 1000;
        const avgPrefixSumMs = (sumPrefixSum / iters) * timestampPeriod * 1000;
        const avgWriteMs = (sumWrite / iters) * timestampPeriod * 1000;
        const avgTotalMs = avgDiagonalsMs + avgCountMs + avgPrefixSumMs + avgWriteMs;

        console.log(
            `[V1] GPU timestamps avg over ${iters} iterations:`,
            `diagonals = ${avgDiagonalsMs.toFixed(4)} ms,`,
            `count = ${avgCountMs.toFixed(4)} ms,`,
            `prefixSum = ${avgPrefixSumMs.toFixed(4)} ms,`,
            `write = ${avgWriteMs.toFixed(4)} ms,`,
            `total = ${avgTotalMs.toFixed(4)} ms`
        );

        // Validate count result
        this.validateCount(lastTotalCount, setA, setB);

        bufferA.destroy();
        bufferB.destroy();

        return {
            aIndices: lastAIndices!,
            bIndices: lastBIndices!,
            stars: lastStars!,
            counts: lastCounts!,
            offsets: lastOffsets!,
            result: lastResult!,
            totalCount: lastTotalCount,
            partitionSize,
            numWg,
            avgDiagonalsMs,
            avgCountMs,
            avgPrefixSumMs,
            avgWriteMs,
            avgTotalMs
        };
    }

    /**
     * Phase 1: Compute Diagonals (Balanced Path)
     * Computes merge path partition points with star bits for duplicate handling
     */
    public async runComputeDiagonalsPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number
    ): Promise<{ aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[]; bufferDPI: GPUBuffer }> {

        const device = this.device;

        // DPI buffer: 2 * (num_wg + 1) for a_indices and b_indices
        const dpiSize = 2 * (numWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform buffers
        const bufferALength = device.createBuffer({
            label: 'a_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

        const bufferBLength = device.createBuffer({
            label: 'b_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

        const bufferNumWg = device.createBuffer({
            label: 'num_wg uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWorkgroups]));

        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutDiagonals,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ],
        });

        // 2D dispatch for large workgroup counts
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        const commandEncoder = device.createCommandEncoder({ label: 'Compute Diagonals' });
        const base = this.getQueryBaseOffset();
        const pass = commandEncoder.beginComputePass(
            this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1)
        );
        pass.setPipeline(this.computeDiagonalsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back DPI buffer
        const readbackBuffer = device.createBuffer({
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferDPI, 0, readbackBuffer, 0, dpiSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const dpiData = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Parse DPI data
        const aIndices = new Uint32Array(numWorkgroups + 1);
        const bIndices = new Uint32Array(numWorkgroups + 1);
        const stars: boolean[] = [];

        for (let i = 0; i <= numWorkgroups; i++) {
            const packedA = dpiData[i];
            aIndices[i] = packedA & ~STAR_MASK;
            stars.push((packedA & STAR_MASK) !== 0);
            bIndices[i] = dpiData[numWorkgroups + 1 + i];
        }

        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();
        readbackBuffer.destroy();

        return { aIndices, bIndices, stars, bufferDPI };
    }

    /**
     * Phase 2: Count Phase
     * Executes SerialSetIntersection in each workgroup to count matches
     */
    public async runCountPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        bufferDPI: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number
    ): Promise<{ counts: Uint32Array; totalCount: number; bufferCounts: GPUBuffer }> {

        const device = this.device;

        // Aligned buffer size for prefix sum
        const scan = new ExclusiveScanPipeline(device);
        const alignedSize = scan.getAlignedSize(numWorkgroups);

        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Uniform buffers
        const bufferALength = device.createBuffer({
            label: 'a_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

        const bufferBLength = device.createBuffer({
            label: 'b_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

        const bufferNumWgTotal = device.createBuffer({
            label: 'num_wg_total uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWorkgroups]));

        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutCount,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        const commandEncoder = device.createCommandEncoder({ label: 'Count Intersections (v1)' });
        const base = this.getQueryBaseOffset();
        const countBase = base + 2;
        const pass = commandEncoder.beginComputePass(
            this.timestampQueryManager.createComputePassDescriptor(countBase, countBase + 1)
        );
        pass.setPipeline(this.countPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back counts
        const readbackBuffer = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferCounts, 0, readbackBuffer, 0, numWorkgroups * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        let totalCount = 0;
        for (let i = 0; i < numWorkgroups; i++) {
            totalCount += counts[i];
        }

        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackBuffer.destroy();

        return { counts, totalCount, bufferCounts };
    }

    /**
     * Phase 3: Prefix Sum (Exclusive Scan)
     * Transform counts[] into offsets[] for write phase
     */
    public async runPrefixSumPhase(
        bufferCounts: GPUBuffer,
        numWorkgroups: number
    ): Promise<{ offsets: Uint32Array; bufferOffsets: GPUBuffer }> {
        const device = this.device;

        const scan = new ExclusiveScanPipeline(device);
        const alignedSize = scan.getAlignedSize(numWorkgroups);

        // Pad buffer with zeros if needed
        if (alignedSize > numWorkgroups) {
            const padCount = alignedSize - numWorkgroups;
            device.queue.writeBuffer(
                bufferCounts,
                numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
                new Uint32Array(padCount)
            );
        }

        const base = this.getQueryBaseOffset();
        const scanBaseIndex = base + 4;

        const scanner = scan.prepareGPUInput(bufferCounts, alignedSize);
        await scanner.scan(numWorkgroups, this.timestampQueryManager, scanBaseIndex);

        await device.queue.onSubmittedWorkDone();

        // Read back offsets
        const readbackBuffer = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            bufferCounts, 0,
            readbackBuffer, 0,
            numWorkgroups * Uint32Array.BYTES_PER_ELEMENT
        );
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        readbackBuffer.destroy();

        return { offsets, bufferOffsets: bufferCounts };
    }

    /**
     * Phase 4: Write Phase
     * Re-execute SerialSetIntersection and scatter results to output
     */
    public async runWritePhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        bufferDPI: GPUBuffer,
        bufferOffsets: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number,
        totalCount: number
    ): Promise<{ result: Uint32Array }> {
        const device = this.device;

        const outputSize = Math.max(totalCount, 1);
        const bufferOutput = device.createBuffer({
            label: 'Output buffer',
            size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform buffers
        const bufferALength = device.createBuffer({
            label: 'a_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

        const bufferBLength = device.createBuffer({
            label: 'b_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

        const bufferNumWgTotal = device.createBuffer({
            label: 'num_wg_total uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWorkgroups]));

        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutWrite,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOffsets } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALength } },
                { binding: 6, resource: { buffer: bufferBLength } },
                { binding: 7, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        const commandEncoder = device.createCommandEncoder({ label: 'Write Intersections (v1)' });
        const base = this.getQueryBaseOffset();
        const writeBase = base + 6;
        const pass = commandEncoder.beginComputePass(
            this.timestampQueryManager.createComputePassDescriptor(writeBase, writeBase + 1)
        );
        pass.setPipeline(this.writePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back output
        const readbackBuffer = device.createBuffer({
            size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferOutput, 0, readbackBuffer, 0, outputSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        bufferOutput.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackBuffer.destroy();
        bufferOffsets.destroy();

        return { result };
    }

    /**
     * CPU reference implementation for validation (MULTISET intersection)
     */
    public cpuCountIntersection(setA: Uint32Array, setB: Uint32Array): number {
        const countA = new Map<number, number>();
        for (const val of setA) {
            countA.set(val, (countA.get(val) || 0) + 1);
        }

        const countB = new Map<number, number>();
        for (const val of setB) {
            countB.set(val, (countB.get(val) || 0) + 1);
        }

        let total = 0;
        for (const [val, cntA] of countA) {
            const cntB = countB.get(val) || 0;
            total += Math.min(cntA, cntB);
        }
        return total;
    }

    /**
     * Validate GPU count result against CPU reference
     */
    public validateCount(gpuTotalCount: number, setA: Uint32Array, setB: Uint32Array): boolean {
        const cpuCount = this.cpuCountIntersection(setA, setB);
        const valid = gpuTotalCount === cpuCount;

        if (valid) {
            console.log(`[V1] ✔ Count validation PASSED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        } else {
            console.log(`[V1] ✗ Count validation FAILED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        }

        return valid;
    }
}
