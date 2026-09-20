/**
 * Set Intersection using Balanced Path with Atomic Operations (Strategy B)
 *
 * Two-phase algorithm:
 * 1. Balanced Path (Global) - Compute DPI with merge path partition points
 * 2. Atomic Phase - Single dispatch that counts, allocates, and writes using atomic operations
 *
 * This is the single-pass approach where atomicAdd is used for global offset allocation.
 * Each workgroup:
 * - Executes SerialSetIntersection
 * - Performs workgroup-level exclusive scan
 * - Thread 0 uses atomicAdd to get global offset
 * - All threads scatter results to output
 *
 * Advantages: Single dispatch, no prefix sum phase
 * Disadvantages: Output order not deterministic, potential atomic contention
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import atomicShader from './set_availability_intersection_atomic.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const MAXWORKGROUP = 65535;

export class GPUSetIntersectionAtomic {
    computeDiagonalsPipeline: GPUComputePipeline;
    atomicPipeline: GPUComputePipeline;
    device: GPUDevice;
    timestampQueryManager: TimestampQueryManager;
    bindGroupLayoutDiagonals: GPUBindGroupLayout;
    bindGroupLayoutAtomic: GPUBindGroupLayout;

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
        // Phase 2: Atomic Pipeline (single-pass with atomic offset allocation)
        // ========================================================================
        this.bindGroupLayoutAtomic = device.createBindGroupLayout({
            label: 'atomic intersection bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // global_counter (atomic)
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const pipelineLayoutAtomic = device.createPipelineLayout({
            label: 'atomic intersection pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutAtomic]
        });

        const shaderAtomic = device.createShaderModule({
            label: 'atomic intersection shader',
            code: atomicShader
        });

        this.atomicPipeline = device.createComputePipeline({
            label: 'atomic intersection pipeline',
            layout: pipelineLayoutAtomic,
            compute: {
                module: shaderAtomic,
                entryPoint: 'compute_availability_atomic'
            }
        });
    }

    public setIterationIndex(i: number) {
        this.iterationIndex = i;
    }

    public computeQueries(lenA: number, lenB: number) {
        // compute diagonals: 2 + atomic: 2 = 4
        return this.queriesPerIter = 4;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries(lenA, lenB) must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Automatically select optimal partition size
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
     * Main entry point: Compute set intersection using 2-phase atomic algorithm
     */
    public async computeIntersection(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number,
        partitionSize?: number,
        valueRange?: number,
        maxOutputSize?: number  // Upper bound for output buffer allocation
    ) {
        if (partitionSize === undefined) {
            partitionSize = GPUSetIntersectionAtomic.getOptimalPartitionSize(setA.length + setB.length, valueRange);
        }

        if (setA.length === 0 || setB.length === 0) {
            return {
                aIndices: new Uint32Array(0),
                bIndices: new Uint32Array(0),
                stars: [] as boolean[],
                result: new Uint32Array(0),
                totalCount: 0,
                partitionSize,
                numWg: 0,
                avgDiagonalsMs: 0,
                avgAtomicMs: 0,
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

        const numWg = Math.ceil((setA.length + setB.length) / partitionSize);

        // For atomic approach, we need to pre-allocate output buffer
        // Use maxOutputSize if provided, otherwise use min(A, B) as upper bound
        const outputBufferSize = maxOutputSize ?? Math.min(setA.length, setB.length);

        let lastAIndices: Uint32Array;
        let lastBIndices: Uint32Array;
        let lastStars: boolean[];
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

            // Phase 2: ATOMIC - Single-pass intersection with atomic offset allocation
            const atomicResult = await this.runAtomicPhase(
                bufferA, bufferB, diagResult.bufferDPI,
                setA.length, setB.length, numWg, outputBufferSize
            );

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);

            lastAIndices = diagResult.aIndices;
            lastBIndices = diagResult.bIndices;
            lastStars = diagResult.stars;
            lastResult = atomicResult.result;
            lastTotalCount = atomicResult.totalCount;

            diagResult.bufferDPI.destroy();
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `[Atomic] ComputeIntersection x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        // Process GPU timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumDiagonals = 0;
        let sumAtomic = 0;

        for (let i = 0; i < iters; ++i) {
            const base = i * QUERIES_PER_ITER;
            sumDiagonals += timestamps[base + 1] - timestamps[base + 0];
            sumAtomic += timestamps[base + 3] - timestamps[base + 2];
        }

        const timestampPeriod = 1e-9;
        const avgDiagonalsMs = (sumDiagonals / iters) * timestampPeriod * 1000;
        const avgAtomicMs = (sumAtomic / iters) * timestampPeriod * 1000;
        const avgTotalMs = avgDiagonalsMs + avgAtomicMs;

        console.log(
            `[Atomic] GPU timestamps avg over ${iters} iterations:`,
            `diagonals = ${avgDiagonalsMs.toFixed(4)} ms,`,
            `atomic = ${avgAtomicMs.toFixed(4)} ms,`,
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
            result: lastResult!,
            totalCount: lastTotalCount,
            partitionSize,
            numWg,
            avgDiagonalsMs,
            avgAtomicMs,
            avgTotalMs
        };
    }

    /**
     * Phase 1: Compute Diagonals (Balanced Path)
     */
    public async runComputeDiagonalsPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number
    ): Promise<{ aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[]; bufferDPI: GPUBuffer }> {

        const device = this.device;

        const dpiSize = 2 * (numWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

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
     * Phase 2: Atomic Phase
     * Single-pass intersection using atomic operations for global offset allocation
     */
    public async runAtomicPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        bufferDPI: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number,
        outputBufferSize: number
    ): Promise<{ result: Uint32Array; totalCount: number }> {
        const device = this.device;

        // Output buffer (pre-allocated)
        const bufferOutput = device.createBuffer({
            label: 'Output buffer',
            size: Math.max(outputBufferSize, 1) * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Global counter for atomic offset allocation (initialized to 0)
        const bufferCounter = device.createBuffer({
            label: 'Global counter',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferCounter, 0, new Uint32Array([0]));

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
            layout: this.bindGroupLayoutAtomic,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOutput } },
                { binding: 4, resource: { buffer: bufferCounter } },
                { binding: 5, resource: { buffer: bufferALength } },
                { binding: 6, resource: { buffer: bufferBLength } },
                { binding: 7, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        const commandEncoder = device.createCommandEncoder({ label: 'Atomic Intersection' });
        const base = this.getQueryBaseOffset();
        const atomicBase = base + 2;
        const pass = commandEncoder.beginComputePass(
            this.timestampQueryManager.createComputePassDescriptor(atomicBase, atomicBase + 1)
        );
        pass.setPipeline(this.atomicPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back global counter to get total count
        const counterReadback = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const counterCopyEncoder = device.createCommandEncoder();
        counterCopyEncoder.copyBufferToBuffer(bufferCounter, 0, counterReadback, 0, 4);
        device.queue.submit([counterCopyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await counterReadback.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(counterReadback.getMappedRange())[0];
        counterReadback.unmap();

        // Read back output (only the valid portion)
        const resultSize = Math.max(totalCount, 1);
        const readbackBuffer = device.createBuffer({
            size: resultSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferOutput, 0, readbackBuffer, 0, resultSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        bufferOutput.destroy();
        bufferCounter.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        counterReadback.destroy();
        readbackBuffer.destroy();

        return { result, totalCount };
    }

    /**
     * CPU reference implementation for validation
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
            console.log(`[Atomic] ✔ Count validation PASSED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        } else {
            console.log(`[Atomic] ✗ Count validation FAILED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        }

        return valid;
    }
}
