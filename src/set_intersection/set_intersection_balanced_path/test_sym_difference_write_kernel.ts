/**
 * Test file for Complete Set Symmetric Difference Pipeline (v1)
 *
 * Tests the full two-pass approach for symmetric difference ((A \ B) ∪ (B \ A)):
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Count Phase - Count elements per workgroup (emit when pA != pB)
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual symmetric difference results
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_sym_difference_count_v1.wgsl';
import writeShader from './set_availability_sym_difference_write_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Complete Set Symmetric Difference Pipeline Test Class
 */
export class TestSetSymDifferencePipeline {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI pipeline
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;

    // Count pipeline
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;

    // Write pipeline
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;

    // Prefix sum
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // DPI bind group layout
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: 'DPI bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.diagPipeline = device.createComputePipeline({
            label: 'DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Count bind group layout (7 bindings, no debug buffer for symmetric difference)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Symmetric Difference Count kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: 'Symmetric Difference Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write bind group layout (8 bindings)
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: 'Symmetric Difference Write kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.writePipeline = device.createComputePipeline({
            label: 'Symmetric Difference Write kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    /**
     * Run the complete set symmetric difference pipeline.
     * All 4 phases (DPI, Count, Scan, Write) are merged into a single command encoder
     * for minimal CPU-GPU synchronization overhead.
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations to run for timing (default: 1)
     * @returns Result array, total count, and average GPU time in ms
     */
    public async computeSetSymDifference(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 1,
        warmup: number = 1
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        gpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return { result: new Uint32Array(0), totalCount: 0, gpuTimeMs: 0 };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase (buffer creation, not timed) ============

        // Create GPU buffers for input
        const bufferA = device.createBuffer({
            label: 'Buffer A',
            size: Math.max(4, setA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
            label: 'Buffer B',
            size: Math.max(4, setB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        // DPI buffer
        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Counts buffer (aligned for scan)
        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Buffer to preserve original counts (for computing total after scan)
        const bufferCountsCopy = device.createBuffer({
            label: 'Buffer Counts Copy',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Pre-allocate output buffer with max possible size (symmetric difference: at most a_len + b_len)
        const maxOutputSize = a_len + b_len;
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        // Create bind groups
        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },  // Will contain offsets after scan
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        // Prepare scanner (creates internal buffers but doesn't execute)
        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        // Wait for data uploads to complete before timing
        await device.queue.onSubmittedWorkDone();

        // ============ GPU Execution: Run Multiple Iterations ============
        const times: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const t0 = performance.now();

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (Diagonal Path Indices)
            let pass = encoder.beginComputePass({ label: 'DPI' });
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count
            pass = encoder.beginComputePass({ label: 'Count' });
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Copy counts before prefix sum modifies them
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum (records multiple passes to the same encoder)
            scanner.recordScanCommands(encoder, numWg);

            // Phase 4: Write
            pass = encoder.beginComputePass({ label: 'Write' });
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Single submit for all phases
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            times.push(performance.now() - t0);
        }

        // Calculate average time (skip first 'warmup' iterations)
        let gpuTimeMs: number;
        if (iterations > warmup) {
            const timesWithoutWarmup = times.slice(warmup);
            gpuTimeMs = timesWithoutWarmup.reduce((a, b) => a + b, 0) / timesWithoutWarmup.length;
        } else {
            gpuTimeMs = times[times.length - 1] || 0;
        }

        // ============ Readback Phase (not part of GPU timing) ============

        // Read offsets and original counts to compute total
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        // Total = last offset + last count (exclusive scan property)
        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        // Read output (only the valid portion)
        let result = new Uint32Array(0);
        if (totalCount > 0) {
            const outputReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutput, 0, outputReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            await outputReadback.mapAsync(GPUMapMode.READ);
            result = new Uint32Array(outputReadback.getMappedRange().slice(0));
            outputReadback.unmap();
            outputReadback.destroy();
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutput.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return { result, totalCount, gpuTimeMs };
    }

    /**
     * CPU reference implementation for symmetric difference.
     * Returns elements in (A \ B) ∪ (B \ A).
     */
    public cpuSetSymDifference(a: Uint32Array, b: Uint32Array): Uint32Array {
        const result: number[] = [];
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                result.push(a[ai]);  // A not in B
                ai++;
            } else if (a[ai] > b[bi]) {
                result.push(b[bi]);  // B not in A
                bi++;
            } else {
                // Equal: in both, skip
                ai++;
                bi++;
            }
        }
        // Remaining elements from A
        while (ai < a.length) {
            result.push(a[ai]);
            ai++;
        }
        // Remaining elements from B
        while (bi < b.length) {
            result.push(b[bi]);
            bi++;
        }
        return new Uint32Array(result);
    }

    /**
     * CPU reference implementation for symmetric difference count.
     */
    public cpuSetSymDifferenceCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                count++;  // A not in B
                ai++;
            } else if (a[ai] > b[bi]) {
                count++;  // B not in A
                bi++;
            } else {
                // Equal: in both, skip
                ai++;
                bi++;
            }
        }
        // Remaining elements
        count += (a.length - ai) + (b.length - bi);
        return count;
    }

    /**
     * Validate GPU results against CPU
     */
    public validateResults(gpuResult: Uint32Array, a: Uint32Array, b: Uint32Array): boolean {
        const cpuResult = this.cpuSetSymDifference(a, b);

        if (gpuResult.length !== cpuResult.length) {
            console.log(`Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
            return false;
        }

        for (let i = 0; i < cpuResult.length; i++) {
            if (gpuResult[i] !== cpuResult[i]) {
                console.log(`Value mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
                // Show context
                const start = Math.max(0, i - 3);
                const end = Math.min(cpuResult.length, i + 4);
                console.log(`  GPU[${start}..${end}]: ${Array.from(gpuResult.slice(start, end))}`);
                console.log(`  CPU[${start}..${end}]: ${Array.from(cpuResult.slice(start, end))}`);
                return false;
            }
        }

        return true;
    }

    /**
     * Run the complete set symmetric difference pipeline with per-phase profiling.
     * Each phase is submitted separately to measure individual timing.
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations (default: 20, first 10 are warmup)
     * @param warmup - Number of warmup iterations (default: 10)
     */
    public async computeSetSymDifferenceWithProfiling(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 20,
        warmup: number = 10
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            countMs: number;
            scanMs: number;
            writeMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                result: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============
        const bufferA = device.createBuffer({
            label: 'Buffer A',
            size: Math.max(4, setA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
            label: 'Buffer B',
            size: Math.max(4, setB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const bufferCountsCopy = device.createBuffer({
            label: 'Buffer Counts Copy',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = a_len + b_len;
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        // Wait for uploads
        await device.queue.onSubmittedWorkDone();

        // ============ Run Multiple Iterations ============
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const totalStart = performance.now();

            // Phase 1: DPI
            const dpiStart = performance.now();
            let encoder = device.createCommandEncoder();
            let pass = encoder.beginComputePass({ label: 'DPI' });
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            dpiTimes.push(performance.now() - dpiStart);

            // Phase 2: Count
            const countStart = performance.now();
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass({ label: 'Count' });
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            countTimes.push(performance.now() - countStart);

            // Phase 3: Prefix Sum
            const scanStart = performance.now();
            await scanner.scan(numWg);
            await device.queue.onSubmittedWorkDone();
            scanTimes.push(performance.now() - scanStart);

            // Phase 4: Write
            const writeStart = performance.now();
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass({ label: 'Write' });
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            writeTimes.push(performance.now() - writeStart);

            totalTimes.push(performance.now() - totalStart);
        }

        // Calculate average (skip first 'warmup' iterations)
        const avg = (arr: number[]) => {
            if (arr.length <= warmup) return arr[arr.length - 1] || 0;
            const withoutWarmup = arr.slice(warmup);
            return withoutWarmup.reduce((a, b) => a + b, 0) / withoutWarmup.length;
        };

        const dpiMs = avg(dpiTimes);
        const countMs = avg(countTimes);
        const scanMs = avg(scanTimes);
        const writeMs = avg(writeTimes);
        const totalMs = avg(totalTimes);

        // ============ Readback ============
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        let result = new Uint32Array(0);
        if (totalCount > 0) {
            const outputReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutput, 0, outputReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            await outputReadback.mapAsync(GPUMapMode.READ);
            result = new Uint32Array(outputReadback.getMappedRange().slice(0));
            outputReadback.unmap();
            outputReadback.destroy();
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutput.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return {
            result,
            totalCount,
            timing: { dpiMs, countMs, scanMs, writeMs, totalMs }
        };
    }

    /**
     * Run the complete set symmetric difference pipeline with pure GPU timestamp profiling.
     * Uses WebGPU timestamp queries for accurate GPU-only timing.
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations (default: 20, first 10 are warmup)
     * @param warmup - Number of warmup iterations (default: 10)
     */
    public async computeSetSymDifferenceWithGPUProfiling(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 20,
        warmup: number = 10
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            countMs: number;
            scanMs: number;
            writeMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                result: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        // Check if timestamp queries are supported
        if (!this.timestampQueryManager.timestampSupported) {
            console.warn('Timestamp queries not supported, falling back to performance.now()');
            return this.computeSetSymDifferenceWithProfiling(setA, setB, iterations, warmup);
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============
        const bufferA = device.createBuffer({
            label: 'Buffer A',
            size: Math.max(4, setA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
            label: 'Buffer B',
            size: Math.max(4, setB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const bufferCountsCopy = device.createBuffer({
            label: 'Buffer Counts Copy',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = a_len + b_len;
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        // Wait for uploads
        await device.queue.onSubmittedWorkDone();

        // ============ Run Multiple Iterations with GPU Timestamps ============
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count (timestamps 2, 3)
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Copy counts before prefix sum modifies them
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum (timestamps 4, 5)
            scanner.recordScanCommands(encoder, numWg, this.timestampQueryManager, 4, 5);

            // Phase 4: Write (timestamps 6, 7)
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(6, 7)
            );
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Resolve timestamps
            this.timestampQueryManager.resolve(encoder);

            // Submit and wait
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Download timestamps (in nanoseconds)
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            if (timestamps.length >= 8) {
                // Convert nanoseconds to milliseconds
                const dpiNs = timestamps[1] - timestamps[0];
                const countNs = timestamps[3] - timestamps[2];
                const writeNs = timestamps[7] - timestamps[6];
                const totalNs = timestamps[7] - timestamps[0];

                // Calculate Scan time indirectly
                const scanNsIndirect = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNsIndirect / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        // Calculate average (skip first 'warmup' iterations)
        const avg = (arr: number[]) => {
            if (arr.length <= warmup) return arr[arr.length - 1] || 0;
            const withoutWarmup = arr.slice(warmup);
            return withoutWarmup.reduce((a, b) => a + b, 0) / withoutWarmup.length;
        };

        const dpiMs = avg(dpiTimes);
        const countMs = avg(countTimes);
        const scanMs = avg(scanTimes);
        const writeMs = avg(writeTimes);
        const totalMs = avg(totalTimes);

        // ============ Readback ============
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        let result = new Uint32Array(0);
        if (totalCount > 0) {
            const outputReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutput, 0, outputReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            await outputReadback.mapAsync(GPUMapMode.READ);
            result = new Uint32Array(outputReadback.getMappedRange().slice(0));
            outputReadback.unmap();
            outputReadback.destroy();
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutput.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return {
            result,
            totalCount,
            timing: { dpiMs, countMs, scanMs, writeMs, totalMs }
        };
    }
}

/**
 * Run complete pipeline tests for symmetric difference
 */
export async function runSymDifferenceWriteKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE COMPLETE PIPELINE TEST       ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    let allPassed = true;

    // Test Case 1: Small arrays
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Small arrays');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 3, 4, 7, 8, 10]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        // A \ B = [1, 5, 9], B \ A = [2, 4, 8, 10] => Sym Diff = [1, 2, 4, 5, 8, 9, 10]

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        console.log('GPU result:', Array.from(result));
        console.log('CPU result:', Array.from(tester.cpuSetSymDifference(A, B)));

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 2: No overlap (all different)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: No overlap (all different)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);
        // Sym diff = all elements

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        console.log(`Result length: ${result.length}, expected: ${A.length + B.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 3: Complete overlap (all same)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Complete overlap (all same)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([1, 2, 3, 4, 5]);
        // Sym diff = empty

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        console.log(`Result length: ${result.length}, expected: 0`);

        const valid = result.length === 0;
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 4: Multi-workgroup
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: Multi-workgroup (even vs multiples of 3)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i * 2;      // 0, 2, 4, 6, ...
            B[i] = i * 3;      // 0, 3, 6, 9, ...
        }

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        const cpuResult = tester.cpuSetSymDifference(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 5: Duplicates with overlap
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Duplicates with overlap');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 1, 2, 2, 3, 3]);
        const B = new Uint32Array([2, 2, 3, 3, 4, 4]);
        // A \ B = [1, 1], B \ A = [4, 4] => Sym Diff = [1, 1, 4, 4]
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        console.log('GPU result:', Array.from(result));
        console.log('CPU result:', Array.from(tester.cpuSetSymDifference(A, B)));

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 6: Large dataset
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Large dataset (10K elements each)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 10000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i;         // 0, 1, 2, ..., 9999
            B[i] = i + 5000;  // 5000, 5001, ..., 14999
        }
        // Overlap: 5000-9999, Sym diff: 0-4999 + 10000-14999 => 10000 elements

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        const cpuResult = tester.cpuSetSymDifference(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 7: All same value (multi-WG)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: All same value (multi-WG)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(2500);
        const B = new Uint32Array(2500);
        A.fill(42);
        B.fill(42);
        // All same => complete overlap => sym diff = empty

        const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B);
        console.log(`Result length: GPU=${result.length}, expected: 0`);

        const valid = result.length === 0;
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL SYMMETRIC DIFFERENCE PIPELINE TESTS PASSED          ║');
    } else {
        console.log('║  ✗ SOME SYMMETRIC DIFFERENCE PIPELINE TESTS FAILED         ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Benchmark with real datasets from public/data directory
 */
export async function runSymDifferenceBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE DATASET BENCHMARK            ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    // Dataset configurations to test
    const datasets = [
        { size: '1', range: 'e2', desc: '1M elements, range 100 (many duplicates)' },
        { size: '1', range: 'e6', desc: '1M elements, range 1M (few duplicates)' },
        { size: '2', range: 'e2', desc: '2M elements, range 100' },
        { size: '2', range: 'e6', desc: '2M elements, range 1M' },
        { size: '4', range: 'e2', desc: '4M elements, range 100' },
        { size: '4', range: 'e6', desc: '4M elements, range 1M' },
        { size: '8', range: 'e2', desc: '8M elements, range 100' },
        { size: '8', range: 'e6', desc: '8M elements, range 1M' },
        { size: '16', range: 'e2', desc: '16M elements, range 100' },
        { size: '16', range: 'e6', desc: '16M elements, range 1M' },
        { size: '32', range: 'e2', desc: '32M elements, range 100' },
        { size: '32', range: 'e6', desc: '32M elements, range 1M' },
        { size: '64', range: 'e2', desc: '64M elements, range 100' },
        { size: '64', range: 'e6', desc: '64M elements, range 1M' },
        { size: '128', range: 'e2', desc: '128M elements, range 100' },
        { size: '128', range: 'e6', desc: '128M elements, range 1M' },
    ];

    const results: Array<{
        dataset: string;
        aLen: number;
        bLen: number;
        symDiffSize: number;
        gpuTimeMs: number;
        valid: boolean;
    }> = [];

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.log(`Dataset: ${size}${range} - ${desc}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        try {
            console.log(`Loading ${aPath}...`);
            const A = await utils.loadUint32ArrayFromBin(aPath);
            console.log(`Loading ${bPath}...`);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            console.log(`A: ${A.length.toLocaleString()} elements`);
            console.log(`B: ${B.length.toLocaleString()} elements`);

            // Run GPU symmetric difference
            const NUM_ITERATIONS = 10;
            const NUM_WARMUP = 3;
            const t0 = performance.now();
            const { result, totalCount, gpuTimeMs } = await tester.computeSetSymDifference(A, B, NUM_ITERATIONS, NUM_WARMUP);
            const totalTime = performance.now() - t0;

            console.log(`GPU result: ${result.length.toLocaleString()} elements`);
            console.log(`GPU kernel time (avg of ${NUM_ITERATIONS - NUM_WARMUP} runs, excluding ${NUM_WARMUP} warmup): ${gpuTimeMs.toFixed(2)} ms`);
            console.log(`Total time (${NUM_ITERATIONS} iterations + setup): ${totalTime.toFixed(2)} ms`);

            // Validate against CPU (only for smaller datasets)
            let valid = true;
            if (A.length <= 4_000_000) {
                console.log('Validating against CPU...');
                const cpuT0 = performance.now();
                valid = tester.validateResults(result, A, B);
                const cpuTime = performance.now() - cpuT0;
                console.log(`CPU validation time: ${cpuTime.toFixed(2)} ms`);
                console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
            } else {
                console.log('(Skipping CPU validation for large dataset)');
            }

            results.push({
                dataset: `${size}${range}`,
                aLen: A.length,
                bLen: B.length,
                symDiffSize: result.length,
                gpuTimeMs,
                valid
            });

            // Calculate throughput
            const totalElements = A.length + B.length;
            const throughput = totalElements / (gpuTimeMs / 1000) / 1_000_000;
            console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec\n`);

        } catch (error) {
            console.log(`Error loading dataset: ${error}`);
            console.log('');
        }
    }

    // Print summary table
    console.log('\n╔════════════════════════════════════════════════════════════════════════════╗');
    console.log('║                        BENCHMARK SUMMARY                                    ║');
    console.log('╠══════════╤══════════════╤══════════════╤════════════════╤═════════╤════════╣');
    console.log('║ Dataset  │ A size       │ B size       │ Sym Diff Size  │ Time(ms)│ Valid  ║');
    console.log('╠══════════╪══════════════╪══════════════╪════════════════╪═════════╪════════╣');

    for (const r of results) {
        const ds = r.dataset.padEnd(8);
        const aLen = r.aLen.toLocaleString().padStart(12);
        const bLen = r.bLen.toLocaleString().padStart(12);
        const symSize = r.symDiffSize.toLocaleString().padStart(14);
        const time = r.gpuTimeMs.toFixed(2).padStart(7);
        const valid = r.valid ? '  ✔   ' : '  ✗   ';
        console.log(`║ ${ds} │ ${aLen} │ ${bLen} │ ${symSize} │ ${time} │${valid}║`);
    }

    console.log('╚══════════╧══════════════╧══════════════╧════════════════╧═════════╧════════╝\n');
}

/**
 * Run profiling benchmark to identify performance bottlenecks per phase
 */
export async function runSymDifferenceProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE PROFILING BENCHMARK          ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    // Datasets to profile
    const datasets = [
        { size: '1', range: 'e2', desc: '1M elements, range 100' },
        { size: '1', range: 'e6', desc: '1M elements, range 1M' },
        { size: '2', range: 'e2', desc: '2M elements, range 100' },
        { size: '2', range: 'e6', desc: '2M elements, range 1M' },
        { size: '4', range: 'e2', desc: '4M elements, range 100' },
        { size: '4', range: 'e6', desc: '4M elements, range 1M' },
        { size: '8', range: 'e2', desc: '8M elements, range 100' },
        { size: '8', range: 'e6', desc: '8M elements, range 1M' },
        { size: '16', range: 'e2', desc: '16M elements, range 100' },
        { size: '16', range: 'e6', desc: '16M elements, range 1M' },
        { size: '32', range: 'e2', desc: '32M elements, range 100' },
        { size: '32', range: 'e6', desc: '32M elements, range 1M' },
        { size: '64', range: 'e2', desc: '64M elements, range 100' },
        { size: '64', range: 'e6', desc: '64M elements, range 1M' },
        { size: '128', range: 'e2', desc: '128M elements, range 100' },
        { size: '128', range: 'e6', desc: '128M elements, range 1M' },
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Sym Diff Size  │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetSymDifferenceWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const symSize = result.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${symSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');
}

/**
 * Run profiling benchmark using pure GPU timestamps.
 */
export async function runSymDifferenceGPUProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE GPU TIMESTAMP PROFILING      ║');
    console.log('║      (Pure GPU time, no CPU overhead)                      ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        console.log('Falling back to performance.now() based profiling...\n');
        return runSymDifferenceProfilingBenchmark(device);
    }

    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    // Datasets to profile
    const datasets = [
        { size: '1', range: 'e2', desc: '1M elements, range 100' },
        { size: '1', range: 'e6', desc: '1M elements, range 1M' },
        { size: '2', range: 'e2', desc: '2M elements, range 100' },
        { size: '2', range: 'e6', desc: '2M elements, range 1M' },
        { size: '4', range: 'e2', desc: '4M elements, range 100' },
        { size: '4', range: 'e6', desc: '4M elements, range 1M' },
        { size: '8', range: 'e2', desc: '8M elements, range 100' },
        { size: '8', range: 'e6', desc: '8M elements, range 1M' },
        { size: '16', range: 'e2', desc: '16M elements, range 100' },
        { size: '16', range: 'e6', desc: '16M elements, range 1M' },
        { size: '32', range: 'e2', desc: '32M elements, range 100' },
        { size: '32', range: 'e6', desc: '32M elements, range 1M' },
        { size: '64', range: 'e2', desc: '64M elements, range 100' },
        { size: '64', range: 'e6', desc: '64M elements, range 1M' },
        { size: '128', range: 'e2', desc: '128M elements, range 100' },
        { size: '128', range: 'e6', desc: '128M elements, range 1M' },
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Sym Diff Size  │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetSymDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const symSize = result.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${symSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');

    console.log('Analysis hints (GPU timestamp version):');
    console.log('  - These are pure GPU execution times without CPU overhead');
    console.log('  - If Scan(ms) dominates: Prefix sum is the bottleneck');
    console.log('  - If Write(ms) dominates: Output writing is the bottleneck');
    console.log('  - If Count(ms) dominates: Main computation is the bottleneck');
    console.log('  - Total = DPI + Count + Scan + Write (no gaps between phases)\n');
}

/**
 * Run 128e2 specific profiling benchmark
 */
export async function runSymDifference128e2ProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE 128e2 PROFILING              ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const aPath = `./data/A_128e2.bin`;
    const bPath = `./data/B_128e2.bin`;

    try {
        console.log(`Loading ${aPath}...`);
        const A = await utils.loadUint32ArrayFromBin(aPath);
        console.log(`Loading ${bPath}...`);
        const B = await utils.loadUint32ArrayFromBin(bPath);

        console.log(`A: ${A.length.toLocaleString()} elements`);
        console.log(`B: ${B.length.toLocaleString()} elements`);
        console.log(`Running ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup)...\n`);

        const { result, totalCount, timing } = await tester.computeSetSymDifferenceWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

        console.log(`Symmetric Difference size: ${result.length.toLocaleString()}`);
        console.log(`\nPer-phase timing:`);
        console.log(`  DPI:   ${timing.dpiMs.toFixed(3)} ms`);
        console.log(`  Count: ${timing.countMs.toFixed(3)} ms`);
        console.log(`  Scan:  ${timing.scanMs.toFixed(3)} ms`);
        console.log(`  Write: ${timing.writeMs.toFixed(3)} ms`);
        console.log(`  Total: ${timing.totalMs.toFixed(3)} ms`);

        const totalElements = A.length + B.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`\nThroughput: ${throughput.toFixed(2)} M elements/sec`);

    } catch (error) {
        console.log(`Error: ${error}`);
    }
}

/**
 * Run 128e6 specific profiling benchmark
 */
export async function runSymDifference128e6ProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET SYMMETRIC DIFFERENCE 128e6 PROFILING              ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetSymDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const aPath = `./data/A_128e6.bin`;
    const bPath = `./data/B_128e6.bin`;

    try {
        console.log(`Loading ${aPath}...`);
        const A = await utils.loadUint32ArrayFromBin(aPath);
        console.log(`Loading ${bPath}...`);
        const B = await utils.loadUint32ArrayFromBin(bPath);

        console.log(`A: ${A.length.toLocaleString()} elements`);
        console.log(`B: ${B.length.toLocaleString()} elements`);
        console.log(`Running ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup)...\n`);

        const { result, totalCount, timing } = await tester.computeSetSymDifferenceWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

        console.log(`Symmetric Difference size: ${result.length.toLocaleString()}`);
        console.log(`\nPer-phase timing:`);
        console.log(`  DPI:   ${timing.dpiMs.toFixed(3)} ms`);
        console.log(`  Count: ${timing.countMs.toFixed(3)} ms`);
        console.log(`  Scan:  ${timing.scanMs.toFixed(3)} ms`);
        console.log(`  Write: ${timing.writeMs.toFixed(3)} ms`);
        console.log(`  Total: ${timing.totalMs.toFixed(3)} ms`);

        const totalElements = A.length + B.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`\nThroughput: ${throughput.toFixed(2)} M elements/sec`);

    } catch (error) {
        console.log(`Error: ${error}`);
    }
}
