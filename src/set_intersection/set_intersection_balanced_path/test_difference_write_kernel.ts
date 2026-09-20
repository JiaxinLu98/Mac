/**
 * Test file for Complete Set Difference Pipeline (v1)
 *
 * Tests the full two-pass approach for multiset difference (A \ B):
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Count Phase - Count difference elements per workgroup
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual difference results
 *
 * Multiset difference semantics:
 * - If A has 5 copies of x and B has 3 copies, result has 2 copies
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_difference_count_v1.wgsl';
import writeShader from './set_availability_difference_write_v1.wgsl';
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
 * Complete Set Difference Pipeline Test Class
 */
export class TestSetDifferencePipeline {
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

        // Count bind group layout (with debug buffer)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Difference Count kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: 'Difference Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write bind group layout
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: 'Difference Write kernel bind group layout',
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
            label: 'Difference Write kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    /**
     * Run the complete set difference pipeline.
     * All 4 phases (DPI, Count, Scan, Write) are merged into a single command encoder
     * for minimal CPU-GPU synchronization overhead.
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations to run for timing (default: 1)
     * @returns Result array, total count, and average GPU time in ms
     */
    public async computeSetDifference(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 20,
        warmup: number = 10
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

        // Debug buffer for count kernel
        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            label: 'Buffer Debug',
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Pre-allocate output buffer with max possible size (all of A)
        const maxOutputSize = a_len;
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
                { binding: 7, resource: { buffer: bufferDebug } },
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
        bufferDebug.destroy();
        bufferOutput.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return { result, totalCount, gpuTimeMs };
    }

    /**
     * CPU reference implementation for multiset difference
     * If A has 5 copies of x and B has 3 copies, result has 2 copies
     */
    public cpuSetDifference(a: Uint32Array, b: Uint32Array): Uint32Array {
        const result: number[] = [];
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                result.push(a[ai]);
                ai++;
            } else if (a[ai] > b[bi]) {
                bi++;
            } else {
                // Equal: one B cancels one A, no output
                ai++;
                bi++;
            }
        }
        // Remaining A elements are all in difference
        while (ai < a.length) {
            result.push(a[ai]);
            ai++;
        }
        return new Uint32Array(result);
    }

    public cpuSetDifferenceCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                count++;
                ai++;
            } else if (a[ai] > b[bi]) {
                bi++;
            } else {
                ai++;
                bi++;
            }
        }
        count += (a.length - ai);
        return count;
    }

    /**
     * Validate GPU results against CPU
     */
    public validateResults(gpuResult: Uint32Array, a: Uint32Array, b: Uint32Array): boolean {
        const cpuResult = this.cpuSetDifference(a, b);

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
     * Run the complete set difference pipeline with per-phase GPU timestamp profiling.
     * Each phase is submitted separately to measure individual timing using GPU timestamps.
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations (default: 10, first is warmup)
     */
    public async computeSetDifferenceWithProfiling(
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

        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            label: 'Buffer Debug',
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = a_len;
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
                { binding: 7, resource: { buffer: bufferDebug } },
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

        // ============ Run Multiple Iterations with GPU Timestamps (separate submissions) ============
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            // Phase 1: DPI (timestamps 0, 1)
            let encoder = device.createCommandEncoder();
            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            const dpiTimestamps = await this.timestampQueryManager.downloadTimestampResult();
            const dpiNs = dpiTimestamps[1] - dpiTimestamps[0];

            // Phase 2: Count (timestamps 0, 1)
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            const countTimestamps = await this.timestampQueryManager.downloadTimestampResult();
            const countNs = countTimestamps[1] - countTimestamps[0];

            // Phase 3: Prefix Sum (timestamps 0, 1)
            encoder = device.createCommandEncoder();
            scanner.recordScanCommands(encoder, numWg, this.timestampQueryManager, 0, 1);
            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            const scanTimestamps = await this.timestampQueryManager.downloadTimestampResult();
            const scanNs = scanTimestamps[1] - scanTimestamps[0];

            // Phase 4: Write (timestamps 0, 1)
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            const writeTimestamps = await this.timestampQueryManager.downloadTimestampResult();
            const writeNs = writeTimestamps[1] - writeTimestamps[0];

            dpiTimes.push(dpiNs / 1_000_000);
            countTimes.push(countNs / 1_000_000);
            scanTimes.push(scanNs / 1_000_000);
            writeTimes.push(writeNs / 1_000_000);
            totalTimes.push((dpiNs + countNs + scanNs + writeNs) / 1_000_000);
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
        bufferDebug.destroy();
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
     * Run the complete set difference pipeline with pure GPU timestamp profiling.
     * Uses WebGPU timestamp queries for accurate GPU-only timing.
     * All 4 phases are in a single command encoder, submitted once per iteration.
     *
     * Timestamp indices:
     * - DPI: 0 (begin), 1 (end)
     * - Count: 2 (begin), 3 (end)
     * - Scan: 4 (begin), 5 (end)
     * - Write: 6 (begin), 7 (end)
     *
     * @param setA - First sorted input array
     * @param setB - Second sorted input array
     * @param iterations - Number of iterations (default: 10, first is warmup)
     */
    public async computeSetDifferenceWithGPUProfiling(
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

        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            label: 'Buffer Debug',
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = a_len;
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
                { binding: 7, resource: { buffer: bufferDebug } },
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

        // ============ Run Multiple Iterations with GPU Timestamps (merged submission) ============
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

                // Calculate Scan time indirectly: Scan = Total - DPI - Count - Write
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
        bufferDebug.destroy();
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
 * Run complete pipeline tests for set difference
 */
export async function runDifferenceWriteKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET DIFFERENCE COMPLETE PIPELINE TEST                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetDifferencePipeline(device, timestampQueryManager);

    let allPassed = true;

    // Test Case 1: Basic multiset difference
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Basic multiset difference');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 2, 3, 3, 3, 4, 5]);
        const B = new Uint32Array([2, 3, 3, 6]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        // Expected: [1, 2, 3, 4, 5] (one 2 and one 3 remain after B cancels)

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('GPU result:', Array.from(result));
        console.log('CPU result:', Array.from(tester.cpuSetDifference(A, B)));

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 2: No overlap
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: No overlap (all A elements in result)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('GPU result:', Array.from(result));
        console.log(`Result length: ${result.length}, expected: ${A.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 3: Complete overlap
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Complete overlap (empty result)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log(`Result length: ${result.length}, expected: 0`);

        const valid = result.length === 0;
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 4: B has more copies than A
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: B has more copies than A');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([5, 5, 5]);  // 3 copies of 5
        const B = new Uint32Array([5, 5, 5, 5, 5]);  // 5 copies of 5
        // Expected: [] (all A's 5s are cancelled by B)

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('GPU result:', Array.from(result));
        console.log(`Result length: ${result.length}, expected: 0`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 5: A has more copies than B
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: A has more copies than B');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([5, 5, 5, 5, 5]);  // 5 copies of 5
        const B = new Uint32Array([5, 5, 5]);  // 3 copies of 5
        // Expected: [5, 5] (2 copies remain)

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('GPU result:', Array.from(result));
        console.log(`Result length: ${result.length}, expected: 2`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 6: Empty B
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Empty B (all A in result)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([]);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('GPU result:', Array.from(result));
        console.log(`Result length: ${result.length}, expected: ${A.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 7: Multi-workgroup test
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: Multi-workgroup (even vs multiples of 3)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i * 2;      // 0, 2, 4, 6, ...
            B[i] = i * 3;      // 0, 3, 6, 9, ...
        }

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        const cpuResult = tester.cpuSetDifference(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 8: All same value with duplicates
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 8: All same value (100 elements each)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(100);
        const B = new Uint32Array(60);
        A.fill(5);
        B.fill(5);
        // Expected: 40 copies of 5 (100 - 60)

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log(`Result length: ${result.length}, expected: 40`);
        console.log(`First 10 values: ${Array.from(result.slice(0, 10))}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 9: Large duplicates spanning workgroups
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 9: Large duplicates spanning workgroups');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        const B = new Uint32Array(2000);
        A.fill(42);
        B.fill(42);
        // Expected: 1000 copies of 42

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log(`Result length: GPU=${result.length}, expected: 1000`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 10: Star bit trigger case
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 10: Star bit trigger case');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        A[0] = 0;
        A.fill(1, 1);

        const B = new Uint32Array(1500);
        B.fill(1);
        // Expected: [0] + 1499 copies of 1

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        const cpuResult = tester.cpuSetDifference(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 11: Large dataset
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 11: Large dataset (10K elements each)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 10000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i;          // 0, 1, 2, ..., 9999
            B[i] = i * 2;      // 0, 2, 4, ..., 19998
        }

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        const cpuResult = tester.cpuSetDifference(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 12: Complex mixed duplicates
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 12: Complex mixed duplicates');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // A has: 1(x3), 2(x2), 3(x5), 4(x1), 5(x4)
        // B has: 1(x1), 2(x3), 3(x2), 5(x2)
        // Expected: 1(x2), 3(x3), 4(x1), 5(x2)
        const A = new Uint32Array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5]);
        const B = new Uint32Array([1, 2, 2, 2, 3, 3, 5, 5]);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetDifference(A, B);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('GPU result:', Array.from(result));
        console.log('CPU result:', Array.from(tester.cpuSetDifference(A, B)));

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL DIFFERENCE PIPELINE TESTS PASSED                    ║');
    } else {
        console.log('║  ✗ SOME DIFFERENCE PIPELINE TESTS FAILED                   ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Run profiling benchmark for set difference using GPU timestamps (separate submissions)
 * Includes cooling delays between large datasets to prevent thermal throttling.
 */
export async function runDifferenceProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET DIFFERENCE PROFILING BENCHMARK                    ║');
    console.log('║      (GPU timestamps, separate submissions)                ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 50;
    const NUM_WARMUP = 20;
    const COOLING_DELAY_MS = 8000;  // 8 seconds cooling delay for large datasets
    const EXTRA_COOLING_DELAY_MS = 15000;  // 15 seconds extra delay before 128M datasets
    const LARGE_DATASET_THRESHOLD = 32;  // Apply cooling delay for datasets >= 32M

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...`);
    console.log(`Cooling delay: ${COOLING_DELAY_MS}ms between datasets >= ${LARGE_DATASET_THRESHOLD}M elements`);
    console.log(`Extra cooling delay: ${EXTRA_COOLING_DELAY_MS}ms before 128M datasets\n`);

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
    console.log('║ Dataset  │ Input Size   │ Difference     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;
        const sizeNum = parseInt(size);

        // Extra long cooling delay before 128M datasets
        if (sizeNum === 128) {
            console.log(`║ -------- │ Cooling ${EXTRA_COOLING_DELAY_MS / 1000}s before 128M... │ -------- │ ------ │ ------ │ ------ │ ------ │ ------ ║`);
            await new Promise(resolve => setTimeout(resolve, EXTRA_COOLING_DELAY_MS));
        }

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetDifferenceWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const diffSize = result.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${diffSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

            // Cooling delay for large datasets to prevent thermal throttling
            if (sizeNum >= LARGE_DATASET_THRESHOLD) {
                await new Promise(resolve => setTimeout(resolve, COOLING_DELAY_MS));
            }

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');
}

/**
 * Run profiling benchmark for set difference using pure GPU timestamps (merged submission)
 * Includes cooling delays between large datasets to prevent thermal throttling.
 */
export async function runDifferenceGPUProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET DIFFERENCE GPU TIMESTAMP PROFILING                ║');
    console.log('║      (Pure GPU time, merged submission)                    ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    const COOLING_DELAY_MS = 8000;  // 8 seconds cooling delay for large datasets
    const EXTRA_COOLING_DELAY_MS = 15000;  // 15 seconds extra delay before 128M datasets
    const LARGE_DATASET_THRESHOLD = 32;  // Apply cooling delay for datasets >= 32M

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...`);
    console.log(`Cooling delay: ${COOLING_DELAY_MS}ms between datasets >= ${LARGE_DATASET_THRESHOLD}M elements`);
    console.log(`Extra cooling delay: ${EXTRA_COOLING_DELAY_MS}ms before 128M datasets\n`);

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
    console.log('║ Dataset  │ Input Size   │ Difference     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;
        const sizeNum = parseInt(size);

        // Extra long cooling delay before 128M datasets
        if (sizeNum === 128) {
            console.log(`║ -------- │ Cooling ${EXTRA_COOLING_DELAY_MS / 1000}s before 128M... │ -------- │ ------ │ ------ │ ------ │ ------ │ ------ ║`);
            await new Promise(resolve => setTimeout(resolve, EXTRA_COOLING_DELAY_MS));
        }

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const diffSize = result.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${diffSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

            // Cooling delay for large datasets to prevent thermal throttling
            if (sizeNum >= LARGE_DATASET_THRESHOLD) {
                await new Promise(resolve => setTimeout(resolve, COOLING_DELAY_MS));
            }

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
    console.log('  - DPI should always be small relative to others');
    console.log('  - Total = DPI + Count + Scan + Write (no gaps between phases)\n');
}

/**
 * Run benchmark for only 128e6 dataset (to test without thermal throttling)
 */
export async function runDifference128e6Benchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET DIFFERENCE BENCHMARK - 128e6 ONLY                 ║');
    console.log('║      (Cold start test for thermal throttling check)        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const aPath = './data/A_128e6.bin';
    const bPath = './data/B_128e6.bin';

    try {
        console.log(`Loading ${aPath}...`);
        const A = await utils.loadUint32ArrayFromBin(aPath);
        console.log(`Loading ${bPath}...`);
        const B = await utils.loadUint32ArrayFromBin(bPath);

        console.log(`A: ${A.length.toLocaleString()} elements`);
        console.log(`B: ${B.length.toLocaleString()} elements`);
        console.log('');

        console.log('╔══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
        console.log('║ Input Size   │ Difference     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
        console.log('╠══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

        const { result, totalCount, timing } = await tester.computeSetDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

        const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
        const diffSize = result.length.toLocaleString().padStart(14);
        const dpi = timing.dpiMs.toFixed(2).padStart(6);
        const count = timing.countMs.toFixed(2).padStart(6);
        const scan = timing.scanMs.toFixed(2).padStart(6);
        const write = timing.writeMs.toFixed(2).padStart(6);
        const total = timing.totalMs.toFixed(2).padStart(6);

        console.log(`║ ${inputSize} │ ${diffSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);
        console.log('╚══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');

        // Validate result
        console.log('Validating result...');
        const cpuExpectedCount = tester.cpuSetDifferenceCount(A, B);
        if (result.length === cpuExpectedCount) {
            console.log(`✔ Result count matches CPU: ${result.length.toLocaleString()}`);
        } else {
            console.log(`✗ Result count mismatch: GPU=${result.length}, CPU=${cpuExpectedCount}`);
        }

        // Calculate throughput
        const totalElements = A.length + B.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec\n`);

    } catch (error) {
        console.log(`Error: ${error}`);
    }
}

/**
 * Run cold start benchmark for 32e2 and 32e6 datasets
 * Each dataset is tested independently to avoid thermal effects
 */
export async function runDifference32ColdStartBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET DIFFERENCE COLD START BENCHMARK - 32M             ║');
    console.log('║      (Independent tests for 32e2 and 32e6)                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetDifferencePipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const datasets = [
        { size: '32', range: 'e2', desc: '32M elements, range 100' },
        { size: '32', range: 'e6', desc: '32M elements, range 1M' },
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Difference     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            console.log(`Loading ${aPath}...`);
            const A = await utils.loadUint32ArrayFromBin(aPath);
            console.log(`Loading ${bPath}...`);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            console.log(`A: ${A.length.toLocaleString()} elements, B: ${B.length.toLocaleString()} elements`);

            const { result, totalCount, timing } = await tester.computeSetDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const diffSize = result.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${diffSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

            // Validate result
            const cpuExpectedCount = tester.cpuSetDifferenceCount(A, B);
            if (result.length === cpuExpectedCount) {
                console.log(`✔ Result count matches CPU: ${result.length.toLocaleString()}`);
            } else {
                console.log(`✗ Result count mismatch: GPU=${result.length}, CPU=${cpuExpectedCount}`);
            }

            // Calculate throughput
            const totalElements = A.length + B.length;
            const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
            console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec\n`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');
}
