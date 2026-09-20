/**
 * Test file for Complete Multiset Union Pipeline (v1)
 *
 * Tests the full two-pass approach for multiset union (A U B):
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Count Phase - Count union elements per workgroup
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual union results
 *
 * Multiset union semantics:
 * - If A < B: emit A, advance A
 * - If B < A: emit B, advance B
 * - If A == B: emit A (tie goes to A), advance both
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_union_count_v1.wgsl';
import writeShader from './set_availability_union_write_v1.wgsl';
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
 * Complete Multiset Union Pipeline Test Class
 */
export class TestSetUnionPipeline {
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

        // Count bind group layout (no debug buffer for union)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Union Count kernel bind group layout',
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
            label: 'Union Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write bind group layout
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: 'Union Write kernel bind group layout',
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
            label: 'Union Write kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    /**
     * Run the complete multiset union pipeline.
     */
    public async computeSetUnion(
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

        // Buffer to preserve original counts
        const bufferCountsCopy = device.createBuffer({
            label: 'Buffer Counts Copy',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Output buffer: max size is a_len + b_len (no overlap case)
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
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        // Prepare scanner
        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // ============ GPU Execution ============
        const times: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const t0 = performance.now();

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI
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

            // Copy counts before prefix sum
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum
            scanner.recordScanCommands(encoder, numWg);

            // Phase 4: Write
            pass = encoder.beginComputePass({ label: 'Write' });
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            times.push(performance.now() - t0);
        }

        // Calculate average time
        let gpuTimeMs: number;
        if (iterations > warmup) {
            const timesWithoutWarmup = times.slice(warmup);
            gpuTimeMs = timesWithoutWarmup.reduce((a, b) => a + b, 0) / timesWithoutWarmup.length;
        } else {
            gpuTimeMs = times[times.length - 1] || 0;
        }

        // ============ Readback Phase ============
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

        // Read output
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
     * CPU reference implementation for multiset union
     */
    public cpuSetUnion(a: Uint32Array, b: Uint32Array): Uint32Array {
        const result: number[] = [];
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                result.push(a[ai]);
                ai++;
            } else if (a[ai] > b[bi]) {
                result.push(b[bi]);
                bi++;
            } else {
                // Equal: output one (from A), advance both
                result.push(a[ai]);
                ai++;
                bi++;
            }
        }
        // Add remaining elements
        while (ai < a.length) {
            result.push(a[ai]);
            ai++;
        }
        while (bi < b.length) {
            result.push(b[bi]);
            bi++;
        }
        return new Uint32Array(result);
    }

    public cpuSetUnionCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                count++;
                ai++;
            } else if (a[ai] > b[bi]) {
                count++;
                bi++;
            } else {
                count++;
                ai++;
                bi++;
            }
        }
        count += (a.length - ai) + (b.length - bi);
        return count;
    }

    /**
     * Validate GPU results against CPU
     */
    public validateResults(gpuResult: Uint32Array, a: Uint32Array, b: Uint32Array): boolean {
        const cpuResult = this.cpuSetUnion(a, b);

        if (gpuResult.length !== cpuResult.length) {
            console.log(`Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
            return false;
        }

        for (let i = 0; i < cpuResult.length; i++) {
            if (gpuResult[i] !== cpuResult[i]) {
                console.log(`Value mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
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
     * Run the complete multiset union pipeline with per-phase profiling.
     * Each phase is submitted separately to measure individual timing.
     * This is slower than the merged version but provides detailed breakdown.
     */
    public async computeSetUnionWithProfiling(
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

        // Union max output size is a_len + b_len
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
     * Run the complete multiset union pipeline with pure GPU timestamp profiling.
     * Uses WebGPU timestamp queries for accurate GPU-only timing.
     * All 4 phases are in a single command encoder, submitted once per iteration.
     *
     * Timestamp indices:
     * - DPI: 0 (begin), 1 (end)
     * - Count: 2 (begin), 3 (end)
     * - Scan: 4 (begin), 5 (end)
     * - Write: 6 (begin), 7 (end)
     */
    public async computeSetUnionWithGPUProfiling(
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
            return this.computeSetUnionWithProfiling(setA, setB, iterations, warmup);
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

        // Union max output size is a_len + b_len
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

                // Calculate Scan time indirectly: Scan = Total - DPI - Count - Write
                // Note: Direct Scan timestamps (indices 4,5) are unreliable due to
                // WebGPU driver/timing issues when Scan pass follows buffer copies
                const scanNsIndirect = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(Number(dpiNs) / 1_000_000);
                countTimes.push(Number(countNs) / 1_000_000);
                scanTimes.push(Number(scanNsIndirect) / 1_000_000);
                writeTimes.push(Number(writeNs) / 1_000_000);
                totalTimes.push(Number(totalNs) / 1_000_000);
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
 * Run union write kernel test
 */
export async function runUnionWriteKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║           MULTISET UNION WRITE KERNEL TEST                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

    // Test cases
    const testCases = [
        {
            name: 'Simple union',
            a: new Uint32Array([1, 3, 5, 7, 9]),
            b: new Uint32Array([2, 3, 4, 7, 8, 10]),
        },
        {
            name: 'No overlap',
            a: new Uint32Array([1, 2, 3]),
            b: new Uint32Array([4, 5, 6]),
        },
        {
            name: 'Complete overlap',
            a: new Uint32Array([1, 2, 3, 4, 5]),
            b: new Uint32Array([1, 2, 3, 4, 5]),
        },
        {
            name: 'Large no overlap (even/odd)',
            a: new Uint32Array(10000).map((_, i) => i * 2),
            b: new Uint32Array(10000).map((_, i) => i * 2 + 1),
        },
        {
            name: 'Large with overlap',
            a: new Uint32Array(10000).map((_, i) => i),
            b: new Uint32Array(10000).map((_, i) => i + 5000),
        },
        {
            name: 'Large complete overlap',
            a: new Uint32Array(10000).map((_, i) => i),
            b: new Uint32Array(10000).map((_, i) => i),
        },
    ];

    for (const { name, a, b } of testCases) {
        console.log(`\n--- Test: ${name} ---`);
        console.log(`A: ${a.length} elements, B: ${b.length} elements`);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetUnion(a, b);
        const cpuCount = tester.cpuSetUnionCount(a, b);

        console.log(`GPU result count: ${totalCount}`);
        console.log(`CPU expected count: ${cpuCount}`);
        console.log(`Count match: ${totalCount === cpuCount ? '✔' : '✗'}`);

        const valid = tester.validateResults(result, a, b);
        console.log(`Values match: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU time: ${gpuTimeMs.toFixed(2)} ms`);

        if (!valid) {
            console.log(`First 20 GPU results: ${Array.from(result.slice(0, 20))}`);
            const cpuResult = tester.cpuSetUnion(a, b);
            console.log(`First 20 CPU results: ${Array.from(cpuResult.slice(0, 20))}`);
        }
    }
}

/**
 * Run union benchmark on binary data files
 */
export async function runUnionBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║           MULTISET UNION BENCHMARK                         ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 50;
    const NUM_WARMUP = 20;

    const datasets = [
        { size: '1', range: 'e6' },
        { size: '2', range: 'e6' },
        { size: '4', range: 'e6' },
        { size: '8', range: 'e6' },
        { size: '16', range: 'e6' },
        { size: '32', range: 'e6' },
    ];

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup)...\n`);

    console.log('╔══════════╤══════════════╤════════════════╤══════════╤════════════════╗');
    console.log('║ Dataset  │ Input Size   │ Union Size     │ Time(ms) │ Throughput     ║');
    console.log('╠══════════╪══════════════╪════════════════╪══════════╪════════════════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, gpuTimeMs } = await tester.computeSetUnion(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const unionSize = totalCount.toLocaleString().padStart(14);
            const time = gpuTimeMs.toFixed(2).padStart(8);
            const totalElements = A.length + B.length;
            const throughput = `${(totalElements / (gpuTimeMs / 1000) / 1_000_000).toFixed(2)} M/s`.padStart(14);

            console.log(`║ ${ds} │ ${inputSize} │ ${unionSize} │ ${time} │ ${throughput} ║`);

            // Validate on smaller datasets
            if (A.length <= 1_000_000) {
                const valid = tester.validateResults(result, A, B);
                if (!valid) {
                    console.log(`  ✗ Validation failed for ${size}${range}`);
                }
            }
        } catch (error) {
            console.log(`║ ${size}${range} │ Error loading data: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧══════════╧════════════════╝\n');
}

/**
 * Run profiling benchmark to identify performance bottlenecks per phase
 */
export async function runUnionProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      MULTISET UNION PROFILING BENCHMARK                    ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

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
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Union Size     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetUnionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const unionSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${unionSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');

    // Print analysis hints
    console.log('Analysis hints:');
    console.log('  - If Scan(ms) dominates: Prefix sum is the bottleneck');
    console.log('  - If Write(ms) dominates: Output writing is the bottleneck');
    console.log('  - If Count(ms) dominates: Main computation is the bottleneck');
    console.log('  - DPI should always be small relative to others\n');
}

/**
 * Run profiling benchmark using pure GPU timestamps.
 * This measures actual GPU kernel execution time without CPU overhead.
 */
export async function runUnionGPUProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      MULTISET UNION GPU TIMESTAMP PROFILING               ║');
    console.log('║      (Pure GPU time, no CPU overhead)                      ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        console.log('Falling back to performance.now() based profiling...\n');
        return runUnionProfilingBenchmark(device);
    }

    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

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
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Union Size     │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetUnionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const unionSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${unionSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');

    // Print analysis hints
    console.log('Analysis hints (GPU timestamp version):');
    console.log('  - These are pure GPU execution times without CPU overhead');
    console.log('  - If Scan(ms) dominates: Prefix sum is the bottleneck');
    console.log('  - If Write(ms) dominates: Output writing is the bottleneck');
    console.log('  - If Count(ms) dominates: Main computation is the bottleneck');
    console.log('  - DPI should always be small relative to others');
    console.log('  - Total = DPI + Count + Scan + Write (no gaps between phases)\n');
}

/**
 * Run GPU profiling benchmark specifically for 128e2 dataset
 */
export async function runUnion128e2ProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      MULTISET UNION 128e2 GPU PROFILING                    ║');
    console.log('║      (128M elements, value range 100)                      ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const aPath = './data/A_128e2.bin';
    const bPath = './data/B_128e2.bin';

    try {
        console.log('Loading datasets...');
        const A = await utils.loadUint32ArrayFromBin(aPath);
        const B = await utils.loadUint32ArrayFromBin(bPath);
        console.log(`A: ${A.length.toLocaleString()} elements`);
        console.log(`B: ${B.length.toLocaleString()} elements\n`);

        console.log('Running GPU profiling...');
        const { result, totalCount, timing } = await tester.computeSetUnionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                    RESULTS - 128e2                         ║');
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Union Size:    ${totalCount.toLocaleString().padStart(14)} elements              ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  DPI Phase:     ${timing.dpiMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Count Phase:   ${timing.countMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Scan Phase:    ${timing.scanMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Write Phase:   ${timing.writeMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Total:         ${timing.totalMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log('╚════════════════════════════════════════════════════════════╝\n');

        const totalElements = A.length + B.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec`);

    } catch (error) {
        console.log(`Error: ${error}`);
    }
}

/**
 * Run GPU profiling benchmark specifically for 128e6 dataset
 */
export async function runUnion128e6ProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      MULTISET UNION 128e6 GPU PROFILING                    ║');
    console.log('║      (128M elements, value range 1M)                       ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestSetUnionPipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const aPath = './data/A_128e6.bin';
    const bPath = './data/B_128e6.bin';

    try {
        console.log('Loading datasets...');
        const A = await utils.loadUint32ArrayFromBin(aPath);
        const B = await utils.loadUint32ArrayFromBin(bPath);
        console.log(`A: ${A.length.toLocaleString()} elements`);
        console.log(`B: ${B.length.toLocaleString()} elements\n`);

        console.log('Running GPU profiling...');
        const { result, totalCount, timing } = await tester.computeSetUnionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

        console.log('\n╔════════════════════════════════════════════════════════════╗');
        console.log('║                    RESULTS - 128e6                         ║');
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Union Size:    ${totalCount.toLocaleString().padStart(14)} elements              ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  DPI Phase:     ${timing.dpiMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Count Phase:   ${timing.countMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Scan Phase:    ${timing.scanMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log(`║  Write Phase:   ${timing.writeMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Total:         ${timing.totalMs.toFixed(3).padStart(10)} ms                       ║`);
        console.log('╚════════════════════════════════════════════════════════════╝\n');

        const totalElements = A.length + B.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec`);

    } catch (error) {
        console.log(`Error: ${error}`);
    }
}
