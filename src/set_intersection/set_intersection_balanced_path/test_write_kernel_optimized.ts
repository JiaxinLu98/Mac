/**
 * Benchmark file for Optimized Complete Set Intersection Pipeline (v1)
 *
 * Compares original vs optimized write kernel using pure GPU timestamp profiling.
 * 4-phase approach:
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Count Phase - Count matches per workgroup
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual intersection results
 *
 * Optimizations applied to count + write kernels:
 * - Opt 3: Branchless select() + pre-fetch in serial_set_intersection
 * - Opt 2: Unrolled loads in device_load_2_to_shared (7 iterations)
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShaderOriginal from './set_availability_intersection_count_v1.wgsl';
import countShaderOptimized from './set_availability_intersection_count_v1_optimized.wgsl';
import writeShaderOriginal from './set_availability_intersection_write_v1.wgsl';
import writeShaderOptimized from './set_availability_intersection_write_v1_optimized.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const MAXWORKGROUP = 65535;

// ModernGPU constants
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Complete Set Intersection Pipeline Tester
 * Accepts shader code for both count and write phases.
 */
class PipelineTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    public label: string;

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

    constructor(
        device: GPUDevice,
        countShader: string,
        writeShader: string,
        label: string,
        timestampQueryManager: TimestampQueryManager
    ) {
        this.device = device;
        this.label = label;
        this.timestampQueryManager = timestampQueryManager;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // DPI bind group layout
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `${label} DPI bind group layout`,
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
            label: `${label} DPI pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Count bind group layout (with debug buffer)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Count kernel bind group layout`,
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
            label: `${label} Count kernel pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write bind group layout
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Write kernel bind group layout`,
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
            label: `${label} Write kernel pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    /**
     * Run the complete 4-phase pipeline with GPU timestamp profiling.
     *
     * Timestamp indices:
     * - DPI: 0 (begin), 1 (end)
     * - Count: 2 (begin), 3 (end)
     * - Scan: 4 (begin), 5 (end)
     * - Write: 6 (begin), 7 (end)
     */
    public async run(
        setA: Uint32Array,
        setB: Uint32Array,
        numWarmup: number = 10,
        numIterations: number = 10
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
        const tsm = this.timestampQueryManager;
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

        // ============ Create Buffers ============
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

        const maxOutputSize = Math.min(a_len, b_len);
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        // ============ Create Bind Groups ============
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

        // ============ Warmup Runs ============
        for (let w = 0; w < numWarmup; w++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, this.timestampQueryManager, 4, 5);

            pass = encoder.beginComputePass();
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // ============ Timed Runs with GPU Timestamps ============
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < numIterations; iter++) {
            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(
                tsm.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count (timestamps 2, 3)
            pass = encoder.beginComputePass(
                tsm.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Copy counts before prefix sum modifies them
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum (timestamps 4, 5)
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            // Phase 4: Write (timestamps 6, 7)
            pass = encoder.beginComputePass(
                tsm.createComputePassDescriptor(6, 7)
            );
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Resolve timestamps
            tsm.resolve(encoder);

            // Submit and wait
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Download timestamps (in nanoseconds)
            const timestamps = await tsm.downloadTimestampResult();

            if (timestamps.length >= 8) {
                const dpiNs = timestamps[1] - timestamps[0];
                const countNs = timestamps[3] - timestamps[2];
                const writeNs = timestamps[7] - timestamps[6];
                const totalNs = timestamps[7] - timestamps[0];
                const scanNsIndirect = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNsIndirect / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        // Average
        const avg = (arr: number[]) =>
            arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        const timing = {
            dpiMs: avg(dpiTimes),
            countMs: avg(countTimes),
            scanMs: avg(scanTimes),
            writeMs: avg(writeTimes),
            totalMs: avg(totalTimes),
        };

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

        return { result, totalCount, timing };
    }
}

// ============================================================================
// CPU Reference
// ============================================================================
function cpuSetIntersection(a: Uint32Array, b: Uint32Array): Uint32Array {
    const result: number[] = [];
    let ai = 0, bi = 0;
    while (ai < a.length && bi < b.length) {
        if (a[ai] < b[bi]) {
            ai++;
        } else if (a[ai] > b[bi]) {
            bi++;
        } else {
            result.push(a[ai]);
            ai++;
            bi++;
        }
    }
    return new Uint32Array(result);
}

function cpuSetIntersectionCount(a: Uint32Array, b: Uint32Array): number {
    let count = 0;
    let ai = 0, bi = 0;
    while (ai < a.length && bi < b.length) {
        if (a[ai] < b[bi]) {
            ai++;
        } else if (a[ai] > b[bi]) {
            bi++;
        } else {
            count++;
            ai++;
            bi++;
        }
    }
    return count;
}

function validateResults(gpuResult: Uint32Array, cpuResult: Uint32Array): boolean {
    if (gpuResult.length !== cpuResult.length) {
        console.log(`  Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
        return false;
    }
    for (let i = 0; i < cpuResult.length; i++) {
        if (gpuResult[i] !== cpuResult[i]) {
            console.log(`  Value mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
            const start = Math.max(0, i - 3);
            const end = Math.min(cpuResult.length, i + 4);
            console.log(`    GPU[${start}..${end}]: ${Array.from(gpuResult.slice(start, end))}`);
            console.log(`    CPU[${start}..${end}]: ${Array.from(cpuResult.slice(start, end))}`);
            return false;
        }
    }
    return true;
}

// ============================================================================
// Test Cases (same as count kernel test)
// ============================================================================
interface TestCase {
    name: string;
    A: Uint32Array;
    B: Uint32Array;
}

function buildTestCases(): TestCase[] {
    const cases: TestCase[] = [];

    cases.push({
        name: 'Minimal duplicates (single WG)',
        A: new Uint32Array([1, 1, 1, 2, 2, 3]),
        B: new Uint32Array([1, 1, 2, 2, 2, 3]),
    });

    {
        const A = new Uint32Array(100); A.fill(5);
        const B = new Uint32Array(100); B.fill(5);
        cases.push({ name: 'All same value (single WG)', A, B });
    }

    cases.push({
        name: 'Small arrays',
        A: new Uint32Array([1, 3, 3, 5, 7, 9]),
        B: new Uint32Array([2, 3, 3, 6, 7, 8]),
    });

    cases.push({
        name: 'No intersection',
        A: new Uint32Array([1, 3, 5, 7, 9]),
        B: new Uint32Array([2, 4, 6, 8, 10]),
    });

    cases.push({
        name: 'Complete intersection (A == B)',
        A: new Uint32Array([1, 2, 3, 4, 5]),
        B: new Uint32Array([1, 2, 3, 4, 5]),
    });

    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) { A[i] = i * 2; B[i] = i * 3; }
        cases.push({ name: 'Multi-WG (even vs multiples of 3)', A, B });
    }

    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = Math.floor(i / 10) * 3;
            B[i] = Math.floor(i / 8) * 3;
        }
        cases.push({ name: 'Many duplicates', A, B });
    }

    {
        const A = new Uint32Array([100, 200, 300, 400, 500]);
        const B = new Uint32Array(3000);
        for (let i = 0; i < B.length; i++) B[i] = i;
        cases.push({ name: 'Unbalanced (tiny A, huge B)', A, B });
    }

    {
        const a9Len = 3000, b9Len = 3000;
        const A = new Uint32Array(a9Len); A[0] = 0; A.fill(1, 1);
        const B = new Uint32Array(b9Len); B.fill(1);
        cases.push({ name: 'Star bit trigger', A, B });
    }

    {
        const A = new Uint32Array(2500); A.fill(42);
        const B = new Uint32Array(2500); B.fill(42);
        cases.push({ name: 'All same value (multi-WG)', A, B });
    }

    {
        const aLen = 896, bLen = 896;
        const A = new Uint32Array(aLen);
        const B = new Uint32Array(bLen);
        for (let i = 0; i < aLen; i++) A[i] = i * 2;
        for (let i = 0; i < bLen; i++) B[i] = i * 2 + 1;
        cases.push({ name: 'NV boundary (total = NV exactly)', A, B });
    }

    {
        const size = 10000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) { A[i] = i; B[i] = i * 2; }
        cases.push({ name: 'Large dataset (10K each)', A, B });
    }

    {
        const aSize = 6000, bSize = 5500;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);
        A[0] = 0; A[1] = 1; A[2] = 2;
        for (let i = 3; i < aSize - 2; i++) A[i] = 100;
        A[aSize - 2] = 999998; A[aSize - 1] = 999999;
        B[0] = 1; B[1] = 2;
        for (let i = 2; i < bSize - 3; i++) B[i] = 100;
        B[bSize - 3] = 999998; B[bSize - 2] = 999999; B[bSize - 1] = 1000000;
        cases.push({ name: 'Single massive duplicate', A, B });
    }

    {
        const numTargetWg = 6;
        const runLength = NV;
        const aSize = numTargetWg * runLength;
        const bSize = numTargetWg * runLength - 200;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);
        for (let i = 0; i < aSize; i++) A[i] = Math.floor(i / runLength);
        for (let i = 0; i < bSize; i++) B[i] = Math.floor(i / runLength);
        cases.push({ name: 'Maximum star bit triggers', A, B });
    }

    {
        const aSize = 2999, bSize = 3001;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);
        for (let i = 0; i < aSize; i++) A[i] = Math.floor(i * 1.5);
        for (let i = 0; i < bSize; i++) B[i] = i * 2;
        cases.push({ name: 'Prime number sizes', A, B });
    }

    return cases;
}

// ============================================================================
// Main: Run optimized pipeline tests + comparison benchmark
// ============================================================================
export async function runOptimizedWriteKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('  OPTIMIZED WRITE PIPELINE TEST SUITE');
    console.log('  Original vs Optimized (count + write) | GPU Timestamp Query');
    console.log('='.repeat(80) + '\n');

    const tsm = new TimestampQueryManager(device, 16);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const original = new PipelineTester(device, countShaderOriginal, writeShaderOriginal, 'Original', tsm);
    const optimized = new PipelineTester(device, countShaderOptimized, writeShaderOptimized, 'Optimized', tsm);

    const NUM_WARMUP = 10;
    const NUM_ITERATIONS = 10;

    // ========================================================================
    // Correctness tests (1 warmup + 1 iteration)
    // ========================================================================
    console.log('--- Correctness Tests ---\n');

    const testCases = buildTestCases();
    let allPassed = true;

    for (let tc = 0; tc < testCases.length; tc++) {
        const { name, A, B } = testCases[tc];
        const cpuResult = cpuSetIntersection(A, B);

        console.log(`Test ${tc}: ${name}`);
        console.log(`  |A|=${A.length}, |B|=${B.length}, CPU count=${cpuResult.length}`);

        const origResult = await original.run(A, B, 1, 1);
        const optResult = await optimized.run(A, B, 1, 1);

        const origValid = validateResults(origResult.result, cpuResult);
        const optValid = validateResults(optResult.result, cpuResult);

        const status = origValid && optValid ? 'PASS' : 'FAIL';
        console.log(`  Original:  count=${origResult.totalCount} ${origValid ? 'OK' : 'MISMATCH'}`);
        console.log(`  Optimized: count=${optResult.totalCount} ${optValid ? 'OK' : 'MISMATCH'}`);
        console.log(`  [${status}]\n`);

        if (!optValid) allPassed = false;
    }

    // ========================================================================
    // Dataset Benchmark
    // ========================================================================
    console.log('\n' + '='.repeat(80));
    console.log(`  DATASET BENCHMARK: Original vs Optimized Pipeline`);
    console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations, GPU timestamp`);
    console.log('='.repeat(80) + '\n');

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

    type DatasetResult = {
        desc: string;
        origTiming: { dpiMs: number; countMs: number; scanMs: number; writeMs: number; totalMs: number };
        optTiming: { dpiMs: number; countMs: number; scanMs: number; writeMs: number; totalMs: number };
        speedup: string;
        match: string;
    };
    const datasetResults: DatasetResult[] = [];

    console.log(
        `${'Dataset'.padEnd(30)} ` +
        `${'Orig Total'.padStart(12)} ${'Opt Total'.padStart(12)} ${'Speedup'.padStart(10)} ` +
        `${'Orig Count'.padStart(10)} ${'Orig Write'.padStart(10)} ` +
        `${'Opt Count'.padStart(10)} ${'Opt Write'.padStart(10)} ` +
        `${'Match'.padStart(6)}`
    );
    console.log('-'.repeat(120));

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;
        const sizeNum = parseInt(size);

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const origResult = await original.run(A, B, NUM_WARMUP, NUM_ITERATIONS);
            const optResult = await optimized.run(A, B, NUM_WARMUP, NUM_ITERATIONS);

            // Validate output arrays match
            let match = origResult.totalCount === optResult.totalCount;
            if (match && sizeNum <= 32) {
                const cpuResult = cpuSetIntersection(A, B);
                match = validateResults(optResult.result, cpuResult);
            }

            const speedup = origResult.timing.totalMs / optResult.timing.totalMs;
            const matchStr = match ? 'OK' : 'FAIL';

            datasetResults.push({
                desc,
                origTiming: origResult.timing,
                optTiming: optResult.timing,
                speedup: speedup.toFixed(2) + 'x',
                match: matchStr,
            });

            console.log(
                `${desc.padEnd(30)} ` +
                `${(origResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ${(optResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ` +
                `${(origResult.timing.countMs.toFixed(3)).padStart(10)} ${(origResult.timing.writeMs.toFixed(3)).padStart(10)} ` +
                `${(optResult.timing.countMs.toFixed(3)).padStart(10)} ${(optResult.timing.writeMs.toFixed(3)).padStart(10)} ` +
                `${matchStr.padStart(6)}`
            );

            if (!match) {
                console.log(`    WARNING: Output mismatch! orig count=${origResult.totalCount}, opt count=${optResult.totalCount}`);
                allPassed = false;
            }
        } catch (e) {
            console.log(`${desc.padEnd(30)} SKIPPED (file not found)`);
        }
    }

    // ========================================================================
    // Summary table
    // ========================================================================
    console.log('\n' + '='.repeat(80));
    console.log('  DATASET BENCHMARK SUMMARY');
    console.log('='.repeat(80));
    console.log(
        `${'Dataset'.padEnd(30)} ${'Orig Total'.padStart(12)} ${'Opt Total'.padStart(12)} ${'Speedup'.padStart(10)}`
    );
    console.log('-'.repeat(80));
    for (const r of datasetResults) {
        console.log(
            `${r.desc.padEnd(30)} ${(r.origTiming.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
            `${(r.optTiming.totalMs.toFixed(3) + ' ms').padStart(12)} ${r.speedup.padStart(10)}`
        );
    }
    console.log('-'.repeat(80));

    if (allPassed) {
        console.log('\nAll tests (synthetic + dataset) PASSED.');
    } else {
        console.log('\nSome tests FAILED!');
    }
    console.log('');
}

/**
 * Isolated 128M benchmark for the optimized pipeline.
 */
export async function runOptimizedWriteKernel128MTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('  ISOLATED 128M PIPELINE BENCHMARK: Original vs Optimized');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(80) + '\n');

    const tsm = new TimestampQueryManager(device, 16);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const original = new PipelineTester(device, countShaderOriginal, writeShaderOriginal, 'Original', tsm);
    const optimized = new PipelineTester(device, countShaderOptimized, writeShaderOptimized, 'Optimized', tsm);

    const datasets = [
        { size: '128', range: 'e2', desc: '128M elements, range 100' },
        { size: '128', range: 'e6', desc: '128M elements, range 1M' },
    ];

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            console.log(`Loading ${desc}...`);
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);
            console.log(`  |A|=${A.length}, |B|=${B.length}`);

            console.log('  Running Original...');
            const origResult = await original.run(A, B, 10, 10);
            console.log('  Running Optimized...');
            const optResult = await optimized.run(A, B, 10, 10);

            const countsMatch = origResult.totalCount === optResult.totalCount;
            const speedup = origResult.timing.totalMs / optResult.timing.totalMs;

            console.log(`\n  ${desc}:`);
            console.log(`    Original:  ${origResult.timing.totalMs.toFixed(3)} ms total  (DPI=${origResult.timing.dpiMs.toFixed(3)} Count=${origResult.timing.countMs.toFixed(3)} Scan=${origResult.timing.scanMs.toFixed(3)} Write=${origResult.timing.writeMs.toFixed(3)})  count=${origResult.totalCount}`);
            console.log(`    Optimized: ${optResult.timing.totalMs.toFixed(3)} ms total  (DPI=${optResult.timing.dpiMs.toFixed(3)} Count=${optResult.timing.countMs.toFixed(3)} Scan=${optResult.timing.scanMs.toFixed(3)} Write=${optResult.timing.writeMs.toFixed(3)})  count=${optResult.totalCount}`);
            console.log(`    Speedup:   ${speedup.toFixed(2)}x`);
            console.log(`    Match:     ${countsMatch ? 'OK' : 'FAIL'}\n`);
        } catch (e) {
            console.log(`  ${desc}: SKIPPED (file not found)\n`);
        }
    }

    console.log('='.repeat(80));
    console.log('  Done.');
    console.log('='.repeat(80) + '\n');
}
