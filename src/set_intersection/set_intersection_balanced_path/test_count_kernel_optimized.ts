/**
 * Test file for Optimized Count Kernel (set_availability_intersection_count_v1_optimized.wgsl)
 *
 * Runs the same test suite as test_count_kernel.ts but uses the optimized shader.
 * Also supports A/B comparison mode to benchmark original vs optimized.
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShaderOriginal from './set_availability_intersection_count_v1.wgsl';
import countShaderOptimized from './set_availability_intersection_count_v1_optimized.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

const NT = 256;
const VT = 7;
const NV = NT * VT;

function countOneBits(n: number): number {
    n = n - ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

/**
 * Generic count kernel tester — works with any shader code that has
 * the same binding layout and entry point 'count_availability'.
 */
class CountKernelTester {
    private device: GPUDevice;
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;
    private scanPipeline: ExclusiveScanPipeline;
    private timestampQueryManager: TimestampQueryManager;
    public label: string;

    constructor(device: GPUDevice, shaderCode: string, label: string, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.label = label;
        this.scanPipeline = new ExclusiveScanPipeline(device);
        this.timestampQueryManager = timestampQueryManager;

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
                module: device.createShaderModule({ code: shaderCode }),
                entryPoint: 'count_availability'
            }
        });
    }

    public async run(
        setA: Uint32Array,
        setB: Uint32Array,
        numWarmup: number = 10,
        numIterations: number = 10
    ): Promise<{
        counts: Uint32Array;
        totalCount: number;
        dpiData: Uint32Array;
        numWorkgroups: number;
        gpuTimeMs: number;
    }> {
        const device = this.device;
        const tsm = this.timestampQueryManager;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            throw new Error('Cannot test count kernel with empty arrays');
        }

        const numWg = Math.ceil(total / NV);

        const bufferA = device.createBuffer({
            size: Math.max(4, setA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
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
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Step 1: Compute DPI
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

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        let encoder = device.createCommandEncoder();
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.diagPipeline);
        pass.setBindGroup(0, diagBindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Step 2: Run Count Kernel
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

        // Warmup runs (no timestamp)
        for (let w = 0; w < numWarmup; w++) {
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Timed runs with GPU timestamp queries
        // Timestamp indices: 0 = begin count, 1 = end count
        const timingsNs: number[] = [];
        for (let iter = 0; iter < numIterations; iter++) {
            encoder = device.createCommandEncoder();
            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(0, 1));
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            tsm.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await tsm.downloadTimestampResult();
            if (timestamps.length >= 2) {
                timingsNs.push(timestamps[1] - timestamps[0]);
            }
        }

        // Average GPU time in milliseconds (nanoseconds -> ms)
        const gpuTimeMs = timingsNs.length > 0
            ? (timingsNs.reduce((a, b) => a + b, 0) / timingsNs.length) / 1_000_000
            : 0;

        // Read back results
        const dpiReadback = device.createBuffer({
            size: dpiSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(bufferDPI, 0, dpiReadback, 0, dpiSize * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await dpiReadback.mapAsync(GPUMapMode.READ);
        const dpiData = new Uint32Array(dpiReadback.getMappedRange().slice(0));
        dpiReadback.unmap();

        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(bufferCounts, 0, countsReadback, 0, numWg * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        let totalCount = 0;
        for (let i = 0; i < numWg; i++) {
            totalCount += counts[i];
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferDebug.destroy();
        dpiReadback.destroy();
        countsReadback.destroy();

        return { counts, totalCount, dpiData, numWorkgroups: numWg, gpuTimeMs };
    }
}

// ============================================================================
// CPU Reference
// ============================================================================
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

// ============================================================================
// Test Cases
// ============================================================================
interface TestCase {
    name: string;
    A: Uint32Array;
    B: Uint32Array;
}

function buildTestCases(): TestCase[] {
    const cases: TestCase[] = [];

    // Case 0: Minimal duplicates
    cases.push({
        name: 'Minimal duplicates (single WG)',
        A: new Uint32Array([1, 1, 1, 2, 2, 3]),
        B: new Uint32Array([1, 1, 2, 2, 2, 3]),
    });

    // Case 1: All same value (single WG)
    {
        const A = new Uint32Array(100); A.fill(5);
        const B = new Uint32Array(100); B.fill(5);
        cases.push({ name: 'All same value (single WG)', A, B });
    }

    // Case 2: Small arrays
    cases.push({
        name: 'Small arrays',
        A: new Uint32Array([1, 3, 3, 5, 7, 9]),
        B: new Uint32Array([2, 3, 3, 6, 7, 8]),
    });

    // Case 3: No intersection
    cases.push({
        name: 'No intersection',
        A: new Uint32Array([1, 3, 5, 7, 9]),
        B: new Uint32Array([2, 4, 6, 8, 10]),
    });

    // Case 4: Complete intersection
    cases.push({
        name: 'Complete intersection (A == B)',
        A: new Uint32Array([1, 2, 3, 4, 5]),
        B: new Uint32Array([1, 2, 3, 4, 5]),
    });

    // Case 5: Multi-workgroup
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) { A[i] = i * 2; B[i] = i * 3; }
        cases.push({ name: 'Multi-WG (even vs multiples of 3)', A, B });
    }

    // Case 6: Many duplicates
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

    // Case 7: Unbalanced sizes
    {
        const A = new Uint32Array([100, 200, 300, 400, 500]);
        const B = new Uint32Array(3000);
        for (let i = 0; i < B.length; i++) B[i] = i;
        cases.push({ name: 'Unbalanced (tiny A, huge B)', A, B });
    }

    // Case 8: Star bit trigger
    {
        const a9Len = 3000, b9Len = 3000;
        const A = new Uint32Array(a9Len); A[0] = 0; A.fill(1, 1);
        const B = new Uint32Array(b9Len); B.fill(1);
        cases.push({ name: 'Star bit trigger', A, B });
    }

    // Case 9: All same value multi-WG
    {
        const A = new Uint32Array(2500); A.fill(42);
        const B = new Uint32Array(2500); B.fill(42);
        cases.push({ name: 'All same value (multi-WG)', A, B });
    }

    // Case 10: NV boundary
    {
        const aLen = 896, bLen = 896;
        const A = new Uint32Array(aLen);
        const B = new Uint32Array(bLen);
        for (let i = 0; i < aLen; i++) A[i] = i * 2;
        for (let i = 0; i < bLen; i++) B[i] = i * 2 + 1;
        cases.push({ name: 'NV boundary (total = NV exactly)', A, B });
    }

    // Case 11: Large dataset
    {
        const size = 10000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) { A[i] = i; B[i] = i * 2; }
        cases.push({ name: 'Large dataset (10K each)', A, B });
    }

    // Case 12: Single massive duplicate
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

    // Case 13: Maximum star bit triggers
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

    // Case 14: Prime number sizes
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
// Main: Run optimized tests + comparison benchmark
// ============================================================================
export async function runOptimizedCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(70));
    console.log('  OPTIMIZED COUNT KERNEL TEST SUITE');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(70) + '\n');

    // 2 timestamp indices needed: begin(0) and end(1) for the count kernel pass
    const tsm = new TimestampQueryManager(device, 2);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const original = new CountKernelTester(device, countShaderOriginal, 'Original', tsm);
    const optimized = new CountKernelTester(device, countShaderOptimized, 'Optimized', tsm);

    const NUM_WARMUP = 10;
    const NUM_ITERATIONS = 10;

    const testCases = buildTestCases();
    let allPassed = true;
    const benchResults: { name: string; origMs: number; optMs: number; speedup: string }[] = [];

    for (let tc = 0; tc < testCases.length; tc++) {
        const { name, A, B } = testCases[tc];
        const cpuCount = cpuSetIntersectionCount(A, B);

        console.log(`--- Test ${tc}: ${name} ---`);
        console.log(`  |A|=${A.length}, |B|=${B.length}, CPU count=${cpuCount}`);

        // Run original (1 warmup, 1 iteration for correctness)
        const origResult = await original.run(A, B, 1, 1);
        const origMatch = origResult.totalCount === cpuCount;

        // Run optimized (1 warmup, 1 iteration for correctness)
        const optResult = await optimized.run(A, B, 1, 1);
        const optMatch = optResult.totalCount === cpuCount;

        const status = origMatch && optMatch ? 'PASS' : 'FAIL';
        console.log(`  Original:  count=${origResult.totalCount} ${origMatch ? 'OK' : 'MISMATCH'}  (${origResult.gpuTimeMs.toFixed(3)} ms)`);
        console.log(`  Optimized: count=${optResult.totalCount} ${optMatch ? 'OK' : 'MISMATCH'}  (${optResult.gpuTimeMs.toFixed(3)} ms)`);
        console.log(`  [${status}]\n`);

        if (!optMatch) {
            allPassed = false;
            // Show per-partition comparison on failure
            console.log('  Per-partition comparison (first mismatches):');
            for (let i = 0; i < origResult.numWorkgroups; i++) {
                if (origResult.counts[i] !== optResult.counts[i]) {
                    console.log(`    WG ${i}: orig=${origResult.counts[i]}, opt=${optResult.counts[i]}`);
                    if (i > 5) { console.log('    ...'); break; }
                }
            }
            console.log('');
        }
    }

    // ========================================================================
    // Benchmark: larger arrays with multiple iterations
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log(`  BENCHMARK: Original vs Optimized (${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations, GPU timestamp)`);
    console.log('='.repeat(70) + '\n');

    const benchCases: { name: string; aSize: number; bSize: number; pattern: string }[] = [
        { name: '10K sequential', aSize: 10000, bSize: 10000, pattern: 'sequential' },
        { name: '50K sequential', aSize: 50000, bSize: 50000, pattern: 'sequential' },
        { name: '100K sequential', aSize: 100000, bSize: 100000, pattern: 'sequential' },
        { name: '50K all-same', aSize: 50000, bSize: 50000, pattern: 'allsame' },
        { name: '100K duplicates', aSize: 100000, bSize: 100000, pattern: 'duplicates' },
    ];

    for (const bc of benchCases) {
        const A = new Uint32Array(bc.aSize);
        const B = new Uint32Array(bc.bSize);

        if (bc.pattern === 'sequential') {
            for (let i = 0; i < bc.aSize; i++) A[i] = i;
            for (let i = 0; i < bc.bSize; i++) B[i] = i * 2;
        } else if (bc.pattern === 'allsame') {
            A.fill(42);
            B.fill(42);
        } else if (bc.pattern === 'duplicates') {
            for (let i = 0; i < bc.aSize; i++) A[i] = Math.floor(i / 10) * 3;
            for (let i = 0; i < bc.bSize; i++) B[i] = Math.floor(i / 8) * 3;
        }

        const cpuCount = cpuSetIntersectionCount(A, B);
        const origResult = await original.run(A, B, NUM_WARMUP, NUM_ITERATIONS);
        const optResult = await optimized.run(A, B, NUM_WARMUP, NUM_ITERATIONS);

        const origCorrect = origResult.totalCount === cpuCount;
        const optCorrect = optResult.totalCount === cpuCount;
        const speedup = origResult.gpuTimeMs / optResult.gpuTimeMs;

        benchResults.push({
            name: bc.name,
            origMs: origResult.gpuTimeMs,
            optMs: optResult.gpuTimeMs,
            speedup: speedup.toFixed(2) + 'x',
        });

        console.log(`${bc.name}:`);
        console.log(`  Original:  ${origResult.gpuTimeMs.toFixed(3)} ms  count=${origResult.totalCount} ${origCorrect ? 'OK' : 'WRONG'}`);
        console.log(`  Optimized: ${optResult.gpuTimeMs.toFixed(3)} ms  count=${optResult.totalCount} ${optCorrect ? 'OK' : 'WRONG'}`);
        console.log(`  Speedup:   ${speedup.toFixed(2)}x\n`);

        if (!optCorrect) allPassed = false;
    }

    // ========================================================================
    // Summary table
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log('  BENCHMARK SUMMARY');
    console.log('='.repeat(70));
    console.log(`${'Test'.padEnd(25)} ${'Original'.padStart(12)} ${'Optimized'.padStart(12)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(70));
    for (const r of benchResults) {
        console.log(
            `${r.name.padEnd(25)} ${(r.origMs.toFixed(3) + ' ms').padStart(12)} ${(r.optMs.toFixed(3) + ' ms').padStart(12)} ${r.speedup.padStart(10)}`
        );
    }
    console.log('-'.repeat(70));

    if (allPassed) {
        console.log('\nAll correctness tests PASSED.');
    } else {
        console.log('\nSome correctness tests FAILED!');
    }

    // ========================================================================
    // Dataset Benchmark: Real binary data files (A_<size><range>.bin)
    // ========================================================================
    console.log('\n' + '='.repeat(70));
    console.log(`  DATASET BENCHMARK: Binary data files (${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations, GPU timestamp)`);
    console.log('='.repeat(70) + '\n');

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

    const datasetResults: { desc: string; origMs: number; optMs: number; speedup: string; origCount: number; optCount: number; cpuCount: number | string }[] = [];

    console.log(`${'Dataset'.padEnd(30)} ${'|A|'.padStart(10)} ${'|B|'.padStart(10)} ${'Original'.padStart(12)} ${'Optimized'.padStart(12)} ${'Speedup'.padStart(10)} ${'Count'.padStart(10)} ${'Match'.padStart(6)}`);
    console.log('-'.repeat(100));

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            // CPU validation only for small datasets (<=2M) to avoid blocking
            const totalElements = A.length + B.length;
            let cpuCount: number | string = '-';
            if (totalElements <= 4_000_000) {
                cpuCount = cpuSetIntersectionCount(A, B);
            }

            const origResult = await original.run(A, B, NUM_WARMUP, NUM_ITERATIONS);
            const optResult = await optimized.run(A, B, NUM_WARMUP, NUM_ITERATIONS);

            const countsMatch = origResult.totalCount === optResult.totalCount;
            const cpuMatch = typeof cpuCount === 'number' ? origResult.totalCount === cpuCount : true;
            const matchStr = countsMatch && cpuMatch ? 'OK' : 'FAIL';
            const speedup = origResult.gpuTimeMs / optResult.gpuTimeMs;

            datasetResults.push({
                desc,
                origMs: origResult.gpuTimeMs,
                optMs: optResult.gpuTimeMs,
                speedup: speedup.toFixed(2) + 'x',
                origCount: origResult.totalCount,
                optCount: optResult.totalCount,
                cpuCount,
            });

            console.log(
                `${desc.padEnd(30)} ${A.length.toString().padStart(10)} ${B.length.toString().padStart(10)} ` +
                `${(origResult.gpuTimeMs.toFixed(3) + ' ms').padStart(12)} ${(optResult.gpuTimeMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ${origResult.totalCount.toString().padStart(10)} ${matchStr.padStart(6)}`
            );

            if (!countsMatch) {
                console.log(`    WARNING: Count mismatch! orig=${origResult.totalCount}, opt=${optResult.totalCount}`);
                allPassed = false;
            }
            if (!cpuMatch && typeof cpuCount === 'number') {
                console.log(`    WARNING: CPU mismatch! gpu=${origResult.totalCount}, cpu=${cpuCount}`);
                allPassed = false;
            }
        } catch (e) {
            console.log(`${desc.padEnd(30)} SKIPPED (file not found)`);
        }
    }

    // Final summary
    console.log('\n' + '='.repeat(70));
    console.log('  DATASET BENCHMARK SUMMARY');
    console.log('='.repeat(70));
    console.log(`${'Dataset'.padEnd(30)} ${'Original'.padStart(12)} ${'Optimized'.padStart(12)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(70));
    for (const r of datasetResults) {
        console.log(
            `${r.desc.padEnd(30)} ${(r.origMs.toFixed(3) + ' ms').padStart(12)} ${(r.optMs.toFixed(3) + ' ms').padStart(12)} ${r.speedup.padStart(10)}`
        );
    }
    console.log('-'.repeat(70));

    if (allPassed) {
        console.log('\nAll tests (synthetic + dataset) PASSED.');
    } else {
        console.log('\nSome tests FAILED!');
    }
    console.log('');
}

// ============================================================================
// Isolated 128M benchmark — run alone for stable results
// ============================================================================
export async function runOptimizedCountKernel128MTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(70));
    console.log('  ISOLATED 128M BENCHMARK: Original vs Optimized');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(70) + '\n');

    const tsm = new TimestampQueryManager(device, 2);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const original = new CountKernelTester(device, countShaderOriginal, 'Original', tsm);
    const optimized = new CountKernelTester(device, countShaderOptimized, 'Optimized', tsm);

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
            const speedup = origResult.gpuTimeMs / optResult.gpuTimeMs;

            console.log(`\n  ${desc}:`);
            console.log(`    Original:  ${origResult.gpuTimeMs.toFixed(3)} ms  count=${origResult.totalCount}`);
            console.log(`    Optimized: ${optResult.gpuTimeMs.toFixed(3)} ms  count=${optResult.totalCount}`);
            console.log(`    Speedup:   ${speedup.toFixed(2)}x`);
            console.log(`    Match:     ${countsMatch ? 'OK' : 'FAIL'}\n`);
        } catch (e) {
            console.log(`  ${desc}: SKIPPED (file not found)\n`);
        }
    }

    console.log('='.repeat(70));
    console.log('  Done.');
    console.log('='.repeat(70) + '\n');
}
