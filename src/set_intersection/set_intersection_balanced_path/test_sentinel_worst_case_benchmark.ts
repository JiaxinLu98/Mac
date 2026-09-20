/**
 * Worst Case Benchmark for 2-Phase Sentinel Intersection
 *
 * Tests the sentinel-optimized pipeline under worst-case scenarios
 * to measure performance under maximum stress conditions.
 *
 * Algorithm Bottleneck Analysis:
 * | Phase              | Bottleneck         | Worst Case Trigger              |
 * |--------------------|--------------------|---------------------------------|
 * | DPI (Diagonal)     | Binary search depth| Large scale + complex dist      |
 * | Lookback (Main)    | Spin-wait chain    | WG completion time variance     |
 * | Scatter (Output)   | Memory bandwidth   | High intersection -> many writes|
 *
 * Key Finding: Lookback is the main bottleneck at 128M scale (55-60% of total, 8-10ms)
 *
 * Worst Case Scenarios:
 *
 * Tier 1 - Primary Worst Cases:
 * W1: 100% Match (A = B)
 *     - Maximum output writes, every element written out
 *     - Dense lookback state propagation
 *     - commit_mask = 0xFFF, full scatter loop execution
 *
 * W2: All Duplicates (A = B = [v, v, ..., v])
 *     - Pathological behavior in balanced path
 *     - All keys equal, binary search boundary handling
 *     - Frequent star bit adjustments
 *
 * W3: 128M e2 Stress (existing binary files)
 *     - Maximum workgroup count (41,664 WGs)
 *     - Maximum spin-wait depth
 *     - Combined scale + output pressure
 *
 * Tier 2 - Secondary Worst Cases:
 * W4: Skewed Size (|A| >> |B|)
 *     - Unbalanced DPI partitions
 *     - Edge case for serial intersection
 *     - Tests algorithm correctness at boundaries
 *
 * W5: Near-Match Offset (A=[0..N-1], B=[1..N])
 *     - ~100% intersection with offset
 *     - Frequent pointer alternation
 *     - Branch prediction stress
 *
 * Pipeline: DPI (VT=12, NV=3072) -> Decoupled Lookback Sentinel
 * Profiling: 10 warmup + 100 timed iterations with GPU timestamps
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import sentinelLookbackShader from './set_availability_intersection_decoupled_lookback_sentinel.wgsl';
import * as utils from '../../utils';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT_SENT = 12;
const NV_SENT = NT * VT_SENT; // 3072

// Sizes adjusted so that total (|A| + |B|) is NV (3072) multiple
// perArray = number of elements per array, total = 2 * perArray
// We want 2 * perArray to be NV multiple, so perArray = k * (NV/2) = k * 1536
const HALF_NV = NV_SENT / 2; // 1536
const BENCHMARK_SIZES = [
    { label: '8M', perArray: 5208 * HALF_NV },    // 7,999,488 per array, 15,998,976 total
    { label: '32M', perArray: 20832 * HALF_NV },  // 31,997,952 per array, 63,995,904 total
    { label: '64M', perArray: 41664 * HALF_NV },  // 63,995,904 per array, 127,991,808 total
];

// Skewed size benchmark: Large vs Small
const SKEWED_SIZES = [
    { label: 'Skewed', largeSize: 64 * 1024 * 1024, smallSize: 64 * 1024, maxVal: 100000 },
];

const NUM_WARMUP = 10;
const NUM_ITERATIONS = 100;

class SentinelWorstCaseTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        const sentinelDpiShader = computeDiagonalsShader.replace('const VT: u32 = 7u;', `const VT: u32 = ${VT_SENT}u;`);

        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: 'WorstCase Sentinel DPI bind group layout',
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
            label: 'WorstCase Sentinel DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: sentinelDpiShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'WorstCase Sentinel Lookback bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: 'WorstCase Sentinel Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: sentinelLookbackShader }),
                entryPoint: 'intersection_decoupled_lookback'
            }
        });
    }

    public async run(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number,
        warmup: number
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        timing: { dpiMs: number; lookbackMs: number; totalMs: number };
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                result: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV_SENT);
        const maxOutputSize = Math.min(a_len, b_len);

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

        const bufferState = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferOutput = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferTotalCount = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

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

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferState } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferTotalCount } },
                { binding: 6, resource: { buffer: bufferALen } },
                { binding: 7, resource: { buffer: bufferBLen } },
                { binding: 8, resource: { buffer: bufferNumWg } },
            ]
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        await device.queue.onSubmittedWorkDone();

        // Warmup
        for (let iter = 0; iter < warmup; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed iterations
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            if (timestamps.length >= 4) {
                const dpiNs = timestamps[1] - timestamps[0];
                const lookbackNs = timestamps[3] - timestamps[2];
                const totalNs = timestamps[3] - timestamps[0];

                dpiTimes.push(dpiNs / 1_000_000);
                lookbackTimes.push(lookbackNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        // Readback total count
        const totalCountReadback = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferTotalCount, 0, totalCountReadback, 0, 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await totalCountReadback.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(totalCountReadback.getMappedRange().slice(0))[0];
        totalCountReadback.unmap();

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

        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferState.destroy();
        bufferOutput.destroy();
        bufferTotalCount.destroy();
        totalCountReadback.destroy();

        return {
            result,
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                lookbackMs: avg(lookbackTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

// ============================================================================
// Synthetic Data Generators
// ============================================================================

/**
 * W1: 100% Match
 * A = B = [0, 1, 2, ..., N-1]
 *
 * Worst case because:
 * - Maximum output writes (every element)
 * - Dense lookback state propagation
 * - commit_mask = 0xFFF for every tile
 * - Maximum scatter loop iterations
 */
function generate100PercentMatch(n: number): { A: Uint32Array; B: Uint32Array; expectedCount: number } {
    const A = new Uint32Array(n);
    const B = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i;
    }
    return { A, B, expectedCount: n };
}

/**
 * W2: All Duplicates
 * A = B = [value, value, ..., value]
 *
 * Worst case because:
 * - Pathological balanced path behavior (all keys equal)
 * - Binary search boundary edge cases
 * - Star bit adjustment triggered at every boundary
 * - Different from W1: merge path hits same value everywhere
 */
function generateAllDuplicates(n: number, value: number = 1): { A: Uint32Array; B: Uint32Array; expectedCount: number } {
    const A = new Uint32Array(n).fill(value);
    const B = new Uint32Array(n).fill(value);
    return { A, B, expectedCount: n };
}

/**
 * W4: Skewed Size (|A| >> |B|)
 * A = largeSize sorted elements
 * B = smallSize sorted elements
 *
 * Tests:
 * - Unbalanced DPI partitions
 * - Early termination in serial intersection
 * - Edge case handling for size asymmetry
 *
 * Note: This is more of a correctness/boundary test than pure worst case
 */
function generateSkewedSize(largeSize: number, smallSize: number, maxVal: number): { A: Uint32Array; B: Uint32Array } {
    // Generate sorted arrays with values in [0, maxVal)
    const generateSorted = (size: number): Uint32Array => {
        const arr = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            arr[i] = Math.floor(Math.random() * maxVal);
        }
        arr.sort();
        return arr;
    };

    return {
        A: generateSorted(largeSize),
        B: generateSorted(smallSize)
    };
}

/**
 * W5: Near-Match Offset
 * A = [0, 1, 2, ..., N-1]
 * B = [1, 2, 3, ..., N]
 * Intersection = [1, 2, ..., N-1] (N-1 matches, ~100% match rate)
 *
 * Worst case because:
 * - Frequent pointer alternation (A or B advances each step)
 * - Maximum comparison count per element
 * - First element doesn't match, causing offset pattern
 * - Branch prediction stress from alternating advances
 */
function generateNearMatchOffset(n: number): { A: Uint32Array; B: Uint32Array; expectedCount: number } {
    const A = new Uint32Array(n);
    const B = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        A[i] = i;        // A = [0, 1, 2, ..., N-1]
        B[i] = i + 1;    // B = [1, 2, 3, ..., N]
    }
    // Intersection = [1, 2, ..., N-1], which has N-1 elements
    return { A, B, expectedCount: n - 1 };
}

/**
 * CPU set intersection for validation (merge-based for sorted arrays)
 */
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

// ============================================================================
// Output Helpers
// ============================================================================

function formatNumber(n: number): string {
    return n.toLocaleString();
}

function printHeader() {
    console.log(
        `${'Size'.padEnd(8)} ` +
        `${'|A|'.padStart(14)} ` +
        `${'|B|'.padStart(14)} ` +
        `${'Total'.padStart(14)} ` +
        `${'Result'.padStart(12)} ` +
        `${'DPI(ms)'.padStart(10)} ` +
        `${'LB(ms)'.padStart(10)} ` +
        `${'Total(ms)'.padStart(12)} ` +
        `${'Throughput'.padStart(14)} ` +
        `${'Valid'.padStart(6)}`
    );
    console.log('-'.repeat(130));
}

async function runBenchmark(
    tester: SentinelWorstCaseTester,
    label: string,
    A: Uint32Array,
    B: Uint32Array,
    skipCpuValidation: boolean,
    expectedCount: number = -1  // -1 means no expected value check
): Promise<{ throughput: number; totalMs: number }> {
    const result = await tester.run(A, B, NUM_ITERATIONS, NUM_WARMUP);

    const totalElements = A.length + B.length;
    const throughput = totalElements / result.timing.totalMs / 1000; // M elem/sec

    let validStr = '-';
    if (!skipCpuValidation) {
        // Full CPU validation
        const cpuResult = cpuSetIntersection(A, B);
        const valid = result.totalCount === cpuResult.length;
        validStr = valid ? 'OK' : 'FAIL';
        if (!valid) {
            console.log(`    VALIDATION FAILED: GPU=${result.totalCount}, CPU=${cpuResult.length}`);
        }
    } else if (expectedCount >= 0) {
        // Validate against expected value (for large datasets where CPU is too slow)
        const valid = result.totalCount === expectedCount;
        validStr = valid ? 'OK*' : 'FAIL';  // OK* = validated by expected value, not full CPU
        if (!valid) {
            console.log(`    VALIDATION FAILED: GPU=${result.totalCount}, expected=${expectedCount}`);
        }
    }

    console.log(
        `${label.padEnd(8)} ` +
        `${formatNumber(A.length).padStart(14)} ` +
        `${formatNumber(B.length).padStart(14)} ` +
        `${formatNumber(totalElements).padStart(14)} ` +
        `${formatNumber(result.totalCount).padStart(12)} ` +
        `${result.timing.dpiMs.toFixed(3).padStart(10)} ` +
        `${result.timing.lookbackMs.toFixed(3).padStart(10)} ` +
        `${result.timing.totalMs.toFixed(3).padStart(12)} ` +
        `${(throughput.toFixed(2) + ' M/s').padStart(14)} ` +
        `${validStr.padStart(6)}`
    );

    return { throughput, totalMs: result.timing.totalMs };
}

// ============================================================================
// Main Benchmark Entry Point
// ============================================================================

export async function runSentinelWorstCaseBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(130));
    console.log('  2-PHASE SENTINEL: WORST CASE BENCHMARK');
    console.log('  Pipeline: DPI -> Decoupled Lookback Sentinel');
    console.log(`  VT=${VT_SENT}, NV=${NV_SENT} | ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations`);
    console.log('='.repeat(130));

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('\nERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new SentinelWorstCaseTester(device, timestampQueryManager);

    // Track results for comparison
    const results: { scenario: string; size: string; throughput: number; totalMs: number }[] = [];

    // ========================================
    // W1: 100% Match (A = B)
    // ========================================
    console.log('\n' + '-'.repeat(130));
    console.log('W1: 100% MATCH (A = B)');
    console.log('A = [0, 1, 2, ..., N-1], B = [0, 1, 2, ..., N-1]');
    console.log('100% intersection | Maximum output writes | Dense lookback propagation');
    console.log('-'.repeat(130) + '\n');
    printHeader();

    for (const { label, perArray } of BENCHMARK_SIZES) {
        const { A, B, expectedCount } = generate100PercentMatch(perArray);
        const skipCpu = perArray > 32_000_000;
        const { throughput, totalMs } = await runBenchmark(tester, label, A, B, skipCpu, expectedCount);
        results.push({ scenario: 'W1: 100% Match', size: label, throughput, totalMs });
    }

    // ========================================
    // W2: All Duplicates
    // ========================================
    console.log('\n' + '-'.repeat(130));
    console.log('W2: ALL DUPLICATES');
    console.log('A = [1, 1, 1, ..., 1], B = [1, 1, 1, ..., 1]');
    console.log('Pathological balanced path | All keys equal | Star bit stress');
    console.log('-'.repeat(130) + '\n');
    printHeader();

    for (const { label, perArray } of BENCHMARK_SIZES) {
        const { A, B, expectedCount } = generateAllDuplicates(perArray, 1);
        const skipCpu = perArray > 32_000_000;
        const { throughput, totalMs } = await runBenchmark(tester, label, A, B, skipCpu, expectedCount);
        results.push({ scenario: 'W2: All Duplicates', size: label, throughput, totalMs });
    }

    // ========================================
    // W3: 128M e2 Stress (from binary files)
    // ========================================
    console.log('\n' + '-'.repeat(130));
    console.log('W3: 128M e2 STRESS TEST');
    console.log('Loading from ./data/A_128e2.bin, B_128e2.bin');
    console.log('Maximum scale (41,664 WGs) | High intersection (~99%) | Combined pressure');
    console.log('-'.repeat(130) + '\n');
    printHeader();

    try {
        const A = await utils.loadUint32ArrayFromBin('./data/A_128e2.bin');
        const B = await utils.loadUint32ArrayFromBin('./data/B_128e2.bin');
        // For 128M e2, we skip CPU validation and expected count due to memory constraints
        const { throughput, totalMs } = await runBenchmark(tester, '128M e2', A, B, true, -1);
        results.push({ scenario: 'W3: 128M e2', size: '128M', throughput, totalMs });
    } catch (error) {
        console.log(`    ERROR loading 128M e2 data: ${error}`);
    }

    // ========================================
    // W4: Skewed Size (|A| >> |B|)
    // ========================================
    console.log('\n' + '-'.repeat(130));
    console.log('W4: SKEWED SIZE (|A| >> |B|)');
    console.log('A = 64M elements, B = 64K elements (1000:1 ratio)');
    console.log('Unbalanced DPI partitions | Boundary condition test');
    console.log('-'.repeat(130) + '\n');
    printHeader();

    for (const { label, largeSize, smallSize, maxVal } of SKEWED_SIZES) {
        const { A, B } = generateSkewedSize(largeSize, smallSize, maxVal);
        // Sort arrays (generateSkewedSize already returns sorted)
        // CPU validation for skewed is feasible since output is bounded by min(|A|, |B|)
        const { throughput, totalMs } = await runBenchmark(tester, label, A, B, false, -1);
        results.push({ scenario: 'W4: Skewed', size: label, throughput, totalMs });
    }

    // ========================================
    // W5: Near-Match Offset
    // ========================================
    console.log('\n' + '-'.repeat(130));
    console.log('W5: NEAR-MATCH OFFSET');
    console.log('A = [0, 1, 2, ..., N-1], B = [1, 2, 3, ..., N]');
    console.log('~100% match with offset | Pointer alternation | Branch prediction stress');
    console.log('-'.repeat(130) + '\n');
    printHeader();

    for (const { label, perArray } of BENCHMARK_SIZES) {
        const { A, B, expectedCount } = generateNearMatchOffset(perArray);
        const skipCpu = perArray > 32_000_000;
        const { throughput, totalMs } = await runBenchmark(tester, label, A, B, skipCpu, expectedCount);
        results.push({ scenario: 'W5: Near-Match', size: label, throughput, totalMs });
    }

    // ========================================
    // Summary Comparison Table
    // ========================================
    console.log('\n' + '='.repeat(130));
    console.log('  WORST CASE COMPARISON SUMMARY');
    console.log('='.repeat(130));

    // Group by size for comparison
    const sizes = ['8M', '32M', '64M', '128M', 'Skewed'];

    console.log('\n' + '-'.repeat(80));
    console.log('Throughput Comparison (M elements/s) - Lower = Worse');
    console.log('-'.repeat(80));
    console.log(
        'Scenario'.padEnd(25) +
        '8M'.padStart(12) +
        '32M'.padStart(12) +
        '64M'.padStart(12)
    );
    console.log('-'.repeat(80));

    const scenarios = ['W1: 100% Match', 'W2: All Duplicates', 'W5: Near-Match'];
    for (const scenario of scenarios) {
        const row = [scenario.padEnd(25)];
        for (const size of ['8M', '32M', '64M']) {
            const r = results.find(x => x.scenario === scenario && x.size === size);
            row.push(r ? r.throughput.toFixed(2).padStart(12) : '-'.padStart(12));
        }
        console.log(row.join(''));
    }

    console.log('\n' + '-'.repeat(80));
    console.log('Total Time (ms) - Higher = Worse');
    console.log('-'.repeat(80));
    console.log(
        'Scenario'.padEnd(25) +
        '8M'.padStart(12) +
        '32M'.padStart(12) +
        '64M'.padStart(12)
    );
    console.log('-'.repeat(80));

    for (const scenario of scenarios) {
        const row = [scenario.padEnd(25)];
        for (const size of ['8M', '32M', '64M']) {
            const r = results.find(x => x.scenario === scenario && x.size === size);
            row.push(r ? r.totalMs.toFixed(3).padStart(12) : '-'.padStart(12));
        }
        console.log(row.join(''));
    }

    // ========================================
    // Analysis Notes
    // ========================================
    console.log('\n' + '='.repeat(130));
    console.log('  ANALYSIS NOTES');
    console.log('='.repeat(130));
    console.log(`
  Worst Case Characteristics:

  W1 (100% Match):
    - Tests MAXIMUM OUTPUT BANDWIDTH
    - Every element in A matches B, all elements written
    - Dense prefix sum values, maximum lookback state propagation
    - commit_mask = 0xFFF for every tile

  W2 (All Duplicates):
    - Tests ALGORITHM PATHOLOGY
    - All keys equal -> binary search boundary handling
    - Star bit adjustment at every partition boundary
    - Different from W1: uniform value vs sequential values

  W3 (128M e2):
    - Tests MAXIMUM SCALE + HIGH INTERSECTION
    - 41,664 workgroups -> longest lookback chain possible
    - ~99% intersection with narrow value range
    - Combined scheduling and output pressure

  W4 (Skewed Size):
    - Tests BOUNDARY CONDITIONS
    - Highly unbalanced partition distribution
    - Many partitions with only A elements
    - Not strictly "slowest" but important for correctness

  W5 (Near-Match Offset):
    - Tests COMPARISON INTENSITY
    - Pointers alternate advancement (A then B then A...)
    - Maximum comparisons per matching element
    - Branch prediction cache pressure

  Validation Key:
    OK  = Full CPU validation passed
    OK* = Expected count validation (CPU too slow)
    -   = No validation performed

  Compare these results with Best Case (test_sentinel_best_case_benchmark.ts)
  to measure the performance impact of worst-case data patterns.
`);
}

/**
 * Quick worst case test (smaller sizes for fast iteration)
 */
export async function runSentinelWorstCaseQuickTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(100));
    console.log('  2-PHASE SENTINEL: WORST CASE QUICK TEST');
    console.log('='.repeat(100));

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('\nERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new SentinelWorstCaseTester(device, timestampQueryManager);

    // Quick test sizes (smaller for fast iteration)
    const quickSizes = [
        { label: '1M', perArray: 651 * HALF_NV },      // ~1M per array
        { label: '4M', perArray: 2604 * HALF_NV },     // ~4M per array
    ];

    console.log('\n--- W1: 100% Match ---');
    printHeader();
    for (const { label, perArray } of quickSizes) {
        const { A, B, expectedCount } = generate100PercentMatch(perArray);
        await runBenchmark(tester, label, A, B, false, expectedCount);
    }

    console.log('\n--- W2: All Duplicates ---');
    printHeader();
    for (const { label, perArray } of quickSizes) {
        const { A, B, expectedCount } = generateAllDuplicates(perArray, 42);
        await runBenchmark(tester, label, A, B, false, expectedCount);
    }

    console.log('\n--- W5: Near-Match Offset ---');
    printHeader();
    for (const { label, perArray } of quickSizes) {
        const { A, B, expectedCount } = generateNearMatchOffset(perArray);
        await runBenchmark(tester, label, A, B, false, expectedCount);
    }

    console.log('\nQuick test complete. Run runSentinelWorstCaseBenchmark() for full benchmark.\n');
}

/**
 * Compare best case vs worst case for the same sizes
 */
export async function runSentinelBestVsWorstComparison(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(130));
    console.log('  2-PHASE SENTINEL: BEST CASE vs WORST CASE COMPARISON');
    console.log('='.repeat(130));

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('\nERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new SentinelWorstCaseTester(device, timestampQueryManager);

    // Test both best and worst case patterns at each size
    const testSizes = [
        { label: '8M', perArray: 5208 * HALF_NV },
        { label: '32M', perArray: 20832 * HALF_NV },
    ];

    interface CompareResult {
        size: string;
        pattern: string;
        throughput: number;
        totalMs: number;
        intersectionRate: string;
    }

    const compareResults: CompareResult[] = [];

    for (const { label, perArray } of testSizes) {
        console.log(`\n--- Size: ${label} (${formatNumber(perArray)} per array) ---`);
        printHeader();

        // Best Case: Interleaved (0% intersection)
        const bestA = new Uint32Array(perArray);
        const bestB = new Uint32Array(perArray);
        for (let i = 0; i < perArray; i++) {
            bestA[i] = i * 2 + 1;  // odd: 1, 3, 5, ...
            bestB[i] = (i + 1) * 2; // even: 2, 4, 6, ...
        }
        const bestResult = await runBenchmark(tester, 'Best(0%)', bestA, bestB, perArray > 32_000_000, 0);
        compareResults.push({
            size: label,
            pattern: 'Best (Interleaved 0%)',
            throughput: bestResult.throughput,
            totalMs: bestResult.totalMs,
            intersectionRate: '0%'
        });

        // Worst Case: 100% Match
        const { A: worstA, B: worstB, expectedCount } = generate100PercentMatch(perArray);
        const worstResult = await runBenchmark(tester, 'Worst(100%)', worstA, worstB, perArray > 32_000_000, expectedCount);
        compareResults.push({
            size: label,
            pattern: 'Worst (100% Match)',
            throughput: worstResult.throughput,
            totalMs: worstResult.totalMs,
            intersectionRate: '100%'
        });

        // Calculate degradation
        const degradation = ((bestResult.throughput - worstResult.throughput) / bestResult.throughput * 100).toFixed(1);
        console.log(`    -> Performance degradation: ${degradation}% (Best: ${bestResult.throughput.toFixed(2)} M/s, Worst: ${worstResult.throughput.toFixed(2)} M/s)`);
    }

    // Summary table
    console.log('\n' + '='.repeat(100));
    console.log('  COMPARISON SUMMARY');
    console.log('='.repeat(100));
    console.log(
        'Size'.padEnd(10) +
        'Pattern'.padEnd(25) +
        'Intersection'.padStart(12) +
        'Throughput'.padStart(14) +
        'Time(ms)'.padStart(12)
    );
    console.log('-'.repeat(100));

    for (const r of compareResults) {
        console.log(
            r.size.padEnd(10) +
            r.pattern.padEnd(25) +
            r.intersectionRate.padStart(12) +
            (r.throughput.toFixed(2) + ' M/s').padStart(14) +
            r.totalMs.toFixed(3).padStart(12)
        );
    }

    console.log('\n');
}
