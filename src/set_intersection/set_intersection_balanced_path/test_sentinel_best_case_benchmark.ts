/**
 * Best Case Benchmark for 2-Phase Sentinel Intersection
 *
 * Tests the sentinel-optimized pipeline under best-case scenarios
 * to measure realistic peak throughput on large-scale data.
 *
 * Best Case Definition:
 * - 0% intersection: no scatter writes, minimal lookback propagation
 * - Uniform A/B distribution per partition: balanced workload, minimal spin-wait
 * - Sizes are NV (3072) multiples: no tail workgroup underfill
 *
 * Test Patterns:
 * 1. Disjoint: A=[0..N-1], B=[N..2N-1]
 *    - 0% intersection, merge path extremely regular
 *    - Front half all A, back half all B
 *    - Minimal comparisons (block skipping)
 *
 * 2. Interleaved: A=odd [1,3,5,...], B=even [2,4,6,...]
 *    - 0% intersection with alternating A/B elements
 *    - Perfect workload balance across all workgroups
 *    - A/B pointers alternate advancement
 *
 * Test Sizes (per array, total = 2*perArray is NV=3072 multiple):
 * - 1M -> 999,936 per array (total 1,999,872 = 651 * 3072)
 * - 8M -> 7,999,488 per array (total 15,998,976 = 5208 * 3072)
 * - 32M -> 31,997,952 per array (total 63,995,904 = 20832 * 3072)
 * - 64M -> 63,995,904 per array (total 127,991,808 = 41664 * 3072)
 *
 * Pipeline: DPI (VT=12, NV=3072) -> Decoupled Lookback Sentinel
 * Profiling: 10 warmup + 100 timed iterations with GPU timestamps
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import sentinelLookbackShader from './set_availability_intersection_decoupled_lookback_sentinel.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT_SENT = 12;
const NV_SENT = NT * VT_SENT; // 3072

// Sizes adjusted so that total (|A| + |B|) is NV (3072) multiple
// perArray = number of elements per array, total = 2 * perArray
// We want 2 * perArray to be NV multiple, so perArray = k * (NV/2) = k * 1536
//
// Target sizes (per array):
// 1M  -> 999,936  = 651 * 1536 (total = 1,999,872 = 651 * 3072)
// 8M  -> 7,999,488 = 5208 * 1536 (total = 15,998,976 = 5208 * 3072)
// 32M -> 31,997,952 = 20832 * 1536 (total = 63,995,904 = 20832 * 3072)
// 64M -> 63,995,904 = 41664 * 1536 (total = 127,991,808 = 41664 * 3072)
const HALF_NV = NV_SENT / 2; // 1536
const BENCHMARK_SIZES = [
    { label: '1M', perArray: 651 * HALF_NV },     // 999,936 per array, 1,999,872 total
    { label: '8M', perArray: 5208 * HALF_NV },    // 7,999,488 per array, 15,998,976 total
    { label: '32M', perArray: 20832 * HALF_NV },  // 31,997,952 per array, 63,995,904 total
    { label: '64M', perArray: 41664 * HALF_NV },  // 63,995,904 per array, 127,991,808 total
];

const NUM_WARMUP = 10;
const NUM_ITERATIONS = 100;

class SentinelBestCaseTester {
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
            label: 'BestCase Sentinel DPI bind group layout',
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
            label: 'BestCase Sentinel DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: sentinelDpiShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'BestCase Sentinel Lookback bind group layout',
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
            label: 'BestCase Sentinel Lookback pipeline',
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

// --- Synthetic Data Generators ---

/**
 * Pattern 1: Disjoint
 * A = [0, 1, 2, ..., N-1]
 * B = [N, N+1, N+2, ..., 2N-1]
 *
 * Best case because:
 * - 0% intersection (A and B completely disjoint)
 * - Merge path extremely regular: first half all A, second half all B
 * - Each partition is either all A or all B (except boundary)
 * - Minimal serial intersection comparisons: quick skip of A or B blocks
 */
function generateDisjoint(n: number): { A: Uint32Array; B: Uint32Array } {
    const A = new Uint32Array(n);
    const B = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        A[i] = i;
        B[i] = n + i;
    }
    return { A, B };
}

/**
 * Pattern 2: Interleaved
 * A = [1, 3, 5, 7, ...] (odd numbers)
 * B = [2, 4, 6, 8, ...] (even numbers)
 *
 * Best case because:
 * - 0% intersection (odd and even never match)
 * - A and B elements uniformly alternate in each partition
 * - Workload perfectly balanced across all workgroups
 * - Serial intersection comparisons are regular: A and B pointers alternate
 */
function generateInterleaved(n: number): { A: Uint32Array; B: Uint32Array } {
    const A = new Uint32Array(n);
    const B = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
        A[i] = i * 2 + 1;  // 1, 3, 5, 7, ...
        B[i] = (i + 1) * 2; // 2, 4, 6, 8, ...
    }
    return { A, B };
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

// --- Output Helpers ---

function formatNumber(n: number): string {
    return n.toLocaleString();
}

function printHeader() {
    console.log(
        `${'Size'.padEnd(8)} ` +
        `${'|A|'.padStart(14)} ` +
        `${'|B|'.padStart(14)} ` +
        `${'Total'.padStart(14)} ` +
        `${'Result'.padStart(10)} ` +
        `${'DPI(ms)'.padStart(10)} ` +
        `${'LB(ms)'.padStart(10)} ` +
        `${'Total(ms)'.padStart(12)} ` +
        `${'Throughput'.padStart(14)} ` +
        `${'Valid'.padStart(6)}`
    );
    console.log('-'.repeat(120));
}

async function runBenchmark(
    tester: SentinelBestCaseTester,
    label: string,
    A: Uint32Array,
    B: Uint32Array,
    skipCpuValidation: boolean,
    expectedCount: number = -1  // -1 means no expected value check
): Promise<void> {
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
        `${formatNumber(result.totalCount).padStart(10)} ` +
        `${result.timing.dpiMs.toFixed(3).padStart(10)} ` +
        `${result.timing.lookbackMs.toFixed(3).padStart(10)} ` +
        `${result.timing.totalMs.toFixed(3).padStart(12)} ` +
        `${(throughput.toFixed(2) + ' M/s').padStart(14)} ` +
        `${validStr.padStart(6)}`
    );
}

// --- Main Benchmark Entry Point ---

export async function runSentinelBestCaseBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(120));
    console.log('  2-PHASE SENTINEL: BEST CASE BENCHMARK');
    console.log('  Pipeline: DPI -> Decoupled Lookback Sentinel');
    console.log(`  VT=${VT_SENT}, NV=${NV_SENT} | ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations`);
    console.log('='.repeat(120));

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('\nERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new SentinelBestCaseTester(device, timestampQueryManager);

    // ========================================
    // Pattern 1: Disjoint
    // ========================================
    console.log('\n' + '-'.repeat(120));
    console.log('PATTERN 1: DISJOINT');
    console.log('A = [0, 1, 2, ..., N-1], B = [N, N+1, N+2, ..., 2N-1]');
    console.log('0% intersection | Merge path: front half all A, back half all B');
    console.log('-'.repeat(120) + '\n');
    printHeader();

    for (const { label, perArray } of BENCHMARK_SIZES) {
        const { A, B } = generateDisjoint(perArray);
        const skipCpu = perArray > 32_000_000;
        // Expected count is 0 for disjoint pattern (0% intersection)
        await runBenchmark(tester, label, A, B, skipCpu, 0);
    }

    // ========================================
    // Pattern 2: Interleaved
    // ========================================
    console.log('\n' + '-'.repeat(120));
    console.log('PATTERN 2: INTERLEAVED');
    console.log('A = [1, 3, 5, ...] (odd), B = [2, 4, 6, ...] (even)');
    console.log('0% intersection | Perfect workload balance (A/B alternate in each partition)');
    console.log('-'.repeat(120) + '\n');
    printHeader();

    for (const { label, perArray } of BENCHMARK_SIZES) {
        const { A, B } = generateInterleaved(perArray);
        const skipCpu = perArray > 32_000_000;
        // Expected count is 0 for interleaved pattern (0% intersection)
        await runBenchmark(tester, label, A, B, skipCpu, 0);
    }

    // ========================================
    // Summary
    // ========================================
    console.log('\n' + '='.repeat(120));
    console.log('  SUMMARY');
    console.log('='.repeat(120));
    console.log('  Best case scenarios for 2-Phase Sentinel intersection:');
    console.log('');
    console.log('  Disjoint Pattern:');
    console.log('    - 0% intersection -> no scatter writes');
    console.log('    - Block-level skipping -> minimal comparisons');
    console.log('    - Tests DPI + lookback with zero output overhead');
    console.log('');
    console.log('  Interleaved Pattern:');
    console.log('    - 0% intersection -> no scatter writes');
    console.log('    - Perfect workload distribution -> minimal spin-wait');
    console.log('    - All workgroups finish at nearly the same time');
    console.log('');
    console.log('  All sizes are NV (3072) multiples for maximum GPU occupancy.');
    console.log('  Validation: OK = full CPU validation, OK* = validated by expected count (0).');
    console.log('  CPU validation for sizes <= 32M; expected value check for 64M.');
    console.log('');
}
