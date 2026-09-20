/**
 * Test file for Sentinel-Optimized Count Kernel comparison
 *
 * Compares set_availability_intersection_count_v1_optimized.wgsl (baseline)
 * with set_availability_intersection_count_v1_sentinel.wgsl (sentinel optimization)
 *
 * Sentinel optimization:
 * - Adds sentinel values (NEG_INF=0 at start, POS_INF=0xFFFFFFFF at end) to eliminate bounds checking
 * - All data indices shifted by +1 to accommodate leading sentinel
 * - Expected benefit: eliminates boundary checks in binary search loops
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShaderOptimized from './set_availability_intersection_count_v1_optimized.wgsl';
import countShaderSentinel from './set_availability_intersection_count_v1_sentinel.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const MAXWORKGROUP = 65535;

const NT = 256;
const VT = 7;
const NV = NT * VT;

/**
 * Generic count kernel tester - works with any shader code that has
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

        // Average GPU time in milliseconds
        const gpuTimeMs = timingsNs.length > 0
            ? (timingsNs.reduce((a, b) => a + b, 0) / timingsNs.length) / 1_000_000
            : 0;

        // Read back counts
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
        countsReadback.destroy();

        return { counts, totalCount, numWorkgroups: numWg, gpuTimeMs };
    }
}

// ============================================================================
// Main: Run v1_optimized vs v1_sentinel dataset benchmark
// ============================================================================
export async function runCountV1VsSentinelTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(70));
    console.log('  V1_OPTIMIZED vs V1_SENTINEL COUNT KERNEL COMPARISON');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(70) + '\n');

    const tsm = new TimestampQueryManager(device, 2);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const optimized = new CountKernelTester(device, countShaderOptimized, 'v1_optimized', tsm);
    const sentinel = new CountKernelTester(device, countShaderSentinel, 'v1_sentinel', tsm);

    const NUM_WARMUP = 10;
    const NUM_ITERATIONS = 10;

    let allPassed = true;

    // ========================================================================
    // Dataset Benchmark: Real binary data files
    // ========================================================================
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

    const datasetResults: { desc: string; optMs: number; sentMs: number; speedup: string; optCount: number; sentCount: number }[] = [];

    console.log(`${'Dataset'.padEnd(30)} ${'|A|'.padStart(10)} ${'|B|'.padStart(10)} ${'v1_opt'.padStart(12)} ${'v1_sent'.padStart(12)} ${'Speedup'.padStart(10)} ${'Count'.padStart(10)} ${'Match'.padStart(6)}`);
    console.log('-'.repeat(100));

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const optResult = await optimized.run(A, B, NUM_WARMUP, NUM_ITERATIONS);
            const sentResult = await sentinel.run(A, B, NUM_WARMUP, NUM_ITERATIONS);

            const countsMatch = optResult.totalCount === sentResult.totalCount;
            const matchStr = countsMatch ? 'OK' : 'FAIL';
            const speedup = optResult.gpuTimeMs / sentResult.gpuTimeMs;

            datasetResults.push({
                desc,
                optMs: optResult.gpuTimeMs,
                sentMs: sentResult.gpuTimeMs,
                speedup: speedup.toFixed(2) + 'x',
                optCount: optResult.totalCount,
                sentCount: sentResult.totalCount,
            });

            console.log(
                `${desc.padEnd(30)} ${A.length.toString().padStart(10)} ${B.length.toString().padStart(10)} ` +
                `${(optResult.gpuTimeMs.toFixed(3) + ' ms').padStart(12)} ${(sentResult.gpuTimeMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ${optResult.totalCount.toString().padStart(10)} ${matchStr.padStart(6)}`
            );

            if (!countsMatch) {
                console.log(`    WARNING: Count mismatch! opt=${optResult.totalCount}, sent=${sentResult.totalCount}`);
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
    console.log(`${'Dataset'.padEnd(30)} ${'v1_optimized'.padStart(14)} ${'v1_sentinel'.padStart(14)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(70));
    for (const r of datasetResults) {
        console.log(
            `${r.desc.padEnd(30)} ${(r.optMs.toFixed(3) + ' ms').padStart(14)} ${(r.sentMs.toFixed(3) + ' ms').padStart(14)} ${r.speedup.padStart(10)}`
        );
    }
    console.log('-'.repeat(70));

    if (allPassed) {
        console.log('\nAll tests PASSED.');
    } else {
        console.log('\nSome tests FAILED!');
    }
    console.log('');
}
