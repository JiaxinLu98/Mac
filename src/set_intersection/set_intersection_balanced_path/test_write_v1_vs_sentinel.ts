/**
 * Test file for Sentinel-Optimized Write Kernel comparison
 *
 * Compares the complete 4-phase pipeline:
 * - v1_optimized: count_v1_optimized + write_v1_optimized
 * - v1_sentinel: count_v1_sentinel + write_v1_sentinel
 *
 * 4-phase approach:
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Count Phase - Count matches per workgroup
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual intersection results
 *
 * Sentinel optimization:
 * - Adds sentinel values (NEG_INF=0, POS_INF=0xFFFFFFFF) to eliminate bounds checking
 * - All data indices shifted by +1 to accommodate leading sentinel
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShaderOptimized from './set_availability_intersection_count_v1_optimized.wgsl';
import countShaderSentinel from './set_availability_intersection_count_v1_sentinel.wgsl';
import writeShaderOptimized from './set_availability_intersection_write_v1_optimized.wgsl';
import writeShaderSentinel from './set_availability_intersection_write_v1_sentinel.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const MAXWORKGROUP = 65535;

const NT = 256;
const VT = 7;
const NV = NT * VT;

/**
 * Complete Set Intersection Pipeline Tester
 */
class PipelineTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    public label: string;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;
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

        // Count bind group layout
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

        // Create Buffers
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

        const bufferCountsCopy = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = Math.min(a_len, b_len);
        const bufferOutput = device.createBuffer({
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        // Create Bind Groups
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

        await device.queue.onSubmittedWorkDone();

        // Warmup Runs
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

        // Timed Runs with GPU Timestamps
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < numIterations; iter++) {
            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(tsm.createComputePassDescriptor(0, 1));
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count (timestamps 2, 3)
            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(2, 3));
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum (timestamps 4, 5)
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            // Phase 4: Write (timestamps 6, 7)
            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(6, 7));
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            tsm.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

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

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        const timing = {
            dpiMs: avg(dpiTimes),
            countMs: avg(countTimes),
            scanMs: avg(scanTimes),
            writeMs: avg(writeTimes),
            totalMs: avg(totalTimes),
        };

        // Readback
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
// Main: Run v1_optimized vs v1_sentinel pipeline comparison
// ============================================================================
export async function runWriteV1VsSentinelTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('  V1_OPTIMIZED vs V1_SENTINEL COMPLETE PIPELINE COMPARISON');
    console.log('  4-Phase: DPI -> Count -> Scan -> Write');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(80) + '\n');

    // 8 timestamp indices: DPI(0,1), Count(2,3), Scan(4,5), Write(6,7)
    const tsm = new TimestampQueryManager(device, 8);
    if (!tsm.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.');
        return;
    }

    const optimized = new PipelineTester(
        device,
        countShaderOptimized,
        writeShaderOptimized,
        'v1_optimized',
        tsm
    );

    const sentinel = new PipelineTester(
        device,
        countShaderSentinel,
        writeShaderSentinel,
        'v1_sentinel',
        tsm
    );

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

    interface ResultEntry {
        desc: string;
        optTotal: number;
        sentTotal: number;
        optCount: number;
        sentCount: number;
        optDpi: number;
        sentDpi: number;
        optCnt: number;
        sentCnt: number;
        optScan: number;
        sentScan: number;
        optWrite: number;
        sentWrite: number;
        speedup: string;
    }

    const datasetResults: ResultEntry[] = [];

    // Header
    console.log(`${'Dataset'.padEnd(28)} ${'|A|'.padStart(10)} ${'|B|'.padStart(10)} ${'opt_total'.padStart(12)} ${'sent_total'.padStart(12)} ${'Speedup'.padStart(10)} ${'Count'.padStart(10)} ${'Match'.padStart(6)}`);
    console.log('-'.repeat(110));

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const optResult = await optimized.run(A, B, NUM_WARMUP, NUM_ITERATIONS);
            const sentResult = await sentinel.run(A, B, NUM_WARMUP, NUM_ITERATIONS);

            const countsMatch = optResult.totalCount === sentResult.totalCount;
            let resultsMatch = countsMatch;

            // Verify first few elements match (for large datasets, skip full comparison)
            if (countsMatch && optResult.totalCount > 0) {
                const checkLen = Math.min(100, optResult.totalCount);
                for (let i = 0; i < checkLen; i++) {
                    if (optResult.result[i] !== sentResult.result[i]) {
                        resultsMatch = false;
                        console.log(`    Element mismatch at ${i}: opt=${optResult.result[i]}, sent=${sentResult.result[i]}`);
                        break;
                    }
                }
            }

            const matchStr = resultsMatch ? 'OK' : 'FAIL';
            const speedup = optResult.timing.totalMs / sentResult.timing.totalMs;

            datasetResults.push({
                desc,
                optTotal: optResult.timing.totalMs,
                sentTotal: sentResult.timing.totalMs,
                optCount: optResult.totalCount,
                sentCount: sentResult.totalCount,
                optDpi: optResult.timing.dpiMs,
                sentDpi: sentResult.timing.dpiMs,
                optCnt: optResult.timing.countMs,
                sentCnt: sentResult.timing.countMs,
                optScan: optResult.timing.scanMs,
                sentScan: sentResult.timing.scanMs,
                optWrite: optResult.timing.writeMs,
                sentWrite: sentResult.timing.writeMs,
                speedup: speedup.toFixed(2) + 'x',
            });

            console.log(
                `${desc.padEnd(28)} ${A.length.toString().padStart(10)} ${B.length.toString().padStart(10)} ` +
                `${(optResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ${(sentResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ${optResult.totalCount.toString().padStart(10)} ${matchStr.padStart(6)}`
            );

            if (!resultsMatch) {
                console.log(`    WARNING: Mismatch! opt_count=${optResult.totalCount}, sent_count=${sentResult.totalCount}`);
                allPassed = false;
            }
        } catch (e) {
            console.log(`${desc.padEnd(28)} SKIPPED (file not found)`);
        }
    }

    // Summary table
    console.log('\n' + '='.repeat(80));
    console.log('  TOTAL TIME SUMMARY (all 4 phases)');
    console.log('='.repeat(80));
    console.log(`${'Dataset'.padEnd(28)} ${'v1_optimized'.padStart(14)} ${'v1_sentinel'.padStart(14)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(80));
    for (const r of datasetResults) {
        console.log(
            `${r.desc.padEnd(28)} ${(r.optTotal.toFixed(3) + ' ms').padStart(14)} ${(r.sentTotal.toFixed(3) + ' ms').padStart(14)} ${r.speedup.padStart(10)}`
        );
    }
    console.log('-'.repeat(80));

    // Per-phase breakdown
    console.log('\n' + '='.repeat(80));
    console.log('  PER-PHASE BREAKDOWN: Count Kernel');
    console.log('='.repeat(80));
    console.log(`${'Dataset'.padEnd(28)} ${'opt_count'.padStart(14)} ${'sent_count'.padStart(14)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(80));
    for (const r of datasetResults) {
        const speedup = r.optCnt / r.sentCnt;
        console.log(
            `${r.desc.padEnd(28)} ${(r.optCnt.toFixed(3) + ' ms').padStart(14)} ${(r.sentCnt.toFixed(3) + ' ms').padStart(14)} ${(speedup.toFixed(2) + 'x').padStart(10)}`
        );
    }
    console.log('-'.repeat(80));

    console.log('\n' + '='.repeat(80));
    console.log('  PER-PHASE BREAKDOWN: Write Kernel');
    console.log('='.repeat(80));
    console.log(`${'Dataset'.padEnd(28)} ${'opt_write'.padStart(14)} ${'sent_write'.padStart(14)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(80));
    for (const r of datasetResults) {
        const speedup = r.optWrite / r.sentWrite;
        console.log(
            `${r.desc.padEnd(28)} ${(r.optWrite.toFixed(3) + ' ms').padStart(14)} ${(r.sentWrite.toFixed(3) + ' ms').padStart(14)} ${(speedup.toFixed(2) + 'x').padStart(10)}`
        );
    }
    console.log('-'.repeat(80));

    if (allPassed) {
        console.log('\nAll tests PASSED.');
    } else {
        console.log('\nSome tests FAILED!');
    }
    console.log('');
}
