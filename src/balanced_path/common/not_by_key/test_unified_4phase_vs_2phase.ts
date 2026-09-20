/**
 * Unified Test: 4-Phase vs 2-Phase Pipeline for All 4 Set Operations
 *
 * Keys-only variant: compares and outputs plain u32 arrays (no values).
 *
 * Tests the unified common shaders with OP_MODE string replacement:
 *   0 = intersection (A ∩ B)
 *   1 = difference   (A \ B)
 *   2 = union        (A ∪ B)
 *   3 = sym_difference ((A\B) ∪ (B\A))
 *
 * 4-Phase Pipeline:
 *   1. DPI - Compute diagonal path indices
 *   2. Count - Count matches per workgroup
 *   3. Prefix Sum - Exclusive scan for output offsets
 *   4. Write - Write result elements to compacted output
 *
 * 2-Phase Pipeline:
 *   1. DPI - Compute diagonal path indices
 *   2. Decoupled Lookback - Single-pass count + write with built-in prefix sum
 */

import TimestampQueryManager from '../../../TimestampQueryManager';
import * as utils from '../../../utils';
import { setOpMode } from '../../../utils';
import computeDiagonalsShader from '../balanced_path_biased.wgsl';
import countShaderBase from '../set_availability_count.wgsl';
import writeShaderBase from './set_availability_write.wgsl';
import { ExclusiveScanPipeline } from '../prefix_sum/exclusive_scan';
import { TwoPhasePipelineTester, getMaxOutputSize, MAXWORKGROUP, NT, VT, NV, DPI_WG_SIZE } from './two_phase_pipeline';

const OP_NAMES = ['intersection', 'difference', 'union', 'sym_difference'] as const;
type OpName = typeof OP_NAMES[number];

/** CPU validation functions (keys-only). */
const CPU_FUNCTIONS: Record<OpName, (a: Uint32Array, b: Uint32Array) => Uint32Array> = {
    intersection: utils.setIntersectionCPU,
    difference: utils.setDifferenceCPU,
    union: utils.setUnionCPU,
    sym_difference: utils.setSymmetricDifferenceCPU,
};

/**
 * 4-Phase Pipeline Tester (DPI -> Count -> Scan -> Write)
 */
class FourPhasePipelineTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    // GPU span of every timed run of the last run() call (first pass start to last pass end)
    public lastTotalTimes: number[] = [];
    // Per-phase GPU times of every timed run of the last run() call. Scan is the span minus
    // the Partition (DPI), Count and Write passes, so it also holds the gaps between passes.
    public lastPhaseTimes: { dpi: number[]; count: number[]; scan: number[]; write: number[] } =
        { dpi: [], count: [], scan: [], write: [] };
    private label: string;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager, label: string, opMode: number) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.label = label;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // Count shader
        const countShader = setOpMode(countShaderBase, opMode);
        // Write shader
        const writeShader = setOpMode(writeShaderBase, opMode);

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

        // Write bind group layout (8 bindings)
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Write kernel bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // offsets
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // b_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // num_wg_total
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
        aKeys: Uint32Array,
        bKeys: Uint32Array,
        iterations: number,
        warmup: number
    ): Promise<{
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
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV);

        // Create input buffers
        const bufferA = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(aKeys));

        const bufferB = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(bKeys));

        // Uniform buffers
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

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        // DPI and Count bind groups
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

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // First pass: DPI + Count + Scan to determine output size
        {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Read back counts to determine actual output size
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

        // Create output buffer
        const bufferOutput = device.createBuffer({
            size: Math.max(4, totalCount * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Write bind group (8 bindings)
        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },    // offsets (prefix sum result)
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        // Warmup runs
        for (let w = 0; w < warmup; w++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            pass = encoder.beginComputePass();
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Timed runs
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(tsm.createComputePassDescriptor(0, 1));
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(2, 3));
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

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
                const scanNs = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNs / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
        this.lastTotalTimes = totalTimes;
        this.lastPhaseTimes = { dpi: dpiTimes, count: countTimes, scan: scanTimes, write: writeTimes };

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
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                countMs: avg(countTimes),
                scanMs: avg(scanTimes),
                writeMs: avg(writeTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

// ============================================================================
// Exported test runners
// ============================================================================

/**
 * Run correctness + benchmark for a single set operation.
 */
async function runSingleOpTest(
    device: GPUDevice,
    opMode: number,
    opName: OpName
): Promise<void> {
    const NUM_ITERATIONS = 100;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhaseTester = new FourPhasePipelineTester(device, timestampQueryManager, `4P-${opName}`, opMode);
    const twoPhaseTester = new TwoPhasePipelineTester(device, timestampQueryManager, `2P-${opName}`, opMode);

    // ========================================================================
    // Performance Benchmarks
    // ========================================================================
    console.log(`  Performance (${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations):\n`);

    console.log('  Dataset   | 4-Phase Total(ms) | 2-Phase Total(ms) | Winner  | Match | Output Count');
    console.log('  ----------|-------------------|-------------------|---------|-------|-------------');

    const datasets = [
        { size: '1', range: 'e2' },
        { size: '1', range: 'e6' },
        { size: '2', range: 'e2' },
        { size: '2', range: 'e6' },
        { size: '4', range: 'e2' },
        { size: '4', range: 'e6' },
        { size: '8', range: 'e2' },
        { size: '8', range: 'e6' },
        { size: '16', range: 'e2' },
        { size: '16', range: 'e6' },
        { size: '32', range: 'e2' },
        { size: '32', range: 'e6' },
        { size: '64', range: 'e2' },
        { size: '64', range: 'e6' },
        { size: '128', range: 'e2' },
        { size: '128', range: 'e6' },
    ];

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const aKeys = await utils.loadUint32ArrayFromBin(aPath);
            const bKeys = await utils.loadUint32ArrayFromBin(bPath);

            const fourResult = await fourPhaseTester.run(aKeys, bKeys, NUM_ITERATIONS, NUM_WARMUP);
            const twoResult = await twoPhaseTester.run(aKeys, bKeys, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = fourResult.totalCount === twoResult.totalCount;

            const ds = `${size}M${range}`.padEnd(8);
            const f_total = fourResult.timing.totalMs.toFixed(3).padStart(17);
            const t_total = twoResult.timing.totalMs.toFixed(3).padStart(17);

            let winner: string;
            if (fourResult.timing.totalMs < twoResult.timing.totalMs) {
                const ratio = (twoResult.timing.totalMs / fourResult.timing.totalMs).toFixed(2);
                winner = `4P ${ratio}x`;
            } else {
                const ratio = (fourResult.timing.totalMs / twoResult.timing.totalMs).toFixed(2);
                winner = `2P ${ratio}x`;
            }
            const matchStr = countMatch ? 'OK  ' : 'FAIL';
            const countStr = twoResult.totalCount.toLocaleString().padStart(13);

            console.log(`  ${ds} |${f_total} |${t_total} | ${winner.padEnd(7)} | ${matchStr} |${countStr}`);

        } catch (error) {
            const ds = `${size}M${range}`.padEnd(8);
            console.log(`  ${ds} | Error loading dataset`);
        }
    }

    console.log('');
}

/**
 * Run all 4 set operations: correctness + performance comparison.
 */
export async function runUnified4PhaseVs2PhaseTest(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  UNIFIED 4-PHASE vs 2-PHASE TEST (All 4 Set Operations)');
    console.log('  Using common shaders with OP_MODE replacement');
    console.log('========================================================================\n');

    for (let opMode = 0; opMode < 4; opMode++) {
        const opName = OP_NAMES[opMode];

        console.log(`--- ${opName.toUpperCase()} (OP_MODE=${opMode}) ---\n`);
        await runSingleOpTest(device, opMode, opName);
    }

    console.log('All 4 operations tested.\n');
}

/**
 * Run a single operation test (for selective testing).
 */
export async function runUnifiedSingleOpTest(device: GPUDevice, opMode: number): Promise<void> {
    const opName = OP_NAMES[opMode];

    console.log(`\n--- ${opName.toUpperCase()} (OP_MODE=${opMode}) ---\n`);
    await runSingleOpTest(device, opMode, opName);
}

/**
 * Pipeline fusion benchmark (Table tab:fusion): the 4-step and the 2-step pipeline on the
 * same inputs. Both are timed as the GPU span of the whole pipeline, from the start of its
 * first pass to the end of its last pass, recorded with timestamp queries. Every timed run
 * is printed in a [fusion-result] JSON line.
 * URL: ?app=fusion[&ops=0,2][&range=e2][&sizes=1,2,4,8,16,32,64,128][&w=10][&n=100]
 */
export async function runFusionBenchmark(device: GPUDevice): Promise<void> {
    const params = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);
    const ops = (params.get('ops') ?? '0,2').split(',').map(Number);
    const range = params.get('range') ?? 'e2';
    const sizes = (params.get('sizes') ?? '1,2,4,8,16,32,64,128').split(',');
    const NUM_WARMUP = parseInt(params.get('w') ?? '10', 10);
    const NUM_ITERATIONS = parseInt(params.get('n') ?? '100', 10);

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported.\n');
        return;
    }
    const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / Math.max(v.length, 1);

    for (const opMode of ops) {
        const opName = OP_NAMES[opMode];
        const fourPhaseTester = new FourPhasePipelineTester(device, timestampQueryManager, `4P-${opName}`, opMode);
        const twoPhaseTester = new TwoPhasePipelineTester(device, timestampQueryManager, `2P-${opName}`, opMode);
        const cpuFn = CPU_FUNCTIONS[opName];
        console.log(`\n--- FUSION ${opName.toUpperCase()} (${range}, ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} runs) ---`);

        for (const size of sizes) {
            try {
                const aKeys = await utils.loadUint32ArrayFromBin(`./data/A_${size}${range}.bin`);
                const bKeys = await utils.loadUint32ArrayFromBin(`./data/B_${size}${range}.bin`);

                const fourResult = await fourPhaseTester.run(aKeys, bKeys, NUM_ITERATIONS, NUM_WARMUP);
                const fourRuns = fourPhaseTester.lastTotalTimes;
                const fourPhases = fourPhaseTester.lastPhaseTimes;
                const twoResult = await twoPhaseTester.run(aKeys, bKeys, NUM_ITERATIONS, NUM_WARMUP);
                const twoRuns = twoResult.runs.spanMs;

                let check = fourResult.totalCount === twoResult.totalCount ? 'match' : 'MISMATCH';
                if (aKeys.length <= 64_000_000 && cpuFn) {
                    const reference = cpuFn(aKeys, bKeys).length;
                    check = fourResult.totalCount === reference && twoResult.totalCount === reference ? 'PASS' : 'FAIL';
                }

                console.log(`  ${size}M${range}: 4-step ${mean(fourRuns).toFixed(3)} ms, 2-step ${mean(twoRuns).toFixed(3)} ms, ` +
                    `speedup ${(mean(fourRuns) / mean(twoRuns)).toFixed(2)}x, count ${twoResult.totalCount}, ${check}`);
                console.log(`[fusion-result] ${JSON.stringify({
                    op: opName, dataset: `${size}${range}`, warmup: NUM_WARMUP, runs: twoRuns.length,
                    count: twoResult.totalCount, check, four_step_ms: fourRuns, two_step_ms: twoRuns,
                    four_step_phases_ms: fourPhases,
                })}`);
            } catch (e) {
                console.log(`  ${size}M${range}: Error ${e}`);
            }
        }
    }
    console.log('');
}

/**
 * Run 2-Phase only benchmark for a single set operation.
 */
export async function run2PhaseOnlyTest(device: GPUDevice, opMode: number, sizes?: string[]): Promise<void> {
    const opName = OP_NAMES[opMode];
    // ?w=<warmup>&n=<timed runs>&ph=<preheat ms> override the defaults: 10 warmup + 100 timed
    // runs, no unrelated GPU work (preheat) before each size.
    const params = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);
    const NUM_ITERATIONS = parseInt(params.get('n') ?? '100', 10);
    const NUM_WARMUP = parseInt(params.get('w') ?? '10', 10);
    const PREHEAT_MS = parseInt(params.get('ph') ?? '0', 10);

    console.log(`\n--- 2-PHASE ONLY: ${opName.toUpperCase()} (OP_MODE=${opMode}) ---`);
    console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations\n`);

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported.\n');
        return;
    }

    let twoPhaseTester: TwoPhasePipelineTester;
    try {
        twoPhaseTester = new TwoPhasePipelineTester(device, timestampQueryManager, `2P-${opName}`, opMode);
    } catch (e) {
        console.log(`  ERROR creating pipeline: ${e}`);
        return;
    }
    const cpuFn = CPU_FUNCTIONS[opName];

    console.log('  Dataset   | DPI(ms)    | Lookback(ms) | Total(ms)  | Output Count     | Match');
    console.log('  ----------|------------|--------------|------------|------------------|------');

    // Rebuttal: trimmed to the e6 sizes used for the variance/CV check, so there
    // are few allocations before 128M and the GPU device stays clean.
    // Pass `sizes` (millions of elements, e6 range) to run the full 1M-128M scalability sweep.
    const datasets = sizes
        ? sizes.map(size => ({ size, range: 'e6' }))
        : [
            { size: '16', range: 'e6' },
            { size: '64', range: 'e6' },
            { size: '128', range: 'e6' },
        ];

    for (const { size, range } of datasets) {
        try {
            const aKeys = await utils.loadUint32ArrayFromBin(`./data/A_${size}${range}.bin`);
            const bKeys = await utils.loadUint32ArrayFromBin(`./data/B_${size}${range}.bin`);

            const result = await twoPhaseTester.run(aKeys, bKeys, NUM_ITERATIONS, NUM_WARMUP, PREHEAT_MS);

            // CPU validation
            let matchStr = 'skip';
            if (aKeys.length <= 64_000_000) {
                const cpuResult = cpuFn(aKeys, bKeys);
                matchStr = result.totalCount === cpuResult.length ? 'OK' : 'FAIL';
            }

            const ds = `${size}M${range}`.padEnd(8);
            const dpiCol = result.timing.dpiMs.toFixed(3).padStart(10);
            const lbCol = result.timing.lookbackMs.toFixed(3).padStart(12);
            const totalCol = result.timing.totalMs.toFixed(3).padStart(10);
            const countCol = result.totalCount.toLocaleString().padStart(16);

            console.log(`  ${ds} |${dpiCol} |${lbCol} |${totalCol} |${countCol} | ${matchStr}`);
            console.log(`[micro-result] ${JSON.stringify({
                impl: 'webgpu', op: opName, dataset: `${size}${range}`, preheat_ms: PREHEAT_MS, warmup: NUM_WARMUP,
                runs: result.runs.kernelMs.length, count: result.totalCount, check: matchStr,
                dpi_ms: result.runs.dpiMs, lookback_ms: result.runs.lookbackMs,
                kernel_ms: result.runs.kernelMs, span_ms: result.runs.spanMs,
            })}`);
        } catch (e) {
            console.log(`  ${size}M${range}`.padEnd(10) + ` | Error: ${e}`);
        }
    }
    console.log('');
}
