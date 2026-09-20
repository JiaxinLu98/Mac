/**
 * Test file for Decoupled Lookback Kernel - Original vs Optimized Comparison
 *
 * Compares the original decoupled lookback kernel with the optimized version.
 * Optimizations applied:
 * - Opt 1: Branchless select() + pre-fetch in serial_set_intersection
 * - Opt 2: Unrolled loads in device_load_2_to_shared (7 iterations)
 *
 * This test runs both kernels on the same data and compares:
 * - Correctness (both should produce identical results)
 * - Performance (GPU timestamp profiling)
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import originalLookbackShader from './set_availability_intersection_decoupled_lookback.wgsl';
import optimizedLookbackShader from './set_availability_intersection_decoupled_lookback_optimized.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Tester class that can run either original or optimized lookback shader.
 */
class LookbackKernelTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private label: string;

    // Pipelines
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(
        device: GPUDevice,
        timestampQueryManager: TimestampQueryManager,
        lookbackShader: string,
        label: string
    ) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.label = label;

        // DPI pipeline (shared between original and optimized)
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `DPI bind group layout (${label})`,
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
            label: `DPI pipeline (${label})`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback pipeline (uses provided shader)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: `Decoupled Lookback bind group layout (${label})`,
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
            label: `Decoupled Lookback pipeline (${label})`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: lookbackShader }),
                entryPoint: 'intersection_decoupled_lookback'
            }
        });
    }

    /**
     * Run intersection with GPU timestamp profiling.
     */
    public async computeIntersectionWithProfiling(
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

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = Math.min(a_len, b_len);

        // ============ Buffer Setup ============
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

        const bufferState = device.createBuffer({
            label: 'Buffer State',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferTotalCount = device.createBuffer({
            label: 'Buffer Total Count',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

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

        // ============ Warmup (no timestamps) ============
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

        // ============ Timed iterations ============
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Decoupled Lookback (timestamps 2, 3)
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

        // Calculate average
        const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

        const dpiMs = avg(dpiTimes);
        const lookbackMs = avg(lookbackTimes);
        const totalMs = avg(totalTimes);

        // ============ Readback ============
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

        // Cleanup
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
            timing: { dpiMs, lookbackMs, totalMs }
        };
    }
}

/**
 * CPU reference implementation for validation.
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

/**
 * Run Original vs Optimized comparison on all datasets.
 */
export async function runOptimizedDecoupledLookbackTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════════════════════╗');
    console.log('║    DECOUPLED LOOKBACK: ORIGINAL vs OPTIMIZED COMPARISON                   ║');
    console.log('║    Optimizations: Branchless select + pre-fetch, Unrolled loads           ║');
    console.log('╚════════════════════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 10;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const originalTester = new LookbackKernelTester(
        device, timestampQueryManager, originalLookbackShader, 'Original'
    );
    const optimizedTester = new LookbackKernelTester(
        device, timestampQueryManager, optimizedLookbackShader, 'Optimized'
    );

    console.log(`Running ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations per kernel...\n`);

    // ========================================================================
    // Correctness Tests (synthetic data)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('CORRECTNESS TESTS');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

    const syntheticTests = [
        { name: 'Small arrays', A: new Uint32Array([1, 3, 3, 5, 7, 9]), B: new Uint32Array([2, 3, 3, 6, 7, 8]) },
        { name: 'No intersection', A: new Uint32Array([1, 3, 5, 7, 9]), B: new Uint32Array([2, 4, 6, 8, 10]) },
        { name: 'Complete match', A: new Uint32Array([1, 2, 3, 4, 5]), B: new Uint32Array([1, 2, 3, 4, 5]) },
        { name: 'All same value (2500)', A: new Uint32Array(2500).fill(42), B: new Uint32Array(2500).fill(42) },
        {
            name: 'Multi-WG (2000 each)',
            A: (() => { const a = new Uint32Array(2000); for (let i = 0; i < 2000; i++) a[i] = i * 2; return a; })(),
            B: (() => { const b = new Uint32Array(2000); for (let i = 0; i < 2000; i++) b[i] = i * 3; return b; })()
        },
        {
            name: 'Large (10K each)',
            A: (() => { const a = new Uint32Array(10000); for (let i = 0; i < 10000; i++) a[i] = i; return a; })(),
            B: (() => { const b = new Uint32Array(10000); for (let i = 0; i < 10000; i++) b[i] = i * 2; return b; })()
        },
    ];

    let allCorrect = true;

    for (const test of syntheticTests) {
        const cpuResult = cpuSetIntersection(test.A, test.B);
        const cpuCount = cpuResult.length;

        // Run optimized only for correctness (original is already validated)
        const optResult = await optimizedTester.computeIntersectionWithProfiling(test.A, test.B, 1, 0);

        const countMatch = optResult.totalCount === cpuCount;
        let valuesMatch = true;
        if (countMatch && optResult.totalCount > 0) {
            const sortedGpu = [...optResult.result].sort((a, b) => a - b);
            const sortedCpu = [...cpuResult].sort((a, b) => a - b);
            for (let i = 0; i < optResult.totalCount; i++) {
                if (sortedGpu[i] !== sortedCpu[i]) {
                    valuesMatch = false;
                    break;
                }
            }
        }

        const status = (countMatch && valuesMatch) ? '✔' : '✗';
        console.log(`  ${status} ${test.name}: GPU=${optResult.totalCount}, CPU=${cpuCount}`);

        if (!countMatch || !valuesMatch) allCorrect = false;
    }

    console.log(allCorrect ? '\n  All correctness tests PASSED\n' : '\n  Some correctness tests FAILED\n');

    // ========================================================================
    // Performance Benchmarks (binary datasets)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('PERFORMANCE BENCHMARKS');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

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

    console.log('╔══════════╤══════════════════════════════╤══════════════════════════════╤═════════╤═══════╗');
    console.log('║          │ Original                     │ Optimized                    │         │       ║');
    console.log('║ Dataset  │ DPI(ms)│Lookbk(ms)│Total(ms) │ DPI(ms)│Lookbk(ms)│Total(ms) │ Speedup │ Match ║');
    console.log('╠══════════╪════════╪══════════╪══════════╪════════╪══════════╪══════════╪═════════╪═══════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            // Run original first
            const origResult = await originalTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            // Run optimized
            const optResult = await optimizedTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            // Check if counts match
            const countMatch = origResult.totalCount === optResult.totalCount;

            const ds = `${size}${range}`.padEnd(8);
            const origDpi = origResult.timing.dpiMs.toFixed(2).padStart(6);
            const origLookback = origResult.timing.lookbackMs.toFixed(2).padStart(8);
            const origTotal = origResult.timing.totalMs.toFixed(2).padStart(8);
            const optDpi = optResult.timing.dpiMs.toFixed(2).padStart(6);
            const optLookback = optResult.timing.lookbackMs.toFixed(2).padStart(8);
            const optTotal = optResult.timing.totalMs.toFixed(2).padStart(8);
            const speedup = (origResult.timing.totalMs / optResult.timing.totalMs).toFixed(2) + 'x';
            const matchStr = countMatch ? '  ✔  ' : '  ✗  ';

            console.log(`║ ${ds} │ ${origDpi} │ ${origLookback} │ ${origTotal} │ ${optDpi} │ ${optLookback} │ ${optTotal} │ ${speedup.padStart(7)} │${matchStr}║`);

        } catch (error) {
            const ds = `${size}${range}`.padEnd(8);
            console.log(`║ ${ds} │ Error loading dataset                                                          ║`);
        }
    }

    console.log('╚══════════╧════════╧══════════╧══════════╧════════╧══════════╧══════════╧═════════╧═══════╝\n');

    console.log('Notes:');
    console.log('  - DPI time should be nearly identical (same shader)');
    console.log('  - Lookback time shows the effect of optimizations');
    console.log('  - Speedup > 1.0x means optimized is faster');
    console.log('  - Match column verifies both produce same count\n');
}

/**
 * Run 128M-only comparison test.
 */
export async function runOptimizedDecoupledLookback128MTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════════════════════╗');
    console.log('║    DECOUPLED LOOKBACK: ORIGINAL vs OPTIMIZED (128M ONLY)                  ║');
    console.log('╚════════════════════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 10;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const originalTester = new LookbackKernelTester(
        device, timestampQueryManager, originalLookbackShader, 'Original'
    );
    const optimizedTester = new LookbackKernelTester(
        device, timestampQueryManager, optimizedLookbackShader, 'Optimized'
    );

    console.log(`Running ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations per kernel...\n`);

    const datasets = [
        { size: '128', range: 'e2' },
        { size: '128', range: 'e6' },
    ];

    console.log('╔══════════╤══════════════════════════════╤══════════════════════════════╤═════════╤═══════╗');
    console.log('║          │ Original                     │ Optimized                    │         │       ║');
    console.log('║ Dataset  │ DPI(ms)│Lookbk(ms)│Total(ms) │ DPI(ms)│Lookbk(ms)│Total(ms) │ Speedup │ Match ║');
    console.log('╠══════════╪════════╪══════════╪══════════╪════════╪══════════╪══════════╪═════════╪═══════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const origResult = await originalTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);
            const optResult = await optimizedTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = origResult.totalCount === optResult.totalCount;

            const ds = `${size}${range}`.padEnd(8);
            const origDpi = origResult.timing.dpiMs.toFixed(2).padStart(6);
            const origLookback = origResult.timing.lookbackMs.toFixed(2).padStart(8);
            const origTotal = origResult.timing.totalMs.toFixed(2).padStart(8);
            const optDpi = optResult.timing.dpiMs.toFixed(2).padStart(6);
            const optLookback = optResult.timing.lookbackMs.toFixed(2).padStart(8);
            const optTotal = optResult.timing.totalMs.toFixed(2).padStart(8);
            const speedup = (origResult.timing.totalMs / optResult.timing.totalMs).toFixed(2) + 'x';
            const matchStr = countMatch ? '  ✔  ' : '  ✗  ';

            console.log(`║ ${ds} │ ${origDpi} │ ${origLookback} │ ${origTotal} │ ${optDpi} │ ${optLookback} │ ${optTotal} │ ${speedup.padStart(7)} │${matchStr}║`);

        } catch (error) {
            const ds = `${size}${range}`.padEnd(8);
            console.log(`║ ${ds} │ Error loading dataset                                                          ║`);
        }
    }

    console.log('╚══════════╧════════╧══════════╧══════════╧════════╧══════════╧══════════╧═════════╧═══════╝\n');
}

/**
 * Run 64M e6-only comparison test (isolated cold start).
 */
export async function runOptimizedDecoupledLookback64e6Test(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════════════════════╗');
    console.log('║    DECOUPLED LOOKBACK: ORIGINAL vs OPTIMIZED (64M e6 ONLY - COLD START)   ║');
    console.log('╚════════════════════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 10;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const originalTester = new LookbackKernelTester(
        device, timestampQueryManager, originalLookbackShader, 'Original'
    );
    const optimizedTester = new LookbackKernelTester(
        device, timestampQueryManager, optimizedLookbackShader, 'Optimized'
    );

    console.log(`Running ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations per kernel...\n`);

    const datasets = [
        { size: '64', range: 'e6' },
    ];

    console.log('╔══════════╤══════════════════════════════╤══════════════════════════════╤═════════╤═══════╗');
    console.log('║          │ Original                     │ Optimized                    │         │       ║');
    console.log('║ Dataset  │ DPI(ms)│Lookbk(ms)│Total(ms) │ DPI(ms)│Lookbk(ms)│Total(ms) │ Speedup │ Match ║');
    console.log('╠══════════╪════════╪══════════╪══════════╪════════╪══════════╪══════════╪═════════╪═══════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const origResult = await originalTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);
            const optResult = await optimizedTester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = origResult.totalCount === optResult.totalCount;

            const ds = `${size}${range}`.padEnd(8);
            const origDpi = origResult.timing.dpiMs.toFixed(2).padStart(6);
            const origLookback = origResult.timing.lookbackMs.toFixed(2).padStart(8);
            const origTotal = origResult.timing.totalMs.toFixed(2).padStart(8);
            const optDpi = optResult.timing.dpiMs.toFixed(2).padStart(6);
            const optLookback = optResult.timing.lookbackMs.toFixed(2).padStart(8);
            const optTotal = optResult.timing.totalMs.toFixed(2).padStart(8);
            const speedup = (origResult.timing.totalMs / optResult.timing.totalMs).toFixed(2) + 'x';
            const matchStr = countMatch ? '  ✔  ' : '  ✗  ';

            console.log(`║ ${ds} │ ${origDpi} │ ${origLookback} │ ${origTotal} │ ${optDpi} │ ${optLookback} │ ${optTotal} │ ${speedup.padStart(7)} │${matchStr}║`);

        } catch (error) {
            const ds = `${size}${range}`.padEnd(8);
            console.log(`║ ${ds} │ Error loading dataset                                                          ║`);
        }
    }

    console.log('╚══════════╧════════╧══════════╧══════════╧════════╧══════════╧══════════╧═════════╧═══════╝\n');
}
