/**
 * VT Parameterization Benchmark for Sentinel Decoupled Lookback Shader
 *
 * Tests VT = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 on the sentinel decoupled lookback kernel.
 * Each VT variant uses a separate WGSL shader with unrolled loops matching VT.
 * The DPI shader is dynamically patched via string replacement to match each VT.
 *
 * Outputs:
 * - Correctness tests (synthetic data, all VT variants vs CPU reference)
 * - Performance benchmarks (binary datasets, timing comparison table)
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';

// DPI shader (will be patched per-VT at runtime)
import computeDiagonalsShaderBase from './balanced_path_biased.wgsl';

// Sentinel lookback shaders for each VT
import sentinelShaderVT6 from './set_availability_intersection_decoupled_lookback_sentinel_vt6.wgsl';
import sentinelShaderVT7 from './set_availability_intersection_decoupled_lookback_sentinel_vt7.wgsl';
import sentinelShaderVT8 from './set_availability_intersection_decoupled_lookback_sentinel_vt8.wgsl';
import sentinelShaderVT9 from './set_availability_intersection_decoupled_lookback_sentinel_vt9.wgsl';
import sentinelShaderVT10 from './set_availability_intersection_decoupled_lookback_sentinel_vt10.wgsl';
import sentinelShaderVT11 from './set_availability_intersection_decoupled_lookback_sentinel_vt11.wgsl';
import sentinelShaderVT12 from './set_availability_intersection_decoupled_lookback_sentinel_vt12.wgsl';
import sentinelShaderVT13 from './set_availability_intersection_decoupled_lookback_sentinel_vt13.wgsl';
import sentinelShaderVT14 from './set_availability_intersection_decoupled_lookback_sentinel_vt14.wgsl';
import sentinelShaderVT15 from './set_availability_intersection_decoupled_lookback_sentinel_vt15.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;

interface VTConfig {
    vt: number;
    shader: string;
}

const VT_CONFIGS: VTConfig[] = [
    { vt: 6, shader: sentinelShaderVT6 },
    { vt: 7, shader: sentinelShaderVT7 },
    { vt: 8, shader: sentinelShaderVT8 },
    { vt: 9, shader: sentinelShaderVT9 },
    { vt: 10, shader: sentinelShaderVT10 },
    { vt: 11, shader: sentinelShaderVT11 },
    { vt: 12, shader: sentinelShaderVT12 },
    { vt: 13, shader: sentinelShaderVT13 },
    { vt: 14, shader: sentinelShaderVT14 },
    { vt: 15, shader: sentinelShaderVT15 },
];

/**
 * Patch the DPI shader to use a different VT value.
 * Only modifies the in-memory string — the original file stays untouched.
 */
function createDPIShaderForVT(baseShader: string, vt: number): string {
    return baseShader.replace('const VT: u32 = 7u;', `const VT: u32 = ${vt}u;`);
}

/**
 * Tester class for a single VT variant.
 */
class VTLookbackTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private label: string;
    private vt: number;
    private nv: number;

    // Pipelines
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(
        device: GPUDevice,
        timestampQueryManager: TimestampQueryManager,
        lookbackShader: string,
        vt: number
    ) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.vt = vt;
        this.nv = NT * vt;
        this.label = `Sentinel-VT${vt}`;

        // Create DPI shader with matching VT
        const dpiShader = createDPIShaderForVT(computeDiagonalsShaderBase, vt);

        // DPI pipeline
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `DPI bind group layout (${this.label})`,
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
            label: `DPI pipeline (${this.label})`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: dpiShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback pipeline
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: `Decoupled Lookback bind group layout (${this.label})`,
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
            label: `Decoupled Lookback pipeline (${this.label})`,
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

        const numWg = Math.ceil(total / this.nv);
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
        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

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
 * Approximate workgroup memory usage for a given VT.
 */
function estimateWorkgroupMemoryBytes(vt: number): number {
    const nv = NT * vt;
    const keysSharedSize = nv + vt + 6;  // keys_shared array
    const scanSize = NT;                  // shared_scan array
    const miscSize = 8;                   // wg_* variables (approx)
    return (keysSharedSize + scanSize + miscSize) * 4;
}

/**
 * Run VT parameterization benchmark for sentinel decoupled lookback.
 */
export async function runDecoupledLookbackVTBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(90));
    console.log('  SENTINEL DECOUPLED LOOKBACK: VT PARAMETERIZATION BENCHMARK');
    console.log('  Testing VT = 6, 7, 8, 9, 10, 11, 12, 13, 14, 15');
    console.log('  GPU Timestamp Query | 10 warmup + 10 iterations');
    console.log('='.repeat(90) + '\n');

    const NUM_ITERATIONS = 10;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    // ========================================================================
    // Create testers for each VT variant
    // ========================================================================
    console.log('Creating pipelines for each VT variant...\n');

    interface VTTesterEntry {
        vt: number;
        tester: VTLookbackTester;
        memBytes: number;
    }

    const testers: VTTesterEntry[] = [];

    for (const config of VT_CONFIGS) {
        const memBytes = estimateWorkgroupMemoryBytes(config.vt);
        console.log(`  VT=${config.vt.toString().padStart(2)}: NV=${(NT * config.vt).toString().padStart(4)}, ~${memBytes} bytes workgroup memory`);

        try {
            const tester = new VTLookbackTester(
                device,
                timestampQueryManager,
                config.shader,
                config.vt
            );
            testers.push({ vt: config.vt, tester, memBytes });
            console.log(`         Pipeline created successfully`);
        } catch (error) {
            console.log(`         SKIPPED: Pipeline creation failed (likely exceeds workgroup memory limit)`);
            console.log(`         Error: ${error}`);
        }
    }

    console.log(`\n  ${testers.length}/${VT_CONFIGS.length} VT variants available\n`);

    if (testers.length === 0) {
        console.log('ERROR: No VT variants could be created.\n');
        return;
    }

    // ========================================================================
    // Correctness Tests (synthetic data)
    // ========================================================================
    console.log('-'.repeat(90));
    console.log('CORRECTNESS TESTS');
    console.log('-'.repeat(90) + '\n');

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

        const vtResults: string[] = [];
        let testPassed = true;

        for (const { vt, tester } of testers) {
            try {
                const gpuResult = await tester.computeIntersectionWithProfiling(test.A, test.B, 1, 0);
                const countMatch = gpuResult.totalCount === cpuCount;

                let valuesMatch = true;
                if (countMatch && gpuResult.totalCount > 0) {
                    for (let i = 0; i < gpuResult.totalCount; i++) {
                        if (gpuResult.result[i] !== cpuResult[i]) {
                            valuesMatch = false;
                            break;
                        }
                    }
                }

                const ok = countMatch && valuesMatch;
                vtResults.push(`VT${vt}=${ok ? 'OK' : 'FAIL'}(${gpuResult.totalCount})`);
                if (!ok) testPassed = false;
            } catch (error) {
                vtResults.push(`VT${vt}=ERR`);
                testPassed = false;
            }
        }

        const status = testPassed ? 'OK' : 'FAIL';
        console.log(`  ${status.padEnd(5)} ${test.name} (cpu=${cpuCount}): ${vtResults.join(', ')}`);
        if (!testPassed) allCorrect = false;
    }

    console.log(allCorrect ? '\n  All correctness tests PASSED\n' : '\n  Some correctness tests FAILED\n');

    // ========================================================================
    // Performance Benchmarks (binary datasets)
    // ========================================================================
    console.log('-'.repeat(90));
    console.log('PERFORMANCE BENCHMARKS');
    console.log('-'.repeat(90) + '\n');

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

    // Build header with VT columns
    const vtLabels = testers.map(t => `VT${t.vt}`);
    const colWidth = 12;

    // ---- Total Time Table ----
    console.log('  TOTAL TIME (DPI + Lookback) in ms');
    console.log('');
    let header = 'Dataset'.padEnd(10);
    for (const label of vtLabels) {
        header += label.padStart(colWidth);
    }
    header += '  Count'.padStart(10);
    console.log(header);
    console.log('-'.repeat(10 + colWidth * vtLabels.length + 10));

    interface DatasetResult {
        dataset: string;
        vtTimings: Map<number, { dpiMs: number; lookbackMs: number; totalMs: number }>;
        count: number;
        allMatch: boolean;
    }

    const allResults: DatasetResult[] = [];

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;
        const ds = `${size}${range}`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const vtTimings = new Map<number, { dpiMs: number; lookbackMs: number; totalMs: number }>();
            let referenceCount = -1;
            let allMatch = true;

            let row = ds.padEnd(10);

            for (const { vt, tester } of testers) {
                try {
                    const result = await tester.computeIntersectionWithProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);
                    vtTimings.set(vt, result.timing);

                    if (referenceCount === -1) {
                        referenceCount = result.totalCount;
                    } else if (result.totalCount !== referenceCount) {
                        allMatch = false;
                    }

                    row += (result.timing.totalMs.toFixed(3)).padStart(colWidth);
                } catch (error) {
                    row += 'ERR'.padStart(colWidth);
                }
            }

            row += referenceCount.toString().padStart(10);
            if (!allMatch) row += '  MISMATCH';
            console.log(row);

            allResults.push({ dataset: ds, vtTimings, count: referenceCount, allMatch });

        } catch (error) {
            console.log(`${ds.padEnd(10)} Error loading dataset`);
        }
    }

    // ---- Lookback Kernel Time Only Table ----
    console.log('\n');
    console.log('  LOOKBACK KERNEL TIME ONLY (ms)');
    console.log('');
    header = 'Dataset'.padEnd(10);
    for (const label of vtLabels) {
        header += label.padStart(colWidth);
    }
    console.log(header);
    console.log('-'.repeat(10 + colWidth * vtLabels.length));

    for (const dr of allResults) {
        let row = dr.dataset.padEnd(10);
        for (const { vt } of testers) {
            const timing = dr.vtTimings.get(vt);
            if (timing) {
                row += timing.lookbackMs.toFixed(3).padStart(colWidth);
            } else {
                row += 'N/A'.padStart(colWidth);
            }
        }
        console.log(row);
    }

    // ---- DPI Kernel Time Only Table ----
    console.log('\n');
    console.log('  DPI KERNEL TIME ONLY (ms)');
    console.log('');
    header = 'Dataset'.padEnd(10);
    for (const label of vtLabels) {
        header += label.padStart(colWidth);
    }
    console.log(header);
    console.log('-'.repeat(10 + colWidth * vtLabels.length));

    for (const dr of allResults) {
        let row = dr.dataset.padEnd(10);
        for (const { vt } of testers) {
            const timing = dr.vtTimings.get(vt);
            if (timing) {
                row += timing.dpiMs.toFixed(3).padStart(colWidth);
            } else {
                row += 'N/A'.padStart(colWidth);
            }
        }
        console.log(row);
    }

    // ---- Speedup relative to VT=7 ----
    console.log('\n');
    console.log('  SPEEDUP vs VT=7 (total time, >1.0 = faster than VT=7)');
    console.log('');
    header = 'Dataset'.padEnd(10);
    for (const label of vtLabels) {
        header += label.padStart(colWidth);
    }
    console.log(header);
    console.log('-'.repeat(10 + colWidth * vtLabels.length));

    for (const dr of allResults) {
        let row = dr.dataset.padEnd(10);
        const vt7Timing = dr.vtTimings.get(7);

        for (const { vt } of testers) {
            const timing = dr.vtTimings.get(vt);
            if (timing && vt7Timing && vt7Timing.totalMs > 0) {
                const speedup = vt7Timing.totalMs / timing.totalMs;
                row += (speedup.toFixed(2) + 'x').padStart(colWidth);
            } else {
                row += 'N/A'.padStart(colWidth);
            }
        }
        console.log(row);
    }

    // ---- Summary ----
    console.log('\n' + '='.repeat(90));
    console.log('  SUMMARY');
    console.log('='.repeat(90));

    const allMatch = allResults.every(r => r.allMatch);
    console.log(`  Correctness: ${allMatch ? 'All VT variants produce matching counts' : 'MISMATCH detected!'}`);
    console.log(`  VT variants tested: ${testers.map(t => t.vt).join(', ')}`);
    console.log(`  Workgroup memory estimates:`);
    for (const t of testers) {
        const warn = t.memBytes > 16384 ? ' (WARNING: may exceed 16KB limit!)' : '';
        console.log(`    VT=${t.vt.toString().padStart(2)}: ~${t.memBytes} bytes${warn}`);
    }

    console.log('\n  Notes:');
    console.log('  - DPI time varies with VT because partition size (NV) changes');
    console.log('  - Lookback kernel time is the primary metric to compare');
    console.log('  - Speedup > 1.0x means that VT is faster than VT=7');
    console.log('  - VT=15 may fail on GPUs with 16KB workgroup storage limit');
    console.log('');
}
