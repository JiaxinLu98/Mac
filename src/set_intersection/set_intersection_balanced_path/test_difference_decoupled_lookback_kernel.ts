/**
 * Test file for Difference Decoupled Lookback Kernel
 *
 * Tests the decoupled lookback implementation of set difference.
 * This kernel combines Count + Scan + Write in a single pass using
 * the decoupled lookback technique for streaming prefix sum.
 */

import { GPUSetDifferenceDecoupledLookback } from './set_difference_decoupled_lookback';
import { TestSetDifferencePipeline } from './test_difference_write_kernel';
import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import decoupledLookbackShader from './set_availability_difference_decoupled_lookback.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Test class for the difference decoupled lookback kernel.
 */
export class TestDifferenceDecoupledLookbackKernel {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private difference: GPUSetDifferenceDecoupledLookback;

    // Pipelines for GPU profiling
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.difference = new GPUSetDifferenceDecoupledLookback(device, timestampQueryManager);

        // Create pipelines for GPU profiling
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

        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'Difference Decoupled Lookback bind group layout',
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
            label: 'Difference Decoupled Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: decoupledLookbackShader }),
                entryPoint: 'difference_decoupled_lookback'
            }
        });
    }

    /**
     * Run the decoupled lookback difference test.
     */
    public async testDecoupledLookback(
        setA: Uint32Array,
        setB: Uint32Array
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        numWorkgroups: number;
        gpuTimeMs: number;
    }> {
        const result = await this.difference.computeDifference(setA, setB, 1);
        return {
            result: result.result,
            totalCount: result.totalCount,
            numWorkgroups: result.numWorkgroups,
            gpuTimeMs: result.avgTotalMs
        };
    }

    // ========================================================================
    // CPU Reference Implementation
    // ========================================================================

    /**
     * CPU merge-path set difference.
     * Returns the difference elements (A \ B).
     */
    public cpuSetDifference(a: Uint32Array, b: Uint32Array): Uint32Array {
        const result: number[] = [];
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                result.push(a[ai]);
                ai++;
            } else if (a[ai] > b[bi]) {
                bi++;
            } else {
                // Equal: one B cancels one A, no output
                ai++;
                bi++;
            }
        }
        // Remaining A elements are all in difference
        while (ai < a.length) {
            result.push(a[ai]);
            ai++;
        }
        return new Uint32Array(result);
    }

    /**
     * CPU set difference count.
     */
    public cpuSetDifferenceCount(a: Uint32Array, b: Uint32Array): number {
        return this.cpuSetDifference(a, b).length;
    }

    /**
     * Validate GPU results against CPU reference.
     */
    public validateResults(
        result: { result: Uint32Array; totalCount: number },
        a: Uint32Array,
        b: Uint32Array
    ): { countMatch: boolean; valuesMatch: boolean } {
        const cpuResult = this.cpuSetDifference(a, b);
        const cpuCount = cpuResult.length;

        console.log('\n=== Validation ===');
        console.log(`CPU total count: ${cpuCount}`);
        console.log(`GPU total count: ${result.totalCount}`);

        const countMatch = result.totalCount === cpuCount;
        console.log(`Count match: ${countMatch ? '✔ YES' : '✗ NO'}`);

        let valuesMatch = true;
        if (countMatch && result.totalCount > 0) {
            for (let i = 0; i < result.totalCount; i++) {
                if (result.result[i] !== cpuResult[i]) {
                    valuesMatch = false;
                    console.log(`  Value mismatch at index ${i}: GPU=${result.result[i]}, CPU=${cpuResult[i]}`);
                    // Show context
                    const start = Math.max(0, i - 3);
                    const end = Math.min(cpuResult.length, i + 4);
                    console.log(`  GPU[${start}..${end}]: ${Array.from(result.result.slice(start, end))}`);
                    console.log(`  CPU[${start}..${end}]: ${Array.from(cpuResult.slice(start, end))}`);
                    break;
                }
            }
        }

        console.log(`Values match: ${valuesMatch ? '✔ YES' : '✗ NO'}`);
        console.log('==================\n');

        return { countMatch, valuesMatch };
    }

    /**
     * Print results summary.
     */
    public printResults(
        result: { result: Uint32Array; totalCount: number; numWorkgroups: number; gpuTimeMs: number },
        a: Uint32Array,
        b: Uint32Array
    ): void {
        console.log('\n=== Difference Decoupled Lookback Results ===');
        console.log(`Input sizes: A=${a.length}, B=${b.length}, total=${a.length + b.length}`);
        console.log(`Number of workgroups: ${result.numWorkgroups}`);
        console.log(`GPU total count: ${result.totalCount}`);
        console.log(`GPU time: ${result.gpuTimeMs.toFixed(4)} ms`);

        if (result.totalCount <= 20) {
            console.log(`Result values: [${Array.from(result.result).join(', ')}]`);
        } else {
            console.log(`Result values (first 20): [${Array.from(result.result.slice(0, 20)).join(', ')}...]`);
        }
        console.log('==============================================\n');
    }

    /**
     * Compute difference with GPU timestamp profiling.
     */
    public async computeDifferenceWithGPUProfiling(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 20,
        warmup: number = 10
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
        const maxOutputSize = a_len;  // Difference max output = all of A

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

        // Wait for uploads
        await device.queue.onSubmittedWorkDone();

        // ============ Run Multiple Iterations with GPU Timestamps ============
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            // Reset state buffer and total count before each iteration
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

            // Resolve timestamps
            this.timestampQueryManager.resolve(encoder);

            // Submit and wait
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Download timestamps (in nanoseconds)
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

        // Calculate average (skip first 'warmup' iterations)
        const avg = (arr: number[]) => {
            if (arr.length <= warmup) return arr[arr.length - 1] || 0;
            const withoutWarmup = arr.slice(warmup);
            return withoutWarmup.reduce((a, b) => a + b, 0) / withoutWarmup.length;
        };

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
 * Run all difference decoupled lookback test cases.
 */
export async function runDifferenceDecoupledLookbackTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    DIFFERENCE DECOUPLED LOOKBACK KERNEL TEST SUITE        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestDifferenceDecoupledLookbackKernel(device, timestampQueryManager);

    let allPassed = true;

    // ========================================================================
    // Test Case 1: Basic multiset difference
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Basic multiset difference');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 2, 3, 3, 3, 4, 5]);
        const B = new Uint32Array([2, 3, 3, 6]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [1, 2, 3, 4, 5]');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 2: No overlap
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: No overlap (all A elements in result)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [1, 3, 5, 7, 9]');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 3: Complete overlap (empty result)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Complete overlap (empty result)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [] (empty)');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 4: A has more copies than B
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: A has more copies than B');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([5, 5, 5, 5, 5]);  // 5 copies
        const B = new Uint32Array([5, 5, 5]);          // 3 copies
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [5, 5] (2 copies remain)');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 5: B has more copies than A
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: B has more copies than A');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([5, 5, 5]);          // 3 copies
        const B = new Uint32Array([5, 5, 5, 5, 5]);  // 5 copies
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [] (all cancelled)');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 6: Multi-workgroup
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Multi-workgroup (even vs multiples of 3)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i * 2;
            B[i] = i * 3;
        }
        console.log(`A: [0, 2, 4, ..., ${A[size-1]}] (${size} elements)`);
        console.log(`B: [0, 3, 6, ..., ${B[size-1]}] (${size} elements)`);

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 7: All same value spanning workgroups
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: All same value spanning workgroups');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        const B = new Uint32Array(2000);
        A.fill(42);
        B.fill(42);
        console.log(`A: [42 x 3000], B: [42 x 2000]`);
        console.log('Expected: [42 x 1000]');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 8: Star bit trigger case
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 8: Star bit trigger case');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        A[0] = 0;
        A.fill(1, 1);

        const B = new Uint32Array(1500);
        B.fill(1);

        const result = await tester.testDecoupledLookback(A, B);
        const cpuResult = tester.cpuSetDifference(A, B);
        console.log(`Result length: GPU=${result.totalCount}, CPU=${cpuResult.length}`);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 9: Large dataset
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 9: Large dataset (10K elements each)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 10000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i;
            B[i] = i * 2;
        }
        console.log(`A: [0, 1, 2, ..., ${size-1}] (${size} elements)`);
        console.log(`B: [0, 2, 4, ..., ${(size-1)*2}] (${size} elements)`);

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 10: Complex mixed duplicates
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 10: Complex mixed duplicates');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // A has: 1(x3), 2(x2), 3(x5), 4(x1), 5(x4)
        // B has: 1(x1), 2(x3), 3(x2), 5(x2)
        // Expected: 1(x2), 3(x3), 4(x1), 5(x2)
        const A = new Uint32Array([1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5]);
        const B = new Uint32Array([1, 2, 2, 2, 3, 3, 5, 5]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: [1, 1, 3, 3, 3, 4, 5, 5]');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Summary
    // ========================================================================
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL DIFFERENCE LOOKBACK TESTS PASSED                   ║');
    } else {
        console.log('║  ✗ SOME DIFFERENCE LOOKBACK TESTS FAILED                  ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Run GPU timestamp profiling benchmark for difference decoupled lookback.
 */
export async function runDifferenceDecoupledLookbackBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    DIFFERENCE DECOUPLED LOOKBACK GPU PROFILING            ║');
    console.log('║    (Pure GPU time via timestamp queries)                  ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestDifferenceDecoupledLookbackKernel(device, timestampQueryManager);

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

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

    console.log('╔══════════╤══════════════╤════════════════╤════════╤══════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Difference     │ DPI(ms)│Lookbk(ms)│Total(ms)│ Valid  ║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪══════════╪════════╪════════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeDifferenceWithGPUProfiling(
                A, B, NUM_ITERATIONS, NUM_WARMUP
            );

            // Validate against CPU
            let validStr = '  -   ';
            if (A.length <= 32_000_000) {
                const cpuCount = tester.cpuSetDifferenceCount(A, B);
                const countMatch = totalCount === cpuCount;

                if (countMatch) {
                    validStr = '  ✔   ';
                } else {
                    validStr = '  ✗   ';
                    console.log(`  [${size}${range}] Count mismatch: GPU=${totalCount}, CPU=${cpuCount}`);
                }
            }

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const diffSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const lookback = timing.lookbackMs.toFixed(2).padStart(8);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${diffSize} │ ${dpi} │ ${lookback} │ ${total} │${validStr}║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧══════════╧════════╧════════╝\n');

    console.log('Analysis hints (Difference Decoupled Lookback - 2 phases):');
    console.log('  - DPI: Diagonal partition index computation');
    console.log('  - Lookback: Combined Count + Scan + Write (decoupled lookback)');
    console.log('  - Total = DPI + Lookback');
    console.log('  - Compare with 4-phase (DPI+Count+Scan+Write) version\n');
}

/**
 * Back-to-back comparison: 4-Phase vs Decoupled Lookback for set difference.
 */
export async function runDifference4PhaseVsLookbackComparison(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    DIFFERENCE: 4-PHASE vs LOOKBACK COMPARISON             ║');
    console.log('║    (Same run, same GPU state, pure GPU timestamps)        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhase = new TestSetDifferencePipeline(device, timestampQueryManager);
    const lookback = new TestDifferenceDecoupledLookbackKernel(device, timestampQueryManager);

    console.log(`Running ${NUM_ITERATIONS} iterations per method (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

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

    // Header
    console.log('╔══════════╤════════════════════════════════════════╤══════════════════════════╤═════════╗');
    console.log('║          │ 4-Phase                                │ Lookback                 │         ║');
    console.log('║ Dataset  │ DPI(ms)│Count  │ Scan  │Write  │Total  │ DPI(ms)│Lookbk │ Total  │ Speedup ║');
    console.log('╠══════════╪════════╪═══════╪═══════╪═══════╪═══════╪════════╪═══════╪════════╪═════════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            // --- Run 4-Phase first ---
            const fp = await fourPhase.computeSetDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            // --- Run Lookback second ---
            const lb = await lookback.computeDifferenceWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const fpDpi   = fp.timing.dpiMs.toFixed(2).padStart(6);
            const fpCount = fp.timing.countMs.toFixed(2).padStart(5);
            const fpScan  = fp.timing.scanMs.toFixed(2).padStart(5);
            const fpWrite = fp.timing.writeMs.toFixed(2).padStart(5);
            const fpTotal = fp.timing.totalMs.toFixed(2).padStart(5);
            const lbDpi   = lb.timing.dpiMs.toFixed(2).padStart(6);
            const lbLook  = lb.timing.lookbackMs.toFixed(2).padStart(5);
            const lbTotal = lb.timing.totalMs.toFixed(2).padStart(6);
            const speedup = (fp.timing.totalMs / lb.timing.totalMs).toFixed(2) + 'x';

            console.log(`║ ${ds} │ ${fpDpi} │ ${fpCount} │ ${fpScan} │ ${fpWrite} │ ${fpTotal} │ ${lbDpi} │ ${lbLook} │ ${lbTotal} │ ${speedup.padStart(7)} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧════════╧═══════╧═══════╧═══════╧═══════╧════════╧═══════╧════════╧═════════╝\n');

    console.log('Key: Speedup > 1.0x means Lookback is faster than 4-Phase.');
    console.log('Compare DPI columns to verify measurement fairness (should be nearly identical).\n');
}
