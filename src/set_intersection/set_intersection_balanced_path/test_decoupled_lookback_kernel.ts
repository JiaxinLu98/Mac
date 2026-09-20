/**
 * Test file for Decoupled Lookback Kernel
 *
 * Tests the decoupled lookback implementation of set intersection.
 * This kernel combines Count + Scan + Write in a single pass using
 * the decoupled lookback technique for streaming prefix sum.
 */

import { GPUSetIntersectionDecoupledLookback } from './set_intersection_decoupled_lookback';
import { TestSetIntersectionPipeline } from './test_write_kernel';
import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import decoupledLookbackShader from './set_availability_intersection_decoupled_lookback.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Test class for the decoupled lookback kernel.
 */
export class TestDecoupledLookbackKernel {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private intersection: GPUSetIntersectionDecoupledLookback;

    // Pipelines for GPU profiling
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.intersection = new GPUSetIntersectionDecoupledLookback(device, timestampQueryManager);

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
            label: 'Decoupled Lookback bind group layout',
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
            label: 'Decoupled Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: decoupledLookbackShader }),
                entryPoint: 'intersection_decoupled_lookback'
            }
        });
    }

    /**
     * Run the decoupled lookback test.
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
        const result = await this.intersection.computeIntersection(setA, setB, 1);
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
     * CPU merge-path set intersection.
     * Returns the intersection elements.
     */
    public cpuSetIntersection(a: Uint32Array, b: Uint32Array): Uint32Array {
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
     * CPU set intersection count.
     */
    public cpuSetIntersectionCount(a: Uint32Array, b: Uint32Array): number {
        return this.cpuSetIntersection(a, b).length;
    }

    /**
     * Validate GPU results against CPU reference.
     */
    public validateResults(
        result: { result: Uint32Array; totalCount: number },
        a: Uint32Array,
        b: Uint32Array
    ): { countMatch: boolean; valuesMatch: boolean } {
        const cpuResult = this.cpuSetIntersection(a, b);
        const cpuCount = cpuResult.length;

        console.log('\n=== Validation ===');
        console.log(`CPU total count: ${cpuCount}`);
        console.log(`GPU total count: ${result.totalCount}`);

        const countMatch = result.totalCount === cpuCount;
        console.log(`Count match: ${countMatch ? '✔ YES' : '✗ NO'}`);

        // Compare values (sort both since GPU output order may differ within workgroups)
        let valuesMatch = true;
        if (countMatch && result.totalCount > 0) {
            const sortedGpu = [...result.result].sort((a, b) => a - b);
            const sortedCpu = [...cpuResult].sort((a, b) => a - b);

            for (let i = 0; i < result.totalCount; i++) {
                if (sortedGpu[i] !== sortedCpu[i]) {
                    valuesMatch = false;
                    console.log(`  Value mismatch at index ${i}: GPU=${sortedGpu[i]}, CPU=${sortedCpu[i]}`);
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
        console.log('\n=== Decoupled Lookback Results ===');
        console.log(`Input sizes: A=${a.length}, B=${b.length}, total=${a.length + b.length}`);
        console.log(`Number of workgroups: ${result.numWorkgroups}`);
        console.log(`GPU total count: ${result.totalCount}`);
        console.log(`GPU time: ${result.gpuTimeMs.toFixed(4)} ms`);

        if (result.totalCount <= 20) {
            console.log(`Result values: [${Array.from(result.result).join(', ')}]`);
        } else {
            console.log(`Result values (first 20): [${Array.from(result.result.slice(0, 20)).join(', ')}...]`);
        }
        console.log('==================================\n');
    }

    /**
     * Compute intersection with GPU timestamp profiling.
     * Runs multiple iterations and returns per-phase timing using GPU timestamps.
     */
    public async computeIntersectionWithGPUProfiling(
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
 * Run all decoupled lookback test cases.
 */
export async function runDecoupledLookbackTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║       DECOUPLED LOOKBACK KERNEL TEST SUITE                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestDecoupledLookbackKernel(device, timestampQueryManager);

    let allPassed = true;

    // ========================================================================
    // Test Case 1: Small arrays (single workgroup)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Small arrays (single workgroup)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 3, 3, 6, 7, 8]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected intersection: [3, 3, 7] -> count = 3');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 2: No intersection
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: No intersection');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected intersection: [] -> count = 0');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 3: Complete intersection (A == B)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Complete intersection (A == B)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([1, 2, 3, 4, 5]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected intersection: [1,2,3,4,5] -> count = 5');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 4: Multi-workgroup
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: Multi-workgroup (even vs multiples of 3)');
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
        console.log('Expected: multiples of 6 within range');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 5: All same value
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: All same value (maximum duplicates)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(2500);
        const B = new Uint32Array(2500);
        A.fill(42);
        B.fill(42);

        console.log(`A: [42, 42, ..., 42] (${A.length} elements)`);
        console.log(`B: [42, 42, ..., 42] (${B.length} elements)`);
        console.log(`Expected count: min(${A.length}, ${B.length}) = ${Math.min(A.length, B.length)}`);

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 6: Large dataset
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Large dataset (10K elements each)');
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
        console.log('Expected: all even numbers in A');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch || !validation.valuesMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 7: Duplicates spanning workgroup boundaries
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: Duplicates spanning workgroup boundaries');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const NV = 256 * 7; // 1792
        const numWg = 4;
        const runLength = NV;

        const aSize = numWg * runLength;
        const bSize = numWg * runLength - 200;

        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        for (let i = 0; i < aSize; i++) {
            A[i] = Math.floor(i / runLength);
        }
        for (let i = 0; i < bSize; i++) {
            B[i] = Math.floor(i / runLength);
        }

        console.log(`A: ${aSize} elements, values 0-${numWg - 1}, each repeats ${runLength} times`);
        console.log(`B: ${bSize} elements, same pattern`);
        console.log('Every WG boundary falls within a duplicate run');

        const result = await tester.testDecoupledLookback(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.countMatch) allPassed = false;
    }

    // ========================================================================
    // Test Case 8: Unbalanced sizes
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 8: Unbalanced sizes (tiny A, huge B)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([100, 200, 300, 400, 500]);
        const B = new Uint32Array(3000);
        for (let i = 0; i < B.length; i++) {
            B[i] = i;
        }
        console.log(`A: [100, 200, 300, 400, 500] (5 elements)`);
        console.log(`B: [0, 1, 2, ..., 2999] (${B.length} elements)`);
        console.log('Expected: [100, 200, 300, 400, 500] -> count = 5');

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
        console.log('║  ✔ ALL TESTS PASSED                                        ║');
    } else {
        console.log('║  ✗ SOME TESTS FAILED                                       ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Quick test for the decoupled lookback kernel.
 */
export async function quickDecoupledLookbackTest(device: GPUDevice): Promise<boolean> {
    const timestampQueryManager = new TimestampQueryManager(device, 8);
    const tester = new TestDecoupledLookbackKernel(device, timestampQueryManager);

    const A = new Uint32Array([1, 3, 3, 5, 7, 9]);
    const B = new Uint32Array([2, 3, 3, 6, 7, 8]);

    console.log('\n=== Quick Decoupled Lookback Test ===');
    console.log('A:', Array.from(A));
    console.log('B:', Array.from(B));

    const result = await tester.testDecoupledLookback(A, B);
    const cpuCount = tester.cpuSetIntersectionCount(A, B);

    console.log(`GPU total count: ${result.totalCount}`);
    console.log(`CPU total count: ${cpuCount}`);
    console.log(`Match: ${result.totalCount === cpuCount ? '✔ YES' : '✗ NO'}`);
    console.log('=====================================\n');

    return result.totalCount === cpuCount;
}

/**
 * Run GPU timestamp profiling benchmark for decoupled lookback kernel.
 * Uses pure GPU timestamps for accurate timing (no CPU overhead).
 */
export async function runDecoupledLookbackGPUProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    DECOUPLED LOOKBACK GPU TIMESTAMP PROFILING              ║');
    console.log('║    (Pure GPU time via timestamp queries)                   ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    // Need 4 queries per call: DPI start/end (0,1), Lookback start/end (2,3)
    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestDecoupledLookbackKernel(device, timestampQueryManager);

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    // Datasets to profile
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

    console.log('╔══════════╤══════════════╤════════════════╤════════╤══════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Intersection   │ DPI(ms)│Lookbk(ms)│Total(ms)│ Valid  ║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪══════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            // Run with GPU timestamp profiling (handles warmup internally)
            const { result, totalCount, timing } = await tester.computeIntersectionWithGPUProfiling(
                A, B, NUM_ITERATIONS, NUM_WARMUP
            );

            // Validate against CPU
            let validStr = '  -   ';
            if (A.length <= 32_000_000) {
                // For small datasets, validate count and values
                const cpuResult = tester.cpuSetIntersection(A, B);
                const cpuCount = cpuResult.length;

                let countMatch = totalCount === cpuCount;
                let valuesMatch = true;

                if (countMatch && totalCount > 0) {
                    // Sort both arrays for comparison (GPU output order may vary within workgroups)
                    const sortedGpu = [...result].sort((a, b) => a - b);
                    const sortedCpu = [...cpuResult].sort((a, b) => a - b);

                    for (let i = 0; i < totalCount; i++) {
                        if (sortedGpu[i] !== sortedCpu[i]) {
                            valuesMatch = false;
                            console.log(`  [${size}${range}] Value mismatch at ${i}: GPU=${sortedGpu[i]}, CPU=${sortedCpu[i]}`);
                            break;
                        }
                    }
                }

                if (countMatch && valuesMatch) {
                    validStr = '  ✔   ';
                } else {
                    validStr = '  ✗   ';
                    if (!countMatch) {
                        console.log(`  [${size}${range}] Count mismatch: GPU=${totalCount}, CPU=${cpuCount}`);
                    }
                }
            }

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const intSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const lookback = timing.lookbackMs.toFixed(2).padStart(8);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${intSize} │ ${dpi} │ ${lookback} │ ${total} │${validStr}║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧══════════╧════════╧════════╝\n');

    // Print analysis hints
    console.log('Analysis hints (Decoupled Lookback - 2 phases):');
    console.log('  - DPI: Diagonal partition index computation');
    console.log('  - Lookback: Combined Count + Scan + Write (decoupled lookback)');
    console.log('  - Total = DPI + Lookback');
    console.log('  - Compare with V1 (4-phase: DPI+Count+Scan+Write) and Atomic versions\n');
}

/**
 * Cold start benchmark for 8M-16M datasets only.
 * Run independently to avoid thermal throttling from prior datasets.
 */
export async function runDecoupledLookback8M16MColdBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    DECOUPLED LOOKBACK COLD START - 8M & 16M               ║');
    console.log('║    (Pure GPU time, independent cold start)                ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 8);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestDecoupledLookbackKernel(device, timestampQueryManager);

    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const datasets = [
        { size: '8', range: 'e2', desc: '8M elements, range 100' },
        { size: '8', range: 'e6', desc: '8M elements, range 1M' },
        { size: '16', range: 'e2', desc: '16M elements, range 100' },
        { size: '16', range: 'e6', desc: '16M elements, range 1M' },
    ];

    console.log('╔══════════╤══════════════╤════════════════╤════════╤══════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Intersection   │ DPI(ms)│Lookbk(ms)│Total(ms)│ Valid  ║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪══════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeIntersectionWithGPUProfiling(
                A, B, NUM_ITERATIONS, NUM_WARMUP
            );

            // Validate against CPU
            let validStr = '  -   ';
            const cpuResult = tester.cpuSetIntersection(A, B);
            const cpuCount = cpuResult.length;

            let countMatch = totalCount === cpuCount;
            let valuesMatch = true;

            if (countMatch && totalCount > 0) {
                const sortedGpu = [...result].sort((a, b) => a - b);
                const sortedCpu = [...cpuResult].sort((a, b) => a - b);

                for (let i = 0; i < totalCount; i++) {
                    if (sortedGpu[i] !== sortedCpu[i]) {
                        valuesMatch = false;
                        console.log(`  [${size}${range}] Value mismatch at ${i}: GPU=${sortedGpu[i]}, CPU=${sortedCpu[i]}`);
                        break;
                    }
                }
            }

            if (countMatch && valuesMatch) {
                validStr = '  ✔   ';
            } else {
                validStr = '  ✗   ';
                if (!countMatch) {
                    console.log(`  [${size}${range}] Count mismatch: GPU=${totalCount}, CPU=${cpuCount}`);
                }
            }

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const intSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const lookback = timing.lookbackMs.toFixed(2).padStart(8);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${intSize} │ ${dpi} │ ${lookback} │ ${total} │${validStr}║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧══════════╧════════╧════════╝\n');
}

/**
 * Back-to-back comparison: 4-Phase vs Decoupled Lookback on the same datasets
 * in a single run, ensuring identical GPU thermal/clock state for fair DPI comparison.
 */
export async function run4PhaseVsLookbackComparison(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    4-PHASE vs LOOKBACK BACK-TO-BACK COMPARISON            ║');
    console.log('║    (Same run, same GPU state, pure GPU timestamps)        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    // 4-Phase needs 8 queries (DPI:0,1 Count:2,3 Scan:4,5 Write:6,7)
    // Lookback needs 4 queries (DPI:0,1 Lookback:2,3)
    // Use 16 to cover both
    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhase = new TestSetIntersectionPipeline(device, timestampQueryManager);
    const lookback = new TestDecoupledLookbackKernel(device, timestampQueryManager);

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
            const fp = await fourPhase.computeSetIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            // --- Run Lookback second ---
            const lb = await lookback.computeIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

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

/**
 * Back-to-back comparison for 128M datasets only (128e2 + 128e6).
 */
export async function run4PhaseVsLookback128Comparison(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║    4-PHASE vs LOOKBACK COMPARISON - 128M ONLY             ║');
    console.log('║    (Same run, same GPU state, pure GPU timestamps)        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhase = new TestSetIntersectionPipeline(device, timestampQueryManager);
    const lookbackTester = new TestDecoupledLookbackKernel(device, timestampQueryManager);

    console.log(`Running ${NUM_ITERATIONS} iterations per method (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    const datasets = [
        { size: '128', range: 'e2' },
        { size: '128', range: 'e6' },
    ];

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

            const fp = await fourPhase.computeSetIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);
            const lb = await lookbackTester.computeIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

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
}
