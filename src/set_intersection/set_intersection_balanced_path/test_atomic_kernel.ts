/**
 * Test file for Atomic Set Intersection Pipeline
 *
 * Tests the two-phase atomic approach:
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries
 * 2. Atomic Phase - Single-pass intersection with atomic offset allocation
 *
 * Advantages over 4-phase approach:
 * - Only 2 kernel dispatches instead of 4
 * - No separate prefix sum phase needed
 * - Potentially lower latency for smaller datasets
 *
 * Disadvantages:
 * - Output order not deterministic (workgroups write in arbitrary order)
 * - Atomic contention may hurt performance at scale
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import atomicShader from './set_availability_intersection_atomic.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';
import { TestSetIntersectionPipeline } from './test_write_kernel';
import { GPUSorter, SortBuffers } from './radix_sort/sort';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Atomic Set Intersection Pipeline Test Class
 */
export class TestAtomicIntersectionPipeline {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI pipeline
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;

    // Atomic pipeline
    private atomicPipeline: GPUComputePipeline;
    private atomicBindGroupLayout: GPUBindGroupLayout;

    // Radix sort for output
    private sorter: GPUSorter;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager, subgroupSize: number = 32) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.sorter = new GPUSorter(device, subgroupSize, timestampQueryManager);

        // DPI bind group layout
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

        // Atomic bind group layout
        this.atomicBindGroupLayout = device.createBindGroupLayout({
            label: 'Atomic intersection bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // global_counter
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        this.atomicPipeline = device.createComputePipeline({
            label: 'Atomic intersection pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.atomicBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: atomicShader }),
                entryPoint: 'compute_availability_atomic'
            }
        });
    }

    /**
     * Run the atomic set intersection pipeline.
     * Both phases (DPI, Atomic) are merged into a single command encoder.
     */
    public async computeSetIntersection(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 1,
        warmup: number = 1
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        gpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return { result: new Uint32Array(0), totalCount: 0, gpuTimeMs: 0 };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============

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

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        // DPI buffer
        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Output buffer (pre-allocated with max possible size)
        const maxOutputSize = Math.min(a_len, b_len);
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Global counter for atomic offset allocation
        const bufferCounter = device.createBuffer({
            label: 'Global Counter',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

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

        const atomicBindGroup = device.createBindGroup({
            layout: this.atomicBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOutput } },
                { binding: 4, resource: { buffer: bufferCounter } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        // Wait for data uploads
        await device.queue.onSubmittedWorkDone();

        // ============ GPU Execution ============
        const times: number[] = [];
        let lastTotalCount = 0;

        for (let iter = 0; iter < iterations; iter++) {
            // Reset global counter to 0
            device.queue.writeBuffer(bufferCounter, 0, new Uint32Array([0]));
            await device.queue.onSubmittedWorkDone();

            const t0 = performance.now();

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI
            let pass = encoder.beginComputePass({ label: 'DPI' });
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Atomic
            pass = encoder.beginComputePass({ label: 'Atomic' });
            pass.setPipeline(this.atomicPipeline);
            pass.setBindGroup(0, atomicBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            times.push(performance.now() - t0);

            // Read counter on last iteration
            if (iter === iterations - 1) {
                const counterReadback = device.createBuffer({
                    size: 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const copyEncoder = device.createCommandEncoder();
                copyEncoder.copyBufferToBuffer(bufferCounter, 0, counterReadback, 0, 4);
                device.queue.submit([copyEncoder.finish()]);
                await device.queue.onSubmittedWorkDone();
                await counterReadback.mapAsync(GPUMapMode.READ);
                lastTotalCount = new Uint32Array(counterReadback.getMappedRange())[0];
                counterReadback.unmap();
                counterReadback.destroy();
            }
        }

        // Calculate average time
        let gpuTimeMs: number;
        if (iterations > warmup) {
            const timesWithoutWarmup = times.slice(warmup);
            gpuTimeMs = timesWithoutWarmup.reduce((a, b) => a + b, 0) / timesWithoutWarmup.length;
        } else {
            gpuTimeMs = times[times.length - 1] || 0;
        }

        // ============ Readback ============
        let result = new Uint32Array(0);
        if (lastTotalCount > 0) {
            const outputReadback = device.createBuffer({
                size: lastTotalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutput, 0, outputReadback, 0, lastTotalCount * 4);
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
        bufferOutput.destroy();
        bufferCounter.destroy();

        return { result, totalCount: lastTotalCount, gpuTimeMs };
    }

    /**
     * Run with GPU timestamp profiling for per-phase timing.
     * Includes sorting phase to produce sorted output.
     */
    public async computeSetIntersectionWithGPUProfiling(
        setA: Uint32Array,
        setB: Uint32Array,
        iterations: number = 10,
        warmup: number = 1
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            atomicMs: number;
            sortMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                result: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, atomicMs: 0, sortMs: 0, totalMs: 0 }
            };
        }

        // Fallback if timestamps not supported
        if (!this.timestampQueryManager.timestampSupported) {
            console.warn('Timestamp queries not supported, using performance.now()');
            const { result, totalCount, gpuTimeMs } = await this.computeSetIntersection(setA, setB, iterations, warmup);
            return {
                result,
                totalCount,
                timing: { dpiMs: gpuTimeMs / 3, atomicMs: gpuTimeMs / 3, sortMs: gpuTimeMs / 3, totalMs: gpuTimeMs }
            };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============
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

        const maxOutputSize = Math.min(a_len, b_len);
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferCounter = device.createBuffer({
            label: 'Global Counter',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Pre-allocate sort buffers for max possible output size
        const sortBuffers = this.sorter.createSortBuffers(maxOutputSize);

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

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

        const atomicBindGroup = device.createBindGroup({
            layout: this.atomicBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOutput } },
                { binding: 4, resource: { buffer: bufferCounter } },
                { binding: 5, resource: { buffer: bufferALen } },
                { binding: 6, resource: { buffer: bufferBLen } },
                { binding: 7, resource: { buffer: bufferNumWg } },
            ]
        });

        await device.queue.onSubmittedWorkDone();

        // ============ Run with GPU Timestamps ============
        // Timestamp indices: DPI (0,1), Atomic (2,3)
        const dpiTimes: number[] = [];
        const atomicTimes: number[] = [];
        const sortTimes: number[] = [];
        const totalTimes: number[] = [];
        let lastTotalCount = 0;

        for (let iter = 0; iter < iterations; iter++) {
            // Reset counter
            device.queue.writeBuffer(bufferCounter, 0, new Uint32Array([0]));
            await device.queue.onSubmittedWorkDone();

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Atomic (timestamps 2, 3)
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.atomicPipeline);
            pass.setBindGroup(0, atomicBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Resolve timestamps
            this.timestampQueryManager.resolve(encoder);

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Download timestamps for DPI + Atomic
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            let dpiNs = 0, atomicNs = 0;
            if (timestamps.length >= 4) {
                dpiNs = timestamps[1] - timestamps[0];
                atomicNs = timestamps[3] - timestamps[2];
            }

            // Read counter to get actual output count
            const counterReadback = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const copyEncoder = device.createCommandEncoder();
            copyEncoder.copyBufferToBuffer(bufferCounter, 0, counterReadback, 0, 4);
            device.queue.submit([copyEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            await counterReadback.mapAsync(GPUMapMode.READ);
            const actualCount = new Uint32Array(counterReadback.getMappedRange())[0];
            counterReadback.unmap();
            counterReadback.destroy();

            // Phase 3: Sort (using GPU timestamps)
            // Sort uses timestamp indices 0 and 1 (begin and end)
            let sortNs = 0;
            if (actualCount > 0) {
                // Copy atomic output to sort buffer
                const copySortEncoder = device.createCommandEncoder();
                copySortEncoder.copyBufferToBuffer(bufferOutput, 0, sortBuffers.keysA, 0, actualCount * 4);
                device.queue.submit([copySortEncoder.finish()]);
                await device.queue.onSubmittedWorkDone();

                // Run sort with GPU timestamps enabled
                await this.sorter.sort(device.queue, sortBuffers, actualCount, true);

                // Download sort timestamps
                const sortTimestamps = await this.timestampQueryManager.downloadTimestampResult();
                if (sortTimestamps.length >= 2) {
                    // Total sort time = end (index 1) - start (index 0)
                    sortNs = sortTimestamps[1] - sortTimestamps[0];
                }
            }

            dpiTimes.push(dpiNs / 1_000_000);
            atomicTimes.push(atomicNs / 1_000_000);
            sortTimes.push(sortNs / 1_000_000);
            totalTimes.push((dpiNs + atomicNs + sortNs) / 1_000_000);

            lastTotalCount = actualCount;
        }

        // Calculate averages
        const avg = (arr: number[]) => {
            if (arr.length <= warmup) return arr[arr.length - 1] || 0;
            const withoutWarmup = arr.slice(warmup);
            return withoutWarmup.reduce((a, b) => a + b, 0) / withoutWarmup.length;
        };

        const dpiMs = avg(dpiTimes);
        const atomicMs = avg(atomicTimes);
        const sortMs = avg(sortTimes);
        const totalMs = avg(totalTimes);

        // ============ Readback sorted result ============
        let result = new Uint32Array(0);
        if (lastTotalCount > 0) {
            const outputReadback = device.createBuffer({
                size: lastTotalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(sortBuffers.keysA, 0, outputReadback, 0, lastTotalCount * 4);
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
        bufferOutput.destroy();
        bufferCounter.destroy();
        sortBuffers.destroy();

        return {
            result,
            totalCount: lastTotalCount,
            timing: { dpiMs, atomicMs, sortMs, totalMs }
        };
    }

    /**
     * CPU reference implementation (multiset intersection)
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
     * Validate GPU results against CPU.
     * Note: Atomic output may be in different order, so we sort both before comparing.
     */
    public validateResults(gpuResult: Uint32Array, a: Uint32Array, b: Uint32Array): boolean {
        const cpuResult = this.cpuSetIntersection(a, b);

        if (gpuResult.length !== cpuResult.length) {
            console.log(`Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
            return false;
        }

        // Sort both results for comparison (atomic output order is non-deterministic)
        const gpuSorted = new Uint32Array(gpuResult).sort();
        const cpuSorted = new Uint32Array(cpuResult).sort();

        for (let i = 0; i < cpuSorted.length; i++) {
            if (gpuSorted[i] !== cpuSorted[i]) {
                console.log(`Value mismatch at sorted index ${i}: GPU=${gpuSorted[i]}, CPU=${cpuSorted[i]}`);
                return false;
            }
        }

        return true;
    }

    /**
     * Validate count only (for large datasets where full validation is too slow)
     */
    public validateCount(gpuCount: number, a: Uint32Array, b: Uint32Array): boolean {
        const cpuResult = this.cpuSetIntersection(a, b);
        return gpuCount === cpuResult.length;
    }
}

/**
 * Run atomic pipeline tests
 */
export async function runAtomicKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      ATOMIC SET INTERSECTION PIPELINE TEST                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 12);
    const tester = new TestAtomicIntersectionPipeline(device, timestampQueryManager);

    let allPassed = true;

    // Test Case 1: Small arrays
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Small arrays');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 3, 3, 6, 7, 8]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        console.log('GPU result:', Array.from(result));
        console.log('GPU result (sorted):', Array.from(new Uint32Array(result).sort()));
        console.log('CPU result:', Array.from(tester.cpuSetIntersection(A, B)));

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 2: All same values
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: All same values (100 elements each)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(100);
        const B = new Uint32Array(100);
        A.fill(5);
        B.fill(5);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        console.log(`Result length: ${result.length}, expected: 100`);
        console.log(`First 10 values: ${Array.from(result.slice(0, 10))}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 3: No intersection
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: No intersection');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        console.log(`Result length: ${result.length}, expected: 0`);

        const valid = result.length === 0;
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 4: Multi-workgroup
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

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        const cpuResult = tester.cpuSetIntersection(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 5: Star bit trigger
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Star bit trigger case');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        A[0] = 0;
        A.fill(1, 1);

        const B = new Uint32Array(3000);
        B.fill(1);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        const cpuResult = tester.cpuSetIntersection(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 6: Large dataset
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

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        const cpuResult = tester.cpuSetIntersection(A, B);
        console.log(`Result length: GPU=${result.length}, CPU=${cpuResult.length}`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 7: Maximum duplicates (multi-workgroup)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: Maximum duplicates (all same value, multi-WG)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(2500);
        const B = new Uint32Array(2500);
        A.fill(42);
        B.fill(42);

        const { result, totalCount, gpuTimeMs } = await tester.computeSetIntersection(A, B);
        console.log(`Result length: GPU=${result.length}, expected: 2500`);

        const valid = tester.validateResults(result, A, B);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL ATOMIC PIPELINE TESTS PASSED                        ║');
    } else {
        console.log('║  ✗ SOME ATOMIC PIPELINE TESTS FAILED                       ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Run GPU timestamp profiling benchmark for atomic pipeline
 */
export async function runAtomicGPUProfilingBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      ATOMIC INTERSECTION GPU TIMESTAMP PROFILING           ║');
    console.log('║      (3-phase: DPI + Atomic + Sort)                        ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 12);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const tester = new TestAtomicIntersectionPipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

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

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Intersection   │ DPI(ms)│Atomic(ms)│Sort(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            const { result, totalCount, timing } = await tester.computeSetIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const intSize = totalCount.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const atomic = timing.atomicMs.toFixed(2).padStart(6);
            const sort = timing.sortMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${intSize} │ ${dpi} │ ${atomic} │ ${sort} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╝\n');

    console.log('Analysis notes:');
    console.log('  - Atomic approach: 3 phases (DPI + Atomic + Sort)');
    console.log('  - Atomic phase includes: count, workgroup scan, global atomic alloc, write');
    console.log('  - Sort phase: GPU radix sort to produce sorted output');
    console.log('  - Compare Total with 4-phase (DPI + Count + Scan + Write) which produces sorted output directly\n');
}

/**
 * Compare atomic vs 4-phase pipeline performance
 */
export async function runAtomicVs4PhaseBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      ATOMIC vs 4-PHASE PIPELINE COMPARISON                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const atomicTester = new TestAtomicIntersectionPipeline(device, timestampQueryManager);
    const fourPhaseTester = new TestSetIntersectionPipeline(device, timestampQueryManager);

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup)...\n`);

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

    console.log('╔══════════╤══════════════╤════════════════╤════════════╤════════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Intersection   │Atomic+Sort │  4-Phase   │ Speedup║');
    console.log('╠══════════╪══════════════╪════════════════╪════════════╪════════════╪════════╣');

    for (const { size, range } of datasets) {
        const aPath = `./data/A_${size}${range}.bin`;
        const bPath = `./data/B_${size}${range}.bin`;

        try {
            const A = await utils.loadUint32ArrayFromBin(aPath);
            const B = await utils.loadUint32ArrayFromBin(bPath);

            // Run atomic
            const atomicResult = await atomicTester.computeSetIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            // Run 4-phase
            const fourPhaseResult = await fourPhaseTester.computeSetIntersectionWithGPUProfiling(A, B, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}${range}`.padEnd(8);
            const inputSize = `${(A.length / 1_000_000).toFixed(0)}M+${(B.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const intSize = atomicResult.totalCount.toLocaleString().padStart(14);
            const atomicTime = atomicResult.timing.totalMs.toFixed(2).padStart(8);
            const fourPhaseTime = fourPhaseResult.timing.totalMs.toFixed(2).padStart(8);
            const speedup = (fourPhaseResult.timing.totalMs / atomicResult.timing.totalMs).toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${intSize} │ ${atomicTime}   │ ${fourPhaseTime}   │ ${speedup}x ║`);

        } catch (error) {
            console.log(`║ ${size}${range} │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════════╧════════════╧════════╝\n');

    console.log('Notes:');
    console.log('  - Atomic+Sort = DPI + Atomic kernel + Radix Sort (produces sorted output)');
    console.log('  - 4-Phase = DPI + Count + Scan + Write (produces sorted output directly)');
    console.log('  - Speedup > 1.0 means Atomic+Sort is faster than 4-Phase\n');
}

/**
 * Test the keys-only radix sort implementation
 */
export async function runRadixSortTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      RADIX SORT (KEYS-ONLY) TEST                           ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const sorter = new GPUSorter(device, 32, timestampQueryManager);

    let allPassed = true;

    // Test Case 1: Small array
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Small array (10 elements)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const input = new Uint32Array([9, 3, 7, 1, 5, 8, 2, 6, 4, 0]);
        const expected = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        console.log('Input:', Array.from(input));

        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        // Read back result
        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        console.log('Output:', Array.from(result));
        console.log('Expected:', Array.from(expected));

        const valid = result.every((v, i) => v === expected[i]);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 2: Already sorted
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: Already sorted array');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const input = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const expected = new Uint32Array(input);
        console.log('Input:', Array.from(input));

        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        console.log('Output:', Array.from(result));

        const valid = result.every((v, i) => v === expected[i]);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 3: Reverse sorted
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Reverse sorted array');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const input = new Uint32Array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
        const expected = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        console.log('Input:', Array.from(input));

        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        console.log('Output:', Array.from(result));

        const valid = result.every((v, i) => v === expected[i]);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 4: All same values
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: All same values');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const input = new Uint32Array(100);
        input.fill(42);
        const expected = new Uint32Array(input);
        console.log(`Input: 100 elements, all value 42`);

        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        const valid = result.every((v, i) => v === expected[i]);
        console.log(`Output: all values = ${result[0]}`);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 5: Random large array
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Random array (10K elements)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 10000;
        const input = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            input[i] = Math.floor(Math.random() * 1000000);
        }
        const expected = new Uint32Array(input).sort((a, b) => a - b);
        console.log(`Input: ${size} random elements`);

        const t0 = performance.now();
        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const t1 = performance.now();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        const valid = result.every((v, i) => v === expected[i]);
        console.log(`GPU Sort Time: ${(t1 - t0).toFixed(2)} ms`);
        console.log(`First 10 sorted: ${Array.from(result.slice(0, 10))}`);
        console.log(`Last 10 sorted: ${Array.from(result.slice(-10))}`);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 6: Large array (1M elements)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Large array (1M elements)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 1000000;
        const input = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            input[i] = Math.floor(Math.random() * 100000000);
        }
        console.log(`Input: ${size} random elements`);

        const t0 = performance.now();
        const sortBuffers = sorter.createSortBuffers(input.length);
        device.queue.writeBuffer(sortBuffers.keysA, 0, input);
        await sorter.sort(device.queue, sortBuffers);

        const readbackBuffer = device.createBuffer({
            size: input.length * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(sortBuffers.keysA, 0, readbackBuffer, 0, input.length * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const t1 = performance.now();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        sortBuffers.destroy();

        // Verify sorted order (check adjacent pairs)
        let isSorted = true;
        for (let i = 1; i < result.length; i++) {
            if (result[i] < result[i - 1]) {
                isSorted = false;
                console.log(`Sort error at index ${i}: ${result[i - 1]} > ${result[i]}`);
                break;
            }
        }

        console.log(`GPU Sort Time: ${(t1 - t0).toFixed(2)} ms`);
        console.log(`Throughput: ${(size / (t1 - t0) / 1000).toFixed(2)} M elements/ms`);
        console.log(`First 10 sorted: ${Array.from(result.slice(0, 10))}`);
        console.log(`Last 10 sorted: ${Array.from(result.slice(-10))}`);
        console.log(`Validation: ${isSorted ? '✔ PASS' : '✗ FAIL'}\n`);
        if (!isSorted) allPassed = false;
    }

    // Summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL RADIX SORT TESTS PASSED                             ║');
    } else {
        console.log('║  ✗ SOME RADIX SORT TESTS FAILED                            ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}
