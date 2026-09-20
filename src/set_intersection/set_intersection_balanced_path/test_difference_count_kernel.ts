/**
 * Test file for Difference Count Kernel (set_availability_difference_count_v1.wgsl)
 *
 * Tests the count phase of the ModernGPU-style set difference algorithm.
 * This kernel computes multiset difference A \ B:
 * - If A < B: emit A (A not in B), advance A
 * - If B < A: advance B only (skip B element)
 * - If A == B: advance both (one B cancels one A, no emit)
 *
 * Multiset semantics: If A has 5 copies of x and B has 3 copies, result has 2 copies.
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_difference_count_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// Helper: count 1 bits in a 32-bit integer
function countOneBits(n: number): number {
    n = n - ((n >>> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >>> 2) & 0x33333333);
    return (((n + (n >>> 4)) & 0x0F0F0F0F) * 0x01010101) >>> 24;
}

// ModernGPU constants - must match WGSL shader
const NT = 256;       // Threads per workgroup
const VT = 7;         // Values per thread
const NV = NT * VT;   // Elements per workgroup = 1792

/**
 * Test class for the difference count kernel.
 */
export class TestDifferenceCountKernel {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI computation pipeline
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;

    // Count kernel pipeline
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;

    // For aligned buffer sizes
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // Create DPI bind group layout
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

        // Create count kernel bind group layout
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Difference count kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // debug buffer
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: 'Difference count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });
    }

    /**
     * Run the difference count kernel test.
     * Returns per-workgroup counts and total count.
     */
    public async testCountKernel(
        setA: Uint32Array,
        setB: Uint32Array
    ): Promise<{
        counts: Uint32Array;
        totalCount: number;
        dpiData: Uint32Array;
        numWorkgroups: number;
        gpuTimeMs: number;
        debugData: Uint32Array;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;
        const total = a_len + b_len;

        if (total === 0) {
            throw new Error('Cannot test count kernel with empty arrays');
        }

        const numWg = Math.ceil(total / NV);

        // Create GPU buffers
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

        // Counts buffer (aligned for scan)
        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Debug buffer: 7 values per thread, NT threads per workgroup
        const debugSize = numWg * NT * 7;
        const bufferDebug = device.createBuffer({
            label: 'Buffer Debug',
            size: debugSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // ============ Step 1: Compute DPI ============
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

        // ============ Step 2: Run Count Kernel ============
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

        const t0 = performance.now();
        encoder = device.createCommandEncoder();
        pass = encoder.beginComputePass();
        pass.setPipeline(this.countPipeline);
        pass.setBindGroup(0, countBindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const gpuTimeMs = performance.now() - t0;

        // ============ Read back results ============
        // Read DPI
        const dpiReadback = device.createBuffer({
            size: dpiSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(bufferDPI, 0, dpiReadback, 0, dpiSize * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await dpiReadback.mapAsync(GPUMapMode.READ);
        const dpiData = new Uint32Array(dpiReadback.getMappedRange().slice(0));
        dpiReadback.unmap();

        // Read counts
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

        // Read debug data
        const debugReadback = device.createBuffer({
            size: debugSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(bufferDebug, 0, debugReadback, 0, debugSize * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await debugReadback.mapAsync(GPUMapMode.READ);
        const debugData = new Uint32Array(debugReadback.getMappedRange().slice(0));
        debugReadback.unmap();

        // Calculate total
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
        dpiReadback.destroy();
        countsReadback.destroy();
        debugReadback.destroy();

        return { counts, totalCount, dpiData, numWorkgroups: numWg, gpuTimeMs, debugData };
    }

    // ========================================================================
    // CPU Reference Implementation
    // ========================================================================

    /**
     * CPU merge-path set difference count.
     * Returns total count of elements in A \ B (multiset difference).
     */
    public cpuSetDifferenceCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                // A < B: A is not in B, emit A
                count++;
                ai++;
            } else if (a[ai] > b[bi]) {
                // B < A: skip B
                bi++;
            } else {
                // A == B: one B cancels one A, no emit
                ai++;
                bi++;
            }
        }
        // Remaining elements in A (B exhausted)
        count += (a.length - ai);
        return count;
    }

    /**
     * CPU set difference - returns the actual result array.
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
                ai++;
                bi++;
            }
        }
        // Remaining elements in A
        while (ai < a.length) {
            result.push(a[ai]);
            ai++;
        }
        return new Uint32Array(result);
    }

    /**
     * CPU per-partition count using DPI boundaries.
     * This simulates what each workgroup should compute.
     */
    public cpuPerPartitionCount(
        a: Uint32Array,
        b: Uint32Array,
        dpiData: Uint32Array,
        numWorkgroups: number
    ): number[] {
        const counts: number[] = [];

        for (let wg = 0; wg < numWorkgroups; wg++) {
            const packedA0 = dpiData[wg];
            const packedA1 = dpiData[wg + 1];
            const a0 = packedA0 & INDEX_MASK;
            const a1 = packedA1 & INDEX_MASK;
            const star0 = (packedA0 & STAR_MASK) !== 0;
            const star1 = (packedA1 & STAR_MASK) !== 0;
            const b0 = dpiData[numWorkgroups + 1 + wg] + (star0 ? 1 : 0);
            const b1 = dpiData[numWorkgroups + 1 + wg + 1] + (star1 ? 1 : 0);

            // Count difference elements in this partition
            let count = 0;
            let ai = a0, bi = b0;
            while (ai < a1 && bi < b1) {
                if (a[ai] < b[bi]) {
                    count++;
                    ai++;
                } else if (a[ai] > b[bi]) {
                    bi++;
                } else {
                    ai++;
                    bi++;
                }
            }
            // Remaining A elements in this partition
            count += (a1 - ai);
            counts.push(count);
        }

        return counts;
    }

    /**
     * Print detailed results.
     */
    public printResults(
        result: { counts: Uint32Array; totalCount: number; dpiData: Uint32Array; numWorkgroups: number },
        a: Uint32Array,
        b: Uint32Array
    ): void {
        const { counts, totalCount, dpiData, numWorkgroups } = result;

        console.log('\n=== Difference Count Kernel Results ===');
        console.log(`Input sizes: A=${a.length}, B=${b.length}, total=${a.length + b.length}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);
        console.log('');

        // Print DPI info
        console.log('Partition boundaries (from DPI):');
        for (let i = 0; i <= numWorkgroups; i++) {
            const packedA = dpiData[i];
            const aIdx = packedA & INDEX_MASK;
            const star = (packedA & STAR_MASK) !== 0;
            const bIdx = dpiData[numWorkgroups + 1 + i];
            console.log(`  Partition ${i}: aIdx=${aIdx}, bIdx=${bIdx}, star=${star}`);
        }
        console.log('');

        // Print per-workgroup counts
        console.log('Per-workgroup counts:');
        for (let i = 0; i < numWorkgroups; i++) {
            const a0 = dpiData[i] & INDEX_MASK;
            const a1 = dpiData[i + 1] & INDEX_MASK;
            const star0 = (dpiData[i] & STAR_MASK) !== 0;
            const star1 = (dpiData[i + 1] & STAR_MASK) !== 0;
            const b0 = dpiData[numWorkgroups + 1 + i] + (star0 ? 1 : 0);
            const b1 = dpiData[numWorkgroups + 1 + i + 1] + (star1 ? 1 : 0);
            console.log(`  WG ${i}: A[${a0}..${a1}), B[${b0}..${b1}) -> count=${counts[i]}`);
        }
        console.log('');

        console.log(`Total GPU count: ${totalCount}`);
        console.log('===============================\n');
    }

    /**
     * Print debug data for each thread.
     * Debug format: [tid, a0tid, b0tid, star, b_adjust, diag, commit]
     */
    public printDebugData(
        debugData: Uint32Array,
        numWorkgroups: number,
        maxThreads: number = 30
    ): void {
        console.log('\n=== Thread Debug Data ===');
        console.log('Format: tid | a0tid | b0tid | star | b_adjust | diag | commit (binary)');
        console.log('');

        for (let wg = 0; wg < numWorkgroups; wg++) {
            console.log(`--- Workgroup ${wg} ---`);
            const baseOffset = wg * NT * 7;

            let totalCommits = 0;
            for (let t = 0; t < Math.min(NT, maxThreads); t++) {
                const offset = baseOffset + t * 7;
                const tid = debugData[offset + 0];
                const a0tid = debugData[offset + 1];
                const b0tid = debugData[offset + 2];
                const star = debugData[offset + 3];
                const bAdjust = debugData[offset + 4];
                const diag = debugData[offset + 5];
                const commit = debugData[offset + 6];
                const commitBits = countOneBits(commit);
                totalCommits += commitBits;

                if (commitBits > 0 || t < 5 || bAdjust > 0) {
                    const commitBinary = commit.toString(2).padStart(7, '0');
                    console.log(`  t${tid.toString().padStart(3)}: a0=${a0tid.toString().padStart(3)}, b0=${b0tid.toString().padStart(3)}, star=${star}, b_adj=${bAdjust}, diag=${diag.toString().padStart(3)}, commit=${commitBinary} (${commitBits})`);
                }
            }
            if (maxThreads < NT) {
                console.log(`  ... (showing first ${maxThreads} threads)`);
            }
            console.log(`  Total commits in WG ${wg}: ${totalCommits}`);
            console.log('');
        }
    }

    /**
     * Validate GPU results against CPU reference.
     */
    public validateResults(
        result: { counts: Uint32Array; totalCount: number; dpiData: Uint32Array; numWorkgroups: number },
        a: Uint32Array,
        b: Uint32Array
    ): { totalMatch: boolean; perPartitionMatch: boolean } {
        const { counts, totalCount, dpiData, numWorkgroups } = result;

        // CPU total count
        const cpuTotalCount = this.cpuSetDifferenceCount(a, b);

        // CPU per-partition counts
        const cpuPartitionCounts = this.cpuPerPartitionCount(a, b, dpiData, numWorkgroups);

        console.log('\n=== Validation ===');
        console.log(`CPU total count: ${cpuTotalCount}`);
        console.log(`GPU total count: ${totalCount}`);
        console.log(`Total match: ${totalCount === cpuTotalCount ? '✔ YES' : '✗ NO'}`);
        console.log('');

        // Compare per-partition
        let perPartitionMatch = true;
        for (let i = 0; i < numWorkgroups; i++) {
            const gpuCount = counts[i];
            const cpuCount = cpuPartitionCounts[i];
            const match = gpuCount === cpuCount;
            if (!match) {
                console.log(`  WG ${i}: GPU=${gpuCount}, CPU=${cpuCount} ✗`);
                perPartitionMatch = false;
            }
        }

        if (perPartitionMatch) {
            console.log(`Per-partition match: ✔ All ${numWorkgroups} workgroups match`);
        } else {
            console.log(`Per-partition match: ✗ Some workgroups differ`);
        }

        console.log('==================\n');

        return {
            totalMatch: totalCount === cpuTotalCount,
            perPartitionMatch
        };
    }
}

/**
 * Run all difference count kernel test cases.
 */
export async function runDifferenceCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║       DIFFERENCE COUNT KERNEL TEST SUITE                   ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestDifferenceCountKernel(device, timestampQueryManager);

    let allPassed = true;

    // ========================================================================
    // Test Case 0: Basic multiset difference
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 0: Basic multiset difference');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 1, 1, 2, 3]);
        const B = new Uint32Array([1, 2, 4]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: A \\ B = [1, 1, 3] -> count = 3');
        console.log('(3 ones minus 1 one = 2 ones, plus 1 three)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            console.log('>>> DEBUG: Thread-level data <<<');
            tester.printDebugData(result.debugData, result.numWorkgroups, 20);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 1: No overlap (A \\ B = A)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: No overlap (A \\ B = A)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 3, 5, 7, 9]);
        const B = new Uint32Array([2, 4, 6, 8, 10]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: A \\ B = [1, 3, 5, 7, 9] -> count = 5');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 2: Complete overlap (A \\ B = empty)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: Complete overlap (A \\ B = empty)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5]);
        const B = new Uint32Array([1, 2, 3, 4, 5]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: A \\ B = [] -> count = 0');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 3: B has more copies (multiset)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: B has more copies than A');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 1, 2]);
        const B = new Uint32Array([1, 1, 1, 1, 2, 2, 2]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: A \\ B = [] -> count = 0');
        console.log('(A has 2 ones, B has 4 ones: 2-4=0; A has 1 two, B has 3 twos: 1-3=0)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 4: All same value (multiset difference)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: All same value (multiset difference)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(100);
        const B = new Uint32Array(60);
        A.fill(5);
        B.fill(5);
        console.log(`A: [5, 5, ..., 5] (${A.length} elements)`);
        console.log(`B: [5, 5, ..., 5] (${B.length} elements)`);
        console.log(`Expected count: ${A.length} - ${B.length} = ${A.length - B.length}`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 30);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 5: Multi-workgroup with mixed patterns
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Multi-workgroup with mixed patterns');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i * 2;       // 0, 2, 4, 6, ...
            B[i] = i * 3;       // 0, 3, 6, 9, ...
        }
        console.log(`A: [0, 2, 4, ..., ${A[size-1]}] (${size} elements, even numbers)`);
        console.log(`B: [0, 3, 6, ..., ${B[size-1]}] (${size} elements, multiples of 3)`);
        console.log('Expected: A \\ B = even numbers that are NOT multiples of 6');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 6: B empty (A \\ B = A)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: B empty (A \\ B = A)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        const B = new Uint32Array([]);
        console.log('A:', Array.from(A));
        console.log('B: []');
        console.log(`Expected: A \\ B = A -> count = ${A.length}`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 7: Large duplicates spanning workgroups
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: Large duplicates spanning workgroups');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(3000);
        const B = new Uint32Array(2000);
        A.fill(42);
        B.fill(42);
        console.log(`A: [42, 42, ..., 42] (${A.length} elements)`);
        console.log(`B: [42, 42, ..., 42] (${B.length} elements)`);
        console.log(`Expected count: ${A.length} - ${B.length} = ${A.length - B.length}`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 15);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 8: Unbalanced sizes (tiny A, huge B)
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
        console.log('Expected: A \\ B = [] -> count = 0 (all A elements are in B)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
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
            A[i] = i;           // 0, 1, 2, ..., 9999
            B[i] = i * 2;       // 0, 2, 4, ..., 19998
        }
        console.log(`A: [0, 1, 2, ..., ${size-1}] (${size} elements)`);
        console.log(`B: [0, 2, 4, ..., ${(size-1)*2}] (${size} elements)`);
        console.log('Expected: A \\ B = all odd numbers in A');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 10: Complex mixed duplicates
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 10: Complex mixed duplicates');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // A: more 1s and 2s than B
        // B: more 3s than A
        const A = new Uint32Array([1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 5, 5]);
        const B = new Uint32Array([1, 1, 2, 3, 3, 3, 3, 5]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: A \\ B = [1, 1, 1, 2, 2, 4, 5] -> count = 7');
        console.log('  1: 5 in A, 2 in B -> 3 remain');
        console.log('  2: 3 in A, 1 in B -> 2 remain');
        console.log('  3: 1 in A, 4 in B -> 0 remain');
        console.log('  4: 1 in A, 0 in B -> 1 remain');
        console.log('  5: 2 in A, 1 in B -> 1 remain');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 30);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
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
 * Quick test for the difference count kernel.
 */
export async function quickDifferenceCountKernelTest(device: GPUDevice): Promise<boolean> {
    const timestampQueryManager = new TimestampQueryManager(device, 4);
    const tester = new TestDifferenceCountKernel(device, timestampQueryManager);

    // Simple test case: multiset difference
    const A = new Uint32Array([1, 1, 1, 2, 3]);
    const B = new Uint32Array([1, 2, 4]);

    console.log('\n=== Quick Difference Count Kernel Test ===');
    console.log('A:', Array.from(A));
    console.log('B:', Array.from(B));

    const result = await tester.testCountKernel(A, B);
    const cpuCount = tester.cpuSetDifferenceCount(A, B);
    const cpuResult = tester.cpuSetDifference(A, B);

    console.log(`CPU difference result: [${Array.from(cpuResult)}]`);
    console.log(`GPU total count: ${result.totalCount}`);
    console.log(`CPU total count: ${cpuCount}`);
    console.log(`Match: ${result.totalCount === cpuCount ? '✔ YES' : '✗ NO'}`);
    console.log('================================\n');

    return result.totalCount === cpuCount;
}
