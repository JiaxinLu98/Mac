/**
 * Test file for Count Kernel (set_availability_intersection_count_v1.wgsl)
 *
 * Tests the count phase of the ModernGPU-style set intersection algorithm.
 * This kernel:
 * 1. Reads partition boundaries from DPI
 * 2. Loads A and B data into shared memory
 * 3. Each thread runs Local BalancedPath to find its starting position
 * 4. Each thread runs SerialSetIntersection to count matches
 * 5. Workgroup reduction to sum counts
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_intersection_count_v1.wgsl';
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
 * Test class for the count kernel.
 */
export class TestCountKernel {
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
            label: 'Count kernel bind group layout',
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
            label: 'Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });
    }

    /**
     * Run the count kernel test.
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
     * CPU merge-path set intersection count.
     * Returns total count of matching elements.
     */
    public cpuSetIntersectionCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                ai++;
            } else if (a[ai] > b[bi]) {
                bi++;
            } else {
                count++;
                ai++;
                bi++;
            }
        }
        return count;
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

            // Count matches in this partition
            let count = 0;
            let ai = a0, bi = b0;
            while (ai < a1 && bi < b1) {
                if (a[ai] < b[bi]) {
                    ai++;
                } else if (a[ai] > b[bi]) {
                    bi++;
                } else {
                    count++;
                    ai++;
                    bi++;
                }
            }
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

        console.log('\n=== Count Kernel Results ===');
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
        maxThreads: number = 30  // Only print first N threads per workgroup
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

                // Only print threads with commits or first few threads, or if b_adjust > 0
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
        const cpuTotalCount = this.cpuSetIntersectionCount(a, b);

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
 * Calculate the number of workgroups for the count kernel.
 */
export function calculateNumWorkgroups(aLen: number, bLen: number): number {
    const total = aLen + bLen;
    if (total === 0) {
        throw new Error('Cannot calculate workgroups for empty arrays');
    }
    return Math.ceil(total / NV);
}

/**
 * Run all count kernel test cases.
 */
export async function runCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║           COUNT KERNEL TEST SUITE                          ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestCountKernel(device, timestampQueryManager);

    let allPassed = true;

    // ========================================================================
    // Test Case 0: Minimal single-workgroup duplicates (ISOLATION TEST)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 0: MINIMAL - Single workgroup with duplicates');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Minimal test: single workgroup (total < NV=1792), with duplicates
        const A = new Uint32Array([1, 1, 1, 2, 2, 3]);
        const B = new Uint32Array([1, 1, 2, 2, 2, 3]);
        console.log('A:', Array.from(A));
        console.log('B:', Array.from(B));
        console.log('Expected: merge-path matches -> count = 6');
        console.log('(All elements match when traversing merge path)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            console.log('⚠️  MINIMAL TEST FAILED - Problem is in local algorithm, NOT star bit!');
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 0b: All same value, single workgroup
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 0b: MINIMAL - All same value, single workgroup');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(100);
        const B = new Uint32Array(100);
        A.fill(5);
        B.fill(5);
        console.log(`A: [5, 5, ..., 5] (${A.length} elements)`);
        console.log(`B: [5, 5, ..., 5] (${B.length} elements)`);
        console.log(`Expected count: min(${A.length}, ${B.length}) = ${Math.min(A.length, B.length)}`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);

        // Print debug data for this failing test case
        console.log('\n>>> DEBUG: Thread-level data for Case 0b <<<');
        tester.printDebugData(result.debugData, result.numWorkgroups, 50);

        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            console.log('⚠️  MINIMAL TEST FAILED - Problem is in local algorithm, NOT star bit!');
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

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

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
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

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
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

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 4: Multi-workgroup (even vs multiples of 3)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: Multi-workgroup (even vs multiples of 3)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = i * 2;  // 0, 2, 4, 6, ...
            B[i] = i * 3;  // 0, 3, 6, 9, ...
        }
        console.log(`A: [0, 2, 4, ..., ${A[size-1]}] (${size} elements)`);
        console.log(`B: [0, 3, 6, ..., ${B[size-1]}] (${size} elements)`);
        console.log('Expected: multiples of 6 within range');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 5: Many duplicates (stress test)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Many duplicates (stress test)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 2000;
        const A = new Uint32Array(size);
        const B = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            A[i] = Math.floor(i / 10) * 3;  // 0,0,...,0, 3,3,...,3, 6,...
            B[i] = Math.floor(i / 8) * 3;   // 0,0,...,0, 3,3,...,3, 6,...
        }
        console.log(`A: values repeat 10 times each (${size} elements)`);
        console.log(`B: values repeat 8 times each (${size} elements)`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 6: Unbalanced sizes
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Unbalanced sizes (A tiny, B huge)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array([100, 200, 300, 400, 500]);
        const B = new Uint32Array(3000);
        for (let i = 0; i < B.length; i++) {
            B[i] = i;  // 0, 1, 2, ..., 2999
        }
        console.log(`A: [100, 200, 300, 400, 500] (5 elements)`);
        console.log(`B: [0, 1, 2, ..., 2999] (${B.length} elements)`);
        console.log('Expected: [100, 200, 300, 400, 500] -> count = 5');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 7: Star bit trigger case
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: Star bit trigger case');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const a9Len = 3000;
        const b9Len = 3000;

        const A = new Uint32Array(a9Len);
        A[0] = 0;
        A.fill(1, 1);  // A = [0, 1, 1, 1, ..., 1]

        const B = new Uint32Array(b9Len);
        B.fill(1);     // B = [1, 1, 1, ..., 1]

        console.log(`A: [0, 1, 1, ..., 1] (${a9Len} elements)`);
        console.log(`B: [1, 1, ..., 1] (${b9Len} elements)`);

        const result = await tester.testCountKernel(A, B);

        // Check star bits
        const numWg = result.numWorkgroups;
        let starCount = 0;
        for (let i = 0; i <= numWg; i++) {
            if ((result.dpiData[i] & STAR_MASK) !== 0) starCount++;
        }
        console.log(`Star bits triggered: ${starCount} / ${numWg + 1}`);

        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 8: All same value (maximum duplicates)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 8: All same value (maximum duplicates)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const A = new Uint32Array(2500);
        const B = new Uint32Array(2500);
        A.fill(42);
        B.fill(42);

        console.log(`A: [42, 42, ..., 42] (${A.length} elements)`);
        console.log(`B: [42, 42, ..., 42] (${B.length} elements)`);
        console.log(`Expected count: min(${A.length}, ${B.length}) = ${Math.min(A.length, B.length)}`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 9: NV boundary (total = NV exactly)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 9: NV boundary (total = NV exactly)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const aLen = 896;  // NV / 2
        const bLen = 896;  // NV / 2, total = NV = 1792
        const A = new Uint32Array(aLen);
        const B = new Uint32Array(bLen);
        for (let i = 0; i < aLen; i++) A[i] = i * 2;
        for (let i = 0; i < bLen; i++) B[i] = i * 2 + 1;

        console.log(`A: [0, 2, 4, ...] (${aLen} elements)`);
        console.log(`B: [1, 3, 5, ...] (${bLen} elements)`);
        console.log(`Total = ${aLen + bLen} = NV (exactly 1 workgroup)`);
        console.log('Expected count: 0 (no overlap)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 10: Large dataset
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 10: Large dataset (10K elements each)');
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
        console.log('Expected: all even numbers in A');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 11: Complex - Mixed duplicates spanning workgroup boundaries
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 11: Complex - Mixed duplicates spanning WG boundaries');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Design a complex pattern:
        // - Total elements > 3*NV to have 4+ workgroups
        // - Large runs of duplicates that span workgroup boundaries
        // - Variable run lengths (some short, some very long)
        // - Some values only in A, some only in B
        // - Unbalanced sizes (A larger than B)

        const aSize = 4500;
        const bSize = 3800;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        // Build A: pattern with varying duplicate runs
        // [0×100, 1×50, 2×200, 3×1, 5×500, 6×100, 7×300, 8×50, 10×800, 11×200, ...]
        let idx = 0;
        const aPattern = [
            { val: 0, count: 100 },
            { val: 1, count: 50 },     // 1 only in A
            { val: 2, count: 200 },
            { val: 3, count: 1 },      // single element
            { val: 5, count: 500 },    // long run spanning WG boundary
            { val: 6, count: 100 },
            { val: 7, count: 300 },
            { val: 8, count: 50 },
            { val: 10, count: 800 },   // very long run, multiple WG boundaries
            { val: 11, count: 200 },
            { val: 12, count: 300 },
            { val: 15, count: 400 },   // 15 only in A
            { val: 20, count: 500 },
            { val: 25, count: 300 },
            { val: 30, count: 400 },
        ];
        for (const { val, count } of aPattern) {
            for (let i = 0; i < count && idx < aSize; i++) {
                A[idx++] = val;
            }
        }
        // Fill remaining with unique increasing values
        let fillVal = 100;
        while (idx < aSize) {
            A[idx++] = fillVal++;
        }

        // Build B: overlapping pattern with different run lengths
        idx = 0;
        const bPattern = [
            { val: 0, count: 80 },     // fewer 0s than A
            { val: 2, count: 150 },    // fewer 2s than A
            { val: 3, count: 5 },      // more 3s than A
            { val: 4, count: 100 },    // 4 only in B
            { val: 5, count: 600 },    // more 5s than A
            { val: 6, count: 100 },    // same count
            { val: 7, count: 200 },    // fewer 7s than A
            { val: 8, count: 80 },     // more 8s than A
            { val: 9, count: 50 },     // 9 only in B
            { val: 10, count: 700 },   // fewer 10s than A
            { val: 11, count: 250 },   // more 11s than A
            { val: 12, count: 100 },   // fewer 12s than A
            { val: 20, count: 400 },
            { val: 25, count: 350 },
            { val: 30, count: 130 },
        ];
        for (const { val, count } of bPattern) {
            for (let i = 0; i < count && idx < bSize; i++) {
                B[idx++] = val;
            }
        }
        // Fill remaining
        fillVal = 200;
        while (idx < bSize) {
            B[idx++] = fillVal++;
        }

        console.log(`A: ${aSize} elements with mixed duplicate patterns`);
        console.log(`B: ${bSize} elements with mixed duplicate patterns`);
        console.log(`Total: ${aSize + bSize} elements across ${Math.ceil((aSize + bSize) / NV)} workgroups`);
        console.log('Pattern includes:');
        console.log('  - Runs spanning multiple WG boundaries (val=5: 500+600, val=10: 800+700)');
        console.log('  - Values only in A (val=1, 15) and only in B (val=4, 9)');
        console.log('  - Single elements (val=3 in A)');
        console.log('  - Unbalanced sizes (A > B)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);

        // Print detailed star bit info
        const numWg = result.numWorkgroups;
        let starCount = 0;
        for (let i = 0; i <= numWg; i++) {
            if ((result.dpiData[i] & STAR_MASK) !== 0) starCount++;
        }
        console.log(`Star bits triggered: ${starCount} / ${numWg + 1}`);

        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            console.log('\n>>> DEBUG: Thread-level data for Case 11 <<<');
            tester.printDebugData(result.debugData, result.numWorkgroups, 20);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 12: Stress - Maximum star bit triggers
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 12: Stress - Maximum star bit triggers');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Create arrays where every workgroup boundary falls in the middle of duplicates
        // This maximizes star bit triggers
        // NV = 1792, so we design runs that span every 1792 elements

        const numTargetWg = 6;
        const runLength = NV;  // Each value repeats NV times, guaranteeing boundary splits

        // A: [0×1792, 1×1792, 2×1792, 3×1792, 4×1792, 5×1792]
        // B: same pattern but shifted slightly
        const aSize = numTargetWg * runLength;
        const bSize = numTargetWg * runLength - 200;  // Slightly smaller

        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        for (let i = 0; i < aSize; i++) {
            A[i] = Math.floor(i / runLength);
        }
        for (let i = 0; i < bSize; i++) {
            B[i] = Math.floor(i / runLength);
        }

        console.log(`A: ${aSize} elements, values 0-${numTargetWg - 1}, each repeats ${runLength} times`);
        console.log(`B: ${bSize} elements, same pattern`);
        console.log(`Total: ${aSize + bSize} elements`);
        console.log('Every WG boundary should fall within a duplicate run');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);

        // Check star bits
        const numWg = result.numWorkgroups;
        let starCount = 0;
        for (let i = 0; i <= numWg; i++) {
            if ((result.dpiData[i] & STAR_MASK) !== 0) starCount++;
        }
        console.log(`Star bits triggered: ${starCount} / ${numWg + 1} (expecting many)`);

        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            console.log('\n>>> DEBUG: Thread-level data for Case 12 <<<');
            tester.printDebugData(result.debugData, result.numWorkgroups, 15);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 13: Extreme unbalance (tiny A, huge B)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 13: Extreme unbalance (tiny A, huge B with dups)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // A has just 10 elements, B has 8000 elements with duplicates
        const A = new Uint32Array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        const bSize = 8000;
        const B = new Uint32Array(bSize);

        // B: 0-99 repeated 80 times each
        for (let i = 0; i < bSize; i++) {
            B[i] = Math.floor(i / 80);
        }

        console.log(`A: [10, 20, 30, ..., 100] (10 elements)`);
        console.log(`B: [0×80, 1×80, 2×80, ..., 99×80] (${bSize} elements)`);
        console.log('Expected: 10 matches (all of A exists in B)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 10);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 14: Sparse intersection in large arrays
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 14: Sparse intersection (few matches in large arrays)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Large arrays with very few common elements
        const aSize = 5000;
        const bSize = 5000;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        // A: 0, 10, 20, 30, ... (multiples of 10)
        // B: 0, 7, 14, 21, ... (multiples of 7)
        // Intersection: multiples of 70 (LCM of 10 and 7)
        for (let i = 0; i < aSize; i++) {
            A[i] = i * 10;
        }
        for (let i = 0; i < bSize; i++) {
            B[i] = i * 7;
        }

        console.log(`A: [0, 10, 20, ..., ${(aSize - 1) * 10}] (${aSize} elements, multiples of 10)`);
        console.log(`B: [0, 7, 14, ..., ${(bSize - 1) * 7}] (${bSize} elements, multiples of 7)`);
        console.log('Expected: multiples of 70 within range');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 15: Alternating single values with partial overlap
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 15: Alternating values with partial overlap');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // A: [0, 2, 4, 6, 8, ...] even numbers
        // B: [0, 3, 6, 9, 12, ...] multiples of 3
        // Intersection: multiples of 6
        const aSize = 3000;
        const bSize = 2500;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        for (let i = 0; i < aSize; i++) {
            A[i] = i * 2;
        }
        for (let i = 0; i < bSize; i++) {
            B[i] = i * 3;
        }

        console.log(`A: [0, 2, 4, ..., ${(aSize - 1) * 2}] (${aSize} elements, even numbers)`);
        console.log(`B: [0, 3, 6, ..., ${(bSize - 1) * 3}] (${bSize} elements, multiples of 3)`);
        console.log('Expected: multiples of 6 within both ranges');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 16: Single massive duplicate value
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 16: Single massive duplicate (one value dominates)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Most elements are the same value, with a few different values at edges
        const aSize = 6000;
        const bSize = 5500;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        // A: [0, 1, 2, 100×5990, 999998, 999999]
        A[0] = 0;
        A[1] = 1;
        A[2] = 2;
        for (let i = 3; i < aSize - 2; i++) {
            A[i] = 100;
        }
        A[aSize - 2] = 999998;
        A[aSize - 1] = 999999;

        // B: [1, 2, 100×5495, 999998, 999999, 1000000]
        B[0] = 1;
        B[1] = 2;
        for (let i = 2; i < bSize - 3; i++) {
            B[i] = 100;
        }
        B[bSize - 3] = 999998;
        B[bSize - 2] = 999999;
        B[bSize - 1] = 1000000;

        console.log(`A: [0, 1, 2, 100×${aSize - 5}, 999998, 999999] (${aSize} elements)`);
        console.log(`B: [1, 2, 100×${bSize - 5}, 999998, 999999, 1000000] (${bSize} elements)`);
        console.log(`Expected: 2 + min(${aSize - 5}, ${bSize - 5}) + 2 matches`);

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 10);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 17: Prime number sizes (non-aligned)
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 17: Prime number sizes (non-aligned boundaries)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Use prime numbers for array sizes to stress boundary handling
        const aSize = 2999;  // Prime
        const bSize = 3001;  // Prime
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        // Pattern: increasing with some duplicates
        for (let i = 0; i < aSize; i++) {
            A[i] = Math.floor(i * 1.5);  // 0, 1, 3, 4, 6, 7, 9, ...
        }
        for (let i = 0; i < bSize; i++) {
            B[i] = i * 2;  // 0, 2, 4, 6, 8, ...
        }

        console.log(`A: ${aSize} elements (prime), values = floor(i * 1.5)`);
        console.log(`B: ${bSize} elements (prime), values = i * 2`);
        console.log('Testing non-aligned array sizes');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 18: Fibonacci-like run lengths
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 18: Fibonacci-like run lengths');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Run lengths follow Fibonacci sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, ...
        const fibLengths = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597];

        // Build A with Fibonacci run lengths
        const aElements: number[] = [];
        let val = 0;
        for (const len of fibLengths) {
            for (let i = 0; i < len; i++) {
                aElements.push(val);
            }
            val += 2;  // Skip every other value
        }
        const A = new Uint32Array(aElements);

        // Build B with slightly different pattern (offset values)
        const bElements: number[] = [];
        val = 0;
        for (const len of fibLengths) {
            for (let i = 0; i < len; i++) {
                bElements.push(val);
            }
            val += 3;  // Different skip
        }
        const B = new Uint32Array(bElements);

        console.log(`A: ${A.length} elements with Fibonacci run lengths, values += 2`);
        console.log(`B: ${B.length} elements with Fibonacci run lengths, values += 3`);
        console.log('Intersection: values that are multiples of both 2 and 3 (i.e., 6)');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) allPassed = false;
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 19: Clustered matches at workgroup boundaries
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 19: Matches clustered at WG boundaries');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Design arrays where matches occur primarily near workgroup boundaries
        const aSize = 4000;
        const bSize = 4000;
        const A = new Uint32Array(aSize);
        const B = new Uint32Array(bSize);

        // A: mostly unique values, but at positions near NV boundaries, use shared values
        for (let i = 0; i < aSize; i++) {
            const nearBoundary = (i % NV) < 50 || (i % NV) > NV - 50;
            if (nearBoundary) {
                A[i] = Math.floor(i / 100) * 1000;  // Shared values near boundaries
            } else {
                A[i] = i * 10 + 1;  // Unique odd values (won't match B's pattern)
            }
        }

        // B: similar pattern
        for (let i = 0; i < bSize; i++) {
            const nearBoundary = (i % NV) < 50 || (i % NV) > NV - 50;
            if (nearBoundary) {
                B[i] = Math.floor(i / 100) * 1000;  // Same shared values
            } else {
                B[i] = i * 10 + 2;  // Unique even values (won't match A's pattern)
            }
        }

        // Sort to maintain sorted order
        A.sort((a, b) => a - b);
        B.sort((a, b) => a - b);

        console.log(`A: ${aSize} elements with matches clustered near WG boundaries`);
        console.log(`B: ${bSize} elements with matches clustered near WG boundaries`);
        console.log('Testing boundary-heavy matching patterns');

        const result = await tester.testCountKernel(A, B);
        tester.printResults(result, A, B);
        const validation = tester.validateResults(result, A, B);

        if (!validation.totalMatch) {
            tester.printDebugData(result.debugData, result.numWorkgroups, 10);
            allPassed = false;
        }
        console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms\n`);
    }

    // ========================================================================
    // Test Case 20: Interleaved short and long runs
    // ========================================================================
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 20: Interleaved short and long duplicate runs');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        // Alternating pattern: 1 element, 500 elements, 1 element, 500 elements, ...
        const aElements: number[] = [];
        const bElements: number[] = [];
        let val = 0;

        for (let round = 0; round < 10; round++) {
            // Short run (1 element)
            aElements.push(val);
            bElements.push(val);
            val++;

            // Long run (500 elements)
            for (let i = 0; i < 500; i++) {
                aElements.push(val);
            }
            for (let i = 0; i < 450; i++) {  // B has fewer of the long runs
                bElements.push(val);
            }
            val++;
        }

        const A = new Uint32Array(aElements);
        const B = new Uint32Array(bElements);

        console.log(`A: ${A.length} elements (alternating 1 and 500 element runs)`);
        console.log(`B: ${B.length} elements (alternating 1 and 450 element runs)`);
        console.log('Testing rapid transitions between short and long runs');

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
 * Quick test for the count kernel (simplified version for debugging).
 */
export async function quickCountKernelTest(device: GPUDevice): Promise<boolean> {
    const timestampQueryManager = new TimestampQueryManager(device, 4);
    const tester = new TestCountKernel(device, timestampQueryManager);

    // Simple test case
    const A = new Uint32Array([1, 3, 3, 5, 7, 9]);
    const B = new Uint32Array([2, 3, 3, 6, 7, 8]);

    console.log('\n=== Quick Count Kernel Test ===');
    console.log('A:', Array.from(A));
    console.log('B:', Array.from(B));

    const result = await tester.testCountKernel(A, B);
    const cpuCount = tester.cpuSetIntersectionCount(A, B);

    console.log(`GPU total count: ${result.totalCount}`);
    console.log(`CPU total count: ${cpuCount}`);
    console.log(`Match: ${result.totalCount === cpuCount ? '✔ YES' : '✗ NO'}`);
    console.log('================================\n');

    return result.totalCount === cpuCount;
}
