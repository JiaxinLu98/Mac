/**
 * Test file for Union Count Kernel (set_availability_union_count_v1.wgsl)
 *
 * Tests the count phase of the ModernGPU-style multiset union algorithm.
 * This kernel:
 * 1. Reads partition boundaries from DPI
 * 2. Loads A and B data into shared memory
 * 3. Each thread runs Local BalancedPath to find its starting position
 * 4. Each thread runs SerialSetUnion to count output elements
 * 5. Workgroup reduction to sum counts
 *
 * Multiset union semantics:
 * - If A < B: emit A, advance A
 * - If B < A: emit B, advance B
 * - If A == B: emit A (tie goes to A), advance both
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_union_count_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants - must match WGSL shader
const NT = 256;       // Threads per workgroup
const VT = 7;         // Values per thread
const NV = NT * VT;   // Elements per workgroup = 1792

/**
 * Test class for the union count kernel.
 */
export class TestUnionCountKernel {
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

        // Create count kernel bind group layout (no debug buffer for union)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Union Count kernel bind group layout',
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
            label: 'Union Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });
    }

    /**
     * Run the union count kernel test.
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
        dpiReadback.destroy();
        countsReadback.destroy();

        return { counts, totalCount, dpiData, numWorkgroups: numWg, gpuTimeMs };
    }

    // ========================================================================
    // CPU Reference Implementation
    // ========================================================================

    /**
     * CPU multiset union count.
     * Returns total count of elements in the union.
     * Union semantics: merge A and B, when equal values meet, output one and advance both.
     */
    public cpuSetUnionCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                count++;
                ai++;
            } else if (a[ai] > b[bi]) {
                count++;
                bi++;
            } else {
                // Equal: output one, advance both
                count++;
                ai++;
                bi++;
            }
        }
        // Add remaining elements
        count += (a.length - ai) + (b.length - bi);
        return count;
    }

    /**
     * CPU per-partition union count using DPI boundaries.
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

            // Count union elements in this partition
            let count = 0;
            let ai = a0, bi = b0;
            while (ai < a1 && bi < b1) {
                if (a[ai] < b[bi]) {
                    count++;
                    ai++;
                } else if (a[ai] > b[bi]) {
                    count++;
                    bi++;
                } else {
                    count++;
                    ai++;
                    bi++;
                }
            }
            // Add remaining elements in this partition
            count += (a1 - ai) + (b1 - bi);
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

        console.log('\n=== Union Count Kernel Results ===');
        console.log(`Input sizes: A=${a.length}, B=${b.length}, total=${a.length + b.length}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);
        console.log('');

        // Print DPI info
        console.log('Partition boundaries (from DPI):');
        for (let i = 0; i <= Math.min(numWorkgroups, 10); i++) {
            const packedA = dpiData[i];
            const aIdx = packedA & INDEX_MASK;
            const star = (packedA & STAR_MASK) !== 0;
            const bIdx = dpiData[numWorkgroups + 1 + i];
            console.log(`  Partition ${i}: aIdx=${aIdx}, bIdx=${bIdx}, star=${star}`);
        }
        if (numWorkgroups > 10) {
            console.log(`  ... (${numWorkgroups - 10} more partitions)`);
        }
        console.log('');

        // Print per-workgroup counts
        console.log('Per-workgroup counts:');
        for (let i = 0; i < Math.min(numWorkgroups, 10); i++) {
            const a0 = dpiData[i] & INDEX_MASK;
            const a1 = dpiData[i + 1] & INDEX_MASK;
            const star0 = (dpiData[i] & STAR_MASK) !== 0;
            const star1 = (dpiData[i + 1] & STAR_MASK) !== 0;
            const b0 = dpiData[numWorkgroups + 1 + i] + (star0 ? 1 : 0);
            const b1 = dpiData[numWorkgroups + 1 + i + 1] + (star1 ? 1 : 0);
            console.log(`  WG ${i}: A[${a0}..${a1}), B[${b0}..${b1}) -> count=${counts[i]}`);
        }
        if (numWorkgroups > 10) {
            console.log(`  ... (${numWorkgroups - 10} more workgroups)`);
        }
        console.log('');

        console.log(`Total GPU count: ${totalCount}`);
        console.log('===================================\n');
    }
}

/**
 * Run union count kernel test
 */
export async function runUnionCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║           MULTISET UNION COUNT KERNEL TEST                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestUnionCountKernel(device, timestampQueryManager);

    // Test cases
    const testCases = [
        {
            name: 'Simple union',
            a: new Uint32Array([1, 3, 5, 7, 9]),
            b: new Uint32Array([2, 3, 4, 7, 8, 10]),
        },
        {
            name: 'No overlap',
            a: new Uint32Array([1, 2, 3]),
            b: new Uint32Array([4, 5, 6]),
        },
        {
            name: 'Complete overlap',
            a: new Uint32Array([1, 2, 3, 4, 5]),
            b: new Uint32Array([1, 2, 3, 4, 5]),
        },
        {
            name: 'Large random',
            a: new Uint32Array(10000).map((_, i) => i * 2),      // Even numbers
            b: new Uint32Array(10000).map((_, i) => i * 2 + 1),  // Odd numbers
        },
        {
            name: 'Large with overlap',
            a: new Uint32Array(10000).map((_, i) => i),
            b: new Uint32Array(10000).map((_, i) => i + 5000),
        },
    ];

    for (const { name, a, b } of testCases) {
        console.log(`\n--- Test: ${name} ---`);
        console.log(`A: ${a.length} elements, B: ${b.length} elements`);

        const result = await tester.testCountKernel(a, b);
        const cpuCount = tester.cpuSetUnionCount(a, b);

        console.log(`GPU count: ${result.totalCount}`);
        console.log(`CPU count: ${cpuCount}`);
        console.log(`Match: ${result.totalCount === cpuCount ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU time: ${result.gpuTimeMs.toFixed(2)} ms`);

        if (result.totalCount !== cpuCount) {
            tester.printResults(result, a, b);
            const cpuPartitionCounts = tester.cpuPerPartitionCount(a, b, result.dpiData, result.numWorkgroups);
            console.log('CPU per-partition counts:', cpuPartitionCounts.slice(0, 10));
            console.log('GPU per-partition counts:', Array.from(result.counts).slice(0, 10));
        }
    }
}
