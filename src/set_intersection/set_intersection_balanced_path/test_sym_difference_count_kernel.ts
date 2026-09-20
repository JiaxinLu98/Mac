/**
 * Test file for Symmetric Difference Count Kernel (set_availability_sym_difference_count_v1.wgsl)
 *
 * Tests the count phase of the ModernGPU-style symmetric difference algorithm.
 * Symmetric difference ((A \ B) ∪ (B \ A)):
 * - If A < B: emit A, advance A
 * - If B < A: emit B, advance B
 * - If A == B: advance both, no emit (element in both)
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_sym_difference_count_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Test class for the symmetric difference count kernel.
 */
export class TestSymDifferenceCountKernel {
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

        // Create count kernel bind group layout (no debug buffer for symmetric difference)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Symmetric Difference Count kernel bind group layout',
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
            label: 'Symmetric Difference Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });
    }

    /**
     * Run the count kernel test.
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

    /**
     * CPU reference: symmetric difference count.
     * Returns total count of elements in (A \ B) ∪ (B \ A).
     */
    public cpuSymDifferenceCount(a: Uint32Array, b: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < a.length && bi < b.length) {
            if (a[ai] < b[bi]) {
                count++;  // A not in B
                ai++;
            } else if (a[ai] > b[bi]) {
                count++;  // B not in A
                bi++;
            } else {
                // Equal: in both, skip
                ai++;
                bi++;
            }
        }
        // Remaining elements
        count += (a.length - ai) + (b.length - bi);
        return count;
    }

    /**
     * Validate GPU count against CPU reference.
     */
    public validateCount(gpuCount: number, a: Uint32Array, b: Uint32Array): boolean {
        const cpuCount = this.cpuSymDifferenceCount(a, b);
        console.log(`GPU count: ${gpuCount}, CPU count: ${cpuCount}`);
        return gpuCount === cpuCount;
    }
}

/**
 * Run symmetric difference count kernel tests.
 */
export async function runSymDifferenceCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║     SYMMETRIC DIFFERENCE COUNT KERNEL TEST                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSymDifferenceCountKernel(device, timestampQueryManager);

    let allPassed = true;

    // Test cases
    const testCases = [
        {
            name: 'Simple symmetric difference',
            a: new Uint32Array([1, 3, 5, 7, 9]),
            b: new Uint32Array([2, 3, 4, 7, 8, 10]),
            // A \ B = [1, 5, 9], B \ A = [2, 4, 8, 10] => count = 7
        },
        {
            name: 'No overlap (all different)',
            a: new Uint32Array([1, 2, 3]),
            b: new Uint32Array([4, 5, 6]),
            // Sym diff = all elements => count = 6
        },
        {
            name: 'Complete overlap (all same)',
            a: new Uint32Array([1, 2, 3, 4, 5]),
            b: new Uint32Array([1, 2, 3, 4, 5]),
            // Sym diff = empty => count = 0
        },
        {
            name: 'Large no overlap (even vs odd)',
            a: new Uint32Array(1000).map((_, i) => i * 2),      // 0, 2, 4, ...
            b: new Uint32Array(1000).map((_, i) => i * 2 + 1),  // 1, 3, 5, ...
            // No overlap => count = 2000
        },
        {
            name: 'Large with partial overlap',
            a: new Uint32Array(1000).map((_, i) => i),          // 0, 1, 2, ..., 999
            b: new Uint32Array(1000).map((_, i) => i + 500),    // 500, 501, ..., 1499
            // Overlap: 500-999 => sym diff = 0-499 + 1000-1499 => count = 1000
        },
        {
            name: 'Duplicates with overlap',
            a: new Uint32Array([1, 1, 2, 2, 3, 3]),
            b: new Uint32Array([2, 2, 3, 3, 4, 4]),
            // A \ B = [1, 1], B \ A = [4, 4] => count = 4
        },
        {
            name: 'Multi-workgroup test',
            a: new Uint32Array(3000).map((_, i) => i * 2),
            b: new Uint32Array(3000).map((_, i) => i * 3),
            // Sym diff excludes multiples of 6
        },
    ];

    for (const { name, a, b } of testCases) {
        console.log(`\n--- Test: ${name} ---`);
        console.log(`A: ${a.length} elements, B: ${b.length} elements`);

        const result = await tester.testCountKernel(a, b);
        const cpuCount = tester.cpuSymDifferenceCount(a, b);

        console.log(`GPU count: ${result.totalCount}`);
        console.log(`CPU count: ${cpuCount}`);
        console.log(`Match: ${result.totalCount === cpuCount ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU time: ${result.gpuTimeMs.toFixed(2)} ms`);

        if (result.totalCount !== cpuCount) {
            allPassed = false;
        }
    }

    console.log('\n' + '═'.repeat(60));
    if (allPassed) {
        console.log('✔ ALL SYMMETRIC DIFFERENCE COUNT TESTS PASSED');
    } else {
        console.log('✗ SOME TESTS FAILED');
    }
    console.log('═'.repeat(60) + '\n');
}
