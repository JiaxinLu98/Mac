/**
 * Test file for Intersection By Key Count Kernel (set_availability_intersection_by_key_count_v1.wgsl)
 *
 * Tests the count phase of the ModernGPU-style intersection by key algorithm.
 * For intersection by key:
 * - Compare keys only
 * - Count matches (when A_key == B_key)
 * - min(m,n) semantics for duplicates
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_intersection_by_key_count_v1.wgsl';
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
 * Test class for the intersection by key count kernel.
 */
export class TestIntersectionByKeyCountKernel {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI computation pipeline (uses keys only)
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

        // Create DPI bind group layout (uses keys only)
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

        // Create count kernel bind group layout (7 bindings, no debug buffer)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Intersection By Key Count kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // counts
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // a_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg_total
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: 'Intersection By Key Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });
    }

    /**
     * Run the count kernel test.
     * Note: Values are not needed for count phase, but we accept them for interface consistency.
     */
    public async testCountKernel(
        keysA: Uint32Array,
        valuesA: Uint32Array,
        keysB: Uint32Array,
        valuesB: Uint32Array
    ): Promise<{
        counts: Uint32Array;
        totalCount: number;
        dpiData: Uint32Array;
        numWorkgroups: number;
        gpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = keysA.length;
        const b_len = keysB.length;
        const total = a_len + b_len;

        if (total === 0) {
            throw new Error('Cannot test count kernel with empty arrays');
        }

        const numWg = Math.ceil(total / NV);

        // Create GPU buffers for keys only (values not needed for count)
        const bufferAKeys = device.createBuffer({
            label: 'Buffer A Keys',
            size: Math.max(4, keysA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(keysA));

        const bufferBKeys = device.createBuffer({
            label: 'Buffer B Keys',
            size: Math.max(4, keysB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(keysB));

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

        // ============ Step 1: Compute DPI (uses keys only) ============
        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
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
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
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
        bufferAKeys.destroy();
        bufferBKeys.destroy();
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
     * CPU reference: intersection by key count.
     * Returns total count of matching keys (min(m,n) for duplicates).
     * Note: Only keys are compared, values are ignored for counting.
     */
    public cpuIntersectionByKeyCount(keysA: Uint32Array, keysB: Uint32Array): number {
        let count = 0;
        let ai = 0, bi = 0;
        while (ai < keysA.length && bi < keysB.length) {
            if (keysA[ai] < keysB[bi]) {
                ai++;
            } else if (keysA[ai] > keysB[bi]) {
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
     * Validate GPU count against CPU reference.
     */
    public validateCount(gpuCount: number, keysA: Uint32Array, keysB: Uint32Array): boolean {
        const cpuCount = this.cpuIntersectionByKeyCount(keysA, keysB);
        console.log(`GPU count: ${gpuCount}, CPU count: ${cpuCount}`);
        return gpuCount === cpuCount;
    }
}

/**
 * Run intersection by key count kernel tests.
 */
export async function runIntersectionByKeyCountKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║     INTERSECTION BY KEY COUNT KERNEL TEST                  ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestIntersectionByKeyCountKernel(device, timestampQueryManager);

    let allPassed = true;

    // Test cases
    const testCases = [
        {
            name: 'Simple intersection by key',
            keysA: new Uint32Array([1, 3, 5, 7, 9]),
            valuesA: new Uint32Array([10, 30, 50, 70, 90]),
            keysB: new Uint32Array([2, 3, 4, 7, 8]),
            valuesB: new Uint32Array([20, 31, 40, 71, 80]),
            // Keys intersection: [3, 7] => count = 2
        },
        {
            name: 'No intersection',
            keysA: new Uint32Array([1, 3, 5]),
            valuesA: new Uint32Array([10, 30, 50]),
            keysB: new Uint32Array([2, 4, 6]),
            valuesB: new Uint32Array([20, 40, 60]),
            // No matching keys => count = 0
        },
        {
            name: 'Complete intersection',
            keysA: new Uint32Array([1, 2, 3, 4, 5]),
            valuesA: new Uint32Array([10, 20, 30, 40, 50]),
            keysB: new Uint32Array([1, 2, 3, 4, 5]),
            valuesB: new Uint32Array([11, 21, 31, 41, 51]),
            // All keys match => count = 5
        },
        {
            name: 'Duplicates with intersection',
            keysA: new Uint32Array([1, 1, 2, 2, 3, 3]),
            valuesA: new Uint32Array([10, 11, 20, 21, 30, 31]),
            keysB: new Uint32Array([1, 2, 2, 2, 3]),
            valuesB: new Uint32Array([12, 22, 23, 24, 32]),
            // Key 1: min(2,1)=1, Key 2: min(2,3)=2, Key 3: min(2,1)=1 => count = 4
        },
        {
            name: 'Large no overlap (even vs odd keys)',
            keysA: new Uint32Array(1000).map((_, i) => i * 2),
            valuesA: new Uint32Array(1000).map((_, i) => i * 2 + 1000),
            keysB: new Uint32Array(1000).map((_, i) => i * 2 + 1),
            valuesB: new Uint32Array(1000).map((_, i) => i * 2 + 2000),
            // No overlap => count = 0
        },
        {
            name: 'Large with partial overlap',
            keysA: new Uint32Array(1000).map((_, i) => i),
            valuesA: new Uint32Array(1000).map((_, i) => i + 1000),
            keysB: new Uint32Array(1000).map((_, i) => i + 500),
            valuesB: new Uint32Array(1000).map((_, i) => i + 2000),
            // Overlap: 500-999 => count = 500
        },
        {
            name: 'Multi-workgroup test',
            keysA: new Uint32Array(3000).map((_, i) => i * 2),
            valuesA: new Uint32Array(3000).map((_, i) => i),
            keysB: new Uint32Array(3000).map((_, i) => i * 3),
            valuesB: new Uint32Array(3000).map((_, i) => i + 10000),
            // Intersection: multiples of 6
        },
    ];

    for (const { name, keysA, valuesA, keysB, valuesB } of testCases) {
        console.log(`\n--- Test: ${name} ---`);
        console.log(`A: ${keysA.length} key-value pairs, B: ${keysB.length} key-value pairs`);

        const result = await tester.testCountKernel(keysA, valuesA, keysB, valuesB);
        const cpuCount = tester.cpuIntersectionByKeyCount(keysA, keysB);

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
        console.log('✔ ALL INTERSECTION BY KEY COUNT TESTS PASSED');
    } else {
        console.log('✗ SOME TESTS FAILED');
    }
    console.log('═'.repeat(60) + '\n');
}
