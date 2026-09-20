import computeDiagonalsBiasedShader from './balanced_path_biased.wgsl';
import setIntersectionBalancedShader from './set_intersection_balanced.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants - must match balanced_path_biased.wgsl
const NT = 256;          // Threads per workgroup
const VT = 7;            // Values per thread
const NV = NT * VT;      // Elements per workgroup = 1792

/**
 * Calculate the number of workgroups using ModernGPU's formula.
 * num_wg = ceil((aLen + bLen) / NV)
 */
export function calculateNumWorkgroups(aLen: number, bLen: number): number {
    return Math.ceil((aLen + bLen) / NV);
}

/**
 * Test class for the full Balanced Path set intersection implementation.
 *
 * This implementation follows the ModernGPU pattern:
 * 1. ComputeDiagonals: partition work across workgroups (using biased binary search)
 * 2. Cooperative data loading into shared memory
 * 3. (TODO) Local BalancedPath: each thread finds its starting position
 * 4. (TODO) SerialSetIntersection: each thread processes VT elements
 *
 * DPI buffer layout:
 *   dpi[0 .. num_wg]           : packed aIndex (MSB = star flag)
 *   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
 */
export class TestSetIntersectionBalanced {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private computeDiagonalsPipeline: GPUComputePipeline;
    private countIntersectionsPipeline: GPUComputePipeline;
    private bindGroupLayoutDiagonals: GPUBindGroupLayout;
    private bindGroupLayoutCount: GPUBindGroupLayout;

    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // ========== Compute Diagonals Pipeline (from balanced_path_biased.wgsl) ==========
        this.bindGroupLayoutDiagonals = device.createBindGroupLayout({
            label: 'compute diagonals biased bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_uniform
            ]
        });

        const pipelineLayoutDiagonals = device.createPipelineLayout({
            label: 'compute diagonals pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutDiagonals]
        });

        const shaderDiagonals = device.createShaderModule({
            label: 'compute diagonals biased shader',
            code: computeDiagonalsBiasedShader
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals biased pipeline',
            layout: pipelineLayoutDiagonals,
            compute: {
                module: shaderDiagonals,
                entryPoint: 'compute_diagonals'
            }
        });

        // ========== Count Intersections Pipeline (from set_intersection_balanced.wgsl) ==========
        this.bindGroupLayoutCount = device.createBindGroupLayout({
            label: 'count intersections balanced bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // counts
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const pipelineLayoutCount = device.createPipelineLayout({
            label: 'count intersections balanced pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutCount]
        });

        const shaderCount = device.createShaderModule({
            label: 'count intersections balanced shader',
            code: setIntersectionBalancedShader
        });

        this.countIntersectionsPipeline = device.createComputePipeline({
            label: 'count intersections balanced pipeline',
            layout: pipelineLayoutCount,
            compute: {
                module: shaderCount,
                entryPoint: 'count_intersections_balanced'
            }
        });
    }

    public setIterationIndex(i: number): void {
        this.iterationIndex = i;
    }

    public computeQueries(): number {
        // compute_diagonals: 2 + count_intersections: 2 = 4
        return this.queriesPerIter = 4;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries() must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Test set intersection using the full balanced path approach.
     *
     * @param setA - Sorted input array A
     * @param setB - Sorted input array B
     * @param actualNumWorkgroups - Number of partitions (optional, auto-calculated if not provided)
     * @returns Intersection count and timing information
     */
    public async testCountIntersections(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups?: number
    ): Promise<{
        totalCount: number;
        perWorkgroupCounts: Uint32Array;
        gpuTimeMs: number;
        computeDiagonalsTimeMs: number;
        countIntersectionsTimeMs: number;
        numWorkgroups: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // Auto-calculate numWorkgroups using ModernGPU formula: ceil(total / NV)
        const actualNumWorkgroups = numWorkgroups ?? calculateNumWorkgroups(a_len, b_len);

        // Initialize for single iteration
        this.computeQueries();
        this.setIterationIndex(0);

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

        // DPI buffer: size = 2 * (num_wg + 1)
        const dpiSize = 2 * (actualNumWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Counts buffer: one count per workgroup
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: actualNumWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform buffers
        const bufferALength = device.createBuffer({
            label: 'a_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

        const bufferBLength = device.createBuffer({
            label: 'b_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

        const bufferNumWg = device.createBuffer({
            label: 'num_wg uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([actualNumWorkgroups]));

        // Create bind groups
        const bindGroupDiagonals = device.createBindGroup({
            label: 'compute diagonals bind group',
            layout: this.bindGroupLayoutDiagonals,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const bindGroupCount = device.createBindGroup({
            label: 'count intersections bind group',
            layout: this.bindGroupLayoutCount,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        // Calculate dispatch dimensions
        const wgSize = 32;  // compute_diagonals uses workgroup_size(32)
        const dispatchX = Math.min(actualNumWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(actualNumWorkgroups / MAXWORKGROUP);

        // Calculate dispatch for count kernel (256 threads per workgroup)
        const countDispatchX = Math.min(actualNumWorkgroups, MAXWORKGROUP);
        const countDispatchY = Math.ceil(actualNumWorkgroups / MAXWORKGROUP);

        // Readback buffers
        const readbackCounts = device.createBuffer({
            label: 'readback counts',
            size: actualNumWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Encode commands
        const commandEncoder = device.createCommandEncoder({ label: 'test encoder' });
        const baseOffset = this.getQueryBaseOffset();

        // Pass 1: Compute Diagonals
        const passDescDiagonals = this.timestampQueryManager.createComputePassDescriptor(baseOffset, baseOffset + 1);
        const passDiagonals = commandEncoder.beginComputePass({
            label: 'compute diagonals pass',
            ...passDescDiagonals
        });
        passDiagonals.setPipeline(this.computeDiagonalsPipeline);
        passDiagonals.setBindGroup(0, bindGroupDiagonals);
        passDiagonals.dispatchWorkgroups(dispatchX, dispatchY, 1);
        passDiagonals.end();

        // Pass 2: Count Intersections
        const passDescCount = this.timestampQueryManager.createComputePassDescriptor(baseOffset + 2, baseOffset + 3);
        const passCount = commandEncoder.beginComputePass({
            label: 'count intersections pass',
            ...passDescCount
        });
        passCount.setPipeline(this.countIntersectionsPipeline);
        passCount.setBindGroup(0, bindGroupCount);
        passCount.dispatchWorkgroups(countDispatchX, countDispatchY, 1);
        passCount.end();

        // Copy results
        commandEncoder.copyBufferToBuffer(bufferCounts, 0, readbackCounts, 0, actualNumWorkgroups * 4);

        // Resolve timestamps
        this.timestampQueryManager.resolve(commandEncoder);

        // Submit
        device.queue.submit([commandEncoder.finish()]);

        // Read back results
        await readbackCounts.mapAsync(GPUMapMode.READ);
        const countsData = new Uint32Array(readbackCounts.getMappedRange().slice(0));
        readbackCounts.unmap();

        // Calculate total count
        let totalCount = 0;
        for (let i = 0; i < countsData.length; i++) {
            totalCount += countsData[i];
        }

        // Read timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();
        const computeDiagonalsTimeMs = (timestamps[baseOffset + 1] - timestamps[baseOffset]) / 1_000_000;
        const countIntersectionsTimeMs = (timestamps[baseOffset + 3] - timestamps[baseOffset + 2]) / 1_000_000;
        const gpuTimeMs = computeDiagonalsTimeMs + countIntersectionsTimeMs;

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();
        readbackCounts.destroy();

        return {
            totalCount,
            perWorkgroupCounts: countsData,
            gpuTimeMs,
            computeDiagonalsTimeMs,
            countIntersectionsTimeMs,
            numWorkgroups: actualNumWorkgroups
        };
    }

    /**
     * Benchmark set intersection with multiple iterations.
     */
    public async benchmarkCountIntersections(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number,
        warmupIters: number = 5,
        numWorkgroups?: number
    ): Promise<{
        totalCount: number;
        avgGpuTimeMs: number;
        avgComputeDiagonalsTimeMs: number;
        avgCountIntersectionsTimeMs: number;
        numWorkgroups: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // Auto-calculate numWorkgroups using ModernGPU formula: ceil(total / NV)
        const actualNumWorkgroups = numWorkgroups ?? calculateNumWorkgroups(a_len, b_len);

        // Create GPU buffers (reused across iterations)
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

        const dpiSize = 2 * (actualNumWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: actualNumWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferALength = device.createBuffer({
            label: 'a_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

        const bufferBLength = device.createBuffer({
            label: 'b_length uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

        const bufferNumWg = device.createBuffer({
            label: 'num_wg uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([actualNumWorkgroups]));

        // Create bind groups
        const bindGroupDiagonals = device.createBindGroup({
            label: 'compute diagonals bind group',
            layout: this.bindGroupLayoutDiagonals,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const bindGroupCount = device.createBindGroup({
            label: 'count intersections bind group',
            layout: this.bindGroupLayoutCount,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        // Dispatch dimensions
        const dispatchX = Math.min(actualNumWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(actualNumWorkgroups / MAXWORKGROUP);
        const countDispatchX = Math.min(actualNumWorkgroups, MAXWORKGROUP);
        const countDispatchY = Math.ceil(actualNumWorkgroups / MAXWORKGROUP);

        // Warmup
        for (let w = 0; w < warmupIters; w++) {
            const commandEncoder = device.createCommandEncoder();
            const passDiagonals = commandEncoder.beginComputePass();
            passDiagonals.setPipeline(this.computeDiagonalsPipeline);
            passDiagonals.setBindGroup(0, bindGroupDiagonals);
            passDiagonals.dispatchWorkgroups(dispatchX, dispatchY, 1);
            passDiagonals.end();

            const passCount = commandEncoder.beginComputePass();
            passCount.setPipeline(this.countIntersectionsPipeline);
            passCount.setBindGroup(0, bindGroupCount);
            passCount.dispatchWorkgroups(countDispatchX, countDispatchY, 1);
            passCount.end();

            device.queue.submit([commandEncoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Benchmark iterations
        this.computeQueries();
        let totalComputeDiagonalsTime = 0;
        let totalCountTime = 0;
        let totalCount = 0;

        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const baseOffset = this.getQueryBaseOffset();

            const commandEncoder = device.createCommandEncoder();

            const passDescDiagonals = this.timestampQueryManager.createComputePassDescriptor(baseOffset, baseOffset + 1);
            const passDiagonals = commandEncoder.beginComputePass(passDescDiagonals);
            passDiagonals.setPipeline(this.computeDiagonalsPipeline);
            passDiagonals.setBindGroup(0, bindGroupDiagonals);
            passDiagonals.dispatchWorkgroups(dispatchX, dispatchY, 1);
            passDiagonals.end();

            const passDescCount = this.timestampQueryManager.createComputePassDescriptor(baseOffset + 2, baseOffset + 3);
            const passCount = commandEncoder.beginComputePass(passDescCount);
            passCount.setPipeline(this.countIntersectionsPipeline);
            passCount.setBindGroup(0, bindGroupCount);
            passCount.dispatchWorkgroups(countDispatchX, countDispatchY, 1);
            passCount.end();

            this.timestampQueryManager.resolve(commandEncoder);
            device.queue.submit([commandEncoder.finish()]);
        }

        // Read timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();
        for (let i = 0; i < iters; i++) {
            const baseOffset = i * this.queriesPerIter;
            totalComputeDiagonalsTime += timestamps[baseOffset + 1] - timestamps[baseOffset];
            totalCountTime += timestamps[baseOffset + 3] - timestamps[baseOffset + 2];
        }

        // Read final count
        const readbackCounts = device.createBuffer({
            size: actualNumWorkgroups * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferCounts, 0, readbackCounts, 0, actualNumWorkgroups * 4);
        device.queue.submit([copyEncoder.finish()]);

        await readbackCounts.mapAsync(GPUMapMode.READ);
        const countsData = new Uint32Array(readbackCounts.getMappedRange().slice(0));
        readbackCounts.unmap();

        for (let i = 0; i < countsData.length; i++) {
            totalCount += countsData[i];
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();
        readbackCounts.destroy();

        return {
            totalCount,
            avgGpuTimeMs: (totalComputeDiagonalsTime + totalCountTime) / iters / 1_000_000,
            avgComputeDiagonalsTimeMs: totalComputeDiagonalsTime / iters / 1_000_000,
            avgCountIntersectionsTimeMs: totalCountTime / iters / 1_000_000,
            numWorkgroups: actualNumWorkgroups
        };
    }
}
