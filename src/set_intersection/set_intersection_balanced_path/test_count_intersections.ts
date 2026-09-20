import computeDiagonalsShader from './set_intersection.wgsl';
import countIntersectionsBinarySearchShader from './count_intersections_binary_search.wgsl';
import countIntersectionsMergeShader from './count_intersections_merge.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const MAX_DISPATCH_SIZE = 65535;

export type CountMethod = 'binary_search' | 'merge';

export class TestCountIntersections {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // Pipelines
    private computeDiagonalsPipeline: GPUComputePipeline;
    private countIntersectionsBinarySearchPipeline: GPUComputePipeline;
    private countIntersectionsMergePipeline: GPUComputePipeline;

    // Bind group layouts
    private diagonalsBindGroupLayout: GPUBindGroupLayout;
    private countBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // ========== compute_diagonals pipeline ==========
        this.diagonalsBindGroupLayout = device.createBindGroupLayout({
            label: 'compute diagonals bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        const diagonalsPipelineLayout = device.createPipelineLayout({
            label: 'compute diagonals pipeline layout',
            bindGroupLayouts: [this.diagonalsBindGroupLayout]
        });

        const diagonalsShader = device.createShaderModule({
            label: 'compute diagonals shader',
            code: computeDiagonalsShader
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals pipeline',
            layout: diagonalsPipelineLayout,
            compute: {
                module: diagonalsShader,
                entryPoint: 'compute_diagonals'
            }
        });

        // ========== count_intersections pipelines ==========
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'count intersections bind group layout',
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

        const countPipelineLayout = device.createPipelineLayout({
            label: 'count intersections pipeline layout',
            bindGroupLayouts: [this.countBindGroupLayout]
        });

        // Binary search version
        const binarySearchShader = device.createShaderModule({
            label: 'count intersections binary search shader',
            code: countIntersectionsBinarySearchShader
        });

        this.countIntersectionsBinarySearchPipeline = device.createComputePipeline({
            label: 'count intersections binary search pipeline',
            layout: countPipelineLayout,
            compute: {
                module: binarySearchShader,
                entryPoint: 'count_intersections'
            }
        });

        // Merge version
        const mergeShader = device.createShaderModule({
            label: 'count intersections merge shader',
            code: countIntersectionsMergeShader
        });

        this.countIntersectionsMergePipeline = device.createComputePipeline({
            label: 'count intersections merge pipeline',
            layout: countPipelineLayout,
            compute: {
                module: mergeShader,
                entryPoint: 'count_intersections'
            }
        });
    }

    /**
     * Run count_intersections test with specified method
     */
    public async testCountIntersections(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number,
        method: CountMethod = 'binary_search'
    ): Promise<{
        counts: Uint32Array;
        totalCount: number;
        aIndices: Uint32Array;
        bIndices: Uint32Array;
        stars: boolean[];
        diagonalsGpuTimeMs: number;
        countGpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // ========== Create buffers ==========
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

        const bufferC = device.createBuffer({
            label: 'Buffer C (unused)',
            size: 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // DPI buffer: size = 2 * (num_wg + 1)
        const dpiSize = 2 * (numWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Counts buffer
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
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

        const bufferNumWgTotal = device.createBuffer({
            label: 'num_wg_total uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWorkgroups]));

        // ========== Bind groups ==========
        const diagonalsBindGroup = device.createBindGroup({
            layout: this.diagonalsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferC } },
                { binding: 3, resource: { buffer: bufferDPI } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAX_DISPATCH_SIZE);
        const dispatchY = Math.ceil(numWorkgroups / MAX_DISPATCH_SIZE);

        // ========== Pass 1: compute_diagonals ==========
        const commandEncoder1 = device.createCommandEncoder({ label: 'Compute Diagonals' });
        const passDescriptor1 = this.timestampQueryManager.createComputePassDescriptor(0, 1);
        const pass1 = commandEncoder1.beginComputePass(passDescriptor1);
        pass1.setPipeline(this.computeDiagonalsPipeline);
        pass1.setBindGroup(0, diagonalsBindGroup);
        pass1.dispatchWorkgroups(dispatchX, dispatchY);
        pass1.end();
        this.timestampQueryManager.resolve(commandEncoder1);
        device.queue.submit([commandEncoder1.finish()]);
        await device.queue.onSubmittedWorkDone();

        let diagonalsGpuTimeMs = 0;
        try {
            const timestamps1 = await this.timestampQueryManager.downloadTimestampResult();
            diagonalsGpuTimeMs = (timestamps1[1] - timestamps1[0]) * 1e-6;
        } catch (e) { }

        // ========== Pass 2: count_intersections ==========
        const countPipeline = method === 'binary_search'
            ? this.countIntersectionsBinarySearchPipeline
            : this.countIntersectionsMergePipeline;

        const commandEncoder2 = device.createCommandEncoder({ label: 'Count Intersections' });
        const passDescriptor2 = this.timestampQueryManager.createComputePassDescriptor(0, 1);
        const pass2 = commandEncoder2.beginComputePass(passDescriptor2);
        pass2.setPipeline(countPipeline);
        pass2.setBindGroup(0, countBindGroup);
        pass2.dispatchWorkgroups(dispatchX, dispatchY);
        pass2.end();
        this.timestampQueryManager.resolve(commandEncoder2);
        device.queue.submit([commandEncoder2.finish()]);
        await device.queue.onSubmittedWorkDone();

        let countGpuTimeMs = 0;
        try {
            const timestamps2 = await this.timestampQueryManager.downloadTimestampResult();
            countGpuTimeMs = (timestamps2[1] - timestamps2[0]) * 1e-6;
        } catch (e) { }

        // ========== Read back results ==========
        // Read DPI
        const readbackDPI = device.createBuffer({
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        // Read counts
        const readbackCounts = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferDPI, 0, readbackDPI, 0, dpiSize * Uint32Array.BYTES_PER_ELEMENT);
        copyEncoder.copyBufferToBuffer(bufferCounts, 0, readbackCounts, 0, numWorkgroups * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackDPI.mapAsync(GPUMapMode.READ);
        const dpiData = new Uint32Array(readbackDPI.getMappedRange().slice(0));
        readbackDPI.unmap();

        await readbackCounts.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(readbackCounts.getMappedRange().slice(0));
        readbackCounts.unmap();

        // Parse DPI
        const aIndices = new Uint32Array(numWorkgroups + 1);
        const bIndices = new Uint32Array(numWorkgroups + 1);
        const stars: boolean[] = [];

        for (let i = 0; i <= numWorkgroups; i++) {
            const packedA = dpiData[i];
            aIndices[i] = packedA & ~STAR_MASK;
            stars.push((packedA & STAR_MASK) !== 0);
            bIndices[i] = dpiData[numWorkgroups + 1 + i];
        }

        // Calculate total count
        let totalCount = 0;
        for (let i = 0; i < numWorkgroups; i++) {
            totalCount += counts[i];
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferC.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackDPI.destroy();
        readbackCounts.destroy();

        return {
            counts,
            totalCount,
            aIndices,
            bIndices,
            stars,
            diagonalsGpuTimeMs,
            countGpuTimeMs
        };
    }

    /**
     * CPU reference implementation for validation (MULTISET intersection)
     * For each value v: output min(count_A(v), count_B(v)) copies
     */
    public cpuSetIntersection(setA: Uint32Array, setB: Uint32Array): number {
        // Count occurrences in A
        const countA = new Map<number, number>();
        for (const val of setA) {
            countA.set(val, (countA.get(val) || 0) + 1);
        }

        // Count occurrences in B
        const countB = new Map<number, number>();
        for (const val of setB) {
            countB.set(val, (countB.get(val) || 0) + 1);
        }

        // Sum min(countA, countB) for each unique value
        let total = 0;
        for (const [val, cntA] of countA) {
            const cntB = countB.get(val) || 0;
            total += Math.min(cntA, cntB);
        }
        return total;
    }

    /**
     * Print results and validate
     */
    public printResults(
        setA: Uint32Array,
        setB: Uint32Array,
        counts: Uint32Array,
        totalCount: number,
        aIndices: Uint32Array,
        bIndices: Uint32Array,
        stars: boolean[],
        method: CountMethod
    ): boolean {
        const numWorkgroups = counts.length;

        console.log(`\n=== Count Intersections Results (${method}) ===`);
        console.log(`Input sizes: A=${setA.length}, B=${setB.length}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);
        console.log(`Total intersection count (GPU): ${totalCount}`);

        // CPU validation
        const cpuCount = this.cpuSetIntersection(setA, setB);
        console.log(`Total intersection count (CPU): ${cpuCount}`);

        const valid = totalCount === cpuCount;
        if (valid) {
            console.log('VALID: GPU result matches CPU reference');
        } else {
            console.log(`ERROR: GPU (${totalCount}) != CPU (${cpuCount})`);
        }

        // Print per-partition counts (only for small number of partitions)
        if (numWorkgroups <= 16) {
            console.log('\nPer-partition counts:');
            console.log('Partition | aRange         | bRange         | Star | Count | A values        | B values');
            console.log('----------|----------------|----------------|------|-------|-----------------|----------------');
            for (let i = 0; i < numWorkgroups; i++) {
                const aStart = aIndices[i];
                const aEnd = aIndices[i + 1];
                const bStart = bIndices[i];
                const bEnd = bIndices[i + 1];
                const star = stars[i] ? '*' : ' ';

                // Show actual values in partition (for small arrays)
                const aVals = setA.length <= 20 ? Array.from(setA.slice(aStart, aEnd)).join(',') : `${aEnd - aStart} elems`;
                const bVals = setB.length <= 20 ? Array.from(setB.slice(bStart, bEnd)).join(',') : `${bEnd - bStart} elems`;

                console.log(
                    `${String(i).padStart(9)} | ` +
                    `[${String(aStart).padStart(5)},${String(aEnd).padStart(5)}) | ` +
                    `[${String(bStart).padStart(5)},${String(bEnd).padStart(5)}) | ` +
                    `${star.padStart(4)} | ` +
                    `${String(counts[i]).padStart(5)} | ` +
                    `${aVals.padEnd(15)} | ` +
                    `${bVals}`
                );
            }

            // Calculate expected count per partition using CPU
            console.log('\nExpected counts per partition (CPU):');
            let cpuPartitionTotal = 0;
            for (let i = 0; i < numWorkgroups; i++) {
                const aStart = aIndices[i];
                const aEnd = aIndices[i + 1];
                const bStart = bIndices[i];
                const bEnd = bIndices[i + 1];

                const aPartition = setA.slice(aStart, aEnd);
                const bPartition = setB.slice(bStart, bEnd);
                const expected = this.cpuSetIntersection(aPartition, bPartition);
                cpuPartitionTotal += expected;

                console.log(`  Partition ${i}: expected=${expected}, got=${counts[i]}, diff=${counts[i] - expected}`);
            }
            console.log(`  Sum of partition expectations: ${cpuPartitionTotal}`);
            console.log(`  Note: This sum may exceed total unique intersections due to duplicates across partitions`);
        }

        console.log('==========================================\n');
        return valid;
    }
}

/**
 * Run comparison test between binary search and merge methods
 */
export async function runCountIntersectionsComparison(
    device: GPUDevice,
    timestampQueryManager: TimestampQueryManager,
    setA: Uint32Array,
    setB: Uint32Array,
    numWorkgroups: number
): Promise<void> {
    const tester = new TestCountIntersections(device, timestampQueryManager);

    console.log('\n========== Count Intersections Comparison ==========');
    console.log(`A size: ${setA.length}, B size: ${setB.length}, Workgroups: ${numWorkgroups}`);

    // Test binary search method
    const bsResult = await tester.testCountIntersections(setA, setB, numWorkgroups, 'binary_search');
    const bsValid = tester.printResults(
        setA, setB, bsResult.counts, bsResult.totalCount,
        bsResult.aIndices, bsResult.bIndices, bsResult.stars, 'binary_search'
    );

    // Test merge method
    const mergeResult = await tester.testCountIntersections(setA, setB, numWorkgroups, 'merge');
    const mergeValid = tester.printResults(
        setA, setB, mergeResult.counts, mergeResult.totalCount,
        mergeResult.aIndices, mergeResult.bIndices, mergeResult.stars, 'merge'
    );

    // Timing comparison
    console.log('=== Performance Comparison ===');
    console.log(`Method         | Diagonals (ms) | Count (ms) | Total (ms)`);
    console.log('---------------|----------------|------------|----------');
    console.log(
        `Binary Search  | ${bsResult.diagonalsGpuTimeMs.toFixed(4).padStart(14)} | ` +
        `${bsResult.countGpuTimeMs.toFixed(4).padStart(10)} | ` +
        `${(bsResult.diagonalsGpuTimeMs + bsResult.countGpuTimeMs).toFixed(4).padStart(8)}`
    );
    console.log(
        `Merge          | ${mergeResult.diagonalsGpuTimeMs.toFixed(4).padStart(14)} | ` +
        `${mergeResult.countGpuTimeMs.toFixed(4).padStart(10)} | ` +
        `${(mergeResult.diagonalsGpuTimeMs + mergeResult.countGpuTimeMs).toFixed(4).padStart(8)}`
    );

    // Result comparison
    if (bsResult.totalCount === mergeResult.totalCount) {
        console.log(`\nBoth methods agree: ${bsResult.totalCount} intersections`);
    } else {
        console.log(`\nWARNING: Methods disagree! Binary Search: ${bsResult.totalCount}, Merge: ${mergeResult.totalCount}`);
    }

    console.log('====================================================\n');
}
