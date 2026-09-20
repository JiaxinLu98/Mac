import computeDiagonalsShader from './set_intersection.wgsl';
import countIntersectionsMergeShader from './count_intersections_merge.wgsl';
import { ExclusiveScanPipeline } from '../set_intersection_binary_search/prefix_sum/exclusive_scan';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const MAX_DISPATCH_SIZE = 65535;
const BLOCK_SIZE = 512;

/**
 * Aligns a value to the next multiple of alignment
 */
function alignTo(val: number, align: number): number {
    return Math.floor((val + align - 1) / align) * align;
}

export class TestPrefixSum {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // Pipelines
    private computeDiagonalsPipeline: GPUComputePipeline;
    private countIntersectionsPipeline: GPUComputePipeline;

    // Bind group layouts
    private diagonalsBindGroupLayout: GPUBindGroupLayout;
    private countBindGroupLayout: GPUBindGroupLayout;

    // Reusable ExclusiveScanPipeline
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // Create ExclusiveScanPipeline (reuse existing implementation)
        this.scanPipeline = new ExclusiveScanPipeline(device);

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

        // ========== count_intersections pipeline ==========
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'count intersections bind group layout',
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

        const countPipelineLayout = device.createPipelineLayout({
            label: 'count intersections pipeline layout',
            bindGroupLayouts: [this.countBindGroupLayout]
        });

        const countShader = device.createShaderModule({
            label: 'count intersections merge shader',
            code: countIntersectionsMergeShader
        });

        this.countIntersectionsPipeline = device.createComputePipeline({
            label: 'count intersections pipeline',
            layout: countPipelineLayout,
            compute: {
                module: countShader,
                entryPoint: 'count_intersections'
            }
        });
    }

    /**
     * Run the full pipeline: compute_diagonals -> count_intersections -> prefix_sum
     */
    public async testPrefixSum(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number
    ): Promise<{
        counts: Uint32Array;
        offsets: Uint32Array;
        totalCount: number;
        diagonalsGpuTimeMs: number;
        countGpuTimeMs: number;
        prefixSumGpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // Align size for prefix sum (must be multiple of BLOCK_SIZE)
        const alignedSize = alignTo(numWorkgroups, BLOCK_SIZE);

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

        // Counts buffer (aligned for prefix sum, will be transformed in-place)
        // Use the ExclusiveScanPipeline's aligned size requirement
        const scanAlignedSize = this.scanPipeline.getAlignedSize(numWorkgroups);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: scanAlignedSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
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
        const commandEncoder2 = device.createCommandEncoder({ label: 'Count Intersections' });
        const passDescriptor2 = this.timestampQueryManager.createComputePassDescriptor(0, 1);
        const pass2 = commandEncoder2.beginComputePass(passDescriptor2);
        pass2.setPipeline(this.countIntersectionsPipeline);
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

        // Read counts before prefix sum (for validation)
        const readbackCountsBefore = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const copyEncoderCounts = device.createCommandEncoder();
        copyEncoderCounts.copyBufferToBuffer(bufferCounts, 0, readbackCountsBefore, 0, numWorkgroups * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoderCounts.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackCountsBefore.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(readbackCountsBefore.getMappedRange().slice(0));
        readbackCountsBefore.unmap();

        // ========== Pass 3: prefix_sum using ExclusiveScanPipeline ==========
        // Prepare the scanner with the GPU buffer
        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, scanAlignedSize);

        // Run the scan with timestamp
        await scanner.scan(numWorkgroups, this.timestampQueryManager, 0);

        let prefixSumGpuTimeMs = 0;
        try {
            const timestamps3 = await this.timestampQueryManager.downloadTimestampResult();
            prefixSumGpuTimeMs = (timestamps3[1] - timestamps3[0]) * 1e-6;
        } catch (e) { }

        // ========== Read back results ==========
        // Read offsets (prefix sum result)
        const readbackOffsets = device.createBuffer({
            size: scanAlignedSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferCounts, 0, readbackOffsets, 0, scanAlignedSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackOffsets.mapAsync(GPUMapMode.READ);
        const offsetsData = new Uint32Array(readbackOffsets.getMappedRange().slice(0));
        readbackOffsets.unmap();

        // Extract only the relevant offsets (numWorkgroups elements)
        const offsets = offsetsData.slice(0, numWorkgroups);

        // Calculate total count from counts
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
        readbackCountsBefore.destroy();
        readbackOffsets.destroy();

        return {
            counts,
            offsets,
            totalCount,
            diagonalsGpuTimeMs,
            countGpuTimeMs,
            prefixSumGpuTimeMs
        };
    }

    /**
     * CPU reference implementation for exclusive prefix sum
     */
    public cpuExclusiveScan(input: Uint32Array): Uint32Array {
        const output = new Uint32Array(input.length);
        let sum = 0;
        for (let i = 0; i < input.length; i++) {
            output[i] = sum;
            sum += input[i];
        }
        return output;
    }

    /**
     * CPU reference implementation for multiset intersection count
     */
    public cpuSetIntersection(setA: Uint32Array, setB: Uint32Array): number {
        const countA = new Map<number, number>();
        for (const val of setA) {
            countA.set(val, (countA.get(val) || 0) + 1);
        }

        const countB = new Map<number, number>();
        for (const val of setB) {
            countB.set(val, (countB.get(val) || 0) + 1);
        }

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
        offsets: Uint32Array,
        totalCount: number
    ): boolean {
        const numWorkgroups = counts.length;

        console.log(`\n=== Prefix Sum Test Results ===`);
        console.log(`Input sizes: A=${setA.length}, B=${setB.length}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);

        // CPU validation of prefix sum
        const expectedOffsets = this.cpuExclusiveScan(counts);

        let prefixSumValid = true;
        for (let i = 0; i < numWorkgroups; i++) {
            if (offsets[i] !== expectedOffsets[i]) {
                prefixSumValid = false;
                console.log(`  ERROR at index ${i}: expected=${expectedOffsets[i]}, got=${offsets[i]}`);
            }
        }

        if (prefixSumValid) {
            console.log('Prefix Sum: VALID');
        } else {
            console.log('Prefix Sum: ERROR');
        }

        // CPU validation of total count
        const cpuTotalCount = this.cpuSetIntersection(setA, setB);
        const countValid = totalCount === cpuTotalCount;

        console.log(`Total intersection count (GPU): ${totalCount}`);
        console.log(`Total intersection count (CPU): ${cpuTotalCount}`);
        console.log(`Count: ${countValid ? 'VALID' : 'ERROR'}`);

        // Print per-partition details (only for small number of partitions)
        if (numWorkgroups <= 16) {
            console.log('\nPer-partition counts and offsets:');
            console.log('Partition | Count | Offset | Expected Offset');
            console.log('----------|-------|--------|----------------');
            for (let i = 0; i < numWorkgroups; i++) {
                const match = offsets[i] === expectedOffsets[i] ? '' : ' <-- ERROR';
                console.log(
                    `${String(i).padStart(9)} | ` +
                    `${String(counts[i]).padStart(5)} | ` +
                    `${String(offsets[i]).padStart(6)} | ` +
                    `${String(expectedOffsets[i]).padStart(15)}${match}`
                );
            }
        }

        // Verify last offset + last count = total
        const lastIndex = numWorkgroups - 1;
        const expectedTotal = offsets[lastIndex] + counts[lastIndex];
        console.log(`\nFinal check: offsets[${lastIndex}] + counts[${lastIndex}] = ${offsets[lastIndex]} + ${counts[lastIndex]} = ${expectedTotal}`);
        console.log(`Expected total: ${totalCount}`);
        const finalValid = expectedTotal === totalCount;
        console.log(`Final check: ${finalValid ? 'VALID' : 'ERROR'}`);

        console.log('==========================================\n');
        return prefixSumValid && countValid && finalValid;
    }
}

/**
 * Run prefix sum test
 */
export async function runPrefixSumTest(
    device: GPUDevice,
    timestampQueryManager: TimestampQueryManager,
    setA: Uint32Array,
    setB: Uint32Array,
    numWorkgroups: number
): Promise<void> {
    const tester = new TestPrefixSum(device, timestampQueryManager);

    console.log('\n========== Prefix Sum Test ==========');
    console.log(`A size: ${setA.length}, B size: ${setB.length}, Workgroups: ${numWorkgroups}`);

    const result = await tester.testPrefixSum(setA, setB, numWorkgroups);
    const valid = tester.printResults(setA, setB, result.counts, result.offsets, result.totalCount);

    console.log('=== Performance ===');
    console.log(`Diagonals GPU time: ${result.diagonalsGpuTimeMs.toFixed(4)} ms`);
    console.log(`Count GPU time: ${result.countGpuTimeMs.toFixed(4)} ms`);
    console.log(`Prefix Sum GPU time: ${result.prefixSumGpuTimeMs.toFixed(4)} ms`);
    console.log(`Total GPU time: ${(result.diagonalsGpuTimeMs + result.countGpuTimeMs + result.prefixSumGpuTimeMs).toFixed(4)} ms`);

    if (valid) {
        console.log('\nAll tests PASSED!');
    } else {
        console.log('\nSome tests FAILED!');
    }

    console.log('====================================\n');
}
