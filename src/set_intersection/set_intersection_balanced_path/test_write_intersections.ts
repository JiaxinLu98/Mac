import computeDiagonalsShader from './set_intersection.wgsl';
import countIntersectionsMergeShader from './count_intersections_merge.wgsl';
import writeIntersectionsShader from './write_intersections.wgsl';
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

export class TestWriteIntersections {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // Pipelines
    private computeDiagonalsPipeline: GPUComputePipeline;
    private countIntersectionsPipeline: GPUComputePipeline;
    private writeIntersectionsPipeline: GPUComputePipeline;

    // Bind group layouts
    private diagonalsBindGroupLayout: GPUBindGroupLayout;
    private countBindGroupLayout: GPUBindGroupLayout;
    private writeBindGroupLayout: GPUBindGroupLayout;

    // Reusable ExclusiveScanPipeline
    private scanPipeline: ExclusiveScanPipeline;

    // Timestamp query tracking (following set_intersection.ts pattern)
    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // Create ExclusiveScanPipeline
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

        // ========== write_intersections pipeline ==========
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: 'write intersections bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // offsets
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const writePipelineLayout = device.createPipelineLayout({
            label: 'write intersections pipeline layout',
            bindGroupLayouts: [this.writeBindGroupLayout]
        });

        const writeShader = device.createShaderModule({
            label: 'write intersections shader',
            code: writeIntersectionsShader
        });

        this.writeIntersectionsPipeline = device.createComputePipeline({
            label: 'write intersections pipeline',
            layout: writePipelineLayout,
            compute: {
                module: writeShader,
                entryPoint: 'write_intersections'
            }
        });
    }

    /**
     * Compute the number of timestamp queries needed per iteration
     * diagonals(2) + count(2) + scan(2) + write(2) = 8
     */
    public computeQueries(): number {
        return this.queriesPerIter = 8;
    }

    /**
     * Set the current iteration index for timestamp query offset calculation
     */
    public setIterationIndex(i: number): void {
        this.iterationIndex = i;
    }

    /**
     * Get the base offset for timestamp queries in the current iteration
     */
    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries() must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Run the full pipeline: compute_diagonals -> count_intersections -> prefix_sum -> write_intersections
     */
    public async testFullPipeline(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        diagonalsGpuTimeMs: number;
        countGpuTimeMs: number;
        prefixSumGpuTimeMs: number;
        writeGpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // Use the ExclusiveScanPipeline's aligned size requirement
        const scanAlignedSize = this.scanPipeline.getAlignedSize(numWorkgroups);

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

        // Counts/Offsets buffer (aligned for prefix sum)
        const bufferOffsets = device.createBuffer({
            label: 'Buffer Offsets',
            size: scanAlignedSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Output buffer - estimate max size as min(a_len, b_len)
        const maxOutputSize = Math.min(a_len, b_len);
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(4, maxOutputSize * Uint32Array.BYTES_PER_ELEMENT),
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
                { binding: 3, resource: { buffer: bufferOffsets } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOffsets } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALength } },
                { binding: 6, resource: { buffer: bufferBLength } },
                { binding: 7, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAX_DISPATCH_SIZE);
        const dispatchY = Math.ceil(numWorkgroups / MAX_DISPATCH_SIZE);

        // Get timestamp query base offset for this iteration
        const base = this.getQueryBaseOffset();

        // ========== Pass 1: compute_diagonals ==========
        const commandEncoder1 = device.createCommandEncoder({ label: 'Compute Diagonals' });
        const passDescriptor1 = this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1);
        const pass1 = commandEncoder1.beginComputePass(passDescriptor1);
        pass1.setPipeline(this.computeDiagonalsPipeline);
        pass1.setBindGroup(0, diagonalsBindGroup);
        pass1.dispatchWorkgroups(dispatchX, dispatchY);
        pass1.end();
        this.timestampQueryManager.resolve(commandEncoder1);
        device.queue.submit([commandEncoder1.finish()]);
        await device.queue.onSubmittedWorkDone();

        // ========== Pass 2: count_intersections ==========
        const commandEncoder2 = device.createCommandEncoder({ label: 'Count Intersections' });
        const passDescriptor2 = this.timestampQueryManager.createComputePassDescriptor(base + 2, base + 3);
        const pass2 = commandEncoder2.beginComputePass(passDescriptor2);
        pass2.setPipeline(this.countIntersectionsPipeline);
        pass2.setBindGroup(0, countBindGroup);
        pass2.dispatchWorkgroups(dispatchX, dispatchY);
        pass2.end();
        this.timestampQueryManager.resolve(commandEncoder2);
        device.queue.submit([commandEncoder2.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read counts to get total count
        const readbackCounts = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        const copyEncoderCounts = device.createCommandEncoder();
        copyEncoderCounts.copyBufferToBuffer(bufferOffsets, 0, readbackCounts, 0, numWorkgroups * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoderCounts.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackCounts.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(readbackCounts.getMappedRange().slice(0));
        readbackCounts.unmap();

        let totalCount = 0;
        for (let i = 0; i < numWorkgroups; i++) {
            totalCount += counts[i];
        }

        // ========== Pass 3: prefix_sum ==========
        const scanner = this.scanPipeline.prepareGPUInput(bufferOffsets, scanAlignedSize);
        await scanner.scan(numWorkgroups, this.timestampQueryManager, base + 4);

        // ========== Pass 4: write_intersections ==========
        const commandEncoder4 = device.createCommandEncoder({ label: 'Write Intersections' });
        const passDescriptor4 = this.timestampQueryManager.createComputePassDescriptor(base + 6, base + 7);
        const pass4 = commandEncoder4.beginComputePass(passDescriptor4);
        pass4.setPipeline(this.writeIntersectionsPipeline);
        pass4.setBindGroup(0, writeBindGroup);
        pass4.dispatchWorkgroups(dispatchX, dispatchY);
        pass4.end();
        this.timestampQueryManager.resolve(commandEncoder4);
        device.queue.submit([commandEncoder4.finish()]);
        await device.queue.onSubmittedWorkDone();

        // ========== Read timestamps at the end ==========
        let diagonalsGpuTimeMs = 0;
        let countGpuTimeMs = 0;
        let prefixSumGpuTimeMs = 0;
        let writeGpuTimeMs = 0;

        try {
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();
            diagonalsGpuTimeMs = (timestamps[base + 1] - timestamps[base + 0]) * 1e-6;
            countGpuTimeMs = (timestamps[base + 3] - timestamps[base + 2]) * 1e-6;
            prefixSumGpuTimeMs = (timestamps[base + 5] - timestamps[base + 4]) * 1e-6;
            writeGpuTimeMs = (timestamps[base + 7] - timestamps[base + 6]) * 1e-6;
        } catch (e) { }

        // ========== Read back results ==========
        const readbackOutput = device.createBuffer({
            size: Math.max(4, totalCount * Uint32Array.BYTES_PER_ELEMENT),
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferOutput, 0, readbackOutput, 0, Math.max(4, totalCount * Uint32Array.BYTES_PER_ELEMENT));
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackOutput.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackOutput.getMappedRange().slice(0, totalCount * Uint32Array.BYTES_PER_ELEMENT));
        readbackOutput.unmap();

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferC.destroy();
        bufferDPI.destroy();
        bufferOffsets.destroy();
        bufferOutput.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackCounts.destroy();
        readbackOutput.destroy();

        return {
            result,
            totalCount,
            diagonalsGpuTimeMs,
            countGpuTimeMs,
            prefixSumGpuTimeMs,
            writeGpuTimeMs
        };
    }

    /**
     * CPU reference implementation for multiset intersection
     */
    public cpuSetIntersection(setA: Uint32Array, setB: Uint32Array): Uint32Array {
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

        // Build result: for each value, output min(countA, countB) copies
        const result: number[] = [];
        // Need to output in sorted order
        const sortedKeys = Array.from(countA.keys()).sort((a, b) => a - b);
        for (const val of sortedKeys) {
            const cntA = countA.get(val) || 0;
            const cntB = countB.get(val) || 0;
            const cnt = Math.min(cntA, cntB);
            for (let i = 0; i < cnt; i++) {
                result.push(val);
            }
        }
        return new Uint32Array(result);
    }

    /**
     * Compare GPU result with CPU reference
     */
    public compareResults(gpuResult: Uint32Array, cpuResult: Uint32Array): boolean {
        if (gpuResult.length !== cpuResult.length) {
            console.log(`Length mismatch: GPU=${gpuResult.length}, CPU=${cpuResult.length}`);
            return false;
        }

        for (let i = 0; i < gpuResult.length; i++) {
            if (gpuResult[i] !== cpuResult[i]) {
                console.log(`Mismatch at index ${i}: GPU=${gpuResult[i]}, CPU=${cpuResult[i]}`);
                return false;
            }
        }

        return true;
    }
}

/**
 * Full pipeline test result
 */
export interface FullPipelineTestResult {
    valid: boolean;
    diagonalsGpuTimeMs: number;
    countGpuTimeMs: number;
    prefixSumGpuTimeMs: number;
    writeGpuTimeMs: number;
    totalGpuTimeMs: number;
}

/**
 * Run full pipeline test
 */
export async function runFullPipelineTest(
    device: GPUDevice,
    timestampQueryManager: TimestampQueryManager,
    setA: Uint32Array,
    setB: Uint32Array,
    numWorkgroups: number,
    verbose: boolean = true
): Promise<FullPipelineTestResult> {
    const tester = new TestWriteIntersections(device, timestampQueryManager);

    // Initialize timestamp query tracking (following set_intersection.ts pattern)
    tester.computeQueries();
    tester.setIterationIndex(0);

    if (verbose) {
        console.log('\n========== Full Pipeline Test ==========');
        console.log(`A size: ${setA.length}, B size: ${setB.length}, Workgroups: ${numWorkgroups}`);
    }

    const result = await tester.testFullPipeline(setA, setB, numWorkgroups);

    if (verbose) {
        console.log(`GPU result length: ${result.result.length}`);
        console.log(`GPU result (first 20): ${Array.from(result.result.slice(0, 20))}`);
    }

    // CPU validation
    const cpuResult = tester.cpuSetIntersection(setA, setB);
    if (verbose) {
        console.log(`CPU result length: ${cpuResult.length}`);
        console.log(`CPU result (first 20): ${Array.from(cpuResult.slice(0, 20))}`);
    }

    const valid = tester.compareResults(result.result, cpuResult);
    if (verbose) {
        console.log(`Validation: ${valid ? 'PASSED' : 'FAILED'}`);
    }

    const totalGpuTimeMs = result.diagonalsGpuTimeMs + result.countGpuTimeMs + result.prefixSumGpuTimeMs + result.writeGpuTimeMs;

    if (verbose) {
        console.log('\n=== Performance ===');
        console.log(`Diagonals GPU time: ${result.diagonalsGpuTimeMs.toFixed(4)} ms`);
        console.log(`Count GPU time: ${result.countGpuTimeMs.toFixed(4)} ms`);
        console.log(`Prefix Sum GPU time: ${result.prefixSumGpuTimeMs.toFixed(4)} ms`);
        console.log(`Write GPU time: ${result.writeGpuTimeMs.toFixed(4)} ms`);
        console.log(`Total GPU time: ${totalGpuTimeMs.toFixed(4)} ms`);
        console.log('========================================\n');
    }

    return {
        valid,
        diagonalsGpuTimeMs: result.diagonalsGpuTimeMs,
        countGpuTimeMs: result.countGpuTimeMs,
        prefixSumGpuTimeMs: result.prefixSumGpuTimeMs,
        writeGpuTimeMs: result.writeGpuTimeMs,
        totalGpuTimeMs
    };
}
