import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './count_intersections_binary_search.wgsl';
import writeShader from './write_intersections.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const MAXWORKGROUP = 65535;

export class GPUSetIntersectionBP {
    computeDiagonalsPipeline: GPUComputePipeline;
    countPipeline: GPUComputePipeline;
    writePipeline: GPUComputePipeline;
    device: GPUDevice;
    timestampQueryManager: TimestampQueryManager;
    bindGroupLayoutDiagonals: GPUBindGroupLayout;
    bindGroupLayoutCount: GPUBindGroupLayout;
    bindGroupLayoutWrite: GPUBindGroupLayout;

    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;
    private numScanChunks: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // Create bind group layout matching WGSL bindings
        this.bindGroupLayoutDiagonals = device.createBindGroupLayout({
            label: 'compute diagonals bind group layout',
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

        let shader_code_diagonals = `${computeDiagonalsShader}`;
        const shader_diagonals = device.createShaderModule({
            label: 'compute diagonals shader',
            code: shader_code_diagonals
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals pipeline',
            layout: pipelineLayoutDiagonals,
            compute: {
                module: shader_diagonals,
                entryPoint: 'compute_diagonals'
            }
        });

        this.bindGroupLayoutCount = device.createBindGroupLayout({
            label: 'count intersections bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi (只读)
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // counts (读写)
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        const pipelineLayoutCount = device.createPipelineLayout({
            label: 'count intersections pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutCount]
        });

        let shader_code_count = `${countShader}`;
        const shader_count = device.createShaderModule({
            label: 'count intersections shader',
            code: shader_code_count
        });

        this.countPipeline = device.createComputePipeline({
            label: 'count intersections pipeline',
            layout: pipelineLayoutCount,
            compute: {
                module: shader_count,
                entryPoint: 'count_intersections'
            }
        });

        // ========== Write Intersections Pipeline ==========
        this.bindGroupLayoutWrite = device.createBindGroupLayout({
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

        const pipelineLayoutWrite = device.createPipelineLayout({
            label: 'write intersections pipeline layout',
            bindGroupLayouts: [this.bindGroupLayoutWrite]
        });

        const shader_write = device.createShaderModule({
            label: 'write intersections shader',
            code: writeShader
        });

        this.writePipeline = device.createComputePipeline({
            label: 'write intersections pipeline',
            layout: pipelineLayoutWrite,
            compute: {
                module: shader_write,
                entryPoint: 'write_intersections'
            }
        });
    }

    public setIterationIndex(i: number) {
        this.iterationIndex = i;
    }

    public computeQueries(lenA: number, lenB: number) {
        // compute diagonals: 2 + count: 2 + prefix sum: 2 + write: 2 = 8
        return this.queriesPerIter = 8;
    }

	private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries(lenA, lenB) must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Automatically select optimal partition size based on target workgroup count.
     *
     * Design principles:
     * 1. Target workgroup count (not partition size directly) - GPU utilization is
     *    primarily determined by number of workgroups
     * 2. Use continuous function instead of lookup table for better generalization
     * 3. Consider: data size, density, and GPU parallelism sweet spot (~7000-8000 wg)
     *
     * Based on benchmark analysis (2026-01-15):
     * - numWg ≈ 7813 shows best speedup for both sparse and dense data
     * - Partition size alone doesn't correlate well with performance
     *
     * @param totalElements Total number of elements (|A| + |B|)
     * @param valueRange Optional value range to estimate density. If not provided, assumes medium density.
     * @returns Optimal partition size
     */
    public static getOptimalPartitionSize(totalElements: number, valueRange?: number): number {
        // ===== Configuration Constants =====
        const BASE_WG = 4096;        // Base workgroup count for good GPU utilization
        const MIN_WG = 256;          // Minimum workgroups to ensure parallelism
        const MAX_WG = 8000;         // Maximum workgroups (~7813 showed best performance in benchmarks)
        const MAX_PARTITION = 65536; // Maximum partition size for memory efficiency

        // ===== Step 1: Calculate density metrics =====
        // Density = elements per unique value (higher = more duplicates)
        const density = valueRange ? totalElements / valueRange : 100;
        const logDensity = Math.log10(density + 1);  // Logarithmic scale (typically 0-6 range)
        const isDense = density > 1000;  // Dense: many duplicates (e.g., 2M elements / 100 range = 20000)

        // ===== Step 2: Determine MIN_PARTITION based on density and size =====
        // Benchmark analysis (2026-01-15) showed:
        // - Both sparse and dense data need larger partitions for medium-sized datasets
        // - Sparse data (8-16M): optimal PartSize was 16384-32768
        // - Dense data (8-16M): optimal PartSize was 8192-32768
        let minPartition: number;
        if (isDense) {
            // Dense data: different partition sizes for different scales
            // - Small data (≤8M): smaller partition to ensure enough workgroups
            // - Large data (>8M): larger partition to handle duplicate runs efficiently
            if (totalElements <= 16_000_000) {
                minPartition = 4096;   // Small: need more workgroups
            } else if (totalElements <= 64_000_000) {
                minPartition = 8192;   // Medium: balance between wg count and partition size
            } else {
                minPartition = 16384;  // Large: larger partition for efficiency
            }
        } else {
            // Sparse data: medium-sized datasets need larger partitions
            if (totalElements <= 16_000_000) {
                minPartition = 16384;  // Small/medium: use larger partition
            } else {
                minPartition = 8192;   // Large: can use smaller partition
            }
        }

        // ===== Step 3: Calculate target workgroup count =====
        // Size factor: sub-linear growth with sqrt
        // - Larger data needs more workgroups, but not proportionally
        // - sqrt(2M/2M)=1, sqrt(8M/2M)=2, sqrt(32M/2M)=4, sqrt(128M/2M)=8
        const sizeFactor = Math.sqrt(totalElements / 2_000_000);

        // Density factor: denser data benefits from slightly more workgroups
        // - Typical range: 1.0 (sparse) to 1.6 (very dense)
        const densityFactor = 1.0 + logDensity * 0.1;

        // Calculate target workgroups with bounds
        let targetWg = BASE_WG * sizeFactor * densityFactor;
        targetWg = Math.max(MIN_WG, Math.min(MAX_WG, targetWg));

        // ===== Step 4: Derive partition size from target workgroups =====
        const rawPartition = totalElements / targetWg;

        // ===== Step 5: Round to nearest power of 2 for memory alignment =====
        const log2Partition = Math.log2(rawPartition);
        let partitionSize = Math.pow(2, Math.round(log2Partition));

        // ===== Step 6: Apply practical bounds (density-aware) =====
        partitionSize = Math.max(minPartition, Math.min(MAX_PARTITION, partitionSize));

        return partitionSize;
    }

    /**
     * Calculate the target workgroup count for given parameters.
     * Useful for debugging and analysis.
     */
    public static getTargetWorkgroupCount(totalElements: number, valueRange?: number): number {
        const partitionSize = this.getOptimalPartitionSize(totalElements, valueRange);
        return Math.ceil(totalElements / partitionSize);
    }

    // =========================================================================
    // [DEPRECATED] Old hardcoded lookup table strategy (commented out 2026-01-15)
    // Replaced by workgroup-count-based continuous function above
    // =========================================================================
    // public static getOptimalPartitionSize_OLD(totalElements: number, valueRange?: number): number {
    //     // Calculate density ratio if valueRange provided
    //     const densityRatio = valueRange ? totalElements / valueRange : 100;
    //     const isDense = densityRatio > 1000; // High duplicate rate (e.g., 2M elements / 100 range = 20000)
    //
    //     // Small datasets: use 4096 to minimize overhead
    //     if (totalElements <= 8_000_000) {
    //         return 4096;
    //     }
    //
    //     if (isDense) {
    //         // Dense data (e2 type - many duplicates, high intersection rate)
    //         // 8M->8192, 16M->32768, 32M->8192, 64M->16384, 128M->16384
    //         if (totalElements <= 16_000_000) {
    //             return 8192;
    //         } else if (totalElements <= 32_000_000) {
    //             return 32768;
    //         } else if (totalElements <= 64_000_000) {
    //             return 8192;
    //         } else {
    //             return 16384;
    //         }
    //     } else {
    //         // Sparse data (e6 type - few duplicates, low intersection rate)
    //         // 8M->32768, 16M->16384, 32M->8192, 64M->8192, 128M->16384
    //         if (totalElements <= 16_000_000) {
    //             return 32768;
    //         } else if (totalElements <= 32_000_000) {
    //             return 16384;
    //         } else if (totalElements <= 128_000_000) {
    //             return 8192;
    //         } else {
    //             return 16384;
    //         }
    //     }
    // }

    public async computeIntersection(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number,
        partitionSize?: number,  // Optional, auto-select if not provided
        valueRange?: number      // Optional, used for adaptive partition size selection
    ) {
        // If partitionSize not specified, use adaptive selection
        if (partitionSize === undefined) {
            partitionSize = GPUSetIntersectionBP.getOptimalPartitionSize(setA.length + setB.length, valueRange);
        }

        if (setA.length === 0 || setB.length === 0) {
            return {
                aIndices: new Uint32Array(0),
                bIndices: new Uint32Array(0),
                stars: [] as boolean[],
                counts: new Uint32Array(0),
                offsets: new Uint32Array(0),
                result: new Uint32Array(0),
                totalCount: 0,
                partitionSize,
                numWg: 0,
                avgDiagonalsMs: 0,
                avgCountMs: 0,
                avgPrefixSumMs: 0,
                avgWriteMs: 0,
                avgTotalMs: 0
            };
        }

        const device = this.device;

        this.computeQueries(setA.length, setB.length);
        if (this.queriesPerIter === 0) {
            throw new Error("queriesPerIter is 0; make sure computeQueries(lenA, lenB) was called.");
        }
        const QUERIES_PER_ITER = this.queriesPerIter;

        // Create GPU buffers
        const bufferA = device.createBuffer({
            label: 'A buffer',
            size: setA.length * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

        const bufferB = device.createBuffer({
            label: 'B buffer',
            size: setB.length * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        // Calculate number of workgroups based on partition size
        const numWg = Math.ceil((setA.length + setB.length) / partitionSize);

        let lastAIndices: Uint32Array;
        let lastBIndices: Uint32Array;
        let lastStars: boolean[];
        let lastCounts: Uint32Array;
        let lastOffsets: Uint32Array;
        let lastResult: Uint32Array;
        let lastTotalCount: number = 0;
        let wallTotalMs = 0;

        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const t0 = performance.now();

            // Phase 1: COMPUTE DIAGONALS
            const diagResult = await this.runComputeDiagonalsPhase(bufferA, bufferB, setA.length, setB.length, numWg);

            // Phase 2: COUNT INTERSECTIONS
            const countResult = await this.runCountPhase(bufferA, bufferB, diagResult.bufferDPI, setA.length, setB.length, numWg);

            // Phase 3: PREFIX SUM (Exclusive Scan)
            const prefixSumResult = await this.runPrefixSumPhase(countResult.bufferCounts, numWg);

            // Phase 4: WRITE INTERSECTIONS
            const writeResult = await this.runWritePhase(
                bufferA, bufferB, diagResult.bufferDPI, prefixSumResult.bufferOffsets,
                setA.length, setB.length, numWg, countResult.totalCount
            );

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);

            lastAIndices = diagResult.aIndices;
            lastBIndices = diagResult.bIndices;
            lastStars = diagResult.stars;
            lastCounts = countResult.counts;
            lastOffsets = prefixSumResult.offsets;
            lastResult = writeResult.result;
            lastTotalCount = countResult.totalCount;

            // Cleanup DPI buffer after use
            diagResult.bufferDPI.destroy();
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `ComputeIntersection x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        // Process GPU timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumDiagonals = 0;
        let sumCount = 0;
        let sumPrefixSum = 0;
        let sumWrite = 0;

        for (let i = 0; i < iters; ++i) {
            const base = i * QUERIES_PER_ITER;

            // compute diagonals: [base + 0, base + 1]
            const diagonalsTicks = timestamps[base + 1] - timestamps[base + 0];
            // count intersections: [base + 2, base + 3]
            const countTicks = timestamps[base + 3] - timestamps[base + 2];
            // prefix sum: [base + 4, base + 5]
            const prefixSumTicks = timestamps[base + 5] - timestamps[base + 4];
            // write: [base + 6, base + 7]
            const writeTicks = timestamps[base + 7] - timestamps[base + 6];

            sumDiagonals += diagonalsTicks;
            sumCount += countTicks;
            sumPrefixSum += prefixSumTicks;
            sumWrite += writeTicks;
        }

        const avgTicks = {
            diagonals: sumDiagonals / iters,
            count: sumCount / iters,
            prefixSum: sumPrefixSum / iters,
            write: sumWrite / iters,
        };

        const timestampPeriod = 1e-9;

        const avgSeconds = {
            diagonals: avgTicks.diagonals * timestampPeriod,
            count: avgTicks.count * timestampPeriod,
            prefixSum: avgTicks.prefixSum * timestampPeriod,
            write: avgTicks.write * timestampPeriod,
        };

        const avgDiagonalsMs = avgSeconds.diagonals * 1000;
        const avgCountMs = avgSeconds.count * 1000;
        const avgPrefixSumMs = avgSeconds.prefixSum * 1000;
        const avgWriteMs = avgSeconds.write * 1000;
        const avgTotalMs = avgDiagonalsMs + avgCountMs + avgPrefixSumMs + avgWriteMs;

        console.log(
            `GPU timestamps avg over ${iters} iterations:`,
            `diagonals = ${avgDiagonalsMs.toFixed(4)} ms,`,
            `count = ${avgCountMs.toFixed(4)} ms,`,
            `prefixSum = ${avgPrefixSumMs.toFixed(4)} ms,`,
            `write = ${avgWriteMs.toFixed(4)} ms,`,
            `total = ${avgTotalMs.toFixed(4)} ms`
        );

        // Validate count result
        this.validateCount(lastTotalCount, setA, setB);

        bufferA.destroy();
        bufferB.destroy();

        return {
            aIndices: lastAIndices!,
            bIndices: lastBIndices!,
            stars: lastStars!,
            counts: lastCounts!,
            offsets: lastOffsets!,
            result: lastResult!,
            totalCount: lastTotalCount,
            partitionSize,
            numWg,
            avgDiagonalsMs,
            avgCountMs,
            avgPrefixSumMs,
            avgWriteMs,
            avgTotalMs
        };

    }

    public async runComputeDiagonalsPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number
    ): Promise<{ aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[]; bufferDPI: GPUBuffer }> {

        const device = this.device;

        // DPI buffer: size = 2 * (num_wg + 1)
        const dpiSize = 2 * (numWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Uniform buffers for lengths
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

        // Uniform buffer for num_wg (to handle 2D dispatch correctly)
        const bufferNumWg = device.createBuffer({
            label: 'num_wg uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWorkgroups]));

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutDiagonals,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        // Dispatch compute shader with timestamp queries
        const commandEncoder = device.createCommandEncoder({ label: 'Compute Diagonals' });
        const base = this.getQueryBaseOffset();
        const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1));
        pass.setPipeline(this.computeDiagonalsPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back DPI buffer
        const readbackBuffer = device.createBuffer({
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferDPI, 0, readbackBuffer, 0, dpiSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const dpiData = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Parse DPI data
        // Layout: dpi[0..num_wg] = packed aIndex, dpi[num_wg+1..2*num_wg+1] = bIndex
        const aIndices = new Uint32Array(numWorkgroups + 1);
        const bIndices = new Uint32Array(numWorkgroups + 1);
        const stars: boolean[] = [];

        for (let i = 0; i <= numWorkgroups; i++) {
            const packedA = dpiData[i];
            aIndices[i] = packedA & ~STAR_MASK;
            stars.push((packedA & STAR_MASK) !== 0);
            bIndices[i] = dpiData[numWorkgroups + 1 + i];
        }

        // Cleanup
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();
        readbackBuffer.destroy();

        return { aIndices, bIndices, stars, bufferDPI };
    }

    public async runCountPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        bufferDPI: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number
    ): Promise<{ counts: Uint32Array; totalCount: number; bufferCounts: GPUBuffer }> {

        const device = this.device;

        // Counts buffer: one count per workgroup
        // Use aligned size for prefix sum (must be multiple of 512)
        const scan = new ExclusiveScanPipeline(device);
        const alignedSize = scan.getAlignedSize(numWorkgroups);

        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * Uint32Array.BYTES_PER_ELEMENT,
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

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutCount,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALength } },
                { binding: 5, resource: { buffer: bufferBLength } },
                { binding: 6, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        // Dispatch compute shader with timestamp queries
        const commandEncoder = device.createCommandEncoder({ label: 'Count Intersections' });
        const base = this.getQueryBaseOffset();
        const countBase = base + 2;
        const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(countBase, countBase + 1));
        pass.setPipeline(this.countPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back counts buffer
        const readbackBuffer = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferCounts, 0, readbackBuffer, 0, numWorkgroups * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Calculate total count
        let totalCount = 0;
        for (let i = 0; i < numWorkgroups; i++) {
            totalCount += counts[i];
        }

        // Cleanup (keep bufferCounts for prefix sum phase)
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackBuffer.destroy();

        return { counts, totalCount, bufferCounts };
    }

    /**
     * Phase 3: Prefix Sum (Exclusive Scan)
     * Transform counts into write offsets for each workgroup
     */
    public async runPrefixSumPhase(
        bufferCounts: GPUBuffer,
        numWorkgroups: number
    ): Promise<{ offsets: Uint32Array; bufferOffsets: GPUBuffer }> {
        const device = this.device;

        const scan = new ExclusiveScanPipeline(device);
        const alignedSize = scan.getAlignedSize(numWorkgroups);

        // Pad buffer with zeros if needed
        if (alignedSize > numWorkgroups) {
            const padCount = alignedSize - numWorkgroups;
            device.queue.writeBuffer(
                bufferCounts,
                numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
                new Uint32Array(padCount)
            );
        }

        // Calculate number of scan chunks for timestamp tracking
        const numChunks = Math.ceil(alignedSize / scan.maxScanSize);
        this.numScanChunks = numChunks;

        // Get timestamp base offset
        const base = this.getQueryBaseOffset();
        const scanBaseIndex = base + 4; // After diagonals(2) + count(2)

        // Run exclusive scan
        const scanner = scan.prepareGPUInput(bufferCounts, alignedSize);
        await scanner.scan(numWorkgroups, this.timestampQueryManager, scanBaseIndex);

        await device.queue.onSubmittedWorkDone();

        // Read back offsets
        const readbackBuffer = device.createBuffer({
            size: numWorkgroups * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(
            bufferCounts, 0,
            readbackBuffer, 0,
            numWorkgroups * Uint32Array.BYTES_PER_ELEMENT
        );
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Cleanup readback buffer (keep bufferCounts as bufferOffsets for write phase)
        readbackBuffer.destroy();

        return { offsets, bufferOffsets: bufferCounts };
    }

    /**
     * Phase 4: Write Intersections
     * Write the actual intersection elements to output buffer
     */
    public async runWritePhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        bufferDPI: GPUBuffer,
        bufferOffsets: GPUBuffer,
        a_len: number,
        b_len: number,
        numWorkgroups: number,
        totalCount: number
    ): Promise<{ result: Uint32Array }> {
        const device = this.device;

        // Output buffer
        const outputSize = Math.max(totalCount, 1); // At least 1 to avoid 0-size buffer
        const bufferOutput = device.createBuffer({
            label: 'Output buffer',
            size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
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

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayoutWrite,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferOffsets } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferALength } },
                { binding: 6, resource: { buffer: bufferBLength } },
                { binding: 7, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        // Dispatch compute shader with timestamp queries
        const commandEncoder = device.createCommandEncoder({ label: 'Write Intersections' });
        const base = this.getQueryBaseOffset();
        const writeBase = base + 6; // After diagonals(2) + count(2) + prefixSum(2)
        const pass = commandEncoder.beginComputePass(
            this.timestampQueryManager.createComputePassDescriptor(writeBase, writeBase + 1)
        );
        pass.setPipeline(this.writePipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // Read back output buffer
        const readbackBuffer = device.createBuffer({
            size: outputSize * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        const copyEncoder = device.createCommandEncoder();
        copyEncoder.copyBufferToBuffer(bufferOutput, 0, readbackBuffer, 0, outputSize * Uint32Array.BYTES_PER_ELEMENT);
        device.queue.submit([copyEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Cleanup
        bufferOutput.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackBuffer.destroy();
        bufferOffsets.destroy();

        return { result };
    }

    /**
     * CPU reference implementation for validation (MULTISET intersection)
     * For each value v: output min(count_A(v), count_B(v)) copies
     */
    public cpuCountIntersection(setA: Uint32Array, setB: Uint32Array): number {
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
     * Validate GPU count result against CPU reference
     */
    public validateCount(gpuTotalCount: number, setA: Uint32Array, setB: Uint32Array): boolean {
        const cpuCount = this.cpuCountIntersection(setA, setB);
        const valid = gpuTotalCount === cpuCount;

        if (valid) {
            console.log(`✔ Count validation PASSED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        } else {
            console.log(`✗ Count validation FAILED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        }

        return valid;
    }

    public printDiagonals(
        aIndices: Uint32Array,
        bIndices: Uint32Array,
        stars: boolean[],
        a_len: number,
        b_len: number
    ): void {
        const numWorkgroups = aIndices.length - 1;
        const total = a_len + b_len;

        console.log('\n=== Compute Diagonals Results ===');
        console.log(`Input sizes: A=${a_len}, B=${b_len}, total=${total}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);
        console.log('');

        console.log('Partition | Diagonal | aIndex | bIndex | Star | Segment Size');
        console.log('----------|----------|--------|--------|------|-------------');

        for (let i = 0; i <= numWorkgroups; i++) {
            const diag = Math.floor((i * total) / numWorkgroups);
            const aIdx = aIndices[i];
            const bIdx = bIndices[i];
            const star = stars[i] ? '*' : ' ';

            // Calculate segment size (elements in this partition)
            let segmentSize = '-';
            if (i < numWorkgroups) {
                const nextAIdx = aIndices[i + 1];
                const nextBIdx = bIndices[i + 1];
                const aCount = nextAIdx - aIdx;
                const bCount = nextBIdx - bIdx;
                segmentSize = `A:${aCount} B:${bCount} (${aCount + bCount})`;
            }

            console.log(
                `${String(i).padStart(9)} | ` +
                `${String(diag).padStart(8)} | ` +
                `${String(aIdx).padStart(6)} | ` +
                `${String(bIdx).padStart(6)} | ` +
                `${star.padStart(4)} | ` +
                `${segmentSize}`
            );
        }

        // Verify consistency
        console.log('\n=== Verification ===');
        let valid = true;

        // Check boundary conditions
        if (aIndices[0] !== 0) {
            console.log(`ERROR: aIndex[0] should be 0, got ${aIndices[0]}`);
            valid = false;
        }
        if (bIndices[0] !== 0) {
            console.log(`ERROR: bIndex[0] should be 0, got ${bIndices[0]}`);
            valid = false;
        }
        if (aIndices[numWorkgroups] !== a_len) {
            console.log(`ERROR: aIndex[${numWorkgroups}] should be ${a_len}, got ${aIndices[numWorkgroups]}`);
            valid = false;
        }
        if (bIndices[numWorkgroups] !== b_len) {
            console.log(`ERROR: bIndex[${numWorkgroups}] should be ${b_len}, got ${bIndices[numWorkgroups]}`);
            valid = false;
        }

        // Check monotonicity and diagonal constraint
        for (let i = 0; i < numWorkgroups; i++) {
            const diag = Math.floor((i * total) / numWorkgroups);
            const aIdx = aIndices[i];
            const bIdx = bIndices[i];

            // aIndex + bIndex should equal diagonal
            if (aIdx + bIdx !== diag) {
                console.log(`ERROR: Partition ${i}: aIndex(${aIdx}) + bIndex(${bIdx}) = ${aIdx + bIdx} != diagonal(${diag})`);
                valid = false;
            }

            // Monotonicity
            if (aIndices[i + 1] < aIdx) {
                console.log(`ERROR: aIndex not monotonic at partition ${i}`);
                valid = false;
            }
            if (bIndices[i + 1] < bIdx) {
                console.log(`ERROR: bIndex not monotonic at partition ${i}`);
                valid = false;
            }
        }

        if (valid) {
            console.log('All checks passed!');
        }

        console.log('=================================\n');
    }
}