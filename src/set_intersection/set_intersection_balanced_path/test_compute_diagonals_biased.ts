import computeDiagonalsBiasedShader from './balanced_path_biased.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const MAXWORKGROUP = 65535;

// ModernGPU constants - must match WGSL shader
const NV = 1792;  // NT * VT = 256 * 7, elements per workgroup

/**
 * Test class for the biased balanced path compute diagonals shader.
 *
 * This implementation uses BiasedBinarySearch (ModernGPU-style) for finding
 * duplicate run boundaries, which biases the search towards the END of the range.
 *
 * DPI buffer layout:
 *   dpi[0 .. num_wg]           : packed aIndex (MSB = star flag)
 *   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
 */
export class TestComputeDiagonalsBiased {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private computeDiagonalsPipeline: GPUComputePipeline;
    private bindGroupLayout: GPUBindGroupLayout;

    // Timestamp query management (following set_intersection.ts pattern)
    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // Create bind group layout matching balanced_path_biased.wgsl bindings
        // binding 0: a (storage, read)
        // binding 1: b (storage, read)
        // binding 2: dpi (storage, read_write)
        // binding 3: a_length (uniform)
        // binding 4: b_length (uniform)
        // binding 5: num_wg_uniform (uniform)
        this.bindGroupLayout = device.createBindGroupLayout({
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

        const pipelineLayout = device.createPipelineLayout({
            label: 'compute diagonals biased pipeline layout',
            bindGroupLayouts: [this.bindGroupLayout]
        });

        const shader = device.createShaderModule({
            label: 'compute diagonals biased shader',
            code: computeDiagonalsBiasedShader
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals biased pipeline',
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: 'compute_diagonals'
            }
        });
    }

    /**
     * Set the current iteration index for timestamp query offset calculation.
     */
    public setIterationIndex(i: number): void {
        this.iterationIndex = i;
    }

    /**
     * Compute the number of timestamp queries needed per iteration.
     * compute_diagonals uses 2 queries (start and end).
     */
    public computeQueries(): number {
        return this.queriesPerIter = 2;
    }

    /**
     * Get the base offset for timestamp queries based on current iteration.
     */
    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries() must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Compute diagonal partition points using biased balanced path algorithm.
     * Single iteration version for testing.
     *
     * @param setA - Sorted input array A
     * @param setB - Sorted input array B
     * @param numWorkgroups - Number of partitions (workgroups)
     * @returns Partition indices and timing information
     */
    public async testComputeDiagonals(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number
    ): Promise<{ aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[]; gpuTimeMs: number }> {

        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

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
        const dpiSize = 2 * (numWorkgroups + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
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
            label: 'num_wg_uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWorkgroups]));

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ],
        });

        // Calculate 2D dispatch dimensions (handle >65535 workgroups)
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        // Dispatch compute shader with timestamp queries
        const commandEncoder = device.createCommandEncoder({ label: 'Compute Diagonals Biased' });
        const base = this.getQueryBaseOffset();
        const passDescriptor = this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1);
        const pass = commandEncoder.beginComputePass(passDescriptor);
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

        // Get GPU timestamp
        let gpuTimeMs = 0;
        try {
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();
            const gpuTicks = timestamps[base + 1] - timestamps[base + 0];
            gpuTimeMs = gpuTicks * 1e-6;  // nanoseconds to milliseconds
        } catch {
            // Timestamp query not supported
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferDPI.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();
        readbackBuffer.destroy();

        return { aIndices, bIndices, stars, gpuTimeMs };
    }

    /**
     * Benchmark compute diagonals with multiple iterations.
     * Returns average GPU time and detailed per-iteration stats.
     *
     * @param setA - Sorted input array A
     * @param setB - Sorted input array B
     * @param numWorkgroups - Number of partitions (workgroups)
     * @param iters - Number of benchmark iterations
     * @param warmupIters - Number of warmup iterations (not timed) to stabilize GPU state
     * @returns Partition indices and timing statistics
     */
    public async benchmarkComputeDiagonals(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number,
        iters: number,
        warmupIters: number = 5
    ): Promise<{
        aIndices: Uint32Array;
        bIndices: Uint32Array;
        stars: boolean[];
        avgGpuTimeMs: number;
        wallTimeMs: number;
        partitionSize: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        // Initialize timestamp query tracking
        const QUERIES_PER_ITER = this.computeQueries();

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

        // Uniform buffers (reused)
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
            label: 'num_wg_uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWorkgroups]));

        // DPI buffer size
        const dpiSize = 2 * (numWorkgroups + 1);

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        let lastAIndices: Uint32Array = new Uint32Array(0);
        let lastBIndices: Uint32Array = new Uint32Array(0);
        let lastStars: boolean[] = [];
        let wallTotalMs = 0;

        // Warmup runs (not timed) - stabilize GPU state (frequency, cache, etc.)
        for (let w = 0; w < warmupIters; w++) {
            const warmupDPI = device.createBuffer({
                label: `Warmup DPI ${w}`,
                size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });
            const warmupBindGroup = device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: bufferA } },
                    { binding: 1, resource: { buffer: bufferB } },
                    { binding: 2, resource: { buffer: warmupDPI } },
                    { binding: 3, resource: { buffer: bufferALength } },
                    { binding: 4, resource: { buffer: bufferBLength } },
                    { binding: 5, resource: { buffer: bufferNumWg } },
                ],
            });
            const warmupEncoder = device.createCommandEncoder({ label: `Warmup ${w}` });
            const warmupPass = warmupEncoder.beginComputePass();
            warmupPass.setPipeline(this.computeDiagonalsPipeline);
            warmupPass.setBindGroup(0, warmupBindGroup);
            warmupPass.dispatchWorkgroups(dispatchX, dispatchY);
            warmupPass.end();
            device.queue.submit([warmupEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();
            warmupDPI.destroy();
        }

        // Run benchmark iterations
        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const t0 = performance.now();

            // Create DPI buffer for this iteration
            const bufferDPI = device.createBuffer({
                label: `Buffer DPI iter ${i}`,
                size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });

            // Create bind group
            const bindGroup = device.createBindGroup({
                layout: this.bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: bufferA } },
                    { binding: 1, resource: { buffer: bufferB } },
                    { binding: 2, resource: { buffer: bufferDPI } },
                    { binding: 3, resource: { buffer: bufferALength } },
                    { binding: 4, resource: { buffer: bufferBLength } },
                    { binding: 5, resource: { buffer: bufferNumWg } },
                ],
            });

            // Dispatch compute shader with timestamp queries
            const commandEncoder = device.createCommandEncoder({ label: `Compute Diagonals Biased iter ${i}` });
            const base = this.getQueryBaseOffset();
            const passDescriptor = this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1);
            const pass = commandEncoder.beginComputePass(passDescriptor);
            pass.setPipeline(this.computeDiagonalsPipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();
            this.timestampQueryManager.resolve(commandEncoder);

            device.queue.submit([commandEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Read back DPI buffer (only on last iteration)
            if (i === iters - 1) {
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
                lastAIndices = new Uint32Array(numWorkgroups + 1);
                lastBIndices = new Uint32Array(numWorkgroups + 1);
                lastStars = [];

                for (let j = 0; j <= numWorkgroups; j++) {
                    const packedA = dpiData[j];
                    lastAIndices[j] = packedA & ~STAR_MASK;
                    lastStars.push((packedA & STAR_MASK) !== 0);
                    lastBIndices[j] = dpiData[numWorkgroups + 1 + j];
                }

                readbackBuffer.destroy();
            }

            bufferDPI.destroy();

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `ComputeDiagonals (Biased) x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        // Process GPU timestamps
        let avgGpuTimeMs = 0;
        try {
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();
            let sumTicks = 0;
            for (let i = 0; i < iters; i++) {
                const base = i * QUERIES_PER_ITER;
                sumTicks += timestamps[base + 1] - timestamps[base + 0];
            }
            avgGpuTimeMs = (sumTicks / iters) * 1e-6;
            console.log(
                `GPU timestamps avg over ${iters} iterations:`,
                `diagonals = ${avgGpuTimeMs.toFixed(4)} ms`
            );
        } catch {
            console.log('Timestamp queries not supported');
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();

        const partitionSize = Math.ceil((a_len + b_len) / numWorkgroups);

        return {
            aIndices: lastAIndices,
            bIndices: lastBIndices,
            stars: lastStars,
            avgGpuTimeMs,
            wallTimeMs: wallTotalMs,
            partitionSize
        };
    }

    /**
     * Print diagonal partition results with verification.
     */
    public printDiagonals(
        aIndices: Uint32Array,
        bIndices: Uint32Array,
        stars: boolean[],
        a_len: number,
        b_len: number
    ): void {
        const numWorkgroups = aIndices.length - 1;
        const total = a_len + b_len;

        console.log('\n=== Compute Diagonals (Biased) Results ===');
        console.log(`Input sizes: A=${a_len}, B=${b_len}, total=${total}`);
        console.log(`Number of workgroups: ${numWorkgroups}`);
        console.log(`NV (elements per workgroup): ${NV}`);
        console.log('');

        console.log('Partition | Diagonal | aIndex | bIndex | Star | Segment Size');
        console.log('----------|----------|--------|--------|------|-------------');

        for (let i = 0; i <= numWorkgroups; i++) {
            // ModernGPU: diag = NV * k, but clamped to total for the last entry
            const diag = Math.min(NV * i, total);
            const aIdx = aIndices[i];
            const bIdx = bIndices[i];
            const star = stars[i] ? '*' : ' ';

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

        // Verification
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
        // ModernGPU: diag = NV * k, clamped to total
        for (let i = 0; i < numWorkgroups; i++) {
            const diag = Math.min(NV * i, total);
            const aIdx = aIndices[i];
            const bIdx = bIndices[i];

            if (aIdx + bIdx !== diag) {
                console.log(`ERROR: Partition ${i}: aIndex(${aIdx}) + bIndex(${bIdx}) = ${aIdx + bIdx} != diagonal(${diag})`);
                valid = false;
            }

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

        console.log('==========================================\n');
    }

    // ========================================================================
    // CPU Reference Implementation (for validation)
    // ========================================================================

    /**
     * CPU MergePath: Find intersection point on diagonal.
     */
    private cpuMergePath(a: Uint32Array, b: Uint32Array, diag: number): number {
        const aCount = a.length;
        const bCount = b.length;
        let begin = Math.max(0, diag - bCount);
        let end = Math.min(diag, aCount);

        while (begin < end) {
            const mid = (begin + end) >> 1;
            const aKey = a[mid];
            const bKey = b[diag - 1 - mid];
            if (aKey <= bKey) {
                begin = mid + 1;
            } else {
                end = mid;
            }
        }
        return begin;
    }

    private cpuLowerBound(arr: Uint32Array, end: number, key: number): number {
        let lo = 0, hi = end;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (arr[mid] < key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private cpuUpperBound(arr: Uint32Array, begin: number, end: number, key: number): number {
        let lo = begin, hi = end;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (arr[mid] <= key) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    /**
     * CPU BalancedPath: Adjust MergePath result to handle duplicates.
     */
    private cpuBalancedPath(a: Uint32Array, b: Uint32Array, diag: number): { aIndex: number; bIndex: number; star: boolean } {
        const bCount = b.length;
        const p = this.cpuMergePath(a, b, diag);
        let aIndex = p;
        let bIndex = diag - p;
        let star = false;

        if (bIndex < bCount) {
            const x = b[bIndex];
            const aStart = this.cpuLowerBound(a, aIndex, x);
            const bStart = this.cpuLowerBound(b, bIndex, x);
            const aRun = aIndex - aStart;
            const bRun = bIndex - bStart;
            const xCount = aRun + bRun;

            let bAdvance = Math.max(xCount >> 1, xCount - aRun);
            const bEndHint = Math.min(bCount, bStart + bAdvance + 1);
            const bRunEnd = this.cpuUpperBound(b, bIndex, Math.max(bEndHint, Math.min(bCount, bIndex + 1)), x);
            const actualBRun = bRunEnd - bStart;

            bAdvance = Math.min(bAdvance, actualBRun);
            const aAdvance = xCount - bAdvance;
            const roundUp = (aAdvance === bAdvance + 1) && (bAdvance < actualBRun);

            aIndex = aStart + aAdvance;
            bIndex = diag - aIndex;
            star = roundUp;
        }
        return { aIndex, bIndex, star };
    }

    /**
     * Compute all diagonal partitions using CPU reference.
     */
    public cpuComputeDiagonals(a: Uint32Array, b: Uint32Array, numWorkgroups: number): { aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[] } {
        const aIndices = new Uint32Array(numWorkgroups + 1);
        const bIndices = new Uint32Array(numWorkgroups + 1);
        const stars: boolean[] = [];

        for (let k = 0; k <= numWorkgroups; k++) {
            if (k === 0) {
                aIndices[0] = 0; bIndices[0] = 0; stars.push(false);
            } else if (k === numWorkgroups) {
                aIndices[k] = a.length; bIndices[k] = b.length; stars.push(false);
            } else {
                // ModernGPU: diag = NV * k (fixed partition size)
                const diag = NV * k;
                const result = this.cpuBalancedPath(a, b, diag);
                aIndices[k] = result.aIndex;
                bIndices[k] = result.bIndex;
                stars.push(result.star);
            }
        }
        return { aIndices, bIndices, stars };
    }

    /**
     * Validate GPU results against CPU reference.
     */
    public validateAgainstCPU(
        gpuResult: { aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[] },
        a: Uint32Array, b: Uint32Array, numWorkgroups: number
    ): boolean {
        const cpuResult = this.cpuComputeDiagonals(a, b, numWorkgroups);
        let allMatch = true;
        let mismatchCount = 0;

        console.log('\n=== GPU vs CPU Validation ===');
        for (let i = 0; i <= numWorkgroups; i++) {
            const aMatch = gpuResult.aIndices[i] === cpuResult.aIndices[i];
            const bMatch = gpuResult.bIndices[i] === cpuResult.bIndices[i];
            const starMatch = gpuResult.stars[i] === cpuResult.stars[i];

            if (!aMatch || !bMatch || !starMatch) {
                if (mismatchCount < 10) {
                    console.log(`Partition ${i}: GPU(a=${gpuResult.aIndices[i]}, b=${gpuResult.bIndices[i]}, star=${gpuResult.stars[i]}) vs CPU(a=${cpuResult.aIndices[i]}, b=${cpuResult.bIndices[i]}, star=${cpuResult.stars[i]})`);
                }
                allMatch = false;
                mismatchCount++;
            }
        }

        if (allMatch) {
            console.log(`✔ All ${numWorkgroups + 1} partitions match CPU reference!`);
        } else {
            console.log(`✗ ${mismatchCount} / ${numWorkgroups + 1} partitions differ from CPU`);
        }
        console.log('==============================\n');
        return allMatch;
    }

    /**
     * Compare results between biased and non-biased implementations.
     */
    public compareResults(
        biasedResult: { aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[] },
        nonBiasedResult: { aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[] }
    ): boolean {
        const numWg = biasedResult.aIndices.length;
        let allMatch = true;

        console.log('\n=== Comparison: Biased vs Non-Biased ===');

        for (let i = 0; i < numWg; i++) {
            const aMatch = biasedResult.aIndices[i] === nonBiasedResult.aIndices[i];
            const bMatch = biasedResult.bIndices[i] === nonBiasedResult.bIndices[i];
            const starMatch = biasedResult.stars[i] === nonBiasedResult.stars[i];

            if (!aMatch || !bMatch || !starMatch) {
                console.log(`Partition ${i}: Biased(a=${biasedResult.aIndices[i]}, b=${biasedResult.bIndices[i]}, star=${biasedResult.stars[i]}) vs Non-Biased(a=${nonBiasedResult.aIndices[i]}, b=${nonBiasedResult.bIndices[i]}, star=${nonBiasedResult.stars[i]})`);
                allMatch = false;
            }
        }

        if (allMatch) {
            console.log('All partition results match!');
        }

        console.log('=========================================\n');
        return allMatch;
    }

    /**
     * Verify DPI consistency: compare stored B indices with ModernGPU's calculated B indices.
     *
     * ModernGPU calculates B from diagonal: b = gid - a + bit
     * Our implementation stores B directly in DPI.
     * This function verifies they are consistent.
     *
     * @param dpiData - Raw DPI data from GPU
     * @param numWorkgroups - Number of workgroups
     * @param aLen - Length of array A
     * @param bLen - Length of array B
     * @param NV - Elements per workgroup (NT * VT, default 1792 = 256 * 7)
     * @returns true if all B indices match
     */
    public verifyDPIConsistencyWithModernGPU(
        dpiData: Uint32Array,
        numWorkgroups: number,
        aLen: number,
        bLen: number,
        NV: number = 1792  // NT * VT = 256 * 7
    ): boolean {
        let allMatch = true;
        let mismatchCount = 0;
        const maxMismatchesToShow = 10;

        console.log('\n=== DPI Consistency Check (vs ModernGPU formula) ===');
        console.log(`NV (elements per workgroup) = ${NV}`);
        console.log(`Total elements = ${aLen + bLen}`);
        console.log('');

        for (let block = 0; block <= numWorkgroups; block++) {
            // 从 DPI 读取存储的值
            const packed_a = dpiData[block];
            const a_stored = packed_a & ~STAR_MASK;
            const star = (packed_a & STAR_MASK) !== 0;
            const b_stored = dpiData[numWorkgroups + 1 + block];

            // ModernGPU 存储逻辑:
            //   存储时: b = diag - a (不包含 star bit)
            //   使用时: b += bit (在 set_intersection_balanced.wgsl 中加上)
            // 所以验证时，我们检查 b_stored = gid - a_stored (不加 bit)
            //
            // 边界情况: 最后一个 block 的对角线 = min(NV * block, total)
            const total = aLen + bLen;
            const gid = Math.min(NV * block, total);
            const b_calculated = gid - a_stored;  // 存储的 B 不包含 star bit

            // 验证对角线性质: a + b = diagonal (存储时的值)
            const diagonal = a_stored + b_stored;
            const expectedDiagonal = gid;  // min(NV * block, total)

            // 比较
            if (b_stored !== b_calculated) {
                if (mismatchCount < maxMismatchesToShow) {
                    console.log(`Block ${block}: MISMATCH!`);
                    console.log(`  a_stored = ${a_stored}, star = ${star}`);
                    console.log(`  b_stored = ${b_stored}`);
                    console.log(`  b_calculated = gid(${gid}) - a(${a_stored}) = ${b_calculated}`);
                    console.log(`  difference = ${b_stored - b_calculated}`);
                    console.log(`  diagonal check: a + b = ${diagonal}, expected = ${expectedDiagonal}`);
                }
                allMatch = false;
                mismatchCount++;
            }
        }

        if (allMatch) {
            console.log(`✔ All ${numWorkgroups + 1} entries are consistent with ModernGPU formula!`);
        } else {
            console.log(`✗ ${mismatchCount} / ${numWorkgroups + 1} entries have mismatches`);
            if (mismatchCount > maxMismatchesToShow) {
                console.log(`  (showing first ${maxMismatchesToShow} mismatches)`);
            }
        }
        console.log('=====================================================\n');

        return allMatch;
    }

}

/**
 * Run test cases for the biased compute diagonals implementation.
 *
 * Note: numWorkgroups is calculated using ModernGPU formula: ceil(total / NV)
 * For small test arrays (total < NV=1792), this results in numWorkgroups=1,
 * which is the correct behavior matching the WGSL shader logic.
 */
export async function runDiagonalsBiasedTest(device: GPUDevice): Promise<void> {
    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestComputeDiagonalsBiased(device, timestampQueryManager);

    // Test case 1: Simple sorted arrays (small, single workgroup)
    const setA = new Uint32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const setB = new Uint32Array([2, 4, 6, 8, 10, 12]);
    const numWorkgroups = calculateNumWorkgroupsForDPI(setA.length, setB.length);

    console.log('Test Case 1: Simple sorted arrays');
    console.log('A:', Array.from(setA));
    console.log('B:', Array.from(setB));
    console.log(`numWorkgroups: ${numWorkgroups} (calculated from ceil(${setA.length + setB.length} / ${NV}))`);

    const result = await tester.testComputeDiagonals(setA, setB, numWorkgroups);
    tester.printDiagonals(result.aIndices, result.bIndices, result.stars, setA.length, setB.length);
    tester.validateAgainstCPU(result, setA, setB, numWorkgroups);
    console.log(`GPU Time: ${result.gpuTimeMs.toFixed(3)} ms`);
    const starCount = result.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount} / ${result.stars.length}`);
    console.log(`stars[] = [${result.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 2: Arrays with duplicates (small, single workgroup)
    const setA2 = new Uint32Array([1, 1, 2, 2, 3, 3, 4, 4]);
    const setB2 = new Uint32Array([1, 2, 2, 3, 4, 4]);
    const numWorkgroups2 = calculateNumWorkgroupsForDPI(setA2.length, setB2.length);

    console.log('Test Case 2: Arrays with duplicates');
    console.log('A:', Array.from(setA2));
    console.log('B:', Array.from(setB2));
    console.log(`numWorkgroups: ${numWorkgroups2} (calculated from ceil(${setA2.length + setB2.length} / ${NV}))`);

    const result2 = await tester.testComputeDiagonals(setA2, setB2, numWorkgroups2);
    tester.printDiagonals(result2.aIndices, result2.bIndices, result2.stars, setA2.length, setB2.length);
    tester.validateAgainstCPU(result2, setA2, setB2, numWorkgroups2);
    console.log(`GPU Time: ${result2.gpuTimeMs.toFixed(3)} ms`);
    const starCount2 = result2.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount2} / ${result2.stars.length}`);
    console.log(`stars[] = [${result2.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 3: Long duplicate runs (small, single workgroup)
    const setA3 = new Uint32Array([5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10]);
    const setB3 = new Uint32Array([5, 5, 5, 5, 10, 10, 10, 10, 10, 10]);
    const numWorkgroups3 = calculateNumWorkgroupsForDPI(setA3.length, setB3.length);

    console.log('Test Case 3: Long duplicate runs');
    console.log('A:', Array.from(setA3));
    console.log('B:', Array.from(setB3));
    console.log(`numWorkgroups: ${numWorkgroups3} (calculated from ceil(${setA3.length + setB3.length} / ${NV}))`);

    const result3 = await tester.testComputeDiagonals(setA3, setB3, numWorkgroups3);
    tester.printDiagonals(result3.aIndices, result3.bIndices, result3.stars, setA3.length, setB3.length);
    tester.validateAgainstCPU(result3, setA3, setB3, numWorkgroups3);
    console.log(`GPU Time: ${result3.gpuTimeMs.toFixed(3)} ms`);
    const starCount3 = result3.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount3} / ${result3.stars.length}`);
    console.log(`stars[] = [${result3.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 4: Large arrays to test multiple workgroups
    // Generate arrays large enough to require multiple workgroups (total > NV)
    const size4 = 2000;  // Each array has 2000 elements, total = 4000 > 1792
    const setA4 = new Uint32Array(size4);
    const setB4 = new Uint32Array(size4);
    for (let i = 0; i < size4; i++) {
        setA4[i] = i * 2;      // Even numbers: 0, 2, 4, ...
        setB4[i] = i * 2 + 1;  // Odd numbers: 1, 3, 5, ...
    }
    const numWorkgroups4 = calculateNumWorkgroupsForDPI(setA4.length, setB4.length);

    console.log('Test Case 4: Large arrays (multiple workgroups)');
    console.log(`A: [0, 2, 4, ..., ${setA4[size4-1]}] (${setA4.length} elements)`);
    console.log(`B: [1, 3, 5, ..., ${setB4[size4-1]}] (${setB4.length} elements)`);
    console.log(`numWorkgroups: ${numWorkgroups4} (calculated from ceil(${setA4.length + setB4.length} / ${NV}))`);

    const result4 = await tester.testComputeDiagonals(setA4, setB4, numWorkgroups4);
    tester.printDiagonals(result4.aIndices, result4.bIndices, result4.stars, setA4.length, setB4.length);
    tester.validateAgainstCPU(result4, setA4, setB4, numWorkgroups4);
    console.log(`GPU Time: ${result4.gpuTimeMs.toFixed(3)} ms`);
    const starCount4 = result4.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount4} / ${result4.stars.length}`);
    console.log(`stars[] = [${result4.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 5: Large arrays with duplicates (multiple workgroups)
    const size5 = 2000;
    const setA5 = new Uint32Array(size5);
    const setB5 = new Uint32Array(size5);
    for (let i = 0; i < size5; i++) {
        setA5[i] = Math.floor(i / 10);  // 0,0,0,...,0, 1,1,1,...,1, 2,2,2,...
        setB5[i] = Math.floor(i / 10);  // Same pattern, creates many duplicates
    }
    const numWorkgroups5 = calculateNumWorkgroupsForDPI(setA5.length, setB5.length);

    console.log('Test Case 5: Large arrays with duplicates (multiple workgroups)');
    console.log(`A: [0,0,...,0, 1,1,...,1, ...] (${setA5.length} elements, 10 copies each)`);
    console.log(`B: [0,0,...,0, 1,1,...,1, ...] (${setB5.length} elements, 10 copies each)`);
    console.log(`numWorkgroups: ${numWorkgroups5} (calculated from ceil(${setA5.length + setB5.length} / ${NV}))`);

    const result5 = await tester.testComputeDiagonals(setA5, setB5, numWorkgroups5);
    tester.printDiagonals(result5.aIndices, result5.bIndices, result5.stars, setA5.length, setB5.length);
    tester.validateAgainstCPU(result5, setA5, setB5, numWorkgroups5);
    console.log(`GPU Time: ${result5.gpuTimeMs.toFixed(3)} ms`);
    const starCount5 = result5.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount5} / ${result5.stars.length}`);
    console.log(`stars[] = [${result5.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 6: Highly unbalanced sizes (A tiny, B huge with runs)
    const setA6 = new Uint32Array(32);
    for (let i = 0; i < setA6.length; i++) setA6[i] = i * 3;  // 0, 3, 6, ...

    const b6Len = 50000;
    const setB6 = new Uint32Array(b6Len);
    // Make long runs: value increases every 50 elements -> lots of duplicates
    for (let i = 0; i < b6Len; i++) setB6[i] = Math.floor(i / 50);

    const numWorkgroups6 = calculateNumWorkgroupsForDPI(setA6.length, setB6.length);

    console.log('Test Case 6: Unbalanced sizes (A tiny, B huge with duplicates)');
    console.log(`A.length=${setA6.length}, B.length=${setB6.length}, numWorkgroups=${numWorkgroups6}`);

    const result6 = await tester.testComputeDiagonals(setA6, setB6, numWorkgroups6);
    tester.printDiagonals(result6.aIndices, result6.bIndices, result6.stars, setA6.length, setB6.length);
    tester.validateAgainstCPU(result6, setA6, setB6, numWorkgroups6);
    console.log(`GPU Time: ${result6.gpuTimeMs.toFixed(3)} ms`);
    const starCount6 = result6.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount6} / ${result6.stars.length}`);
    console.log(`stars[] = [${result6.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 7: All duplicates (max stress on duplicate balancing)
    const setA7 = new Uint32Array(6000);
    const setB7 = new Uint32Array(7000);
    setA7.fill(7);
    setB7.fill(7);

    const numWorkgroups7 = calculateNumWorkgroupsForDPI(setA7.length, setB7.length);

    console.log('Test Case 7: All duplicates (A=7..., B=7...)');
    console.log(`A.length=${setA7.length}, B.length=${setB7.length}, numWorkgroups=${numWorkgroups7}`);

    const result7 = await tester.testComputeDiagonals(setA7, setB7, numWorkgroups7);
    tester.printDiagonals(result7.aIndices, result7.bIndices, result7.stars, setA7.length, setB7.length);
    tester.validateAgainstCPU(result7, setA7, setB7, numWorkgroups7);
    console.log(`GPU Time: ${result7.gpuTimeMs.toFixed(3)} ms`);
    const starCount7 = result7.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount7} / ${result7.stars.length}`);
    console.log(`stars[] = [${result7.stars.map(v => v ? '1' : '0').join(', ')}]`);

    // Test case 8: NV boundary totals (NV-1, NV, NV+1)
    const a8Len = 900;
    for (const total8 of [NV - 1, NV, NV + 1]) {
        const b8Len = total8 - a8Len;  // ensure total matches exactly
        const setA8 = new Uint32Array(a8Len);
        const setB8 = new Uint32Array(b8Len);

        // Give both some duplicates to stress boundary + dup handling
        for (let i = 0; i < a8Len; i++) setA8[i] = Math.floor(i / 3);  // runs of 3
        for (let i = 0; i < b8Len; i++) setB8[i] = Math.floor(i / 5);  // runs of 5

        const numWorkgroups8 = calculateNumWorkgroupsForDPI(setA8.length, setB8.length);

        console.log(`Test Case 8: NV boundary total=${total8}`);
        console.log(`A.length=${setA8.length}, B.length=${setB8.length}, numWorkgroups=${numWorkgroups8}`);

        const result8 = await tester.testComputeDiagonals(setA8, setB8, numWorkgroups8);
        tester.printDiagonals(result8.aIndices, result8.bIndices, result8.stars, setA8.length, setB8.length);
        tester.validateAgainstCPU(result8, setA8, setB8, numWorkgroups8);
        console.log(`GPU Time: ${result8.gpuTimeMs.toFixed(3)} ms`);
        const starCount8 = result8.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
        console.log(`starCount = ${starCount8} / ${result8.stars.length}`);
        console.log(`stars[] = [${result8.stars.map(v => v ? '1' : '0').join(', ')}]`);
    }

    // Test case 9: Guaranteed star at partition k=1
    const a9Len = 3000;  // >= NV=1792
    const b9Len = 3000;

    const setA9 = new Uint32Array(a9Len);
    setA9[0] = 0;
    setA9.fill(1, 1);  // A = [0, 1, 1, 1, ..., 1]

    const setB9 = new Uint32Array(b9Len);
    setB9.fill(1);     // B = [1, 1, 1, ..., 1]

    const numWorkgroups9 = calculateNumWorkgroupsForDPI(setA9.length, setB9.length);

    console.log('Test Case 9: Guaranteed star');
    console.log(`A.length=${setA9.length}, B.length=${setB9.length}, numWorkgroups=${numWorkgroups9}`);

    const result9 = await tester.testComputeDiagonals(setA9, setB9, numWorkgroups9);
    tester.printDiagonals(result9.aIndices, result9.bIndices, result9.stars, setA9.length, setB9.length);
    tester.validateAgainstCPU(result9, setA9, setB9, numWorkgroups9);
    console.log(`GPU Time: ${result9.gpuTimeMs.toFixed(3)} ms`);
    const starCount9 = result9.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
    console.log(`starCount = ${starCount9} / ${result9.stars.length}`);
    console.log(`stars[] = [${result9.stars.map(v => v ? '1' : '0').join(', ')}]`);
    console.log(`partition[1] star = ${result9.stars[1]} (should be true if numWorkgroups>=2)`);

    // ========================================================================
    // Test cases matching Count Kernel test cases (for comparison)
    // ========================================================================

    // Test case 10: Match Count Kernel Case 5 - Many duplicates
    // Exact same data as test_count_kernel.ts Case 5
    console.log('\n' + '='.repeat(70));
    console.log('Test Case 10: [Match Count Kernel Case 5] Many duplicates');
    console.log('='.repeat(70));
    {
        const size = 2000;
        const setA10 = new Uint32Array(size);
        const setB10 = new Uint32Array(size);
        for (let i = 0; i < size; i++) {
            setA10[i] = Math.floor(i / 10) * 3;  // 0,0,...,0, 3,3,...,3, 6,...
            setB10[i] = Math.floor(i / 8) * 3;   // 0,0,...,0, 3,3,...,3, 6,...
        }
        const numWorkgroups10 = calculateNumWorkgroupsForDPI(setA10.length, setB10.length);

        console.log(`A: values = floor(i/10)*3, repeat 10 times each (${size} elements)`);
        console.log(`B: values = floor(i/8)*3, repeat 8 times each (${size} elements)`);
        console.log(`numWorkgroups: ${numWorkgroups10}`);

        const result10 = await tester.testComputeDiagonals(setA10, setB10, numWorkgroups10);
        tester.printDiagonals(result10.aIndices, result10.bIndices, result10.stars, setA10.length, setB10.length);
        tester.validateAgainstCPU(result10, setA10, setB10, numWorkgroups10);

        const starCount10 = result10.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
        console.log(`starCount = ${starCount10} / ${result10.stars.length}`);
        console.log(`stars[] = [${result10.stars.map(v => v ? '1' : '0').join(', ')}]`);

        // Show which partitions have star bits
        const starPartitions10 = result10.stars.map((s, i) => s ? i : -1).filter(i => i >= 0);
        console.log(`Partitions with star bit: [${starPartitions10.join(', ')}]`);
    }

    // Test case 11: Match Count Kernel Case 7 - Star bit trigger
    // Same as Test Case 9, but labeled for clarity
    console.log('\n' + '='.repeat(70));
    console.log('Test Case 11: [Match Count Kernel Case 7] Star bit trigger');
    console.log('='.repeat(70));
    {
        const a11Len = 3000;
        const b11Len = 3000;

        const setA11 = new Uint32Array(a11Len);
        setA11[0] = 0;
        setA11.fill(1, 1);  // A = [0, 1, 1, 1, ..., 1]

        const setB11 = new Uint32Array(b11Len);
        setB11.fill(1);     // B = [1, 1, 1, ..., 1]

        const numWorkgroups11 = calculateNumWorkgroupsForDPI(setA11.length, setB11.length);

        console.log(`A: [0, 1, 1, ..., 1] (${a11Len} elements)`);
        console.log(`B: [1, 1, ..., 1] (${b11Len} elements)`);
        console.log(`numWorkgroups: ${numWorkgroups11}`);

        const result11 = await tester.testComputeDiagonals(setA11, setB11, numWorkgroups11);
        tester.printDiagonals(result11.aIndices, result11.bIndices, result11.stars, setA11.length, setB11.length);
        tester.validateAgainstCPU(result11, setA11, setB11, numWorkgroups11);

        const starCount11 = result11.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
        console.log(`starCount = ${starCount11} / ${result11.stars.length}`);
        console.log(`stars[] = [${result11.stars.map(v => v ? '1' : '0').join(', ')}]`);

        const starPartitions11 = result11.stars.map((s, i) => s ? i : -1).filter(i => i >= 0);
        console.log(`Partitions with star bit: [${starPartitions11.join(', ')}]`);
    }

    // Test case 12: Match Count Kernel Case 8 - All same value
    console.log('\n' + '='.repeat(70));
    console.log('Test Case 12: [Match Count Kernel Case 8] All same value');
    console.log('='.repeat(70));
    {
        const setA12 = new Uint32Array(2500);
        const setB12 = new Uint32Array(2500);
        setA12.fill(42);
        setB12.fill(42);

        const numWorkgroups12 = calculateNumWorkgroupsForDPI(setA12.length, setB12.length);

        console.log(`A: [42, 42, ..., 42] (${setA12.length} elements)`);
        console.log(`B: [42, 42, ..., 42] (${setB12.length} elements)`);
        console.log(`numWorkgroups: ${numWorkgroups12}`);

        const result12 = await tester.testComputeDiagonals(setA12, setB12, numWorkgroups12);
        tester.printDiagonals(result12.aIndices, result12.bIndices, result12.stars, setA12.length, setB12.length);
        tester.validateAgainstCPU(result12, setA12, setB12, numWorkgroups12);

        const starCount12 = result12.stars.reduce((s, v) => s + (v ? 1 : 0), 0);
        console.log(`starCount = ${starCount12} / ${result12.stars.length}`);
        console.log(`stars[] = [${result12.stars.map(v => v ? '1' : '0').join(', ')}]`);

        const starPartitions12 = result12.stars.map((s, i) => s ? i : -1).filter(i => i >= 0);
        console.log(`Partitions with star bit: [${starPartitions12.join(', ')}]`);
    }
}

/**
 * Run benchmark with multiple iterations.
 */
export async function runDiagonalsBiasedBenchmark(
    device: GPUDevice,
    setA: Uint32Array,
    setB: Uint32Array,
    numWorkgroups: number,
    iters: number
): Promise<void> {
    const totalQueries = iters * 2;  // 2 queries per iteration
    const timestampQueryManager = new TimestampQueryManager(device, totalQueries);
    const tester = new TestComputeDiagonalsBiased(device, timestampQueryManager);

    console.log(`\nBenchmark: A=${setA.length}, B=${setB.length}, workgroups=${numWorkgroups}, iters=${iters}`);

    const result = await tester.benchmarkComputeDiagonals(setA, setB, numWorkgroups, iters);
    tester.printDiagonals(result.aIndices, result.bIndices, result.stars, setA.length, setB.length);

    console.log(`\nBenchmark Summary:`);
    console.log(`  Partition size: ${result.partitionSize}`);
    console.log(`  Avg GPU time: ${result.avgGpuTimeMs.toFixed(4)} ms`);
    console.log(`  Total wall time: ${result.wallTimeMs.toFixed(2)} ms`);
}

/**
 * Verify DPI consistency with ModernGPU formula.
 * This function runs compute_diagonals and then verifies that:
 *   b_stored == gid - a_stored + bit
 * where gid = NV * block
 */

/**
 * Calculate the number of workgroups using ModernGPU's formula.
 * num_wg = ceil((aLen + bLen) / NV)
 * @throws Error if total is 0 (both arrays are empty)
 */
export function calculateNumWorkgroupsForDPI(aLen: number, bLen: number): number {
    const total = aLen + bLen;
    if (total === 0) {
        throw new Error('Cannot compute DPI for empty arrays (total = 0)');
    }
    return Math.ceil(total / NV);
}

export async function verifyDPIConsistency(
    device: GPUDevice,
    setA: Uint32Array,
    setB: Uint32Array,
    numWorkgroups?: number  // Optional, auto-calculated if not provided
): Promise<boolean> {
    const timestampQueryManager = new TimestampQueryManager(device, 4);
    const tester = new TestComputeDiagonalsBiased(device, timestampQueryManager);

    const a_len = setA.length;
    const b_len = setB.length;

    // Auto-calculate numWorkgroups using ModernGPU formula: ceil(total / NV)
    const actualNumWorkgroups = numWorkgroups ?? calculateNumWorkgroupsForDPI(a_len, b_len);

    // Create GPU buffers
    const bufferA = device.createBuffer({
        size: Math.max(4, setA.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

    const bufferB = device.createBuffer({
        size: Math.max(4, setB.byteLength),
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

    const dpiSize = 2 * (actualNumWorkgroups + 1);
    const bufferDPI = device.createBuffer({
        size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    const bufferALength = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferALength, 0, new Uint32Array([a_len]));

    const bufferBLength = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferBLength, 0, new Uint32Array([b_len]));

    const bufferNumWg = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([actualNumWorkgroups]));

    // Create bind group (need to access the pipeline from tester)
    // For simplicity, let's run the test and get raw data back
    const result = await tester.testComputeDiagonals(setA, setB, actualNumWorkgroups);

    // Re-run to get raw DPI data
    const bindGroup = device.createBindGroup({
        layout: (tester as any).bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferALength } },
            { binding: 4, resource: { buffer: bufferBLength } },
            { binding: 5, resource: { buffer: bufferNumWg } },
        ]
    });

    const dispatchX = Math.min(actualNumWorkgroups, MAXWORKGROUP);
    const dispatchY = Math.ceil(actualNumWorkgroups / MAXWORKGROUP);

    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline((tester as any).computeDiagonalsPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(dispatchX, dispatchY, 1);
    pass.end();
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read back raw DPI data
    const readbackBuffer = device.createBuffer({
        size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const copyEncoder = device.createCommandEncoder();
    copyEncoder.copyBufferToBuffer(bufferDPI, 0, readbackBuffer, 0, dpiSize * 4);
    device.queue.submit([copyEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const dpiData = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
    readbackBuffer.unmap();

    // Run verification
    console.log(`\n=== Verifying DPI Consistency ===`);
    console.log(`A.length = ${a_len}, B.length = ${b_len}`);
    console.log(`numWorkgroups = ${actualNumWorkgroups} (auto-calculated: ceil(${a_len + b_len} / ${NV}))`);
    console.log(`NV (ModernGPU fixed) = ${NV}`);

    const isConsistent = tester.verifyDPIConsistencyWithModernGPU(
        dpiData, actualNumWorkgroups, a_len, b_len, NV
    );

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferDPI.destroy();
    bufferALength.destroy();
    bufferBLength.destroy();
    bufferNumWg.destroy();
    readbackBuffer.destroy();

    return isConsistent;
}
