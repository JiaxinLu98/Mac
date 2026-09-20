import computeDiagonalsShader from './balanced_path.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const MAXWORKGROUP = 65535;

export class TestComputeDiagonals {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private computeDiagonalsPipeline: GPUComputePipeline;
    private bindGroupLayout: GPUBindGroupLayout;

    // Timestamp query management
    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // Create bind group layout matching WGSL bindings
        // balanced_path.wgsl bindings:
        //   binding 0: a (storage, read)
        //   binding 1: b (storage, read)
        //   binding 2: dpi (storage, read_write)
        //   binding 3: a_length (uniform)
        //   binding 4: b_length (uniform)
        //   binding 5: num_wg_uniform (uniform)
        this.bindGroupLayout = device.createBindGroupLayout({
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

        const pipelineLayout = device.createPipelineLayout({
            label: 'compute diagonals pipeline layout',
            bindGroupLayouts: [this.bindGroupLayout]
        });

        const shader = device.createShaderModule({
            label: 'compute diagonals shader',
            code: computeDiagonalsShader
        });

        this.computeDiagonalsPipeline = device.createComputePipeline({
            label: 'compute diagonals pipeline',
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: 'compute_diagonals'
            }
        });
    }

    public setIterationIndex(i: number): void {
        this.iterationIndex = i;
    }

    public computeQueries(): number {
        return this.queriesPerIter = 2;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries() must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    public async testComputeDiagonals(
        setA: Uint32Array,
        setB: Uint32Array,
        numWorkgroups: number
    ): Promise<{ aIndices: Uint32Array; bIndices: Uint32Array; stars: boolean[]; gpuTimeMs: number }> {

        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

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

        // Pass total number of workgroups as uniform (for 2D dispatch support)
        const bufferNumWgTotal = device.createBuffer({
            label: 'num_wg_total uniform',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWorkgroups]));

        // Create bind group
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALength } },
                { binding: 4, resource: { buffer: bufferBLength } },
                { binding: 5, resource: { buffer: bufferNumWgTotal } },
            ],
        });

        // Calculate 2D dispatch dimensions
        const dispatchX = Math.min(numWorkgroups, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWorkgroups / MAXWORKGROUP);

        // Dispatch compute shader with timestamp queries
        const commandEncoder = device.createCommandEncoder({ label: 'Compute Diagonals' });
        const passDescriptor = this.timestampQueryManager.createComputePassDescriptor(0, 1);
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
            const gpuTicks = timestamps[1] - timestamps[0];
            gpuTimeMs = gpuTicks * 1e-6;  // nanoseconds to milliseconds
        } catch (e) {
            // Timestamp query not supported, gpuTimeMs remains 0
        }

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferDPI.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWgTotal.destroy();
        readbackBuffer.destroy();

        return { aIndices, bIndices, stars, gpuTimeMs };
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

    /**
     * Benchmark compute diagonals with multiple iterations.
     * @param warmupIters Number of warmup iterations (not timed) to stabilize GPU state
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

        const dpiSize = 2 * (numWorkgroups + 1);
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

        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const t0 = performance.now();

            const bufferDPI = device.createBuffer({
                label: `Buffer DPI iter ${i}`,
                size: dpiSize * Uint32Array.BYTES_PER_ELEMENT,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });

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

            const commandEncoder = device.createCommandEncoder({ label: `Compute Diagonals iter ${i}` });
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

            // Read back DPI buffer only on last iteration
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
            wallTotalMs += (performance.now() - t0);
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `ComputeDiagonals (Non-Biased) x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

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

        bufferA.destroy();
        bufferB.destroy();
        bufferALength.destroy();
        bufferBLength.destroy();
        bufferNumWg.destroy();

        return {
            aIndices: lastAIndices,
            bIndices: lastBIndices,
            stars: lastStars,
            avgGpuTimeMs,
            wallTimeMs: wallTotalMs,
            partitionSize: Math.ceil((a_len + b_len) / numWorkgroups)
        };
    }
}
