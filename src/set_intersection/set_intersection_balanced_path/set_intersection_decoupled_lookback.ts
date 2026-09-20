/**
 * Set Intersection using Balanced Path with Decoupled Lookback
 *
 * Two-phase algorithm with single GPU submit:
 * 1. Balanced Path (Global) - Compute DPI with merge path partition points
 * 2. Decoupled Lookback - Single kernel that counts, scans, and writes
 *
 * Advantages over 4-phase:
 * - Eliminates separate prefix sum phase
 * - Reduces kernel launch overhead
 * - Single CommandEncoder submit
 *
 * Advantages over atomic version:
 * - Output is in deterministic order
 * - No atomic contention on global counter
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import decoupledLookbackShader from './set_availability_intersection_decoupled_lookback.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants - must match WGSL shader
const NT = 256;       // Threads per workgroup
const VT = 7;         // Values per thread
const NV = NT * VT;   // Elements per workgroup = 1792

export class GPUSetIntersectionDecoupledLookback {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI computation pipeline
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;

    // Decoupled Lookback pipeline
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        // ========================================================================
        // Phase 1: Balanced Path Pipeline (compute DPI)
        // ========================================================================
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

        // ========================================================================
        // Phase 2: Decoupled Lookback Pipeline
        // ========================================================================
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'Decoupled Lookback bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // state
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // output
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // total_count
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // a_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // b_length
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: 'Decoupled Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: decoupledLookbackShader }),
                entryPoint: 'intersection_decoupled_lookback'
            }
        });
    }

    public setIterationIndex(i: number) {
        this.iterationIndex = i;
    }

    public computeQueries(lenA: number, lenB: number) {
        // DPI: 2 + Decoupled Lookback: 2 = 4
        return this.queriesPerIter = 4;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeQueries(lenA, lenB) must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    /**
     * Calculate number of workgroups
     */
    public static calculateNumWorkgroups(aLen: number, bLen: number): number {
        const total = aLen + bLen;
        if (total === 0) return 0;
        return Math.ceil(total / NV);
    }

    /**
     * Main entry point: Compute set intersection using 2-phase decoupled lookback algorithm
     * Single CommandEncoder submit for both phases
     */
    public async computeIntersection(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number = 1
    ): Promise<{
        result: Uint32Array;
        totalCount: number;
        numWorkgroups: number;
        avgDpiMs: number;
        avgLookbackMs: number;
        avgTotalMs: number;
    }> {
        const device = this.device;
        const a_len = setA.length;
        const b_len = setB.length;

        if (a_len === 0 || b_len === 0) {
            return {
                result: new Uint32Array(0),
                totalCount: 0,
                numWorkgroups: 0,
                avgDpiMs: 0,
                avgLookbackMs: 0,
                avgTotalMs: 0
            };
        }

        const numWg = GPUSetIntersectionDecoupledLookback.calculateNumWorkgroups(a_len, b_len);
        this.computeQueries(a_len, b_len);
        const QUERIES_PER_ITER = this.queriesPerIter;

        // Pre-allocate output buffer with maximum possible size
        const maxOutputSize = Math.min(a_len, b_len);

        // Create input buffers
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
        const bufferALen = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const bufferBLen = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        const bufferNumWg = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
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

        // State buffer for decoupled lookback
        const bufferState = device.createBuffer({
            label: 'Buffer State',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Output buffer
        const bufferOutput = device.createBuffer({
            label: 'Buffer Output',
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        // Total count buffer
        const bufferTotalCount = device.createBuffer({
            label: 'Buffer Total Count',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Create bind groups
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

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferState } },
                { binding: 4, resource: { buffer: bufferOutput } },
                { binding: 5, resource: { buffer: bufferTotalCount } },
                { binding: 6, resource: { buffer: bufferALen } },
                { binding: 7, resource: { buffer: bufferBLen } },
                { binding: 8, resource: { buffer: bufferNumWg } },
            ]
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        let lastTotalCount = 0;
        let lastResult: Uint32Array = new Uint32Array(0);
        let wallTotalMs = 0;

        for (let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const base = this.getQueryBaseOffset();

            // Initialize state buffer to 0 (STATUS_NOT_READY)
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            // Initialize total count to 0
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const t0 = performance.now();

            // Single CommandEncoder with both passes
            const encoder = device.createCommandEncoder({ label: 'Decoupled Lookback Intersection' });

            // Pass 1: DPI
            const pass1 = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1)
            );
            pass1.setPipeline(this.diagPipeline);
            pass1.setBindGroup(0, diagBindGroup);
            pass1.dispatchWorkgroups(dispatchX, dispatchY);
            pass1.end();

            // Pass 2: Decoupled Lookback
            const pass2 = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(base + 2, base + 3)
            );
            pass2.setPipeline(this.lookbackPipeline);
            pass2.setBindGroup(0, lookbackBindGroup);
            pass2.dispatchWorkgroups(dispatchX, dispatchY);
            pass2.end();

            // Resolve timestamps and submit
            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);

            // Read back total count
            const totalCountReadback = device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const copyEncoder1 = device.createCommandEncoder();
            copyEncoder1.copyBufferToBuffer(bufferTotalCount, 0, totalCountReadback, 0, 4);
            device.queue.submit([copyEncoder1.finish()]);
            await device.queue.onSubmittedWorkDone();

            await totalCountReadback.mapAsync(GPUMapMode.READ);
            lastTotalCount = new Uint32Array(totalCountReadback.getMappedRange().slice(0))[0];
            totalCountReadback.unmap();
            totalCountReadback.destroy();

            // Read back results (only what we need)
            if (lastTotalCount > 0) {
                const resultReadback = device.createBuffer({
                    size: lastTotalCount * 4,
                    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
                });
                const copyEncoder2 = device.createCommandEncoder();
                copyEncoder2.copyBufferToBuffer(bufferOutput, 0, resultReadback, 0, lastTotalCount * 4);
                device.queue.submit([copyEncoder2.finish()]);
                await device.queue.onSubmittedWorkDone();

                await resultReadback.mapAsync(GPUMapMode.READ);
                lastResult = new Uint32Array(resultReadback.getMappedRange().slice(0));
                resultReadback.unmap();
                resultReadback.destroy();
            } else {
                lastResult = new Uint32Array(0);
            }
        }

        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `[DecoupledLookback] ComputeIntersection x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        // Process GPU timestamps
        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumDpi = 0;
        let sumLookback = 0;

        for (let i = 0; i < iters; ++i) {
            const base = i * QUERIES_PER_ITER;
            sumDpi += timestamps[base + 1] - timestamps[base + 0];
            sumLookback += timestamps[base + 3] - timestamps[base + 2];
        }

        const timestampPeriod = 1e-9;
        const avgDpiMs = (sumDpi / iters) * timestampPeriod * 1000;
        const avgLookbackMs = (sumLookback / iters) * timestampPeriod * 1000;
        const avgTotalMs = avgDpiMs + avgLookbackMs;

        console.log(
            `[DecoupledLookback] GPU timestamps avg over ${iters} iterations:`,
            `dpi = ${avgDpiMs.toFixed(4)} ms,`,
            `lookback = ${avgLookbackMs.toFixed(4)} ms,`,
            `total = ${avgTotalMs.toFixed(4)} ms`
        );

        // Validate result
        this.validateCount(lastTotalCount, setA, setB);

        // Cleanup
        bufferA.destroy();
        bufferB.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferState.destroy();
        bufferOutput.destroy();
        bufferTotalCount.destroy();

        return {
            result: lastResult,
            totalCount: lastTotalCount,
            numWorkgroups: numWg,
            avgDpiMs,
            avgLookbackMs,
            avgTotalMs
        };
    }

    /**
     * CPU reference implementation for validation (MULTISET intersection)
     */
    public cpuCountIntersection(setA: Uint32Array, setB: Uint32Array): number {
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
     * Validate GPU count result against CPU reference
     */
    public validateCount(gpuTotalCount: number, setA: Uint32Array, setB: Uint32Array): boolean {
        const cpuCount = this.cpuCountIntersection(setA, setB);
        const valid = gpuTotalCount === cpuCount;

        if (valid) {
            console.log(`[DecoupledLookback] Count validation PASSED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        } else {
            console.log(`[DecoupledLookback] Count validation FAILED: GPU=${gpuTotalCount}, CPU=${cpuCount}`);
        }

        return valid;
    }
}
