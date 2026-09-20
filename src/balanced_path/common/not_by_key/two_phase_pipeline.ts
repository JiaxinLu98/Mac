/**
 * Reusable 2-Phase Pipeline (DPI -> Decoupled Lookback) for set operations.
 *
 * Extracted from test_unified_4phase_vs_2phase.ts so that other modules
 * (e.g. triangle_query) can use the pipeline without duplicating code.
 */

import TimestampQueryManager from '../../../TimestampQueryManager';
import { gpuPreheat } from '../../../gpu_preheat';
import { setOpMode } from '../../../utils';
import computeDiagonalsShader from '../balanced_path_biased.wgsl';
import lookbackShaderBase from './set_availability_decoupled_lookback.wgsl';

export const MAXWORKGROUP = 65535;
export const NT = 256;
export const VT = 12;
export const NV = NT * VT;  // 3072
export const DPI_WG_SIZE = 256;

/** Max possible output size for each operation. */
export function getMaxOutputSize(opMode: number, aLen: number, bLen: number): number {
    switch (opMode) {
        case 0: return Math.min(aLen, bLen);       // intersection
        case 1: return aLen;                        // difference
        case 2: return aLen + bLen;                 // union
        case 3: return aLen + bLen;                 // sym_difference
        default: return aLen + bLen;
    }
}

/**
 * 2-Phase Pipeline Tester (DPI -> Decoupled Lookback)
 */
export class TwoPhasePipelineTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;
    private label: string;
    private opMode: number;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager, label: string, opMode: number) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.label = label;
        this.opMode = opMode;

        const lookbackShader = setOpMode(lookbackShaderBase, opMode);

        // DPI bind group layout
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: `${label} DPI bind group layout`,
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
            label: `${label} DPI pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback bind group layout (9 bindings)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: `${label} Decoupled Lookback bind group layout`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // state
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // total_count
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // b_length
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // num_wg_total
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: `${label} Decoupled Lookback pipeline`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: lookbackShader }),
                entryPoint: 'decoupled_lookback_kernel'
            }
        });
    }

    public async run(
        aKeys: Uint32Array,
        bKeys: Uint32Array,
        iterations: number,
        warmup: number,
        preheatMs: number = 0
    ): Promise<{
        totalCount: number;
        timing: {
            dpiMs: number;
            lookbackMs: number;
            totalMs: number;
        };
        runs: {
            dpiMs: number[];
            lookbackMs: number[];
            kernelMs: number[];
            spanMs: number[];
        };
    }> {
        const device = this.device;
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                totalCount: 0,
                timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 },
                runs: { dpiMs: [], lookbackMs: [], kernelMs: [], spanMs: [] }
            };
        }

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = getMaxOutputSize(this.opMode, a_len, b_len);

        // Create input buffers
        const bufferA = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(aKeys));

        const bufferB = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, new Uint32Array(bKeys));

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const dpiSize = 2 * (numWg + 1);
        const bufferDPI = device.createBuffer({
            size: dpiSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferState = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        const bufferOutput = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferTotalCount = device.createBuffer({
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

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        await device.queue.onSubmittedWorkDone();

        // Unrelated GPU work first, so the GPU reaches its steady performance state.
        await gpuPreheat(device, preheatMs);

        // Warmup runs
        for (let iter = 0; iter < warmup; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed runs
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();

            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            this.timestampQueryManager.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            if (timestamps.length >= 4) {
                const dpiNs = timestamps[1] - timestamps[0];
                const lookbackNs = timestamps[3] - timestamps[2];
                const totalNs = timestamps[3] - timestamps[0];

                dpiTimes.push(dpiNs / 1_000_000);
                lookbackTimes.push(lookbackNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        // Distribution stats on total time (rebuttal: variance, CI, mean vs median/p95).
        if (totalTimes.length > 1) {
            const n = totalTimes.length;
            const m = avg(totalTimes);
            const sd = Math.sqrt(totalTimes.reduce((a, b) => a + (b - m) * (b - m), 0) / (n - 1));
            const cv = m > 0 ? 100 * sd / m : 0;
            const sem = sd / Math.sqrt(n);
            const s = [...totalTimes].sort((a, b) => a - b);
            const median = s[Math.floor(n / 2)];
            const p95 = s[Math.min(n - 1, Math.floor(n * 0.95))];
            console.log(`    [stats] total mean=${m.toFixed(4)} sd=${sd.toFixed(4)} cv=${cv.toFixed(2)}% sem=${sem.toFixed(4)} (95%CI +/-${(1.96 * sem).toFixed(4)}) median=${median.toFixed(4)} p95=${p95.toFixed(4)}`);
        }

        // Readback
        const totalCountReadback = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferTotalCount, 0, totalCountReadback, 0, 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await totalCountReadback.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(totalCountReadback.getMappedRange().slice(0))[0];
        totalCountReadback.unmap();

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
        totalCountReadback.destroy();

        return {
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                lookbackMs: avg(lookbackTimes),
                totalMs: avg(totalTimes),
            },
            // Every timed run. Kernel time is DPI + Lookback, the unit of the CUDA event
            // timers. Span runs from the start of the DPI pass to the end of the Lookback pass.
            runs: {
                dpiMs: dpiTimes,
                lookbackMs: lookbackTimes,
                kernelMs: dpiTimes.map((d, i) => d + lookbackTimes[i]),
                spanMs: totalTimes,
            }
        };
    }
}
