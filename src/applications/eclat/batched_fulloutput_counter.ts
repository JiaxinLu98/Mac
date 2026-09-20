/**
 * BatchedFullOutputCounter
 *
 * Like BatchedPerPairCounter but also writes intersection elements to an
 * output buffer. Returns per-pair counts AND the flat output array.
 *
 * Used by ECLAT v2 where tidsets must stay on GPU for the next level.
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import batchedDpiShader from '../../balanced_path/common/balanced_path_batched.wgsl';
import batchedFullOutputShader from './set_availability_decoupled_lookback_batched_fulloutput.wgsl';

const MAXWORKGROUP = 65535;
const DPI_WG_SIZE = 256;

export interface BatchedFullOutputTiming {
    dpiMs: number;
    lookbackMs: number;
    totalMs: number;
}

export interface BatchedFullOutputGPUResult {
    pairCounts: Uint32Array;
    outputBuffer: GPUBuffer;
    timing: BatchedFullOutputTiming;
}

export class BatchedFullOutputCounter {
    private device: GPUDevice;
    private tsm: TimestampQueryManager;

    private dpiPipeline: GPUComputePipeline;
    private dpiBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, tsm: TimestampQueryManager) {
        this.device = device;
        this.tsm = tsm;

        // DPI (same as other batched variants)
        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'FullOutput DPI bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.dpiPipeline = device.createComputePipeline({
            label: 'FullOutput DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: batchedDpiShader }),
                entryPoint: 'compute_diagonals_batched'
            }
        });

        // Full output lookback (9 bindings)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'FullOutput Lookback bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // keysA
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // keysB
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // state
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // pairCounts
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // wgInfo
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // pairIdPerWg
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // pairOutputOffsetPerWg
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // totalWg
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: 'FullOutput Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: batchedFullOutputShader }),
                entryPoint: 'decoupled_lookback_batched_kernel'
            }
        });
    }

    async run(
        keysA: Uint32Array,
        keysB: Uint32Array,
        wgInfo: Uint32Array,
        pairIdPerWg: Uint32Array,
        pairOutputOffsetPerWg: Uint32Array,
        totalWg: number,
        totalDpiEntries: number,
        numPairs: number,
        maxOutputSize: number,
    ): Promise<{
        pairCounts: Uint32Array;
        output: Uint32Array;
        timing: BatchedFullOutputTiming;
    }> {
        const device = this.device;

        if (totalWg === 0 || numPairs === 0) {
            return { pairCounts: new Uint32Array(numPairs), output: new Uint32Array(0), timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 } };
        }

        const bufKeysA = device.createBuffer({ size: Math.max(4, keysA.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufKeysA, 0, new Uint32Array(keysA));

        const bufKeysB = device.createBuffer({ size: Math.max(4, keysB.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufKeysB, 0, new Uint32Array(keysB));

        try {
            const gpuResult = await this.runPackedGPU(
                bufKeysA,
                bufKeysB,
                wgInfo,
                pairIdPerWg,
                pairOutputOffsetPerWg,
                totalWg,
                totalDpiEntries,
                numPairs,
                maxOutputSize
            );

            try {
                const output = await this.readOutputBuffer(gpuResult.outputBuffer, maxOutputSize);
                return { pairCounts: gpuResult.pairCounts, output, timing: gpuResult.timing };
            } finally {
                gpuResult.outputBuffer.destroy();
            }
        } finally {
            bufKeysA.destroy();
            bufKeysB.destroy();
        }
    }

    async runPackedGPU(
        bufKeysA: GPUBuffer,
        bufKeysB: GPUBuffer,
        wgInfo: Uint32Array,
        pairIdPerWg: Uint32Array,
        pairOutputOffsetPerWg: Uint32Array,
        totalWg: number,
        totalDpiEntries: number,
        numPairs: number,
        maxOutputSize: number,
    ): Promise<BatchedFullOutputGPUResult> {
        const device = this.device;

        if (totalWg === 0 || numPairs === 0) {
            return {
                pairCounts: new Uint32Array(numPairs),
                outputBuffer: this.createOutputBuffer(1),
                timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 }
            };
        }

        const bufDpi = device.createBuffer({ size: Math.max(4, totalDpiEntries * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const bufWgInfo = device.createBuffer({ size: Math.max(4, wgInfo.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufWgInfo, 0, new Uint32Array(wgInfo));

        const bufPairIdPerWg = device.createBuffer({ size: Math.max(4, pairIdPerWg.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufPairIdPerWg, 0, new Uint32Array(pairIdPerWg));

        const bufPairOutputOffsetPerWg = device.createBuffer({ size: Math.max(4, pairOutputOffsetPerWg.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufPairOutputOffsetPerWg, 0, new Uint32Array(pairOutputOffsetPerWg));

        const bufState = device.createBuffer({ size: Math.max(4, totalWg * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufState, 0, new Uint32Array(totalWg).fill(0));

        const bufPairCounts = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufPairCounts, 0, new Uint32Array(numPairs).fill(0));

        const bufOutput = this.createOutputBuffer(maxOutputSize);

        const bufTotalWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufTotalWg, 0, new Uint32Array([totalWg]));

        // Bind groups
        const dpiBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufWgInfo } },
                { binding: 4, resource: { buffer: bufTotalWg } },
            ]
        });

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufState } },
                { binding: 4, resource: { buffer: bufPairCounts } },
                { binding: 5, resource: { buffer: bufOutput } },
                { binding: 6, resource: { buffer: bufWgInfo } },
                { binding: 7, resource: { buffer: bufPairIdPerWg } },
                { binding: 8, resource: { buffer: bufPairOutputOffsetPerWg } },
                { binding: 9, resource: { buffer: bufTotalWg } },
            ]
        });

        // Dispatch
        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(totalWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);
        const lookbackDispatchX = Math.min(totalWg, MAXWORKGROUP);
        const lookbackDispatchY = Math.ceil(totalWg / MAXWORKGROUP);

        await device.queue.onSubmittedWorkDone();

        const encoder = device.createCommandEncoder();

        let pass = encoder.beginComputePass(this.tsm.createComputePassDescriptor(0, 1));
        pass.setPipeline(this.dpiPipeline);
        pass.setBindGroup(0, dpiBindGroup);
        pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
        pass.end();

        pass = encoder.beginComputePass(this.tsm.createComputePassDescriptor(2, 3));
        pass.setPipeline(this.lookbackPipeline);
        pass.setBindGroup(0, lookbackBindGroup);
        pass.dispatchWorkgroups(lookbackDispatchX, lookbackDispatchY);
        pass.end();

        // Readback pairCounts
        const readbackCounts = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        this.tsm.resolve(encoder);
        encoder.copyBufferToBuffer(bufPairCounts, 0, readbackCounts, 0, numPairs * 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        const timestamps = await this.tsm.downloadTimestampResult();
        let dpiMs = 0, lookbackMs = 0, totalMs = 0;
        if (timestamps.length >= 4) {
            dpiMs = (timestamps[1] - timestamps[0]) / 1_000_000;
            lookbackMs = (timestamps[3] - timestamps[2]) / 1_000_000;
            totalMs = (timestamps[3] - timestamps[0]) / 1_000_000;
        }

        await readbackCounts.mapAsync(GPUMapMode.READ);
        const pairCounts = new Uint32Array(readbackCounts.getMappedRange().slice(0));
        readbackCounts.unmap();

        // Cleanup
        bufDpi.destroy();
        bufWgInfo.destroy(); bufPairIdPerWg.destroy(); bufPairOutputOffsetPerWg.destroy();
        bufState.destroy(); bufPairCounts.destroy();
        bufTotalWg.destroy();
        readbackCounts.destroy();

        return { pairCounts, outputBuffer: bufOutput, timing: { dpiMs, lookbackMs, totalMs } };
    }

    private createOutputBuffer(maxOutputSize: number): GPUBuffer {
        return this.device.createBuffer({
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
    }

    private async readOutputBuffer(outputBuffer: GPUBuffer, maxOutputSize: number): Promise<Uint32Array> {
        if (maxOutputSize === 0) {
            return new Uint32Array(0);
        }

        const device = this.device;
        const readbackOutput = device.createBuffer({
            size: maxOutputSize * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        try {
            const readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(outputBuffer, 0, readbackOutput, 0, maxOutputSize * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await readbackOutput.mapAsync(GPUMapMode.READ);
            const outputData = new Uint32Array(readbackOutput.getMappedRange().slice(0));
            readbackOutput.unmap();
            return outputData;
        } finally {
            readbackOutput.destroy();
        }
    }
}
