/**
 * InPlaceFullOutputCounter
 *
 * Batched intersection with full output for GPU-resident inputs:
 *   - every tidset is read in place from its own buffer (no packing copy), so a
 *     chunk binds exactly one A buffer and one B buffer;
 *   - pair-level metadata is stored once per pair (pairInfo) plus one pair id
 *     per workgroup;
 *   - all chunks passed to run() are encoded into one command buffer and share
 *     pooled scratch buffers through aligned binding windows, with a single wait
 *     for the per-pair counts and timestamps.
 *
 * Used by ECLAT, where tidsets stay on the GPU across levels.
 */

import batchedDpiShader from './balanced_path_batched_pairinfo.wgsl';
import lookbackShader from './set_availability_decoupled_lookback_batched_fulloutput_pairinfo.wgsl';

const MAXWORKGROUP = 65535;
const DPI_WG_SIZE = 256;
export const PAIR_INFO_STRIDE = 8;
// A timestamp query set holds at most 4096 queries; each chunk uses 4.
export const MAX_CHUNKS_PER_SUBMIT = 1024;
const QUERY_COUNT = 4 * MAX_CHUNKS_PER_SUBMIT;

export interface InPlaceChunk {
    keysA: GPUBuffer;
    keysB: GPUBuffer;
    numPairs: number;
    totalWg: number;
    totalDpiEntries: number;
    maxOutputSize: number;
    /** numPairs * PAIR_INFO_STRIDE entries; state/dpi/output offsets are chunk-local */
    pairInfo: Uint32Array;
    /** totalWg entries: chunk-local pair id of every workgroup */
    pairIdPerWg: Uint32Array;
}

export interface InPlaceSubmitResult {
    pairCounts: Uint32Array[];
    outputBuffers: GPUBuffer[];
    // encodeMs: CPU time to write metadata and encode the passes. waitMs: submit until counts are mapped.
    // readbackMs: copying counts and timestamps out of the mapped buffers. runMs: the whole run() call.
    timing: { dpiMs: number; lookbackMs: number; totalMs: number; encodeMs: number; waitMs: number; readbackMs: number; runMs: number };
}

function alignUp(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

/** Aligned windows of one submission's shared scratch buffers, one per chunk. */
export class InPlaceSubmitLayout {
    readonly infoOffsets: number[] = [];
    readonly pairIdOffsets: number[] = [];
    readonly dpiOffsets: number[] = [];
    readonly stateOffsets: number[] = [];
    readonly countsOffsets: number[] = [];
    infoBytes = 0;
    pairIdBytes = 0;
    dpiBytes = 0;
    stateBytes = 0;
    countsBytes = 0;

    constructor(private readonly alignment: number, private readonly limit: number) {}

    get chunkCount(): number {
        return this.infoOffsets.length;
    }

    fits(chunk: InPlaceChunk): boolean {
        const a = this.alignment;
        return this.chunkCount < MAX_CHUNKS_PER_SUBMIT &&
            alignUp(this.infoBytes, a) + chunk.numPairs * PAIR_INFO_STRIDE * 4 <= this.limit &&
            alignUp(this.pairIdBytes, a) + chunk.totalWg * 4 <= this.limit &&
            alignUp(this.dpiBytes, a) + chunk.totalDpiEntries * 4 <= this.limit &&
            alignUp(this.stateBytes, a) + chunk.totalWg * 4 <= this.limit &&
            alignUp(this.countsBytes, a) + chunk.numPairs * 4 <= this.limit;
    }

    add(chunk: InPlaceChunk): void {
        this.infoBytes = this.place(this.infoOffsets, this.infoBytes, chunk.numPairs * PAIR_INFO_STRIDE * 4);
        this.pairIdBytes = this.place(this.pairIdOffsets, this.pairIdBytes, chunk.totalWg * 4);
        this.dpiBytes = this.place(this.dpiOffsets, this.dpiBytes, chunk.totalDpiEntries * 4);
        this.stateBytes = this.place(this.stateOffsets, this.stateBytes, chunk.totalWg * 4);
        this.countsBytes = this.place(this.countsOffsets, this.countsBytes, chunk.numPairs * 4);
    }

    private place(offsets: number[], total: number, bytes: number): number {
        const offset = alignUp(total, this.alignment);
        offsets.push(offset);
        return offset + bytes;
    }
}

/** Scratch buffer reused across submissions; recreated only when it must grow. */
class PooledBuffer {
    private buffer: GPUBuffer | null = null;

    constructor(
        private readonly device: GPUDevice,
        private readonly label: string,
        private readonly usage: GPUBufferUsageFlags
    ) {}

    ensure(byteLength: number): GPUBuffer {
        const size = Math.max(4, alignUp(byteLength, 4));
        if (this.buffer === null || this.buffer.size < size) {
            if (this.buffer !== null) this.buffer.destroy();
            this.buffer = this.device.createBuffer({ label: this.label, size, usage: this.usage });
        }
        return this.buffer;
    }
}

export class InPlaceFullOutputCounter {
    private readonly device: GPUDevice;

    private readonly dpiPipeline: GPUComputePipeline;
    private readonly dpiBindGroupLayout: GPUBindGroupLayout;
    private readonly lookbackPipeline: GPUComputePipeline;
    private readonly lookbackBindGroupLayout: GPUBindGroupLayout;

    private readonly pairInfo: PooledBuffer;
    private readonly pairIdPerWg: PooledBuffer;
    private readonly dpi: PooledBuffer;
    private readonly state: PooledBuffer;
    private readonly pairCounts: PooledBuffer;
    private readonly pairCountsRead: PooledBuffer;
    private readonly totalWg: PooledBuffer;

    private readonly querySet: GPUQuerySet | null = null;
    private readonly timestampResolve: GPUBuffer | null = null;
    private readonly timestampRead: GPUBuffer | null = null;

    constructor(device: GPUDevice) {
        this.device = device;

        const entry = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });

        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'InPlace DPI bind group layout',
            entries: [
                entry(0, 'read-only-storage'),  // keysA
                entry(1, 'read-only-storage'),  // keysB
                entry(2, 'storage'),            // dpi
                entry(3, 'read-only-storage'),  // pairInfo
                entry(4, 'read-only-storage'),  // pairIdPerWg
                entry(5, 'uniform'),            // totalWg
            ]
        });
        this.dpiPipeline = device.createComputePipeline({
            label: 'InPlace DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: { module: device.createShaderModule({ code: batchedDpiShader }), entryPoint: 'compute_diagonals_batched' }
        });

        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'InPlace Lookback bind group layout',
            entries: [
                entry(0, 'read-only-storage'),  // keysA
                entry(1, 'read-only-storage'),  // keysB
                entry(2, 'read-only-storage'),  // dpi
                entry(3, 'storage'),            // state
                entry(4, 'storage'),            // pairCounts
                entry(5, 'storage'),            // output
                entry(6, 'read-only-storage'),  // pairInfo
                entry(7, 'read-only-storage'),  // pairIdPerWg
                entry(8, 'uniform'),            // totalWg
            ]
        });
        this.lookbackPipeline = device.createComputePipeline({
            label: 'InPlace Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: { module: device.createShaderModule({ code: lookbackShader }), entryPoint: 'decoupled_lookback_batched_kernel' }
        });

        this.pairInfo = new PooledBuffer(device, 'InPlace pairInfo', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.pairIdPerWg = new PooledBuffer(device, 'InPlace pairIdPerWg', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.dpi = new PooledBuffer(device, 'InPlace dpi', GPUBufferUsage.STORAGE);
        this.state = new PooledBuffer(device, 'InPlace state', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.pairCounts = new PooledBuffer(device, 'InPlace pairCounts', GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
        this.pairCountsRead = new PooledBuffer(device, 'InPlace pairCounts read', GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        this.totalWg = new PooledBuffer(device, 'InPlace totalWg', GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

        if (device.features.has('timestamp-query')) {
            this.querySet = device.createQuerySet({ type: 'timestamp', count: QUERY_COUNT });
            this.timestampResolve = device.createBuffer({ size: QUERY_COUNT * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            this.timestampRead = device.createBuffer({ size: QUERY_COUNT * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        }
    }

    createLayout(storageLimit: number): InPlaceSubmitLayout {
        return new InPlaceSubmitLayout(this.device.limits.minStorageBufferOffsetAlignment, storageLimit);
    }

    async run(
        chunks: InPlaceChunk[],
        layout: InPlaceSubmitLayout,
        acquireOutput: (bytes: number) => GPUBuffer
    ): Promise<InPlaceSubmitResult> {
        const runT0 = performance.now();
        const device = this.device;
        const n = chunks.length;
        const uniformAlign = device.limits.minUniformBufferOffsetAlignment;

        const infoHost = new Uint32Array(layout.infoBytes / 4);
        const pairIdHost = new Uint32Array(layout.pairIdBytes / 4);
        const totalWgHost = new Uint32Array((n * uniformAlign) / 4);
        for (let c = 0; c < n; c++) {
            infoHost.set(chunks[c].pairInfo, layout.infoOffsets[c] / 4);
            pairIdHost.set(chunks[c].pairIdPerWg, layout.pairIdOffsets[c] / 4);
            totalWgHost[(c * uniformAlign) / 4] = chunks[c].totalWg;
        }

        const bufInfo = this.pairInfo.ensure(layout.infoBytes);
        const bufPairId = this.pairIdPerWg.ensure(layout.pairIdBytes);
        const bufDpi = this.dpi.ensure(layout.dpiBytes);
        const bufState = this.state.ensure(layout.stateBytes);
        const bufCounts = this.pairCounts.ensure(layout.countsBytes);
        const bufCountsRead = this.pairCountsRead.ensure(layout.countsBytes);
        const bufTotalWg = this.totalWg.ensure(n * uniformAlign);

        device.queue.writeBuffer(bufInfo, 0, infoHost);
        device.queue.writeBuffer(bufPairId, 0, pairIdHost);
        device.queue.writeBuffer(bufTotalWg, 0, totalWgHost);

        const encoder = device.createCommandEncoder();
        encoder.clearBuffer(bufState, 0, layout.stateBytes);
        encoder.clearBuffer(bufCounts, 0, layout.countsBytes);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const outputBuffers: GPUBuffer[] = [];

        for (let c = 0; c < n; c++) {
            const chunk = chunks[c];
            const output = acquireOutput(Math.max(4, chunk.maxOutputSize * 4));
            outputBuffers.push(output);

            const dpiWindow: GPUBufferBinding = { buffer: bufDpi, offset: layout.dpiOffsets[c], size: chunk.totalDpiEntries * 4 };
            const infoWindow: GPUBufferBinding = { buffer: bufInfo, offset: layout.infoOffsets[c], size: chunk.numPairs * PAIR_INFO_STRIDE * 4 };
            const pairIdWindow: GPUBufferBinding = { buffer: bufPairId, offset: layout.pairIdOffsets[c], size: chunk.totalWg * 4 };
            const totalWgWindow: GPUBufferBinding = { buffer: bufTotalWg, offset: c * uniformAlign, size: 4 };

            const dpiBindGroup = device.createBindGroup({
                layout: this.dpiBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: chunk.keysA } },
                    { binding: 1, resource: { buffer: chunk.keysB } },
                    { binding: 2, resource: dpiWindow },
                    { binding: 3, resource: infoWindow },
                    { binding: 4, resource: pairIdWindow },
                    { binding: 5, resource: totalWgWindow },
                ]
            });
            const lookbackBindGroup = device.createBindGroup({
                layout: this.lookbackBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: chunk.keysA } },
                    { binding: 1, resource: { buffer: chunk.keysB } },
                    { binding: 2, resource: dpiWindow },
                    { binding: 3, resource: { buffer: bufState, offset: layout.stateOffsets[c], size: chunk.totalWg * 4 } },
                    { binding: 4, resource: { buffer: bufCounts, offset: layout.countsOffsets[c], size: chunk.numPairs * 4 } },
                    { binding: 5, resource: { buffer: output } },
                    { binding: 6, resource: infoWindow },
                    { binding: 7, resource: pairIdWindow },
                    { binding: 8, resource: totalWgWindow },
                ]
            });

            const dpiBlocks = Math.ceil(chunk.totalWg / subgroupsPerWg);
            let pass = encoder.beginComputePass(this.passDescriptor(4 * c));
            pass.setPipeline(this.dpiPipeline);
            pass.setBindGroup(0, dpiBindGroup);
            pass.dispatchWorkgroups(Math.min(dpiBlocks, MAXWORKGROUP), Math.ceil(dpiBlocks / MAXWORKGROUP));
            pass.end();

            pass = encoder.beginComputePass(this.passDescriptor(4 * c + 2));
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(Math.min(chunk.totalWg, MAXWORKGROUP), Math.ceil(chunk.totalWg / MAXWORKGROUP));
            pass.end();
        }

        const timestampBytes = 4 * n * 8;
        if (this.querySet !== null) {
            encoder.resolveQuerySet(this.querySet, 0, 4 * n, this.timestampResolve, 0);
            encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampRead, 0, timestampBytes);
        }
        encoder.copyBufferToBuffer(bufCounts, 0, bufCountsRead, 0, layout.countsBytes);
        const commandBuffer = encoder.finish();
        const encodeMs = performance.now() - runT0;
        const waitT0 = performance.now();
        device.queue.submit([commandBuffer]);

        // The only wait of this submission.
        await Promise.all([
            bufCountsRead.mapAsync(GPUMapMode.READ, 0, layout.countsBytes),
            this.timestampRead !== null ? this.timestampRead.mapAsync(GPUMapMode.READ, 0, timestampBytes) : Promise.resolve(),
        ]);
        const waitMs = performance.now() - waitT0;
        const readbackT0 = performance.now();

        const counts = new Uint32Array(bufCountsRead.getMappedRange(0, layout.countsBytes));
        const pairCounts = chunks.map((chunk, c) =>
            counts.slice(layout.countsOffsets[c] / 4, layout.countsOffsets[c] / 4 + chunk.numPairs));
        bufCountsRead.unmap();

        let dpiMs = 0, lookbackMs = 0, totalMs = 0;
        if (this.timestampRead !== null) {
            const timestamps = new BigUint64Array(this.timestampRead.getMappedRange(0, timestampBytes));
            for (let c = 0; c < n; c++) {
                const t = (i: number) => Number(timestamps[4 * c + i]);
                dpiMs += (t(1) - t(0)) / 1_000_000;
                lookbackMs += (t(3) - t(2)) / 1_000_000;
                totalMs += (t(3) - t(0)) / 1_000_000;
            }
            this.timestampRead.unmap();
        }

        const readbackMs = performance.now() - readbackT0;
        const runMs = performance.now() - runT0;
        return { pairCounts, outputBuffers, timing: { dpiMs, lookbackMs, totalMs, encodeMs, waitMs, readbackMs, runMs } };
    }

    private passDescriptor(beginIndex: number): GPUComputePassDescriptor {
        if (this.querySet === null) return {};
        return {
            timestampWrites: {
                querySet: this.querySet,
                beginningOfPassWriteIndex: beginIndex,
                endOfPassWriteIndex: beginIndex + 1,
            }
        };
    }
}
