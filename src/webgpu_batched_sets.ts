import TimestampQueryManager from './TimestampQueryManager';
import { BatchedFullOutputCounter } from './applications/eclat/batched_fulloutput_counter';
import { InPlaceChunk, InPlaceFullOutputCounter, PAIR_INFO_STRIDE } from './applications/eclat/inplace_fulloutput_counter';
import { BatchedJaccardCounter } from './applications/jaccard_genome/batched_jaccard_counter';

const NV = 256 * 12;
const WG_INFO_STRIDE = 8;
const DEFAULT_MAX_CHUNK_INPUT_BYTES = 512 * 1024 * 1024;
const DEFAULT_MAX_CHUNK_GPU_BYTES_COUNTS = 256 * 1024 * 1024;
const DEFAULT_MAX_CHUNK_GPU_BYTES_OUTPUT = 128 * 1024 * 1024;

class GPUSetBufferOwner {
    private refCount = 0;

    constructor(
        public readonly buffer: GPUBuffer,
        private readonly destroyOnZero: boolean = true,
        private readonly onZero: ((buffer: GPUBuffer) => void) | null = null
    ) {}

    acquire(): GPUSetBufferOwner {
        this.refCount++;
        return this;
    }

    release(): void {
        if (this.refCount === 0) return;
        this.refCount--;
        if (this.refCount === 0) {
            if (this.onZero !== null) {
                this.onZero(this.buffer);
            } else if (this.destroyOnZero) {
                this.buffer.destroy();
            }
        }
    }
}

// Output buffers of one ECLAT run. A released buffer is kept and handed to a later chunk
// that fits in it, so later levels reuse the memory of earlier ones instead of creating buffers.
class OutputBufferPool {
    private free: GPUBuffer[] = [];

    acquire(device: GPUDevice, bytes: number): GPUBuffer {
        let best = -1;
        for (let i = 0; i < this.free.length; i++) {
            if (this.free[i].size >= bytes && (best < 0 || this.free[i].size < this.free[best].size)) best = i;
        }
        if (best >= 0) return this.free.splice(best, 1)[0];
        return device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    }

    release(buffer: GPUBuffer): void {
        this.free.push(buffer);
    }

    destroyAll(): void {
        for (const buffer of this.free) buffer.destroy();
        this.free = [];
    }
}

export class GPUSetHandle {
    private owner: GPUSetBufferOwner | null;
    readonly byteOffset: number;
    readonly length: number;

    constructor(owner: GPUSetBufferOwner, byteOffset: number, length: number) {
        this.owner = owner.acquire();
        this.byteOffset = byteOffset;
        this.length = length;
    }

    get buffer(): GPUBuffer {
        if (this.owner === null) {
            throw new Error('GPUSetHandle has been disposed');
        }
        return this.owner.buffer;
    }

    get byteLength(): number {
        return this.length * 4;
    }

    dispose(): void {
        if (this.owner === null) return;
        this.owner.release();
        this.owner = null;
    }
}

export type BatchedSetInput = Uint32Array | GPUSetHandle;

export interface BatchedSetPair {
    a: Uint32Array;
    b: Uint32Array;
}

export interface BatchedGPUSetPair {
    a: BatchedSetInput;
    b: BatchedSetInput;
}

export interface BatchedIntersectionUnionCountsOptions {
    iterations?: number;
    warmup?: number;
    maxChunkInputBytes?: number;
    maxChunkGpuBytes?: number;
    noReuse?: boolean;
}

export interface BatchedIntersectionWithOutputOptions {
    maxChunkInputBytes?: number;
    maxChunkGpuBytes?: number;
}

export interface BatchedIntersectionUnionCountsTiming {
    batchPrepMs: number;
    dpiMs: number;
    intersectionMs: number;
    unionMs: number;
    totalMs: number;
    /** GPU total of each timed iteration, summed over chunks. */
    totalRunsMs: number[];
}

export interface BatchedIntersectionWithOutputTiming {
    batchPrepMs: number;
    dpiMs: number;
    lookbackMs: number;
    totalMs: number;
    // In-place path only (diagnostic): encoding, submit-to-readback wait, and output handle creation.
    encodeMs?: number;
    waitMs?: number;
    handlesMs?: number;
    readbackMs?: number;
    errorScopeMs?: number;
    loopOtherMs?: number;
}

export interface BatchedIntersectionUnionCountsResult {
    intersectionCounts: Uint32Array;
    unionCounts: Uint32Array;
    chunkCount: number;
    timing: BatchedIntersectionUnionCountsTiming;
}

export interface BatchedIntersectionUnionCountsInPlaceResult {
    intersectionCounts: Uint32Array;
    unionCounts: Uint32Array;
    chunkCount: number;
    /**
     * Kernel times are GPU timestamps summed over chunks. Host segments summed over chunks:
     * batchPrepMs (workgroup metadata), createMs (buffers, bind groups, encoding), waitMs
     * (submit until results are mapped), readMs (copy out, unmap, destroy), errorScopeMs
     * (the error-scope round trips around each submission).
     */
    timing: {
        batchPrepMs: number; dpiMs: number; intersectionMs: number; unionMs: number; kernelMs: number;
        createMs: number; waitMs: number; readMs: number; errorScopeMs: number;
    };
}

export interface BatchedIntersectionWithOutputResult {
    pairCounts: Uint32Array;
    outputs: Uint32Array[];
    chunkCount: number;
    timing: BatchedIntersectionWithOutputTiming;
}

export interface BatchedIntersectionWithOutputGPUResult {
    pairCounts: Uint32Array;
    outputHandles: GPUSetHandle[];
    chunkCount: number;
    timing: BatchedIntersectionWithOutputTiming;
}

export interface BatchedGPUSetHandlePair {
    a: GPUSetHandle;
    b: GPUSetHandle;
}

export interface BatchedIntersectionWithOutputGPUInPlaceResult extends BatchedIntersectionWithOutputGPUResult {
    submitCount: number;
}

interface PairBatchMetrics {
    numPairs: number;
    totalALen: number;
    totalBLen: number;
    totalWg: number;
    totalDpiEntries: number;
    maxOutputSize: number;
    totalInputBytes: number;
}

interface PairMeta {
    aStart: number;
    aLen: number;
    bStart: number;
    bLen: number;
    numWg: number;
    stateOffset: number;
    dpiOffset: number;
}

interface BatchMetadata {
    metas: PairMeta[];
    totalALen: number;
    totalBLen: number;
    totalWg: number;
    totalDpiEntries: number;
    numPairs: number;
}

interface CountsBatchLayout extends BatchMetadata {
    keysA: Uint32Array;
    keysB: Uint32Array;
    wgInfo: Uint32Array;
    pairIdPerWg: Uint32Array;
}

interface FullOutputBatchLayout extends CountsBatchLayout {
    pairOutputOffsetPerWg: Uint32Array;
    pairOutputOffsets: number[];
    maxOutputSize: number;
}

interface FullOutputGPUChunkLayout extends BatchMetadata {
    wgInfo: Uint32Array;
    pairIdPerWg: Uint32Array;
    pairOutputOffsetPerWg: Uint32Array;
    pairOutputOffsets: number[];
    maxOutputSize: number;
}

export class WebGPUBatchedSets {
    private device: GPUDevice;
    private jaccardCounter: BatchedJaccardCounter;
    private fullOutputCounter: BatchedFullOutputCounter;
    private inPlaceCounter: InPlaceFullOutputCounter;
    private emptySetOwner: GPUSetBufferOwner | null = null;
    private outputPool = new OutputBufferPool();
    // First GPU error reported by the device-level listener. The in-place path checks it after each submission
    // instead of awaiting error-scope pops, which cost a round trip to the GPU process per submission.
    private uncapturedGpuError: GPUError | null = null;
    // Wraps each in-place submission in error scopes (default, used for the paper results). When false, the
    // device-level listener above is checked instead. An interleaved A/B on the RTX 3060 showed no e2e difference.
    private readonly inPlaceErrorScopes: boolean;

    constructor(device: GPUDevice, options: { inPlaceErrorScopes?: boolean } = {}) {
        this.device = device;
        this.inPlaceErrorScopes = options.inPlaceErrorScopes ?? true;
        device.addEventListener('uncapturederror', event => {
            if (this.uncapturedGpuError === null) {
                this.uncapturedGpuError = (event as GPUUncapturedErrorEvent).error;
            }
        });
        this.jaccardCounter = new BatchedJaccardCounter(device, new TimestampQueryManager(device, 8));
        this.fullOutputCounter = new BatchedFullOutputCounter(device, new TimestampQueryManager(device, 8));
        this.inPlaceCounter = new InPlaceFullOutputCounter(device);
    }

    /** Destroys the output buffers that disposed in-place results returned to the pool. Call at the end of a run. */
    releaseOutputPool(): void {
        this.outputPool.destroyAll();
    }

    uploadSet(data: Uint32Array): GPUSetHandle {
        if (data.length === 0) {
            return this.createEmptyHandle();
        }

        const buffer = this.device.createBuffer({
            size: Math.max(4, data.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        this.device.queue.writeBuffer(buffer, 0, new Uint32Array(data));
        return new GPUSetHandle(new GPUSetBufferOwner(buffer), 0, data.length);
    }

    /**
     * Upload many sets into shared arena buffers, each within the storage binding
     * limit. Every handle addresses its set by offset, so the in-place batched path
     * reads the sets without copying them.
     */
    uploadSets(datas: Uint32Array[]): GPUSetHandle[] {
        const limit = this.getStorageBufferLimit();
        const handles = new Array<GPUSetHandle>(datas.length);

        let start = 0;
        while (start < datas.length) {
            let end = start;
            let arenaBytes = 0;
            while (end < datas.length && (end === start || arenaBytes + datas[end].byteLength <= limit)) {
                arenaBytes += datas[end].byteLength;
                end++;
            }
            if (arenaBytes > limit) {
                throw new Error(`uploadSets: set ${start} (${arenaBytes} bytes) exceeds storage binding limit ${limit}`);
            }

            if (arenaBytes === 0) {
                for (let i = start; i < end; i++) {
                    handles[i] = this.createEmptyHandle();
                }
            } else {
                const buffer = this.device.createBuffer({
                    size: arenaBytes,
                    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
                });
                const owner = new GPUSetBufferOwner(buffer);
                let byteOffset = 0;
                for (let i = start; i < end; i++) {
                    if (datas[i].length === 0) {
                        handles[i] = this.createEmptyHandle();
                        continue;
                    }
                    this.device.queue.writeBuffer(buffer, byteOffset, datas[i].buffer as ArrayBuffer, datas[i].byteOffset, datas[i].byteLength);
                    handles[i] = new GPUSetHandle(owner, byteOffset, datas[i].length);
                    byteOffset += datas[i].byteLength;
                }
            }
            start = end;
        }

        return handles;
    }

    async readbackSet(handle: GPUSetHandle): Promise<Uint32Array> {
        if (handle.length === 0) {
            return new Uint32Array(0);
        }

        const staging = this.device.createBuffer({
            size: handle.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        try {
            const encoder = this.device.createCommandEncoder();
            encoder.copyBufferToBuffer(handle.buffer, handle.byteOffset, staging, 0, handle.byteLength);
            this.device.queue.submit([encoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await staging.mapAsync(GPUMapMode.READ);
            const data = new Uint32Array(staging.getMappedRange().slice(0));
            staging.unmap();
            return data;
        } finally {
            staging.destroy();
        }
    }

    async batchedIntersectionUnionCounts(
        pairs: BatchedSetPair[],
        options: BatchedIntersectionUnionCountsOptions = {}
    ): Promise<BatchedIntersectionUnionCountsResult> {
        const { iterations = 1, warmup = 0, noReuse = false } = options;
        const chunks = this.splitPairsIntoChunks(pairs, 'counts', options.maxChunkInputBytes, options.maxChunkGpuBytes);

        const intersectionCounts = new Uint32Array(pairs.length);
        const unionCounts = new Uint32Array(pairs.length);

        let batchPrepMs = 0;
        let dpiMs = 0;
        let intersectionMs = 0;
        let unionMs = 0;
        let totalMs = 0;
        const totalRunsMs: number[] = [];
        let pairOffset = 0;

        for (const chunkPairs of chunks) {
            const prepT0 = performance.now();
            const batch = this.prepareCountsBatch(chunkPairs);
            batchPrepMs += performance.now() - prepT0;

            const result = await this.runWithGpuErrorScopes(
                'batchedIntersectionUnionCounts',
                () => this.jaccardCounter.run(
                    batch.keysA,
                    batch.keysB,
                    batch.wgInfo,
                    batch.pairIdPerWg,
                    batch.totalWg,
                    batch.totalDpiEntries,
                    batch.numPairs,
                    iterations,
                    warmup,
                    noReuse
                )
            );

            intersectionCounts.set(result.intersectionCounts, pairOffset);
            unionCounts.set(result.unionCounts, pairOffset);

            dpiMs += result.timing.dpiMs;
            intersectionMs += result.timing.intersectionMs;
            unionMs += result.timing.unionMs;
            totalMs += result.timing.totalMs;
            result.timing.totalRunsMs.forEach((ms, i) => { totalRunsMs[i] = (totalRunsMs[i] ?? 0) + ms; });
            pairOffset += chunkPairs.length;
        }

        return {
            intersectionCounts,
            unionCounts,
            chunkCount: chunks.length,
            timing: { batchPrepMs, dpiMs, intersectionMs, unionMs, totalMs, totalRunsMs }
        };
    }

    /**
     * Intersection and union sizes of many pairs whose sets already live on the GPU, for
     * example uploaded once with uploadSets. Every set is read in place by its offset, and
     * one DPI dispatch serves both operations unless noReuse is set. A chunk ends only where
     * the source buffers change or the workgroup metadata would exceed the storage limit.
     */
    async batchedIntersectionUnionCountsInPlace(
        pairs: Array<{ a: GPUSetHandle; b: GPUSetHandle }>,
        options: { noReuse?: boolean } = {}
    ): Promise<BatchedIntersectionUnionCountsInPlaceResult> {
        const noReuse = options.noReuse ?? false;
        const storageLimit = this.getStorageBufferLimit();
        const intersectionCounts = new Uint32Array(pairs.length);
        const unionCounts = new Uint32Array(pairs.length);
        const timing = {
            batchPrepMs: 0, dpiMs: 0, intersectionMs: 0, unionMs: 0, kernelMs: 0,
            createMs: 0, waitMs: 0, readMs: 0, errorScopeMs: 0
        };
        let chunkCount = 0;

        // A pair with an empty set needs no GPU work.
        const gpuPairs: number[] = [];
        for (let i = 0; i < pairs.length; i++) {
            if (pairs[i].a.length === 0 || pairs[i].b.length === 0) {
                unionCounts[i] = pairs[i].a.length + pairs[i].b.length;
            } else {
                gpuPairs.push(i);
            }
        }

        let start = 0;
        while (start < gpuPairs.length) {
            const prepT0 = performance.now();
            const first = pairs[gpuPairs[start]];
            const bufA = first.a.buffer;
            const bufB = first.b.buffer;
            const metas: PairMeta[] = [];
            let totalWg = 0;
            let totalDpiEntries = 0;
            let end = start;
            while (end < gpuPairs.length) {
                const pair = pairs[gpuPairs[end]];
                if (pair.a.buffer !== bufA || pair.b.buffer !== bufB) break;
                const numWg = this.getNumWorkgroups(pair);
                const nextWg = totalWg + numWg;
                const nextDpi = totalDpiEntries + 2 * (numWg + 1);
                if (end > start && (nextWg * WG_INFO_STRIDE * 4 > storageLimit || nextDpi * 4 > storageLimit)) break;
                metas.push({
                    aStart: pair.a.byteOffset / 4, aLen: pair.a.length,
                    bStart: pair.b.byteOffset / 4, bLen: pair.b.length,
                    numWg, stateOffset: totalWg, dpiOffset: totalDpiEntries
                });
                totalWg = nextWg;
                totalDpiEntries = nextDpi;
                end++;
            }
            const { wgInfo, pairIdPerWg } = this.createWorkgroupMetadata(metas, metas.length, totalWg);
            timing.batchPrepMs += performance.now() - prepT0;

            const callT0 = performance.now();
            const result = await this.runWithGpuErrorScopes(
                'batchedIntersectionUnionCountsInPlace',
                () => this.jaccardCounter.runInPlace(
                    bufA, bufB, wgInfo, pairIdPerWg, totalWg, totalDpiEntries, metas.length, noReuse)
            );
            timing.errorScopeMs += performance.now() - callT0 - result.timing.runMs;
            for (let k = 0; k < metas.length; k++) {
                intersectionCounts[gpuPairs[start + k]] = result.intersectionCounts[k];
                unionCounts[gpuPairs[start + k]] = result.unionCounts[k];
            }
            timing.dpiMs += result.timing.dpiMs;
            timing.intersectionMs += result.timing.intersectionMs;
            timing.unionMs += result.timing.unionMs;
            timing.kernelMs += result.timing.kernelMs;
            timing.createMs += result.timing.createMs;
            timing.waitMs += result.timing.waitMs;
            timing.readMs += result.timing.readMs;
            chunkCount++;
            start = end;
        }

        return { intersectionCounts, unionCounts, chunkCount, timing };
    }

    async batchedIntersectionWithOutput(
        pairs: BatchedSetPair[],
        options: BatchedIntersectionWithOutputOptions = {}
    ): Promise<BatchedIntersectionWithOutputResult> {
        const chunks = this.splitPairsIntoChunks(pairs, 'output_cpu', options.maxChunkInputBytes, options.maxChunkGpuBytes);

        const pairCounts = new Uint32Array(pairs.length);
        const outputs = new Array<Uint32Array>(pairs.length);

        let batchPrepMs = 0;
        let dpiMs = 0;
        let lookbackMs = 0;
        let totalMs = 0;
        let pairOffset = 0;

        for (const chunkPairs of chunks) {
            const prepT0 = performance.now();
            const batch = this.prepareFullOutputBatch(chunkPairs);
            batchPrepMs += performance.now() - prepT0;

            const result = await this.runWithGpuErrorScopes(
                'batchedIntersectionWithOutput',
                () => this.fullOutputCounter.run(
                    batch.keysA,
                    batch.keysB,
                    batch.wgInfo,
                    batch.pairIdPerWg,
                    batch.pairOutputOffsetPerWg,
                    batch.totalWg,
                    batch.totalDpiEntries,
                    batch.numPairs,
                    batch.maxOutputSize
                )
            );

            pairCounts.set(result.pairCounts, pairOffset);

            for (let i = 0; i < chunkPairs.length; i++) {
                const count = result.pairCounts[i];
                const outputOffset = batch.pairOutputOffsets[i];
                outputs[pairOffset + i] = result.output.subarray(outputOffset, outputOffset + count);
            }

            dpiMs += result.timing.dpiMs;
            lookbackMs += result.timing.lookbackMs;
            totalMs += result.timing.totalMs;
            pairOffset += chunkPairs.length;
        }

        return {
            pairCounts,
            outputs,
            chunkCount: chunks.length,
            timing: { batchPrepMs, dpiMs, lookbackMs, totalMs }
        };
    }

    async batchedIntersectionWithOutputGPU(
        pairs: BatchedGPUSetPair[],
        options: BatchedIntersectionWithOutputOptions = {}
    ): Promise<BatchedIntersectionWithOutputGPUResult> {
        const chunks = this.splitPairsIntoChunks(pairs, 'output_gpu', options.maxChunkInputBytes, options.maxChunkGpuBytes);

        const pairCounts = new Uint32Array(pairs.length);
        const outputHandles = new Array<GPUSetHandle>(pairs.length);

        let batchPrepMs = 0;
        let dpiMs = 0;
        let lookbackMs = 0;
        let totalMs = 0;
        let pairOffset = 0;

        for (const chunkPairs of chunks) {
            const prepT0 = performance.now();
            const batch = this.prepareFullOutputGPUChunkLayout(chunkPairs);
            const prepMs = performance.now() - prepT0;

            if (batch.totalWg === 0) {
                batchPrepMs += prepMs;
                pairCounts.fill(0, pairOffset, pairOffset + chunkPairs.length);
                for (let i = 0; i < chunkPairs.length; i++) {
                    outputHandles[pairOffset + i] = this.createEmptyHandle();
                }
                pairOffset += chunkPairs.length;
                continue;
            }

            const chunkResult = await this.runWithGpuErrorScopes(
                'batchedIntersectionWithOutputGPU',
                async () => {
                    const packedInputs = this.createPackedInputBuffers(chunkPairs, batch);
                    try {
                        return await this.fullOutputCounter.runPackedGPU(
                            packedInputs.bufferA,
                            packedInputs.bufferB,
                            batch.wgInfo,
                            batch.pairIdPerWg,
                            batch.pairOutputOffsetPerWg,
                            batch.totalWg,
                            batch.totalDpiEntries,
                            batch.numPairs,
                            batch.maxOutputSize
                        );
                    } finally {
                        packedInputs.bufferA.destroy();
                        packedInputs.bufferB.destroy();
                    }
                }
            );
            batchPrepMs += prepMs;

            pairCounts.set(chunkResult.pairCounts, pairOffset);

            const chunkHandles = this.createOutputHandles(
                chunkResult.outputBuffer,
                batch.pairOutputOffsets,
                chunkResult.pairCounts
            );
            for (let i = 0; i < chunkHandles.length; i++) {
                outputHandles[pairOffset + i] = chunkHandles[i];
            }

            dpiMs += chunkResult.timing.dpiMs;
            lookbackMs += chunkResult.timing.lookbackMs;
            totalMs += chunkResult.timing.totalMs;
            pairOffset += chunkPairs.length;
        }

        return {
            pairCounts,
            outputHandles,
            chunkCount: chunks.length,
            timing: { batchPrepMs, dpiMs, lookbackMs, totalMs }
        };
    }

    /**
     * Batched intersection with GPU-resident output that reads every input set in
     * place (no packing copy). Pairs are split into chunks that bind one A buffer
     * and one B buffer; consecutive chunks are encoded into one command buffer and
     * the per-pair counts come back with a single wait.
     */
    async batchedIntersectionWithOutputGPUInPlace(
        pairs: BatchedGPUSetHandlePair[],
        options: { maxChunkInputBytes?: number } = {}
    ): Promise<BatchedIntersectionWithOutputGPUInPlaceResult> {
        const maxChunkInputBytes = options.maxChunkInputBytes ?? DEFAULT_MAX_CHUNK_INPUT_BYTES;
        const storageLimit = this.getStorageBufferLimit();

        const pairCounts = new Uint32Array(pairs.length);
        const outputHandles = new Array<GPUSetHandle>(pairs.length);

        let batchPrepMs = 0;
        let dpiMs = 0;
        let lookbackMs = 0;
        let totalMs = 0;
        let chunkCount = 0;
        let submitCount = 0;
        let encodeMs = 0, waitMs = 0, handlesMs = 0;
        // Diagnostic: error-scope round trips, readback copies, and the time spent in submitPending.
        let readbackMs = 0, errorScopeMs = 0, submitPendingMs = 0;
        const functionT0 = performance.now();

        let pending: Array<{ chunk: InPlaceChunk; pairStart: number; pairOutputOffsets: number[] }> = [];
        let layout = this.inPlaceCounter.createLayout(storageLimit);

        const submitPending = async () => {
            if (pending.length === 0) return;
            const submitT0 = performance.now();
            const submitted = pending;
            const submittedLayout = layout;
            pending = [];
            layout = this.inPlaceCounter.createLayout(storageLimit);

            const runCallT0 = performance.now();
            const runChunks = () => this.inPlaceCounter.run(submitted.map(p => p.chunk), submittedLayout,
                bytes => this.outputPool.acquire(this.device, bytes));
            const result = this.inPlaceErrorScopes
                ? await this.runWithGpuErrorScopes('batchedIntersectionWithOutputGPUInPlace', runChunks)
                : await runChunks();
            if (this.uncapturedGpuError !== null) {
                throw new Error(`batchedIntersectionWithOutputGPUInPlace: ${this.uncapturedGpuError.message}`);
            }
            errorScopeMs += performance.now() - runCallT0 - result.timing.runMs;
            readbackMs += result.timing.readbackMs;
            submitCount++;
            dpiMs += result.timing.dpiMs;
            lookbackMs += result.timing.lookbackMs;
            totalMs += result.timing.totalMs;
            encodeMs += result.timing.encodeMs;
            waitMs += result.timing.waitMs;

            const handlesT0 = performance.now();
            for (let c = 0; c < submitted.length; c++) {
                const { pairStart, pairOutputOffsets } = submitted[c];
                pairCounts.set(result.pairCounts[c], pairStart);
                const handles = this.createOutputHandles(result.outputBuffers[c], pairOutputOffsets, result.pairCounts[c],
                    buffer => this.outputPool.release(buffer));
                for (let i = 0; i < handles.length; i++) {
                    outputHandles[pairStart + i] = handles[i];
                }
            }
            handlesMs += performance.now() - handlesT0;
            submitPendingMs += performance.now() - submitT0;
        };

        let chunkStart = 0;
        while (chunkStart < pairs.length) {
            const prepT0 = performance.now();
            const keysA = pairs[chunkStart].a.buffer;
            const keysB = pairs[chunkStart].b.buffer;

            let chunkEnd = chunkStart;
            let inputBytes = 0;
            let outputBytes = 0;
            while (chunkEnd < pairs.length) {
                const pair = pairs[chunkEnd];
                if (pair.a.buffer !== keysA || pair.b.buffer !== keysB) break;
                const pairInputBytes = pair.a.byteLength + pair.b.byteLength;
                const pairOutputBytes = Math.min(pair.a.length, pair.b.length) * 4;
                if (chunkEnd > chunkStart &&
                    (inputBytes + pairInputBytes > maxChunkInputBytes || outputBytes + pairOutputBytes > storageLimit)) {
                    break;
                }
                inputBytes += pairInputBytes;
                outputBytes += pairOutputBytes;
                chunkEnd++;
            }
            if (outputBytes > storageLimit) {
                throw new Error(
                    `batchedIntersectionWithOutputGPUInPlace: pair ${chunkStart} output (${outputBytes} bytes) ` +
                    `exceeds storage binding limit ${storageLimit}`
                );
            }

            const { chunk, pairOutputOffsets } = this.prepareInPlaceChunk(pairs, chunkStart, chunkEnd, keysA, keysB);
            batchPrepMs += performance.now() - prepT0;
            chunkCount++;

            if (chunk.totalWg === 0) {
                for (let i = chunkStart; i < chunkEnd; i++) {
                    outputHandles[i] = this.createEmptyHandle();
                }
            } else {
                if (!layout.fits(chunk)) {
                    await submitPending();
                    if (!layout.fits(chunk)) {
                        throw new Error(
                            `batchedIntersectionWithOutputGPUInPlace: chunk at pair ${chunkStart} ` +
                            `exceeds storage binding limit ${storageLimit}`
                        );
                    }
                }
                layout.add(chunk);
                pending.push({ chunk, pairStart: chunkStart, pairOutputOffsets });
            }
            chunkStart = chunkEnd;
        }
        await submitPending();

        return {
            pairCounts,
            outputHandles,
            chunkCount,
            submitCount,
            timing: {
                batchPrepMs, dpiMs, lookbackMs, totalMs, encodeMs, waitMs, handlesMs, readbackMs, errorScopeMs,
                loopOtherMs: performance.now() - functionT0 - batchPrepMs - submitPendingMs
            }
        };
    }

    private prepareInPlaceChunk(
        pairs: BatchedGPUSetHandlePair[],
        start: number,
        end: number,
        keysA: GPUBuffer,
        keysB: GPUBuffer
    ): { chunk: InPlaceChunk; pairOutputOffsets: number[] } {
        const numPairs = end - start;
        const pairInfo = new Uint32Array(numPairs * PAIR_INFO_STRIDE);
        const pairOutputOffsets = new Array<number>(numPairs);
        let totalWg = 0;
        let totalDpiEntries = 0;
        let maxOutputSize = 0;

        for (let i = 0; i < numPairs; i++) {
            const pair = pairs[start + i];
            const numWg = this.getNumWorkgroups(pair);
            pairOutputOffsets[i] = maxOutputSize;
            if (numWg > 0) {
                const base = i * PAIR_INFO_STRIDE;
                pairInfo[base + 0] = pair.a.byteOffset / 4;
                pairInfo[base + 1] = pair.a.length;
                pairInfo[base + 2] = pair.b.byteOffset / 4;
                pairInfo[base + 3] = pair.b.length;
                pairInfo[base + 4] = numWg;
                pairInfo[base + 5] = totalWg;
                pairInfo[base + 6] = totalDpiEntries;
                pairInfo[base + 7] = maxOutputSize;
                totalWg += numWg;
                totalDpiEntries += 2 * (numWg + 1);
            }
            maxOutputSize += Math.min(pair.a.length, pair.b.length);
        }

        const pairIdPerWg = new Uint32Array(totalWg);
        let wg = 0;
        for (let i = 0; i < numPairs; i++) {
            const numWg = pairInfo[i * PAIR_INFO_STRIDE + 4];
            pairIdPerWg.fill(i, wg, wg + numWg);
            wg += numWg;
        }

        return {
            chunk: { keysA, keysB, numPairs, totalWg, totalDpiEntries, maxOutputSize, pairInfo, pairIdPerWg },
            pairOutputOffsets
        };
    }

    private splitPairsIntoChunks<T extends { a: BatchedSetInput; b: BatchedSetInput }>(
        pairs: T[],
        mode: 'counts' | 'output_cpu' | 'output_gpu',
        requestedChunkInputBytes?: number,
        requestedChunkGpuBytes?: number
    ): T[][] {
        if (pairs.length === 0) return [];

        const maxChunkInputBytes = requestedChunkInputBytes ?? DEFAULT_MAX_CHUNK_INPUT_BYTES;
        const maxChunkGpuBytes = requestedChunkGpuBytes ?? this.getDefaultChunkGpuBytes(mode);
        const chunks: T[][] = [];

        let chunkStart = 0;
        let metrics = this.createEmptyMetrics();

        for (let i = 0; i < pairs.length; i++) {
            const nextMetrics = this.addPairToMetrics(metrics, pairs[i]);

            if (!this.fitsInDeviceLimits(nextMetrics, mode, maxChunkInputBytes, maxChunkGpuBytes)) {
                if (i === chunkStart) {
                    throw this.createChunkLimitError(i, pairs[i], mode, maxChunkInputBytes, maxChunkGpuBytes);
                }

                chunks.push(pairs.slice(chunkStart, i));
                chunkStart = i;
                metrics = this.addPairToMetrics(this.createEmptyMetrics(), pairs[i]);

                if (!this.fitsInDeviceLimits(metrics, mode, maxChunkInputBytes, maxChunkGpuBytes)) {
                    throw this.createChunkLimitError(i, pairs[i], mode, maxChunkInputBytes, maxChunkGpuBytes);
                }

                continue;
            }

            metrics = nextMetrics;
        }

        chunks.push(pairs.slice(chunkStart));
        return chunks;
    }

    private createEmptyMetrics(): PairBatchMetrics {
        return {
            numPairs: 0,
            totalALen: 0,
            totalBLen: 0,
            totalWg: 0,
            totalDpiEntries: 0,
            maxOutputSize: 0,
            totalInputBytes: 0
        };
    }

    private addPairToMetrics<T extends { a: BatchedSetInput; b: BatchedSetInput }>(
        metrics: PairBatchMetrics,
        pair: T
    ): PairBatchMetrics {
        const numWg = this.getNumWorkgroups(pair);
        return {
            numPairs: metrics.numPairs + 1,
            totalALen: metrics.totalALen + this.getInputLength(pair.a),
            totalBLen: metrics.totalBLen + this.getInputLength(pair.b),
            totalWg: metrics.totalWg + numWg,
            totalDpiEntries: metrics.totalDpiEntries + (numWg > 0 ? 2 * (numWg + 1) : 0),
            maxOutputSize: metrics.maxOutputSize + Math.min(this.getInputLength(pair.a), this.getInputLength(pair.b)),
            totalInputBytes: metrics.totalInputBytes + this.getInputByteLength(pair.a) + this.getInputByteLength(pair.b)
        };
    }

    private fitsInDeviceLimits(
        metrics: PairBatchMetrics,
        mode: 'counts' | 'output_cpu' | 'output_gpu',
        maxChunkInputBytes: number,
        maxChunkGpuBytes: number
    ): boolean {
        const storageLimit = this.getStorageBufferLimit();
        const wgInfoBytes = Math.max(metrics.totalWg * WG_INFO_STRIDE * 4, 4);
        const pairIdBytes = Math.max(metrics.totalWg * 4, 4);
        const dpiBytes = Math.max(metrics.totalDpiEntries * 4, 4);
        const stateBytes = Math.max(metrics.totalWg * 4, 4);
        const pairCountsBytes = Math.max(metrics.numPairs * 4, 4);
        const outputBytes = Math.max(metrics.maxOutputSize * 4, 4);
        const pairOutputOffsetBytes = Math.max(metrics.totalWg * 4, 4);
        const estimatedGpuBytes = this.estimateChunkGpuBytes(metrics, mode);

        return (
            metrics.totalInputBytes <= maxChunkInputBytes &&
            estimatedGpuBytes <= maxChunkGpuBytes &&
            Math.max(metrics.totalALen * 4, 4) <= storageLimit &&
            Math.max(metrics.totalBLen * 4, 4) <= storageLimit &&
            wgInfoBytes <= storageLimit &&
            pairIdBytes <= storageLimit &&
            dpiBytes <= storageLimit &&
            stateBytes <= storageLimit &&
            pairCountsBytes <= storageLimit &&
            (mode === 'counts' || pairOutputOffsetBytes <= storageLimit) &&
            (mode === 'counts' || outputBytes <= storageLimit)
        );
    }

    private createChunkLimitError<T extends { a: BatchedSetInput; b: BatchedSetInput }>(
        pairIndex: number,
        pair: T,
        mode: 'counts' | 'output_cpu' | 'output_gpu',
        maxChunkInputBytes: number,
        maxChunkGpuBytes: number
    ): Error {
        const storageLimit = this.getStorageBufferLimit();
        const estimatedGpuBytes = this.estimateChunkGpuBytes(this.addPairToMetrics(this.createEmptyMetrics(), pair), mode);
        const opName = mode === 'counts' ? 'batchedIntersectionUnionCounts' : 'batchedIntersectionWithOutput';
        return new Error(
            `${opName}: pair ${pairIndex} exceeds batching limits ` +
            `(aBytes=${this.getInputByteLength(pair.a)}, bBytes=${this.getInputByteLength(pair.b)}, ` +
            `estimatedGpuBytes=${estimatedGpuBytes}, maxChunkInputBytes=${maxChunkInputBytes}, ` +
            `maxChunkGpuBytes=${maxChunkGpuBytes}, storageLimit=${storageLimit})`
        );
    }

    private getDefaultChunkGpuBytes(mode: 'counts' | 'output_cpu' | 'output_gpu'): number {
        return mode === 'counts'
            ? DEFAULT_MAX_CHUNK_GPU_BYTES_COUNTS
            : DEFAULT_MAX_CHUNK_GPU_BYTES_OUTPUT;
    }

    private estimateChunkGpuBytes(
        metrics: PairBatchMetrics,
        mode: 'counts' | 'output_cpu' | 'output_gpu'
    ): number {
        const keysBytes = Math.max(metrics.totalALen * 4, 4) + Math.max(metrics.totalBLen * 4, 4);
        const wgInfoBytes = Math.max(metrics.totalWg * WG_INFO_STRIDE * 4, 4);
        const pairIdBytes = Math.max(metrics.totalWg * 4, 4);
        const dpiBytes = Math.max(metrics.totalDpiEntries * 4, 4);
        const stateBytes = Math.max(metrics.totalWg * 4, 4);
        const pairCountsBytes = Math.max(metrics.numPairs * 4, 4);
        const outputBytes = Math.max(metrics.maxOutputSize * 4, 4);
        const pairOutputOffsetBytes = Math.max(metrics.totalWg * 4, 4);
        const uniformBytes = 4;

        if (mode === 'counts') {
            return keysBytes + wgInfoBytes + pairIdBytes + dpiBytes + (2 * stateBytes) + (4 * pairCountsBytes) + uniformBytes;
        }

        if (mode === 'output_cpu') {
            return keysBytes + wgInfoBytes + pairIdBytes + pairOutputOffsetBytes + dpiBytes + stateBytes + (2 * pairCountsBytes) + (2 * outputBytes) + uniformBytes;
        }

        return keysBytes + wgInfoBytes + pairIdBytes + pairOutputOffsetBytes + dpiBytes + stateBytes + (2 * pairCountsBytes) + outputBytes + uniformBytes;
    }

    private getStorageBufferLimit(): number {
        return Math.min(this.device.limits.maxStorageBufferBindingSize, this.device.limits.maxBufferSize);
    }

    private getInputLength(input: BatchedSetInput): number {
        return input.length;
    }

    private getInputByteLength(input: BatchedSetInput): number {
        return input.byteLength;
    }

    private getNumWorkgroups<T extends { a: BatchedSetInput; b: BatchedSetInput }>(pair: T): number {
        if (this.getInputLength(pair.a) === 0 || this.getInputLength(pair.b) === 0) return 0;
        return Math.ceil((this.getInputLength(pair.a) + this.getInputLength(pair.b)) / NV);
    }

    private prepareBatchMetadata<T extends { a: BatchedSetInput; b: BatchedSetInput }>(pairs: T[]): BatchMetadata {
        let totalALen = 0;
        let totalBLen = 0;
        let totalWg = 0;
        let totalDpiEntries = 0;

        const metas: PairMeta[] = [];

        for (const pair of pairs) {
            const aLen = this.getInputLength(pair.a);
            const bLen = this.getInputLength(pair.b);
            const numWg = this.getNumWorkgroups(pair);
            if (numWg === 0) {
                metas.push({ aStart: 0, aLen, bStart: 0, bLen, numWg: 0, stateOffset: 0, dpiOffset: 0 });
                continue;
            }

            metas.push({
                aStart: totalALen,
                aLen,
                bStart: totalBLen,
                bLen,
                numWg,
                stateOffset: totalWg,
                dpiOffset: totalDpiEntries
            });

            totalALen += aLen;
            totalBLen += bLen;
            totalWg += numWg;
            totalDpiEntries += 2 * (numWg + 1);
        }

        return { metas, totalALen, totalBLen, totalWg, totalDpiEntries, numPairs: pairs.length };
    }

    private createWorkgroupMetadata(metas: PairMeta[], numPairs: number, totalWg: number): {
        wgInfo: Uint32Array;
        pairIdPerWg: Uint32Array;
    } {
        const wgInfo = new Uint32Array(Math.max(totalWg * WG_INFO_STRIDE, 1));
        const pairIdPerWg = new Uint32Array(Math.max(totalWg, 1));

        let globalWgIdx = 0;
        for (let pairId = 0; pairId < numPairs; pairId++) {
            const meta = metas[pairId];
            if (meta.numWg === 0) continue;

            for (let wg = 0; wg < meta.numWg; wg++) {
                const base = globalWgIdx * WG_INFO_STRIDE;
                wgInfo[base + 0] = meta.aStart;
                wgInfo[base + 1] = meta.aLen;
                wgInfo[base + 2] = meta.bStart;
                wgInfo[base + 3] = meta.bLen;
                wgInfo[base + 4] = wg;
                wgInfo[base + 5] = meta.numWg;
                wgInfo[base + 6] = meta.stateOffset;
                wgInfo[base + 7] = meta.dpiOffset;
                pairIdPerWg[globalWgIdx] = pairId;
                globalWgIdx++;
            }
        }

        return { wgInfo, pairIdPerWg };
    }

    private prepareCountsBatch(pairs: BatchedSetPair[]): CountsBatchLayout {
        const metadata = this.prepareBatchMetadata(pairs);
        const keysA = new Uint32Array(Math.max(metadata.totalALen, 1));
        const keysB = new Uint32Array(Math.max(metadata.totalBLen, 1));
        const { wgInfo, pairIdPerWg } = this.createWorkgroupMetadata(
            metadata.metas,
            metadata.numPairs,
            metadata.totalWg
        );

        for (let pairId = 0; pairId < pairs.length; pairId++) {
            const meta = metadata.metas[pairId];
            if (meta.aLen > 0) {
                keysA.set(pairs[pairId].a, meta.aStart);
            }
            if (meta.bLen > 0) {
                keysB.set(pairs[pairId].b, meta.bStart);
            }
        }

        return { ...metadata, keysA, keysB, wgInfo, pairIdPerWg };
    }

    private prepareFullOutputBatch(pairs: BatchedSetPair[]): FullOutputBatchLayout {
        const countsBatch = this.prepareCountsBatch(pairs);
        const pairOutputOffsets: number[] = [];
        let maxOutputSize = 0;

        for (const pair of pairs) {
            pairOutputOffsets.push(maxOutputSize);
            maxOutputSize += Math.min(pair.a.length, pair.b.length);
        }

        const pairOutputOffsetPerWg = new Uint32Array(Math.max(countsBatch.totalWg, 1));

        let globalWgIdx = 0;
        for (let pairId = 0; pairId < pairs.length; pairId++) {
            const numWg = this.getNumWorkgroups(pairs[pairId]);
            for (let wg = 0; wg < numWg; wg++) {
                pairOutputOffsetPerWg[globalWgIdx] = pairOutputOffsets[pairId];
                globalWgIdx++;
            }
        }

        return {
            ...countsBatch,
            pairOutputOffsetPerWg,
            pairOutputOffsets,
            maxOutputSize
        };
    }

    private prepareFullOutputGPUChunkLayout(pairs: BatchedGPUSetPair[]): FullOutputGPUChunkLayout {
        const metadata = this.prepareBatchMetadata(pairs);
        const { wgInfo, pairIdPerWg } = this.createWorkgroupMetadata(
            metadata.metas,
            metadata.numPairs,
            metadata.totalWg
        );

        const pairOutputOffsets: number[] = [];
        let maxOutputSize = 0;
        for (const pair of pairs) {
            pairOutputOffsets.push(maxOutputSize);
            maxOutputSize += Math.min(this.getInputLength(pair.a), this.getInputLength(pair.b));
        }

        const pairOutputOffsetPerWg = new Uint32Array(Math.max(metadata.totalWg, 1));
        let globalWgIdx = 0;
        for (let pairId = 0; pairId < pairs.length; pairId++) {
            const numWg = metadata.metas[pairId].numWg;
            for (let wg = 0; wg < numWg; wg++) {
                pairOutputOffsetPerWg[globalWgIdx] = pairOutputOffsets[pairId];
                globalWgIdx++;
            }
        }

        return {
            ...metadata,
            wgInfo,
            pairIdPerWg,
            pairOutputOffsetPerWg,
            pairOutputOffsets,
            maxOutputSize
        };
    }

    private createPackedInputBuffers(
        pairs: BatchedGPUSetPair[],
        batch: FullOutputGPUChunkLayout
    ): { bufferA: GPUBuffer; bufferB: GPUBuffer } {
        const bufferA = this.device.createBuffer({
            size: Math.max(4, batch.totalALen * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        const bufferB = this.device.createBuffer({
            size: Math.max(4, batch.totalBLen * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        const encoder = this.device.createCommandEncoder();
        let hasGpuCopies = false;

        for (let pairId = 0; pairId < pairs.length; pairId++) {
            const pair = pairs[pairId];
            const meta = batch.metas[pairId];

            hasGpuCopies = this.packInputIntoBuffer(pair.a, meta.aStart * 4, meta.aLen, bufferA, encoder) || hasGpuCopies;
            hasGpuCopies = this.packInputIntoBuffer(pair.b, meta.bStart * 4, meta.bLen, bufferB, encoder) || hasGpuCopies;
        }

        if (hasGpuCopies) {
            this.device.queue.submit([encoder.finish()]);
        }

        return { bufferA, bufferB };
    }

    private packInputIntoBuffer(
        input: BatchedSetInput,
        dstOffsetBytes: number,
        elementCount: number,
        dstBuffer: GPUBuffer,
        encoder: GPUCommandEncoder
    ): boolean {
        if (elementCount === 0) {
            return false;
        }

        if (input instanceof GPUSetHandle) {
            encoder.copyBufferToBuffer(input.buffer, input.byteOffset, dstBuffer, dstOffsetBytes, input.byteLength);
            return true;
        }

        this.device.queue.writeBuffer(dstBuffer, dstOffsetBytes, new Uint32Array(input));
        return false;
    }

    private createOutputHandles(
        outputBuffer: GPUBuffer,
        pairOutputOffsets: number[],
        pairCounts: Uint32Array,
        onZero: ((buffer: GPUBuffer) => void) | null = null
    ): GPUSetHandle[] {
        if (pairCounts.length === 0) {
            if (onZero !== null) onZero(outputBuffer);
            else outputBuffer.destroy();
            return [];
        }

        const owner = new GPUSetBufferOwner(outputBuffer, true, onZero);
        return pairOutputOffsets.map((offset, index) => new GPUSetHandle(owner, offset * 4, pairCounts[index]));
    }

    private createEmptyHandle(): GPUSetHandle {
        if (this.emptySetOwner === null) {
            const emptyBuffer = this.device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            });
            this.emptySetOwner = new GPUSetBufferOwner(emptyBuffer, false);
        }
        return new GPUSetHandle(this.emptySetOwner, 0, 0);
    }

    private async runWithGpuErrorScopes<T>(label: string, work: () => Promise<T>): Promise<T> {
        this.device.pushErrorScope('validation');
        this.device.pushErrorScope('out-of-memory');

        try {
            const result = await work();
            const oomError = await this.device.popErrorScope();
            const validationError = await this.device.popErrorScope();
            if (oomError !== null) {
                throw new Error(`${label}: ${oomError.message}`);
            }
            if (validationError !== null) {
                throw new Error(`${label}: ${validationError.message}`);
            }
            return result;
        } catch (error) {
            const oomError = await this.device.popErrorScope();
            const validationError = await this.device.popErrorScope();
            if (oomError !== null) {
                throw new Error(`${label}: ${oomError.message}`);
            }
            if (validationError !== null) {
                throw new Error(`${label}: ${validationError.message}`);
            }
            throw error;
        }
    }
}
