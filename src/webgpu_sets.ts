/**
 * WebGPUSets — Public API for GPU-accelerated set operations.
 *
 * One class, eight methods. The constructor compiles all pipelines once;
 * subsequent calls reuse them with zero additional compilation cost.
 *
 * Usage:
 *   const sets = new WebGPUSets(device);
 *   const { result, totalCount } = await sets.intersection(A, B);
 *   const { resultKeys, resultValues, totalCount } = await sets.intersectionByKey(kA, vA, kB, vB);
 */

import { setOpMode } from './utils';
import computeDiagonalsShader from './balanced_path/common/balanced_path_biased.wgsl';
import lookbackShaderBase from './balanced_path/common/not_by_key/set_availability_decoupled_lookback.wgsl';
import lookbackByKeyShaderBase from './balanced_path/common/by_key/set_availability_decoupled_lookback_by_key.wgsl';

// Pipeline constants (must match WGSL)
const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 12;
const NV = NT * VT;  // 3072
const DPI_WG_SIZE = 256;

/** Return types */
export interface SetResult {
    result: Uint32Array;
    totalCount: number;
}

export interface SetByKeyResult {
    resultKeys: Uint32Array;
    resultValues: Uint32Array;
    totalCount: number;
}

/**
 * A sorted set that stays in device memory between operations.
 * The handle owns its buffer and must be destroyed by the caller.
 */
export class GPUSet {
    constructor(
        public readonly buffer: GPUBuffer,
        public readonly length: number
    ) {}

    destroy(): void {
        this.buffer.destroy();
    }
}

/** Max possible output size for each operation. */
function getMaxOutputSize(opMode: number, aLen: number, bLen: number): number {
    switch (opMode) {
        case 0: return Math.min(aLen, bLen);   // intersection
        case 1: return aLen;                    // difference
        case 2: return aLen + bLen;             // union
        case 3: return aLen + bLen;             // sym_difference
        default: return aLen + bLen;
    }
}

export class WebGPUSets {
    private device: GPUDevice;

    // Shared: 1 DPI pipeline (no OP_MODE dependency)
    private dpiPipeline: GPUComputePipeline;
    private dpiBindGroupLayout: GPUBindGroupLayout;

    // Keys-only: 4 lookback pipelines (OP_MODE 0-3)
    private lookbackPipelines: GPUComputePipeline[];
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    // By-key: 4 lookback pipelines (OP_MODE 0-3)
    private lookbackByKeyPipelines: GPUComputePipeline[];
    private lookbackByKeyBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice) {
        this.device = device;

        // ---- Shared DPI pipeline (6 bindings) ----
        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'WebGPUSets DPI bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // b_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // num_wg
            ]
        });

        this.dpiPipeline = device.createComputePipeline({
            label: 'WebGPUSets DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // ---- Keys-only: 4 lookback pipelines (9 bindings, shared layout) ----
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'WebGPUSets Lookback bind group layout',
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

        const lookbackPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.lookbackBindGroupLayout]
        });

        this.lookbackPipelines = [0, 1, 2, 3].map(opMode =>
            device.createComputePipeline({
                label: `WebGPUSets Lookback OP_MODE=${opMode}`,
                layout: lookbackPipelineLayout,
                compute: {
                    module: device.createShaderModule({ code: setOpMode(lookbackShaderBase, opMode) }),
                    entryPoint: 'decoupled_lookback_kernel'
                }
            })
        );

        // ---- By-key: 4 lookback pipelines (12 bindings, shared layout) ----
        this.lookbackByKeyBindGroupLayout = device.createBindGroupLayout({
            label: 'WebGPUSets Lookback By-Key bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_values
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // state
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_keys
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // output_values
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // total_count
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // a_length
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg_total
            ]
        });

        const lookbackByKeyPipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [this.lookbackByKeyBindGroupLayout]
        });

        this.lookbackByKeyPipelines = [0, 1, 2, 3].map(opMode =>
            device.createComputePipeline({
                label: `WebGPUSets Lookback By-Key OP_MODE=${opMode}`,
                layout: lookbackByKeyPipelineLayout,
                compute: {
                    module: device.createShaderModule({ code: setOpMode(lookbackByKeyShaderBase, opMode) }),
                    entryPoint: 'decoupled_lookback_by_key_kernel'
                }
            })
        );
    }

    // ================================================================
    //  Public API: 8 methods
    // ================================================================

    async intersection(A: Uint32Array, B: Uint32Array): Promise<SetResult> {
        return this.runKeysOnly(0, A, B);
    }

    async union(A: Uint32Array, B: Uint32Array): Promise<SetResult> {
        return this.runKeysOnly(2, A, B);
    }

    async difference(A: Uint32Array, B: Uint32Array): Promise<SetResult> {
        return this.runKeysOnly(1, A, B);
    }

    async symDifference(A: Uint32Array, B: Uint32Array): Promise<SetResult> {
        return this.runKeysOnly(3, A, B);
    }

    upload(A: Uint32Array): GPUSet {
        const buffer = this.device.createBuffer({
            size: Math.max(4, A.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });
        if (A.byteLength > 0) this.device.queue.writeBuffer(buffer, 0, new Uint32Array(A));
        return new GPUSet(buffer, A.length);
    }

    async download(A: GPUSet): Promise<Uint32Array> {
        return this.readBuffer(A.buffer, A.length);
    }

    async intersectionGPU(A: GPUSet, B: GPUSet): Promise<GPUSet> {
        return this.runKeysOnlyGPU(0, A, B);
    }

    async differenceGPU(A: GPUSet, B: GPUSet): Promise<GPUSet> {
        return this.runKeysOnlyGPU(1, A, B);
    }

    async unionGPU(A: GPUSet, B: GPUSet): Promise<GPUSet> {
        return this.runKeysOnlyGPU(2, A, B);
    }

    async symDifferenceGPU(A: GPUSet, B: GPUSet): Promise<GPUSet> {
        return this.runKeysOnlyGPU(3, A, B);
    }

    async intersectionByKey(kA: Uint32Array, vA: Uint32Array, kB: Uint32Array, vB: Uint32Array): Promise<SetByKeyResult> {
        return this.runByKey(0, kA, vA, kB, vB);
    }

    async unionByKey(kA: Uint32Array, vA: Uint32Array, kB: Uint32Array, vB: Uint32Array): Promise<SetByKeyResult> {
        return this.runByKey(2, kA, vA, kB, vB);
    }

    async differenceByKey(kA: Uint32Array, vA: Uint32Array, kB: Uint32Array, vB: Uint32Array): Promise<SetByKeyResult> {
        return this.runByKey(1, kA, vA, kB, vB);
    }

    async symDifferenceByKey(kA: Uint32Array, vA: Uint32Array, kB: Uint32Array, vB: Uint32Array): Promise<SetByKeyResult> {
        return this.runByKey(3, kA, vA, kB, vB);
    }

    // ================================================================
    //  Private: keys-only execution
    // ================================================================

    private async runKeysOnly(opMode: number, A: Uint32Array, B: Uint32Array): Promise<SetResult> {
        const gpuA = this.upload(A);
        const gpuB = this.upload(B);
        try {
            const gpuResult = await this.runKeysOnlyGPU(opMode, gpuA, gpuB);
            try {
                return {
                    result: await this.download(gpuResult),
                    totalCount: gpuResult.length
                };
            } finally {
                gpuResult.destroy();
            }
        } finally {
            gpuA.destroy();
            gpuB.destroy();
        }
    }

    private async runKeysOnlyGPU(opMode: number, A: GPUSet, B: GPUSet): Promise<GPUSet> {
        const device = this.device;
        const a_len = A.length;
        const b_len = B.length;
        const total = a_len + b_len;

        if (total === 0) {
            return new GPUSet(device.createBuffer({
                size: 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
            }), 0);
        }

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = getMaxOutputSize(opMode, a_len, b_len);

        // --- Create buffers ---
        const bufferA = A.buffer;
        const bufferB = B.buffer;

        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const bufferDPI = device.createBuffer({ size: 2 * (numWg + 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufferState = device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufferOutput = device.createBuffer({ size: Math.max(maxOutputSize, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufferTotalCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        // Zero state and total count
        device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
        device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

        // --- Bind groups ---
        const diagBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
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

        // --- Dispatch ---
        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        const encoder = device.createCommandEncoder();

        // Pass 1: DPI
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.dpiPipeline);
        pass.setBindGroup(0, diagBindGroup);
        pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
        pass.end();

        // Pass 2: Lookback (select pipeline by opMode)
        pass = encoder.beginComputePass();
        pass.setPipeline(this.lookbackPipelines[opMode]);
        pass.setBindGroup(0, lookbackBindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // --- Readback totalCount ---
        const totalCount = await this.readU32(bufferTotalCount);

        // --- Cleanup ---
        bufferALen.destroy(); bufferBLen.destroy(); bufferNumWg.destroy();
        bufferDPI.destroy(); bufferState.destroy();
        bufferTotalCount.destroy();

        return new GPUSet(bufferOutput, totalCount);
    }

    // ================================================================
    //  Private: by-key execution
    // ================================================================

    private async runByKey(
        opMode: number,
        kA: Uint32Array, vA: Uint32Array,
        kB: Uint32Array, vB: Uint32Array
    ): Promise<SetByKeyResult> {
        const device = this.device;
        const a_len = kA.length;
        const b_len = kB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return { resultKeys: new Uint32Array(0), resultValues: new Uint32Array(0), totalCount: 0 };
        }

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = getMaxOutputSize(opMode, a_len, b_len);

        // --- Create buffers ---
        const bufferAKeys = device.createBuffer({ size: Math.max(4, kA.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufferAValues = device.createBuffer({ size: Math.max(4, vA.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufferBKeys = device.createBuffer({ size: Math.max(4, kB.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufferBValues = device.createBuffer({ size: Math.max(4, vB.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(kA));
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(vA));
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(kB));
        device.queue.writeBuffer(bufferBValues, 0, new Uint32Array(vB));

        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufferALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

        const bufferDPI = device.createBuffer({ size: 2 * (numWg + 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufferState = device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufferOutputKeys = device.createBuffer({ size: Math.max(maxOutputSize, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufferOutputValues = device.createBuffer({ size: Math.max(maxOutputSize, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufferTotalCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        // Zero state and total count
        device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
        device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

        // --- Bind groups ---
        // DPI operates on keys only
        const diagBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackByKeyBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferBValues } },
                { binding: 4, resource: { buffer: bufferDPI } },
                { binding: 5, resource: { buffer: bufferState } },
                { binding: 6, resource: { buffer: bufferOutputKeys } },
                { binding: 7, resource: { buffer: bufferOutputValues } },
                { binding: 8, resource: { buffer: bufferTotalCount } },
                { binding: 9, resource: { buffer: bufferALen } },
                { binding: 10, resource: { buffer: bufferBLen } },
                { binding: 11, resource: { buffer: bufferNumWg } },
            ]
        });

        // --- Dispatch ---
        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);

        const encoder = device.createCommandEncoder();

        // Pass 1: DPI (on keys only)
        let pass = encoder.beginComputePass();
        pass.setPipeline(this.dpiPipeline);
        pass.setBindGroup(0, diagBindGroup);
        pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
        pass.end();

        // Pass 2: Lookback By Key (select pipeline by opMode)
        pass = encoder.beginComputePass();
        pass.setPipeline(this.lookbackByKeyPipelines[opMode]);
        pass.setBindGroup(0, lookbackBindGroup);
        pass.dispatchWorkgroups(dispatchX, dispatchY);
        pass.end();

        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        // --- Readback ---
        const totalCount = await this.readU32(bufferTotalCount);
        const resultKeys = await this.readBuffer(bufferOutputKeys, totalCount);
        const resultValues = await this.readBuffer(bufferOutputValues, totalCount);

        // --- Cleanup ---
        bufferAKeys.destroy(); bufferAValues.destroy();
        bufferBKeys.destroy(); bufferBValues.destroy();
        bufferALen.destroy(); bufferBLen.destroy(); bufferNumWg.destroy();
        bufferDPI.destroy(); bufferState.destroy();
        bufferOutputKeys.destroy(); bufferOutputValues.destroy();
        bufferTotalCount.destroy();

        return { resultKeys, resultValues, totalCount };
    }

    // ================================================================
    //  Private: readback helpers
    // ================================================================

    private async readU32(srcBuffer: GPUBuffer): Promise<number> {
        const device = this.device;
        const staging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, 4);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await staging.mapAsync(GPUMapMode.READ);
        const value = new Uint32Array(staging.getMappedRange().slice(0))[0];
        staging.unmap();
        staging.destroy();
        return value;
    }

    private async readBuffer(srcBuffer: GPUBuffer, count: number): Promise<Uint32Array> {
        if (count === 0) return new Uint32Array(0);
        const device = this.device;
        const byteSize = count * 4;
        const staging = device.createBuffer({ size: byteSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(srcBuffer, 0, staging, 0, byteSize);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await staging.mapAsync(GPUMapMode.READ);
        const data = new Uint32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return data;
    }
}
