/**
 * Test file for Complete Set Union By Key Pipeline (v1)
 *
 * Tests the full two-pass approach for union by key:
 * 1. DPI (Diagonal Path Indices) - Compute partition boundaries (keys only)
 * 2. Count Phase - Count outputs per workgroup (keys only)
 * 3. Prefix Sum - Exclusive scan on counts for output offsets
 * 4. Write Phase - Write actual key-value pairs
 *
 * Follows Thrust semantics:
 * - If A has m keys and B has n keys with same value: output max(m,n)
 * - Output all m from A, then max(n-m, 0) from B
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_union_by_key_count_v1.wgsl';
import writeShader from './set_availability_union_by_key_write_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';
import * as utils from '../../utils';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;
const MAXWORKGROUP = 65535;

// ModernGPU constants
const NT = 256;
const VT = 7;
const NV = NT * VT;  // 1792

/**
 * Complete Set Union By Key Pipeline Test Class
 */
export class TestSetUnionByKeyPipeline {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    // DPI pipeline (uses keys only)
    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;

    // Count pipeline (uses keys only)
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;

    // Write pipeline (uses keys and values from both A and B)
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;

    // Prefix sum
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        // DPI bind group layout (6 bindings, keys only)
        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: 'DPI bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // a_length
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg
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

        // Count bind group layout (7 bindings, keys only)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: 'Union By Key Count kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // counts
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // a_length
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // num_wg_total
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: 'Union By Key Count kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write bind group layout (11 bindings - needs both A and B values)
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: 'Union By Key Write kernel bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_keys
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // a_values
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_keys
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // b_values
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // offsets
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // output_keys
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },            // output_values
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // a_length
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },            // b_length
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // num_wg_total
            ]
        });

        this.writePipeline = device.createComputePipeline({
            label: 'Union By Key Write kernel pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    /**
     * Run the complete set union by key pipeline.
     */
    public async computeSetUnionByKey(
        keysA: Uint32Array,
        valuesA: Uint32Array,
        keysB: Uint32Array,
        valuesB: Uint32Array,
        iterations: number = 1,
        warmup: number = 1
    ): Promise<{
        resultKeys: Uint32Array;
        resultValues: Uint32Array;
        totalCount: number;
        gpuTimeMs: number;
    }> {
        const device = this.device;
        const a_len = keysA.length;
        const b_len = keysB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return { resultKeys: new Uint32Array(0), resultValues: new Uint32Array(0), totalCount: 0, gpuTimeMs: 0 };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============

        // Create GPU buffers
        const bufferAKeys = device.createBuffer({
            label: 'Buffer A Keys',
            size: Math.max(4, keysA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(keysA));

        const bufferAValues = device.createBuffer({
            label: 'Buffer A Values',
            size: Math.max(4, valuesA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(valuesA));

        const bufferBKeys = device.createBuffer({
            label: 'Buffer B Keys',
            size: Math.max(4, keysB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(keysB));

        const bufferBValues = device.createBuffer({
            label: 'Buffer B Values',
            size: Math.max(4, valuesB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBValues, 0, new Uint32Array(valuesB));

        // Uniform buffers
        const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
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

        // Counts buffer (aligned for scan)
        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            label: 'Buffer Counts',
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        // Buffer to preserve original counts
        const bufferCountsCopy = device.createBuffer({
            label: 'Buffer Counts Copy',
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Pre-allocate output buffers with max possible size (a_len + b_len for union)
        const maxOutputSize = a_len + b_len;
        const bufferOutputKeys = device.createBuffer({
            label: 'Buffer Output Keys',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bufferOutputValues = device.createBuffer({
            label: 'Buffer Output Values',
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        // Create bind groups
        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferBValues } },
                { binding: 4, resource: { buffer: bufferDPI } },
                { binding: 5, resource: { buffer: bufferCounts } },  // Contains offsets after scan
                { binding: 6, resource: { buffer: bufferOutputKeys } },
                { binding: 7, resource: { buffer: bufferOutputValues } },
                { binding: 8, resource: { buffer: bufferALen } },
                { binding: 9, resource: { buffer: bufferBLen } },
                { binding: 10, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // ============ GPU Execution ============
        const times: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const t0 = performance.now();

            const encoder = device.createCommandEncoder();

            // Phase 1: DPI
            let pass = encoder.beginComputePass({ label: 'DPI' });
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count
            pass = encoder.beginComputePass({ label: 'Count' });
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Copy counts before prefix sum
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum
            scanner.recordScanCommands(encoder, numWg);

            // Phase 4: Write
            pass = encoder.beginComputePass({ label: 'Write' });
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            times.push(performance.now() - t0);
        }

        // Calculate average time
        let gpuTimeMs: number;
        if (iterations > warmup) {
            const timesWithoutWarmup = times.slice(warmup);
            gpuTimeMs = timesWithoutWarmup.reduce((a, b) => a + b, 0) / timesWithoutWarmup.length;
        } else {
            gpuTimeMs = times[times.length - 1] || 0;
        }

        // ============ Readback Phase ============
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        // Read output
        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const keysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const valuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, keysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, valuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await keysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(keysReadback.getMappedRange().slice(0));
            keysReadback.unmap();

            await valuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(valuesReadback.getMappedRange().slice(0));
            valuesReadback.unmap();

            keysReadback.destroy();
            valuesReadback.destroy();
        }

        // Cleanup
        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
        bufferBValues.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutputKeys.destroy();
        bufferOutputValues.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return { resultKeys, resultValues, totalCount, gpuTimeMs };
    }

    /**
     * CPU reference implementation for union by key.
     * Thrust semantics: if A has m keys and B has n keys with same value,
     * output all m from A, then max(n-m, 0) from B. Total: max(m, n).
     */
    public cpuSetUnionByKey(
        keysA: Uint32Array,
        valuesA: Uint32Array,
        keysB: Uint32Array,
        valuesB: Uint32Array
    ): { keys: Uint32Array; values: Uint32Array } {
        const resultKeys: number[] = [];
        const resultValues: number[] = [];
        let ai = 0, bi = 0;

        while (ai < keysA.length && bi < keysB.length) {
            if (keysA[ai] < keysB[bi]) {
                // A < B: emit from A
                resultKeys.push(keysA[ai]);
                resultValues.push(valuesA[ai]);
                ai++;
            } else if (keysA[ai] > keysB[bi]) {
                // B < A: emit from B
                resultKeys.push(keysB[bi]);
                resultValues.push(valuesB[bi]);
                bi++;
            } else {
                // Equal: emit from A (tie goes to A), advance both
                resultKeys.push(keysA[ai]);
                resultValues.push(valuesA[ai]);
                ai++;
                bi++;
            }
        }

        // Remaining elements from A
        while (ai < keysA.length) {
            resultKeys.push(keysA[ai]);
            resultValues.push(valuesA[ai]);
            ai++;
        }

        // Remaining elements from B
        while (bi < keysB.length) {
            resultKeys.push(keysB[bi]);
            resultValues.push(valuesB[bi]);
            bi++;
        }

        return {
            keys: new Uint32Array(resultKeys),
            values: new Uint32Array(resultValues)
        };
    }

    /**
     * Validate GPU results against CPU
     */
    public validateResults(
        gpuKeys: Uint32Array,
        gpuValues: Uint32Array,
        keysA: Uint32Array,
        valuesA: Uint32Array,
        keysB: Uint32Array,
        valuesB: Uint32Array
    ): boolean {
        const cpu = this.cpuSetUnionByKey(keysA, valuesA, keysB, valuesB);

        if (gpuKeys.length !== cpu.keys.length) {
            console.log(`Length mismatch: GPU=${gpuKeys.length}, CPU=${cpu.keys.length}`);
            return false;
        }

        for (let i = 0; i < cpu.keys.length; i++) {
            if (gpuKeys[i] !== cpu.keys[i]) {
                console.log(`Key mismatch at index ${i}: GPU=${gpuKeys[i]}, CPU=${cpu.keys[i]}`);
                return false;
            }
            if (gpuValues[i] !== cpu.values[i]) {
                console.log(`Value mismatch at index ${i}: GPU=${gpuValues[i]}, CPU=${cpu.values[i]}`);
                console.log(`  (Key at this index: ${gpuKeys[i]})`);
                return false;
            }
        }

        return true;
    }

    /**
     * Run the complete set union by key pipeline with GPU timestamp profiling.
     */
    public async computeSetUnionByKeyWithGPUProfiling(
        keysA: Uint32Array,
        valuesA: Uint32Array,
        keysB: Uint32Array,
        valuesB: Uint32Array,
        iterations: number = 20,
        warmup: number = 10
    ): Promise<{
        resultKeys: Uint32Array;
        resultValues: Uint32Array;
        totalCount: number;
        timing: {
            dpiMs: number;
            countMs: number;
            scanMs: number;
            writeMs: number;
            totalMs: number;
        };
    }> {
        const device = this.device;
        const a_len = keysA.length;
        const b_len = keysB.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                resultKeys: new Uint32Array(0),
                resultValues: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV);

        // ============ Setup Phase ============
        const bufferAKeys = device.createBuffer({
            size: Math.max(4, keysA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(keysA));

        const bufferAValues = device.createBuffer({
            size: Math.max(4, valuesA.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(valuesA));

        const bufferBKeys = device.createBuffer({
            size: Math.max(4, keysB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(keysB));

        const bufferBValues = device.createBuffer({
            size: Math.max(4, valuesB.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBValues, 0, new Uint32Array(valuesB));

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

        const alignedSize = this.scanPipeline.getAlignedSize(numWg);
        const bufferCounts = device.createBuffer({
            size: alignedSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        const bufferCountsCopy = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        const maxOutputSize = a_len + b_len;
        const bufferOutputKeys = device.createBuffer({
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bufferOutputValues = device.createBuffer({
            size: Math.max(4, maxOutputSize * 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const diagBindGroup = device.createBindGroup({
            layout: this.diagBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferALen } },
                { binding: 4, resource: { buffer: bufferBLen } },
                { binding: 5, resource: { buffer: bufferNumWg } },
            ]
        });

        const countBindGroup = device.createBindGroup({
            layout: this.countBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferBKeys } },
                { binding: 2, resource: { buffer: bufferDPI } },
                { binding: 3, resource: { buffer: bufferCounts } },
                { binding: 4, resource: { buffer: bufferALen } },
                { binding: 5, resource: { buffer: bufferBLen } },
                { binding: 6, resource: { buffer: bufferNumWg } },
            ]
        });

        const writeBindGroup = device.createBindGroup({
            layout: this.writeBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferBValues } },
                { binding: 4, resource: { buffer: bufferDPI } },
                { binding: 5, resource: { buffer: bufferCounts } },
                { binding: 6, resource: { buffer: bufferOutputKeys } },
                { binding: 7, resource: { buffer: bufferOutputValues } },
                { binding: 8, resource: { buffer: bufferALen } },
                { binding: 9, resource: { buffer: bufferBLen } },
                { binding: 10, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // ============ Run Multiple Iterations with GPU Timestamps ============
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const encoder = device.createCommandEncoder();

            // Phase 1: DPI (timestamps 0, 1)
            let pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(0, 1)
            );
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Phase 2: Count (timestamps 2, 3)
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(2, 3)
            );
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Copy counts before prefix sum
            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);

            // Phase 3: Prefix Sum (timestamps 4, 5)
            scanner.recordScanCommands(encoder, numWg, this.timestampQueryManager, 4, 5);

            // Phase 4: Write (timestamps 6, 7)
            pass = encoder.beginComputePass(
                this.timestampQueryManager.createComputePassDescriptor(6, 7)
            );
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            // Resolve timestamps
            this.timestampQueryManager.resolve(encoder);

            // Submit and wait
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            // Download timestamps (in nanoseconds)
            const timestamps = await this.timestampQueryManager.downloadTimestampResult();

            if (timestamps.length >= 8) {
                const dpiNs = timestamps[1] - timestamps[0];
                const countNs = timestamps[3] - timestamps[2];
                const writeNs = timestamps[7] - timestamps[6];
                const totalNs = timestamps[7] - timestamps[0];
                const scanNsIndirect = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNsIndirect / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        // Calculate average (skip warmup)
        const avg = (arr: number[]) => {
            if (arr.length <= warmup) return arr[arr.length - 1] || 0;
            const withoutWarmup = arr.slice(warmup);
            return withoutWarmup.reduce((a, b) => a + b, 0) / withoutWarmup.length;
        };

        const timing = {
            dpiMs: avg(dpiTimes),
            countMs: avg(countTimes),
            scanMs: avg(scanTimes),
            writeMs: avg(writeTimes),
            totalMs: avg(totalTimes)
        };

        // ============ Readback ============
        const countsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const offsetsReadback = device.createBuffer({
            size: numWg * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        let readEncoder = device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufferCountsCopy, 0, countsReadback, 0, numWg * 4);
        readEncoder.copyBufferToBuffer(bufferCounts, 0, offsetsReadback, 0, numWg * 4);
        device.queue.submit([readEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await countsReadback.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(countsReadback.getMappedRange().slice(0));
        countsReadback.unmap();

        await offsetsReadback.mapAsync(GPUMapMode.READ);
        const offsets = new Uint32Array(offsetsReadback.getMappedRange().slice(0));
        offsetsReadback.unmap();

        const totalCount = offsets[numWg - 1] + counts[numWg - 1];

        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const keysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const valuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, keysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, valuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await keysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(keysReadback.getMappedRange().slice(0));
            keysReadback.unmap();

            await valuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(valuesReadback.getMappedRange().slice(0));
            valuesReadback.unmap();

            keysReadback.destroy();
            valuesReadback.destroy();
        }

        // Cleanup
        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
        bufferBValues.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferCounts.destroy();
        bufferCountsCopy.destroy();
        bufferOutputKeys.destroy();
        bufferOutputValues.destroy();
        countsReadback.destroy();
        offsetsReadback.destroy();

        return { resultKeys, resultValues, totalCount, timing };
    }
}

/**
 * Run complete pipeline tests for union by key
 */
export async function runUnionByKeyWriteKernelTest(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY COMPLETE PIPELINE TEST               ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    let allPassed = true;

    // Test Case 1: Simple union by key
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 1: Simple union by key');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const keysA = new Uint32Array([1, 3, 5, 7, 9]);
        const valuesA = new Uint32Array([10, 30, 50, 70, 90]);
        const keysB = new Uint32Array([2, 3, 4, 7, 8]);
        const valuesB = new Uint32Array([20, 31, 40, 71, 80]);
        // Union: [1,2,3,4,5,7,8,9]
        // Values: [10,20,30,40,50,70,80,90] (3 and 7 from A)

        console.log('A keys:', Array.from(keysA), 'values:', Array.from(valuesA));
        console.log('B keys:', Array.from(keysB), 'values:', Array.from(valuesB));

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        const cpu = tester.cpuSetUnionByKey(keysA, valuesA, keysB, valuesB);
        console.log('GPU result keys:', Array.from(resultKeys), 'values:', Array.from(resultValues));
        console.log('CPU result keys:', Array.from(cpu.keys), 'values:', Array.from(cpu.values));

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 2: No overlap
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 2: No overlap');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const keysA = new Uint32Array([1, 3, 5]);
        const valuesA = new Uint32Array([10, 30, 50]);
        const keysB = new Uint32Array([2, 4, 6]);
        const valuesB = new Uint32Array([20, 40, 60]);
        // Union: [1,2,3,4,5,6]

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        console.log(`Result length: ${resultKeys.length}, expected: 6`);
        console.log('GPU result keys:', Array.from(resultKeys));

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 3: Complete overlap (all keys match)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 3: Complete overlap');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const keysA = new Uint32Array([1, 2, 3, 4, 5]);
        const valuesA = new Uint32Array([10, 20, 30, 40, 50]);
        const keysB = new Uint32Array([1, 2, 3, 4, 5]);
        const valuesB = new Uint32Array([11, 21, 31, 41, 51]);  // Different values
        // Union: [1,2,3,4,5] - all from A since tie goes to A

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        console.log(`Result length: ${resultKeys.length}, expected: 5`);
        console.log('GPU values:', Array.from(resultValues));
        console.log('Expected (A values):', Array.from(valuesA));

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 4: Duplicates with union (max(m,n) semantics)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 4: Duplicates with union (max(m,n) semantics)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const keysA = new Uint32Array([1, 1, 2, 2, 3, 3]);
        const valuesA = new Uint32Array([10, 11, 20, 21, 30, 31]);
        const keysB = new Uint32Array([1, 2, 2, 2, 3]);
        const valuesB = new Uint32Array([12, 22, 23, 24, 32]);
        // Key 1: A has 2, B has 1 -> output 2 (all from A)
        // Key 2: A has 2, B has 3 -> output 3 (2 from A, 1 from B)
        // Key 3: A has 2, B has 1 -> output 2 (all from A)
        // Total: 7

        console.log('A keys:', Array.from(keysA), 'values:', Array.from(valuesA));
        console.log('B keys:', Array.from(keysB), 'values:', Array.from(valuesB));

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        const cpu = tester.cpuSetUnionByKey(keysA, valuesA, keysB, valuesB);
        console.log('GPU result keys:', Array.from(resultKeys), 'values:', Array.from(resultValues));
        console.log('CPU result keys:', Array.from(cpu.keys), 'values:', Array.from(cpu.values));

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 5: Large dataset with partial overlap
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 5: Large dataset with partial overlap');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 1000;
        const keysA = new Uint32Array(size).map((_, i) => i);
        const valuesA = new Uint32Array(size).map((_, i) => i + 1000);
        const keysB = new Uint32Array(size).map((_, i) => i + 500);
        const valuesB = new Uint32Array(size).map((_, i) => i + 2000);
        // A: 0-999, B: 500-1499
        // Union: 0-1499 => 1500 elements

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        const cpu = tester.cpuSetUnionByKey(keysA, valuesA, keysB, valuesB);
        console.log(`Result length: GPU=${resultKeys.length}, CPU=${cpu.keys.length}`);

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 6: Multi-workgroup test
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 6: Multi-workgroup (even keys vs multiples of 3)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const size = 3000;
        const keysA = new Uint32Array(size).map((_, i) => i * 2);
        const valuesA = new Uint32Array(size).map((_, i) => i);
        const keysB = new Uint32Array(size).map((_, i) => i * 3);
        const valuesB = new Uint32Array(size).map((_, i) => i + 10000);

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        const cpu = tester.cpuSetUnionByKey(keysA, valuesA, keysB, valuesB);
        console.log(`Result length: GPU=${resultKeys.length}, CPU=${cpu.keys.length}`);

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Test Case 7: All same keys (multi-WG)
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    console.log('Test Case 7: All same keys (multi-WG)');
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
    {
        const keysA = new Uint32Array(2500).fill(42);
        const valuesA = new Uint32Array(2500).map((_, i) => i);
        const keysB = new Uint32Array(2500).fill(42);
        const valuesB = new Uint32Array(2500).map((_, i) => i + 10000);
        // All keys match => output max(2500, 2500) = 2500

        const { resultKeys, resultValues, totalCount, gpuTimeMs } =
            await tester.computeSetUnionByKey(keysA, valuesA, keysB, valuesB);

        console.log(`Result length: GPU=${resultKeys.length}, expected: 2500`);

        const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
        console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}`);
        console.log(`GPU Time: ${gpuTimeMs.toFixed(3)} ms\n`);
        if (!valid) allPassed = false;
    }

    // Summary
    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL UNION BY KEY PIPELINE TESTS PASSED                  ║');
    } else {
        console.log('║  ✗ SOME UNION BY KEY PIPELINE TESTS FAILED                 ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Benchmark with real key-value datasets from public/data directory
 */
export async function runUnionByKeyDatasetBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY DATASET BENCHMARK                    ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    const datasets = [
        { size: '1', range: '2', desc: '1M elements, range 100 (many duplicates)' },
        { size: '1', range: '6', desc: '1M elements, range 1M (few duplicates)' },
        { size: '2', range: '2', desc: '2M elements, range 100' },
        { size: '2', range: '6', desc: '2M elements, range 1M' },
        { size: '4', range: '2', desc: '4M elements, range 100' },
        { size: '4', range: '6', desc: '4M elements, range 1M' },
        { size: '8', range: '2', desc: '8M elements, range 100' },
        { size: '8', range: '6', desc: '8M elements, range 1M' },
        { size: '16', range: '2', desc: '16M elements, range 100' },
        { size: '16', range: '6', desc: '16M elements, range 1M' },
        { size: '32', range: '2', desc: '32M elements, range 100' },
        { size: '32', range: '6', desc: '32M elements, range 1M' },
        { size: '64', range: '2', desc: '64M elements, range 100' },
        { size: '64', range: '6', desc: '64M elements, range 1M' },
        { size: '128', range: '2', desc: '128M elements, range 100' },
        { size: '128', range: '6', desc: '128M elements, range 1M' },
    ];

    const NUM_ITERATIONS = 20;
    const NUM_WARMUP = 10;
    console.log(`Running ${NUM_ITERATIONS} iterations per dataset (${NUM_WARMUP} warmup, averaging remaining ${NUM_ITERATIONS - NUM_WARMUP})...\n`);

    console.log('╔══════════╤══════════════╤════════════════╤════════╤════════╤════════╤════════╤════════╗');
    console.log('║ Dataset  │ Input Size   │ Union          │ DPI(ms)│Count(ms)│Scan(ms)│Write(ms)│Total(ms)║');
    console.log('╠══════════╪══════════════╪════════════════╪════════╪════════╪════════╪════════╪════════╣');

    for (const { size, range, desc } of datasets) {
        const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
        const aValuesPath = `./data/A_values_${size}e${range}.bin`;
        const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
        const bValuesPath = `./data/B_values_${size}e${range}.bin`;

        try {
            const keysA = await utils.loadUint32ArrayFromBin(aKeysPath);
            const valuesA = await utils.loadUint32ArrayFromBin(aValuesPath);
            const keysB = await utils.loadUint32ArrayFromBin(bKeysPath);
            const valuesB = await utils.loadUint32ArrayFromBin(bValuesPath);

            const { resultKeys, resultValues, totalCount, timing } =
                await tester.computeSetUnionByKeyWithGPUProfiling(keysA, valuesA, keysB, valuesB, NUM_ITERATIONS, NUM_WARMUP);

            const ds = `${size}e${range}`.padEnd(8);
            const inputSize = `${(keysA.length / 1_000_000).toFixed(0)}M+${(keysB.length / 1_000_000).toFixed(0)}M`.padStart(12);
            const unionSize = resultKeys.length.toLocaleString().padStart(14);
            const dpi = timing.dpiMs.toFixed(2).padStart(6);
            const count = timing.countMs.toFixed(2).padStart(6);
            const scan = timing.scanMs.toFixed(2).padStart(6);
            const write = timing.writeMs.toFixed(2).padStart(6);
            const total = timing.totalMs.toFixed(2).padStart(6);

            console.log(`║ ${ds} │ ${inputSize} │ ${unionSize} │ ${dpi} │ ${count} │ ${scan} │ ${write} │ ${total} ║`);

        } catch (error) {
            console.log(`║ ${size}e${range}    │ Error: ${error} ║`);
        }
    }

    console.log('╚══════════╧══════════════╧════════════════╧════════╧════════╧════════╧════════╧════════╝\n');
}

/**
 * Run validation benchmark - tests correctness on smaller datasets
 */
export async function runUnionByKeyValidationBenchmark(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY VALIDATION BENCHMARK                 ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    const datasets = [
        { size: '1', range: '2', desc: '1M elements, range 100' },
        { size: '1', range: '6', desc: '1M elements, range 1M' },
        { size: '2', range: '2', desc: '2M elements, range 100' },
        { size: '2', range: '6', desc: '2M elements, range 1M' },
        { size: '4', range: '2', desc: '4M elements, range 100' },
        { size: '4', range: '6', desc: '4M elements, range 1M' },
    ];

    let allPassed = true;

    for (const { size, range, desc } of datasets) {
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.log(`Dataset: ${size}e${range} - ${desc}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

        try {
            const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
            const aValuesPath = `./data/A_values_${size}e${range}.bin`;
            const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
            const bValuesPath = `./data/B_values_${size}e${range}.bin`;

            console.log('Loading data...');
            const keysA = await utils.loadUint32ArrayFromBin(aKeysPath);
            const valuesA = await utils.loadUint32ArrayFromBin(aValuesPath);
            const keysB = await utils.loadUint32ArrayFromBin(bKeysPath);
            const valuesB = await utils.loadUint32ArrayFromBin(bValuesPath);

            console.log(`A: ${keysA.length.toLocaleString()} key-value pairs`);
            console.log(`B: ${keysB.length.toLocaleString()} key-value pairs`);

            const { resultKeys, resultValues, totalCount, timing } =
                await tester.computeSetUnionByKeyWithGPUProfiling(keysA, valuesA, keysB, valuesB, 1, 0);

            console.log(`GPU result: ${resultKeys.length.toLocaleString()} key-value pairs`);
            console.log(`GPU time: ${timing.totalMs.toFixed(2)} ms`);

            console.log('Running CPU validation...');
            const cpuStart = performance.now();
            const valid = tester.validateResults(resultKeys, resultValues, keysA, valuesA, keysB, valuesB);
            const cpuTime = performance.now() - cpuStart;

            console.log(`CPU validation time: ${cpuTime.toFixed(2)} ms`);
            console.log(`Validation: ${valid ? '✔ PASS' : '✗ FAIL'}\n`);

            if (!valid) allPassed = false;

        } catch (error) {
            console.log(`Error: ${error}\n`);
            allPassed = false;
        }
    }

    console.log('╔════════════════════════════════════════════════════════════╗');
    if (allPassed) {
        console.log('║  ✔ ALL VALIDATION TESTS PASSED                             ║');
    } else {
        console.log('║  ✗ SOME VALIDATION TESTS FAILED                            ║');
    }
    console.log('╚════════════════════════════════════════════════════════════╝\n');
}

/**
 * Test 64e6 dataset (64M elements, range 1M - few duplicates)
 */
export async function runUnionByKey64e6Test(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY - 64e6 TEST                          ║');
    console.log('║      64M elements, range 1M (few duplicates)               ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    const size = '64';
    const range = '6';

    const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
    const aValuesPath = `./data/A_values_${size}e${range}.bin`;
    const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
    const bValuesPath = `./data/B_values_${size}e${range}.bin`;

    try {
        console.log('Loading data files...');
        const keysA = await utils.loadUint32ArrayFromBin(aKeysPath);
        const valuesA = await utils.loadUint32ArrayFromBin(aValuesPath);
        const keysB = await utils.loadUint32ArrayFromBin(bKeysPath);
        const valuesB = await utils.loadUint32ArrayFromBin(bValuesPath);

        console.log(`  A: ${keysA.length.toLocaleString()} elements`);
        console.log(`  B: ${keysB.length.toLocaleString()} elements`);

        const NUM_ITERATIONS = 20;
        const NUM_WARMUP = 10;
        console.log(`\nRunning ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup)...\n`);

        const { resultKeys, resultValues, totalCount, timing } =
            await tester.computeSetUnionByKeyWithGPUProfiling(keysA, valuesA, keysB, valuesB, NUM_ITERATIONS, NUM_WARMUP);

        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║  RESULTS                                                   ║');
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Input A:     ${keysA.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Input B:     ${keysB.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Union size:  ${resultKeys.length.toLocaleString().padStart(15)} elements              ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  DPI:         ${timing.dpiMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Count:       ${timing.countMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Scan:        ${timing.scanMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Write:       ${timing.writeMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Total:       ${timing.totalMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log('╚════════════════════════════════════════════════════════════╝\n');

        // Calculate throughput
        const totalElements = keysA.length + keysB.length;
        const throughput = totalElements / (timing.totalMs / 1000) / 1_000_000;
        console.log(`Throughput: ${throughput.toFixed(2)} M elements/sec\n`);

    } catch (error) {
        console.error(`Error: ${error}`);
    }
}

/**
 * Test 128e2 dataset (128M elements, range 100)
 */
export async function runUnionByKey128e2Test(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY - 128e2 TEST                         ║');
    console.log('║      128M elements, range 100 (many duplicates)            ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    const size = '128';
    const range = '2';

    const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
    const aValuesPath = `./data/A_values_${size}e${range}.bin`;
    const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
    const bValuesPath = `./data/B_values_${size}e${range}.bin`;

    try {
        console.log('Loading data files...');
        const keysA = await utils.loadUint32ArrayFromBin(aKeysPath);
        const valuesA = await utils.loadUint32ArrayFromBin(aValuesPath);
        const keysB = await utils.loadUint32ArrayFromBin(bKeysPath);
        const valuesB = await utils.loadUint32ArrayFromBin(bValuesPath);

        console.log(`  A: ${keysA.length.toLocaleString()} elements`);
        console.log(`  B: ${keysB.length.toLocaleString()} elements`);

        const NUM_ITERATIONS = 20;
        const NUM_WARMUP = 10;
        console.log(`\nRunning ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup)...\n`);

        const { resultKeys, resultValues, totalCount, timing } =
            await tester.computeSetUnionByKeyWithGPUProfiling(keysA, valuesA, keysB, valuesB, NUM_ITERATIONS, NUM_WARMUP);

        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║  RESULTS                                                   ║');
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Input A:     ${keysA.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Input B:     ${keysB.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Union size:  ${resultKeys.length.toLocaleString().padStart(15)} elements              ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  DPI:         ${timing.dpiMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Count:       ${timing.countMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Scan:        ${timing.scanMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Write:       ${timing.writeMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Total:       ${timing.totalMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log('╚════════════════════════════════════════════════════════════╝\n');

    } catch (error) {
        console.error(`Error: ${error}`);
    }
}

/**
 * Test 128e6 dataset (128M elements, range 1M)
 */
export async function runUnionByKey128e6Test(device: GPUDevice): Promise<void> {
    console.log('\n╔════════════════════════════════════════════════════════════╗');
    console.log('║      SET UNION BY KEY - 128e6 TEST                         ║');
    console.log('║      128M elements, range 1M (few duplicates)              ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');

    const timestampQueryManager = new TimestampQueryManager(device, 16);
    const tester = new TestSetUnionByKeyPipeline(device, timestampQueryManager);

    const size = '128';
    const range = '6';

    const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
    const aValuesPath = `./data/A_values_${size}e${range}.bin`;
    const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
    const bValuesPath = `./data/B_values_${size}e${range}.bin`;

    try {
        console.log('Loading data files...');
        const keysA = await utils.loadUint32ArrayFromBin(aKeysPath);
        const valuesA = await utils.loadUint32ArrayFromBin(aValuesPath);
        const keysB = await utils.loadUint32ArrayFromBin(bKeysPath);
        const valuesB = await utils.loadUint32ArrayFromBin(bValuesPath);

        console.log(`  A: ${keysA.length.toLocaleString()} elements`);
        console.log(`  B: ${keysB.length.toLocaleString()} elements`);

        const NUM_ITERATIONS = 20;
        const NUM_WARMUP = 10;
        console.log(`\nRunning ${NUM_ITERATIONS} iterations (${NUM_WARMUP} warmup)...\n`);

        const { resultKeys, resultValues, totalCount, timing } =
            await tester.computeSetUnionByKeyWithGPUProfiling(keysA, valuesA, keysB, valuesB, NUM_ITERATIONS, NUM_WARMUP);

        console.log('╔════════════════════════════════════════════════════════════╗');
        console.log('║  RESULTS                                                   ║');
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  Input A:     ${keysA.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Input B:     ${keysB.length.toLocaleString().padStart(15)} elements              ║`);
        console.log(`║  Union size:  ${resultKeys.length.toLocaleString().padStart(15)} elements              ║`);
        console.log('╠════════════════════════════════════════════════════════════╣');
        console.log(`║  DPI:         ${timing.dpiMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Count:       ${timing.countMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Scan:        ${timing.scanMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Write:       ${timing.writeMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log(`║  Total:       ${timing.totalMs.toFixed(3).padStart(15)} ms                   ║`);
        console.log('╚════════════════════════════════════════════════════════════╝\n');

    } catch (error) {
        console.error(`Error: ${error}`);
    }
}
