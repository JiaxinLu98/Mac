/**
 * Test file for 4-Phase vs 2-Phase Difference By Key Comparison
 *
 * Compares the 4-phase pipeline with the 2-phase sentinel pipeline
 * for difference by key operations (compare keys, output key-value pairs from A).
 *
 * 4-Phase (baseline):
 *   VT=7, NV=1792
 *   1. DPI - Compute diagonal path indices
 *   2. Count - Count difference matches per workgroup (keys only)
 *   3. Prefix Sum - Exclusive scan for output offsets
 *   4. Write - Write difference keys + values
 *
 * 2-Phase Sentinel (optimized):
 *   VT=12, NV=3072
 *   1. DPI - Compute diagonal path indices
 *   2. Decoupled Lookback - Single-pass count + write with sentinel optimization
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import * as utils from '../../utils';
import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countByKeyShader from './set_availability_difference_by_key_count_v1.wgsl';
import writeByKeyShader from './set_availability_difference_by_key_write_v1.wgsl';
import sentinelByKeyLookbackShader from './set_availability_difference_by_key_decoupled_lookback_sentinel.wgsl';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT_4P = 7;
const NV_4P = NT * VT_4P;   // 1792
const VT_SENT = 12;
const NV_SENT = NT * VT_SENT; // 3072

/**
 * 4-Phase By-Key Pipeline Tester (DPI -> Count -> Scan -> Write)
 */
class FourPhaseByKeyTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private countPipeline: GPUComputePipeline;
    private countBindGroupLayout: GPUBindGroupLayout;
    private writePipeline: GPUComputePipeline;
    private writeBindGroupLayout: GPUBindGroupLayout;
    private scanPipeline: ExclusiveScanPipeline;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.scanPipeline = new ExclusiveScanPipeline(device);

        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: '4P-DiffByKey DPI bind group layout',
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
            label: '4P-DiffByKey DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Count: 7 bindings (a_keys, b_keys, dpi, counts, a_length, b_length, num_wg_total)
        this.countBindGroupLayout = device.createBindGroupLayout({
            label: '4P-DiffByKey Count bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.countPipeline = device.createComputePipeline({
            label: '4P-DiffByKey Count pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.countBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: countByKeyShader }),
                entryPoint: 'count_availability'
            }
        });

        // Write: 10 bindings (a_keys, a_values, b_keys, dpi, offsets, output_keys, output_values, a_length, b_length, num_wg_total)
        this.writeBindGroupLayout = device.createBindGroupLayout({
            label: '4P-DiffByKey Write bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.writePipeline = device.createComputePipeline({
            label: '4P-DiffByKey Write pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.writeBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: writeByKeyShader }),
                entryPoint: 'write_availability'
            }
        });
    }

    public async run(
        aKeys: Uint32Array,
        aValues: Uint32Array,
        bKeys: Uint32Array,
        iterations: number,
        warmup: number
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
        const tsm = this.timestampQueryManager;
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                resultKeys: new Uint32Array(0),
                resultValues: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, countMs: 0, scanMs: 0, writeMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV_4P);

        const bufferAKeys = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(aKeys));

        const bufferAValues = device.createBuffer({
            size: Math.max(4, aValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(aValues));

        const bufferBKeys = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(bKeys));

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

        // Difference: maxOutputSize = a_len (all A elements could survive)
        const maxOutputSize = a_len;
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
                { binding: 3, resource: { buffer: bufferDPI } },
                { binding: 4, resource: { buffer: bufferCounts } },
                { binding: 5, resource: { buffer: bufferOutputKeys } },
                { binding: 6, resource: { buffer: bufferOutputValues } },
                { binding: 7, resource: { buffer: bufferALen } },
                { binding: 8, resource: { buffer: bufferBLen } },
                { binding: 9, resource: { buffer: bufferNumWg } },
            ]
        });

        const scanner = this.scanPipeline.prepareGPUInput(bufferCounts, alignedSize);

        await device.queue.onSubmittedWorkDone();

        // Warmup
        for (let w = 0; w < warmup; w++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            pass = encoder.beginComputePass();
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        }

        // Timed iterations
        const dpiTimes: number[] = [];
        const countTimes: number[] = [];
        const scanTimes: number[] = [];
        const writeTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass(tsm.createComputePassDescriptor(0, 1));
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(2, 3));
            pass.setPipeline(this.countPipeline);
            pass.setBindGroup(0, countBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            encoder.copyBufferToBuffer(bufferCounts, 0, bufferCountsCopy, 0, numWg * 4);
            scanner.recordScanCommands(encoder, numWg, tsm, 4, 5);

            pass = encoder.beginComputePass(tsm.createComputePassDescriptor(6, 7));
            pass.setPipeline(this.writePipeline);
            pass.setBindGroup(0, writeBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            tsm.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await tsm.downloadTimestampResult();

            if (timestamps.length >= 8) {
                const dpiNs = timestamps[1] - timestamps[0];
                const countNs = timestamps[3] - timestamps[2];
                const writeNs = timestamps[7] - timestamps[6];
                const totalNs = timestamps[7] - timestamps[0];
                const scanNs = totalNs - dpiNs - countNs - writeNs;

                dpiTimes.push(dpiNs / 1_000_000);
                countTimes.push(countNs / 1_000_000);
                scanTimes.push(scanNs / 1_000_000);
                writeTimes.push(writeNs / 1_000_000);
                totalTimes.push(totalNs / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        // Readback
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
            const outputKeysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const outputValuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, outputKeysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, outputValuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await outputKeysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(outputKeysReadback.getMappedRange().slice(0));
            outputKeysReadback.unmap();

            await outputValuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(outputValuesReadback.getMappedRange().slice(0));
            outputValuesReadback.unmap();

            outputKeysReadback.destroy();
            outputValuesReadback.destroy();
        }

        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
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

        return {
            resultKeys,
            resultValues,
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                countMs: avg(countTimes),
                scanMs: avg(scanTimes),
                writeMs: avg(writeTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

/**
 * 2-Phase Sentinel By-Key Pipeline Tester (DPI -> Decoupled Lookback Sentinel)
 */
class SentinelByKeyTester {
    private device: GPUDevice;
    private timestampQueryManager: TimestampQueryManager;

    private diagPipeline: GPUComputePipeline;
    private diagBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        const sentinelDpiShader = computeDiagonalsShader.replace('const VT: u32 = 7u;', `const VT: u32 = ${VT_SENT}u;`);

        this.diagBindGroupLayout = device.createBindGroupLayout({
            label: 'Sentinel-DiffByKey DPI bind group layout',
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
            label: 'Sentinel-DiffByKey DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.diagBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: sentinelDpiShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback: 11 bindings
        // a_keys(0), a_values(1), b_keys(2), dpi(3), state(4),
        // output_keys(5), output_values(6), total_count(7),
        // a_length(8), b_length(9), num_wg_total(10)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'Sentinel-DiffByKey Lookback bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.lookbackPipeline = device.createComputePipeline({
            label: 'Sentinel-DiffByKey Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: sentinelByKeyLookbackShader }),
                entryPoint: 'difference_by_key_decoupled_lookback'
            }
        });
    }

    public async run(
        aKeys: Uint32Array,
        aValues: Uint32Array,
        bKeys: Uint32Array,
        iterations: number,
        warmup: number
    ): Promise<{
        resultKeys: Uint32Array;
        resultValues: Uint32Array;
        totalCount: number;
        timing: { dpiMs: number; lookbackMs: number; totalMs: number };
    }> {
        const device = this.device;
        const a_len = aKeys.length;
        const b_len = bKeys.length;
        const total = a_len + b_len;

        if (total === 0) {
            return {
                resultKeys: new Uint32Array(0),
                resultValues: new Uint32Array(0),
                totalCount: 0,
                timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 }
            };
        }

        const numWg = Math.ceil(total / NV_SENT);
        // Difference: maxOutputSize = a_len (all A elements could survive)
        const maxOutputSize = a_len;

        const bufferAKeys = device.createBuffer({
            size: Math.max(4, aKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAKeys, 0, new Uint32Array(aKeys));

        const bufferAValues = device.createBuffer({
            size: Math.max(4, aValues.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferAValues, 0, new Uint32Array(aValues));

        const bufferBKeys = device.createBuffer({
            size: Math.max(4, bKeys.byteLength),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferBKeys, 0, new Uint32Array(bKeys));

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

        const bufferOutputKeys = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferOutputValues = device.createBuffer({
            size: Math.max(maxOutputSize, 1) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        const bufferTotalCount = device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

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

        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferAKeys } },
                { binding: 1, resource: { buffer: bufferAValues } },
                { binding: 2, resource: { buffer: bufferBKeys } },
                { binding: 3, resource: { buffer: bufferDPI } },
                { binding: 4, resource: { buffer: bufferState } },
                { binding: 5, resource: { buffer: bufferOutputKeys } },
                { binding: 6, resource: { buffer: bufferOutputValues } },
                { binding: 7, resource: { buffer: bufferTotalCount } },
                { binding: 8, resource: { buffer: bufferALen } },
                { binding: 9, resource: { buffer: bufferBLen } },
                { binding: 10, resource: { buffer: bufferNumWg } },
            ]
        });

        const dispatchX = Math.min(numWg, MAXWORKGROUP);
        const dispatchY = Math.ceil(numWg / MAXWORKGROUP);

        await device.queue.onSubmittedWorkDone();

        // Warmup
        for (let iter = 0; iter < warmup; iter++) {
            device.queue.writeBuffer(bufferState, 0, new Uint32Array(numWg).fill(0));
            device.queue.writeBuffer(bufferTotalCount, 0, new Uint32Array([0]));

            const encoder = device.createCommandEncoder();

            let pass = encoder.beginComputePass();
            pass.setPipeline(this.diagPipeline);
            pass.setBindGroup(0, diagBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            pass = encoder.beginComputePass();
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(dispatchX, dispatchY);
            pass.end();

            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed iterations
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
            pass.dispatchWorkgroups(dispatchX, dispatchY);
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

        const avg = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / arr.length;

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

        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const outputKeysReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            const outputValuesReadback = device.createBuffer({
                size: totalCount * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });
            readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufferOutputKeys, 0, outputKeysReadback, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufferOutputValues, 0, outputValuesReadback, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await outputKeysReadback.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(outputKeysReadback.getMappedRange().slice(0));
            outputKeysReadback.unmap();

            await outputValuesReadback.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(outputValuesReadback.getMappedRange().slice(0));
            outputValuesReadback.unmap();

            outputKeysReadback.destroy();
            outputValuesReadback.destroy();
        }

        bufferAKeys.destroy();
        bufferAValues.destroy();
        bufferBKeys.destroy();
        bufferALen.destroy();
        bufferBLen.destroy();
        bufferNumWg.destroy();
        bufferDPI.destroy();
        bufferState.destroy();
        bufferOutputKeys.destroy();
        bufferOutputValues.destroy();
        bufferTotalCount.destroy();
        totalCountReadback.destroy();

        return {
            resultKeys,
            resultValues,
            totalCount,
            timing: {
                dpiMs: avg(dpiTimes),
                lookbackMs: avg(lookbackTimes),
                totalMs: avg(totalTimes),
            }
        };
    }
}

function cpuSetDifferenceByKey(
    aKeys: Uint32Array,
    aValues: Uint32Array,
    bKeys: Uint32Array
): { keys: Uint32Array; values: Uint32Array } {
    const resultKeys: number[] = [];
    const resultValues: number[] = [];
    let ai = 0, bi = 0;
    while (ai < aKeys.length && bi < bKeys.length) {
        if (aKeys[ai] < bKeys[bi]) {
            resultKeys.push(aKeys[ai]);
            resultValues.push(aValues[ai]);
            ai++;
        } else if (aKeys[ai] > bKeys[bi]) {
            bi++;
        } else {
            // equal: cancel one from each
            ai++;
            bi++;
        }
    }
    // Remaining A elements (B exhausted)
    while (ai < aKeys.length) {
        resultKeys.push(aKeys[ai]);
        resultValues.push(aValues[ai]);
        ai++;
    }
    return {
        keys: new Uint32Array(resultKeys),
        values: new Uint32Array(resultValues)
    };
}

function generateValues(keys: Uint32Array): Uint32Array {
    const values = new Uint32Array(keys.length);
    for (let i = 0; i < keys.length; i++) {
        values[i] = i;
    }
    return values;
}

/**
 * Run 4-Phase vs 2-Phase Difference By Key comparison on all datasets.
 */
export async function run4PhaseVs2PhaseDifferenceByKeyTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('  DIFFERENCE BY KEY: 4-PHASE vs 2-PHASE SENTINEL');
    console.log('  4-Phase: DPI->Count->Scan->Write (VT=7, NV=1792, baseline)');
    console.log('  Sentinel: DPI->DecoupledLookback (VT=12, NV=3072, final)');
    console.log('  GPU Timestamp Query | 10 warmup + 100 iterations');
    console.log('='.repeat(80) + '\n');

    const NUM_ITERATIONS = 100;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhaseTester = new FourPhaseByKeyTester(device, timestampQueryManager);
    const sentinelTester = new SentinelByKeyTester(device, timestampQueryManager);

    console.log(`Running ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations per pipeline...\n`);
    console.log(`  4-Phase: VT=${VT_4P}, NV=${NV_4P} (4 kernel dispatches)`);
    console.log(`  2-Phase Sentinel: VT=${VT_SENT}, NV=${NV_SENT} (2 kernel dispatches)\n`);

    // Correctness Tests
    console.log('-'.repeat(80));
    console.log('CORRECTNESS TESTS (Difference By Key)');
    console.log('-'.repeat(80) + '\n');

    const syntheticTests = [
        {
            name: 'Small arrays',
            aKeys: new Uint32Array([1, 3, 3, 5, 7, 9]),
            bKeys: new Uint32Array([2, 3, 3, 6, 7, 8])
        },
        {
            name: 'No overlap (all A survives)',
            aKeys: new Uint32Array([1, 3, 5, 7, 9]),
            bKeys: new Uint32Array([2, 4, 6, 8, 10])
        },
        {
            name: 'Complete overlap (empty result)',
            aKeys: new Uint32Array([1, 2, 3, 4, 5]),
            bKeys: new Uint32Array([1, 2, 3, 4, 5])
        },
        {
            name: 'A subset of B (empty result)',
            aKeys: new Uint32Array([2, 4, 6]),
            bKeys: new Uint32Array([1, 2, 3, 4, 5, 6, 7])
        },
        {
            name: 'B empty (all A survives)',
            aKeys: new Uint32Array([1, 2, 3, 4, 5]),
            bKeys: new Uint32Array([])
        },
        {
            name: 'All same value (2500)',
            aKeys: new Uint32Array(2500).fill(42),
            bKeys: new Uint32Array(2500).fill(42)
        },
        {
            name: 'Multi-WG (2000 each)',
            aKeys: (() => { const a = new Uint32Array(2000); for (let i = 0; i < 2000; i++) a[i] = i * 2; return a; })(),
            bKeys: (() => { const b = new Uint32Array(2000); for (let i = 0; i < 2000; i++) b[i] = i * 3; return b; })()
        },
        {
            name: 'Large (10K each)',
            aKeys: (() => { const a = new Uint32Array(10000); for (let i = 0; i < 10000; i++) a[i] = i; return a; })(),
            bKeys: (() => { const b = new Uint32Array(10000); for (let i = 0; i < 10000; i++) b[i] = i * 2; return b; })()
        },
    ];

    let allCorrect = true;

    for (const test of syntheticTests) {
        const aValues = generateValues(test.aKeys);
        const cpuResult = cpuSetDifferenceByKey(test.aKeys, aValues, test.bKeys);
        const cpuCount = cpuResult.keys.length;

        const fourResult = await fourPhaseTester.run(test.aKeys, aValues, test.bKeys, 1, 0);
        const sentResult = await sentinelTester.run(test.aKeys, aValues, test.bKeys, 1, 0);

        const fourCountMatch = fourResult.totalCount === cpuCount;
        const sentCountMatch = sentResult.totalCount === cpuCount;
        const bothCountMatch = fourResult.totalCount === sentResult.totalCount;

        let fourKeysMatch = fourCountMatch;
        let sentKeysMatch = sentCountMatch;
        let fourValuesMatch = fourCountMatch;
        let sentValuesMatch = sentCountMatch;

        if (fourCountMatch && cpuCount > 0) {
            for (let i = 0; i < cpuCount; i++) {
                if (fourResult.resultKeys[i] !== cpuResult.keys[i]) { fourKeysMatch = false; break; }
            }
            for (let i = 0; i < cpuCount; i++) {
                if (fourResult.resultValues[i] !== cpuResult.values[i]) { fourValuesMatch = false; break; }
            }
        }
        if (sentCountMatch && cpuCount > 0) {
            for (let i = 0; i < cpuCount; i++) {
                if (sentResult.resultKeys[i] !== cpuResult.keys[i]) { sentKeysMatch = false; break; }
            }
            for (let i = 0; i < cpuCount; i++) {
                if (sentResult.resultValues[i] !== cpuResult.values[i]) { sentValuesMatch = false; break; }
            }
        }

        const allMatch = fourCountMatch && sentCountMatch && bothCountMatch && fourKeysMatch && sentKeysMatch && fourValuesMatch && sentValuesMatch;
        const status = allMatch ? 'OK' : 'FAIL';
        console.log(`  ${status.padEnd(5)} ${test.name}: 4P=${fourResult.totalCount}, sent=${sentResult.totalCount}, cpu=${cpuCount}`);

        if (!allMatch) {
            allCorrect = false;
            if (!fourKeysMatch) console.log(`         4P keys mismatch`);
            if (!fourValuesMatch) console.log(`         4P values mismatch`);
            if (!sentKeysMatch) console.log(`         Sentinel keys mismatch`);
            if (!sentValuesMatch) console.log(`         Sentinel values mismatch`);
        }
    }

    console.log(allCorrect ? '\n  All correctness tests PASSED\n' : '\n  Some correctness tests FAILED\n');

    // Performance Benchmarks
    console.log('-'.repeat(80));
    console.log('PERFORMANCE BENCHMARKS (Difference By Key)');
    console.log('-'.repeat(80) + '\n');

    const datasets = [
        { size: '1', range: '2' },
        { size: '1', range: '6' },
        { size: '2', range: '2' },
        { size: '2', range: '6' },
        { size: '4', range: '2' },
        { size: '4', range: '6' },
        { size: '8', range: '2' },
        { size: '8', range: '6' },
        { size: '16', range: '2' },
        { size: '16', range: '6' },
        { size: '32', range: '2' },
        { size: '32', range: '6' },
        { size: '64', range: '2' },
        { size: '64', range: '6' },
        { size: '128', range: '2' },
        { size: '128', range: '6' },
    ];

    interface BenchmarkResult {
        dataset: string;
        fourTotal: number;
        fourDpi: number;
        fourCount: number;
        fourScan: number;
        fourWrite: number;
        sentTotal: number;
        sentDpi: number;
        sentLookback: number;
        speedup: number;
        match: boolean;
    }

    const results: BenchmarkResult[] = [];

    console.log(`${'Dataset'.padEnd(10)} ${'4P_total'.padStart(12)} ${'(DPI'.padStart(8)} ${'Count'.padStart(8)} ${'Scan'.padStart(8)} ${'Write)'.padStart(8)} ${'Sent_total'.padStart(12)} ${'(DPI'.padStart(8)} ${'LB)'.padStart(8)} ${'Speedup'.padStart(10)} ${'Match'.padStart(6)}`);
    console.log('-'.repeat(120));

    for (const { size, range } of datasets) {
        const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
        const aValuesPath = `./data/A_values_${size}e${range}.bin`;
        const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
        const bValuesPath = `./data/B_values_${size}e${range}.bin`;

        try {
            const aKeys = await utils.loadUint32ArrayFromBin(aKeysPath);
            const aValues = await utils.loadUint32ArrayFromBin(aValuesPath);
            const bKeys = await utils.loadUint32ArrayFromBin(bKeysPath);
            const bValues = await utils.loadUint32ArrayFromBin(bValuesPath);

            const fourResult = await fourPhaseTester.run(aKeys, aValues, bKeys, NUM_ITERATIONS, NUM_WARMUP);
            const sentResult = await sentinelTester.run(aKeys, aValues, bKeys, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = fourResult.totalCount === sentResult.totalCount;
            const ds = `${size}e${range}`;
            const speedup = fourResult.timing.totalMs / sentResult.timing.totalMs;

            results.push({
                dataset: ds,
                fourTotal: fourResult.timing.totalMs,
                fourDpi: fourResult.timing.dpiMs,
                fourCount: fourResult.timing.countMs,
                fourScan: fourResult.timing.scanMs,
                fourWrite: fourResult.timing.writeMs,
                sentTotal: sentResult.timing.totalMs,
                sentDpi: sentResult.timing.dpiMs,
                sentLookback: sentResult.timing.lookbackMs,
                speedup,
                match: countMatch,
            });

            const matchStr = countMatch ? 'OK' : 'FAIL';
            console.log(
                `${ds.padEnd(10)} ${(fourResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${fourResult.timing.dpiMs.toFixed(2).padStart(7)} ${fourResult.timing.countMs.toFixed(2).padStart(7)} ` +
                `${fourResult.timing.scanMs.toFixed(2).padStart(7)} ${fourResult.timing.writeMs.toFixed(2).padStart(7)} ` +
                `${(sentResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${sentResult.timing.dpiMs.toFixed(2).padStart(7)} ${sentResult.timing.lookbackMs.toFixed(2).padStart(7)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ${matchStr.padStart(6)}`
            );
        } catch (error) {
            console.log(`${`${size}e${range}`.padEnd(10)} Error loading dataset`);
        }
    }

    // Summary
    console.log('\n' + '='.repeat(80));
    console.log('  TOTAL TIME SUMMARY (Difference By Key: 4-Phase vs 2-Phase Sentinel)');
    console.log('='.repeat(80));
    console.log(`${'Dataset'.padEnd(10)} ${'4-Phase'.padStart(14)} ${'Sentinel'.padStart(14)} ${'Speedup'.padStart(10)}`);
    console.log('-'.repeat(50));
    for (const r of results) {
        console.log(
            `${r.dataset.padEnd(10)} ${(r.fourTotal.toFixed(3) + ' ms').padStart(14)} ${(r.sentTotal.toFixed(3) + ' ms').padStart(14)} ${(r.speedup.toFixed(2) + 'x').padStart(10)}`
        );
    }
    console.log('-'.repeat(50));

    const allPassed = results.every(r => r.match);
    console.log(allPassed ? '\nAll tests PASSED.' : '\nSome tests FAILED!');

    console.log('\nNotes:');
    console.log('  - Speedup > 1.0x means sentinel (2-phase) is faster');
    console.log('  - This measures the FULL end-to-end optimization gain:');
    console.log('    Architecture: 4 dispatches -> 2 dispatches');
    console.log('    Kernel: VT=7 -> VT=12, sentinel, branchless, unrolled');
    console.log('    By-Key: outputs both keys and values from array A');
    console.log('');
}

/**
 * Run 128M only comparison for difference by key.
 */
export async function run4PhaseVs2PhaseDifferenceByKey128MTest(device: GPUDevice): Promise<void> {
    console.log('\n' + '='.repeat(80));
    console.log('  DIFFERENCE BY KEY: 4-PHASE vs 2-PHASE SENTINEL (128M)');
    console.log('='.repeat(80) + '\n');

    const NUM_ITERATIONS = 100;
    const NUM_WARMUP = 10;

    const timestampQueryManager = new TimestampQueryManager(device, 16);

    if (!timestampQueryManager.timestampSupported) {
        console.log('ERROR: GPU timestamp queries are not supported on this device.\n');
        return;
    }

    const fourPhaseTester = new FourPhaseByKeyTester(device, timestampQueryManager);
    const sentinelTester = new SentinelByKeyTester(device, timestampQueryManager);

    console.log(`Running ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} timed iterations per pipeline...\n`);

    const datasets = [
        { size: '128', range: '2' },
        { size: '128', range: '6' },
    ];

    console.log(`${'Dataset'.padEnd(10)} ${'4P_total'.padStart(12)} ${'Sent_total'.padStart(12)} ${'Speedup'.padStart(10)} ${'4P_detail'.padStart(40)} ${'Sent_detail'.padStart(24)} ${'Match'.padStart(6)}`);
    console.log('-'.repeat(120));

    for (const { size, range } of datasets) {
        const aKeysPath = `./data/A_keys_${size}e${range}.bin`;
        const aValuesPath = `./data/A_values_${size}e${range}.bin`;
        const bKeysPath = `./data/B_keys_${size}e${range}.bin`;
        const bValuesPath = `./data/B_values_${size}e${range}.bin`;

        try {
            const aKeys = await utils.loadUint32ArrayFromBin(aKeysPath);
            const aValues = await utils.loadUint32ArrayFromBin(aValuesPath);
            const bKeys = await utils.loadUint32ArrayFromBin(bKeysPath);
            const bValues = await utils.loadUint32ArrayFromBin(bValuesPath);

            const fourResult = await fourPhaseTester.run(aKeys, aValues, bKeys, NUM_ITERATIONS, NUM_WARMUP);
            const sentResult = await sentinelTester.run(aKeys, aValues, bKeys, NUM_ITERATIONS, NUM_WARMUP);

            const countMatch = fourResult.totalCount === sentResult.totalCount;
            const speedup = fourResult.timing.totalMs / sentResult.timing.totalMs;

            const ds = `${size}e${range}`.padEnd(10);
            const matchStr = countMatch ? 'OK' : 'FAIL';

            const fourDetail = `DPI=${fourResult.timing.dpiMs.toFixed(2)} C=${fourResult.timing.countMs.toFixed(2)} S=${fourResult.timing.scanMs.toFixed(2)} W=${fourResult.timing.writeMs.toFixed(2)}`;
            const sentDetail = `DPI=${sentResult.timing.dpiMs.toFixed(2)} LB=${sentResult.timing.lookbackMs.toFixed(2)}`;

            console.log(
                `${ds} ${(fourResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ${(sentResult.timing.totalMs.toFixed(3) + ' ms').padStart(12)} ` +
                `${(speedup.toFixed(2) + 'x').padStart(10)} ${fourDetail.padStart(40)} ${sentDetail.padStart(24)} ${matchStr.padStart(6)}`
            );
        } catch (error) {
            const ds = `${size}e${range}`.padEnd(10);
            console.log(`${ds} Error loading dataset`);
        }
    }

    console.log('');
}
