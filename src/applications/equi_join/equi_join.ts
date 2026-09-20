/**
 * GPU-accelerated Equi-Join (Inner Join) via Set Intersection By Key
 *
 * Implements: SELECT * FROM A JOIN B ON A.key = B.key
 *
 * Uses the existing 2-Phase by_key pipeline:
 *   1. DPI — compute merge path boundaries on keys
 *   2. Decoupled Lookback By Key (OP_MODE=0) — compare keys, output matched key-value pairs
 *
 * No new shader files needed — reuses balanced_path_biased.wgsl and
 * set_availability_decoupled_lookback_by_key.wgsl.
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import { setOpMode, loadUint32ArrayFromBin, setIntersectionCPUByKey } from '../../utils';

async function loadRawUint32Array(url: string): Promise<Uint32Array> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}`);
    return new Uint32Array(await resp.arrayBuffer());
}
import computeDiagonalsShader from '../../balanced_path/common/balanced_path_biased.wgsl';
import lookbackByKeyShaderBase from '../../balanced_path/common/by_key/set_availability_decoupled_lookback_by_key.wgsl';

const MAXWORKGROUP = 65535;
const NT = 256;
const VT = 12;
const NV = NT * VT;  // 3072
const DPI_WG_SIZE = 256;

export class EquiJoin {
    private device: GPUDevice;
    private tsm: TimestampQueryManager;

    private dpiPipeline: GPUComputePipeline;
    private dpiBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, tsm: TimestampQueryManager) {
        this.device = device;
        this.tsm = tsm;

        // DPI pipeline (operates on keys only)
        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'EquiJoin DPI bind group layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.dpiPipeline = device.createComputePipeline({
            label: 'EquiJoin DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: computeDiagonalsShader }),
                entryPoint: 'compute_diagonals'
            }
        });

        // Lookback By Key pipeline (OP_MODE=0, intersection)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'EquiJoin Lookback By Key bind group layout',
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

        this.lookbackPipeline = device.createComputePipeline({
            label: 'EquiJoin Lookback By Key pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: setOpMode(lookbackByKeyShaderBase, 0) }),
                entryPoint: 'decoupled_lookback_by_key_kernel'
            }
        });
    }

    async run(
        aKeys: Uint32Array,
        aValues: Uint32Array,
        bKeys: Uint32Array,
        bValues: Uint32Array,
        iterations: number = 1,
        warmup: number = 0,
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
            return { resultKeys: new Uint32Array(0), resultValues: new Uint32Array(0), totalCount: 0, timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 } };
        }

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = Math.min(a_len, b_len);  // intersection max

        // Create buffers
        const bufAKeys = device.createBuffer({ size: Math.max(4, aKeys.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufAKeys, 0, new Uint32Array(aKeys));

        const bufAValues = device.createBuffer({ size: Math.max(4, aValues.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufAValues, 0, new Uint32Array(aValues));

        const bufBKeys = device.createBuffer({ size: Math.max(4, bKeys.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufBKeys, 0, new Uint32Array(bKeys));

        const bufBValues = device.createBuffer({ size: Math.max(4, bValues.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufBValues, 0, new Uint32Array(bValues));

        const bufALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        const bufNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufALen, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(bufBLen, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(bufNumWg, 0, new Uint32Array([numWg]));

        const bufDPI = device.createBuffer({ size: 2 * (numWg + 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufState = device.createBuffer({ size: Math.max(4, numWg * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufOutputKeys = device.createBuffer({ size: Math.max(maxOutputSize, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufOutputValues = device.createBuffer({ size: Math.max(maxOutputSize, 1) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufTotalCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        // DPI bind group (keys only)
        const dpiBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufAKeys } },
                { binding: 1, resource: { buffer: bufBKeys } },
                { binding: 2, resource: { buffer: bufDPI } },
                { binding: 3, resource: { buffer: bufALen } },
                { binding: 4, resource: { buffer: bufBLen } },
                { binding: 5, resource: { buffer: bufNumWg } },
            ]
        });

        // Lookback bind group (keys + values)
        const lookbackBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufAKeys } },
                { binding: 1, resource: { buffer: bufAValues } },
                { binding: 2, resource: { buffer: bufBKeys } },
                { binding: 3, resource: { buffer: bufBValues } },
                { binding: 4, resource: { buffer: bufDPI } },
                { binding: 5, resource: { buffer: bufState } },
                { binding: 6, resource: { buffer: bufOutputKeys } },
                { binding: 7, resource: { buffer: bufOutputValues } },
                { binding: 8, resource: { buffer: bufTotalCount } },
                { binding: 9, resource: { buffer: bufALen } },
                { binding: 10, resource: { buffer: bufBLen } },
                { binding: 11, resource: { buffer: bufNumWg } },
            ]
        });

        // Dispatch sizes
        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);
        const lookbackDispatchX = Math.min(numWg, MAXWORKGROUP);
        const lookbackDispatchY = Math.ceil(numWg / MAXWORKGROUP);

        const stateZeros = new Uint32Array(numWg).fill(0);
        await device.queue.onSubmittedWorkDone();

        // Warmup
        for (let w = 0; w < warmup; w++) {
            device.queue.writeBuffer(bufState, 0, stateZeros);
            device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
            const encoder = device.createCommandEncoder();
            let pass = encoder.beginComputePass();
            pass.setPipeline(this.dpiPipeline);
            pass.setBindGroup(0, dpiBindGroup);
            pass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
            pass.end();
            pass = encoder.beginComputePass();
            pass.setPipeline(this.lookbackPipeline);
            pass.setBindGroup(0, lookbackBindGroup);
            pass.dispatchWorkgroups(lookbackDispatchX, lookbackDispatchY);
            pass.end();
            device.queue.submit([encoder.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed runs
        const dpiTimes: number[] = [];
        const lookbackTimes: number[] = [];
        const totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufState, 0, stateZeros);
            device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
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

            this.tsm.resolve(encoder);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            const timestamps = await this.tsm.downloadTimestampResult();
            if (timestamps.length >= 4) {
                dpiTimes.push((timestamps[1] - timestamps[0]) / 1_000_000);
                lookbackTimes.push((timestamps[3] - timestamps[2]) / 1_000_000);
                totalTimes.push((timestamps[3] - timestamps[0]) / 1_000_000);
            }
        }

        const avg = (arr: number[]) => arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;

        // Final run for readback
        device.queue.writeBuffer(bufState, 0, stateZeros);
        device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
        const finalEncoder = device.createCommandEncoder();
        let finalPass = finalEncoder.beginComputePass();
        finalPass.setPipeline(this.dpiPipeline);
        finalPass.setBindGroup(0, dpiBindGroup);
        finalPass.dispatchWorkgroups(dpiDispatchX, dpiDispatchY);
        finalPass.end();
        finalPass = finalEncoder.beginComputePass();
        finalPass.setPipeline(this.lookbackPipeline);
        finalPass.setBindGroup(0, lookbackBindGroup);
        finalPass.dispatchWorkgroups(lookbackDispatchX, lookbackDispatchY);
        finalPass.end();

        const readbackCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        finalEncoder.copyBufferToBuffer(bufTotalCount, 0, readbackCount, 0, 4);
        device.queue.submit([finalEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackCount.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(readbackCount.getMappedRange().slice(0))[0];
        readbackCount.unmap();

        let resultKeys = new Uint32Array(0);
        let resultValues = new Uint32Array(0);
        if (totalCount > 0) {
            const readbackKeys = device.createBuffer({ size: totalCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const readbackValues = device.createBuffer({ size: totalCount * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const readEncoder = device.createCommandEncoder();
            readEncoder.copyBufferToBuffer(bufOutputKeys, 0, readbackKeys, 0, totalCount * 4);
            readEncoder.copyBufferToBuffer(bufOutputValues, 0, readbackValues, 0, totalCount * 4);
            device.queue.submit([readEncoder.finish()]);
            await device.queue.onSubmittedWorkDone();

            await readbackKeys.mapAsync(GPUMapMode.READ);
            resultKeys = new Uint32Array(readbackKeys.getMappedRange().slice(0));
            readbackKeys.unmap();

            await readbackValues.mapAsync(GPUMapMode.READ);
            resultValues = new Uint32Array(readbackValues.getMappedRange().slice(0));
            readbackValues.unmap();

            readbackKeys.destroy();
            readbackValues.destroy();
        }

        // Cleanup
        bufAKeys.destroy(); bufAValues.destroy();
        bufBKeys.destroy(); bufBValues.destroy();
        bufALen.destroy(); bufBLen.destroy(); bufNumWg.destroy();
        bufDPI.destroy(); bufState.destroy();
        bufOutputKeys.destroy(); bufOutputValues.destroy();
        bufTotalCount.destroy(); readbackCount.destroy();

        return {
            resultKeys, resultValues, totalCount,
            timing: { dpiMs: avg(dpiTimes), lookbackMs: avg(lookbackTimes), totalMs: avg(totalTimes) }
        };
    }
}

// --- Test runner ---

export async function runEquiJoinTest(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  EQUI-JOIN (Inner Join) via GPU Set Intersection By Key');
    console.log('  DPI (keys only) + Decoupled Lookback By Key (OP_MODE=0)');
    console.log('========================================================================\n');

    const NUM_ITERATIONS = 10;
    const NUM_WARMUP = 3;

    const tsm = new TimestampQueryManager(device, 8);

    if (!tsm.timestampSupported) {
        console.log('  ERROR: GPU timestamp queries not supported.\n');
        return;
    }

    const joiner = new EquiJoin(device, tsm);

    console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations\n`);

    // --- TPC-H Real Datasets ---
    console.log('  === TPC-H Join Benchmarks ===\n');
    console.log('  Dataset                | |A|          | |B|          | Join size    | DPI(ms)   | Lookback(ms) | Total(ms) | Match');
    console.log('  -----------------------|--------------|--------------|--------------|-----------|--------------|-----------|------');

    const tpchDatasets = [
        { name: 'lineitem×orders SF1', aKeys: 'tpch/lineitem_orders_sf1_A_keys.bin', aValues: 'tpch/lineitem_orders_sf1_A_values.bin', bKeys: 'tpch/lineitem_orders_sf1_B_keys.bin', bValues: 'tpch/lineitem_orders_sf1_B_values.bin' },
        { name: 'lineitem×part SF1', aKeys: 'tpch/lineitem_part_sf1_A_keys.bin', aValues: 'tpch/lineitem_part_sf1_A_values.bin', bKeys: 'tpch/lineitem_part_sf1_B_keys.bin', bValues: 'tpch/lineitem_part_sf1_B_values.bin' },
        { name: 'lineitem×orders SF5', aKeys: 'tpch/lineitem_orders_sf5_A_keys.bin', aValues: 'tpch/lineitem_orders_sf5_A_values.bin', bKeys: 'tpch/lineitem_orders_sf5_B_keys.bin', bValues: 'tpch/lineitem_orders_sf5_B_values.bin' },
        { name: 'lineitem×part SF5', aKeys: 'tpch/lineitem_part_sf5_A_keys.bin', aValues: 'tpch/lineitem_part_sf5_A_values.bin', bKeys: 'tpch/lineitem_part_sf5_B_keys.bin', bValues: 'tpch/lineitem_part_sf5_B_values.bin' },
        { name: 'lineitem×orders SF10', aKeys: 'tpch/lineitem_orders_sf10_A_keys.bin', aValues: 'tpch/lineitem_orders_sf10_A_values.bin', bKeys: 'tpch/lineitem_orders_sf10_B_keys.bin', bValues: 'tpch/lineitem_orders_sf10_B_values.bin' },
        { name: 'lineitem×part SF10', aKeys: 'tpch/lineitem_part_sf10_A_keys.bin', aValues: 'tpch/lineitem_part_sf10_A_values.bin', bKeys: 'tpch/lineitem_part_sf10_B_keys.bin', bValues: 'tpch/lineitem_part_sf10_B_values.bin' },
    ];

    for (const ds of tpchDatasets) {
        try {
            const aKeys = await loadRawUint32Array(`./${ds.aKeys}`);
            const aValues = await loadRawUint32Array(`./${ds.aValues}`);
            const bKeys = await loadRawUint32Array(`./${ds.bKeys}`);
            const bValues = await loadRawUint32Array(`./${ds.bValues}`);

            const result = await joiner.run(aKeys, aValues, bKeys, bValues, NUM_ITERATIONS, NUM_WARMUP);

            // CPU validation (skip for very large)
            let matchStr = 'skip';
            if (aKeys.length <= 64_000_000) {
                const aInterleaved = new Uint32Array(aKeys.length * 2);
                for (let i = 0; i < aKeys.length; i++) {
                    aInterleaved[i * 2] = aKeys[i];
                    aInterleaved[i * 2 + 1] = aValues[i];
                }
                const cpuResult = setIntersectionCPUByKey(aInterleaved, bKeys);
                const cpuCount = cpuResult.length / 2;
                matchStr = result.totalCount === cpuCount ? 'OK' : 'FAIL';
            }

            const nameCol = ds.name.padEnd(22);
            const aCol = aKeys.length.toLocaleString().padStart(12);
            const bCol = bKeys.length.toLocaleString().padStart(12);
            const joinCol = result.totalCount.toLocaleString().padStart(12);
            const dpiCol = result.timing.dpiMs.toFixed(3).padStart(9);
            const lbCol = result.timing.lookbackMs.toFixed(3).padStart(12);
            const totalCol = result.timing.totalMs.toFixed(3).padStart(9);

            console.log(`  ${nameCol} |${aCol} |${bCol} |${joinCol} |${dpiCol} |${lbCol} |${totalCol} | ${matchStr}`);

        } catch {
            console.log(`  ${ds.name.padEnd(22)} | not found`);
        }
    }

    console.log('');
}
