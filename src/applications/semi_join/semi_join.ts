/**
 * GPU-accelerated Semi-Join via Set Intersection By Key
 *
 * Implements: SELECT * FROM A WHERE A.key IN (SELECT key FROM B)
 * Returns rows from A whose key exists in B (no B columns in output).
 *
 * Uses the same 2-Phase by_key pipeline as Equi-Join (OP_MODE=0, intersection).
 * Functionally equivalent to inner join when B has unique keys,
 * but semantically distinct: output contains only A's columns.
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import { setOpMode, setIntersectionCPUByKey } from '../../utils';

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
const NV = NT * VT;
const DPI_WG_SIZE = 256;

class SemiJoin {
    private device: GPUDevice;
    private tsm: TimestampQueryManager;
    private dpiPipeline: GPUComputePipeline;
    private dpiBindGroupLayout: GPUBindGroupLayout;
    private lookbackPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, tsm: TimestampQueryManager) {
        this.device = device;
        this.tsm = tsm;

        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'SemiJoin DPI BGL',
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
            label: 'SemiJoin DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: { module: device.createShaderModule({ code: computeDiagonalsShader }), entryPoint: 'compute_diagonals' }
        });

        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'SemiJoin Lookback BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 11, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        // OP_MODE=0: intersection (same as equi-join)
        this.lookbackPipeline = device.createComputePipeline({
            label: 'SemiJoin Lookback pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] }),
            compute: { module: device.createShaderModule({ code: setOpMode(lookbackByKeyShaderBase, 0) }), entryPoint: 'decoupled_lookback_by_key_kernel' }
        });
    }

    async run(aKeys: Uint32Array, aValues: Uint32Array, bKeys: Uint32Array, iterations = 1, warmup = 0) {
        const device = this.device;
        const a_len = aKeys.length, b_len = bKeys.length, total = a_len + b_len;
        if (total === 0) return { resultKeys: new Uint32Array(0), resultValues: new Uint32Array(0), totalCount: 0, timing: { dpiMs: 0, lookbackMs: 0, totalMs: 0 } };

        const numWg = Math.ceil(total / NV);
        const maxOutputSize = Math.min(a_len, b_len);

        const bufAKeys = device.createBuffer({ size: Math.max(4, aKeys.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufAKeys, 0, new Uint32Array(aKeys));
        const bufAValues = device.createBuffer({ size: Math.max(4, aValues.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufAValues, 0, new Uint32Array(aValues));
        const bufBKeys = device.createBuffer({ size: Math.max(4, bKeys.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufBKeys, 0, new Uint32Array(bKeys));
        // Semi-join: B values not needed (only checking key existence)
        const bufBValues = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

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

        const dpiBindGroup = device.createBindGroup({ layout: this.dpiBindGroupLayout, entries: [
            { binding: 0, resource: { buffer: bufAKeys } }, { binding: 1, resource: { buffer: bufBKeys } },
            { binding: 2, resource: { buffer: bufDPI } }, { binding: 3, resource: { buffer: bufALen } },
            { binding: 4, resource: { buffer: bufBLen } }, { binding: 5, resource: { buffer: bufNumWg } },
        ]});

        const lookbackBindGroup = device.createBindGroup({ layout: this.lookbackBindGroupLayout, entries: [
            { binding: 0, resource: { buffer: bufAKeys } }, { binding: 1, resource: { buffer: bufAValues } },
            { binding: 2, resource: { buffer: bufBKeys } }, { binding: 3, resource: { buffer: bufBValues } },
            { binding: 4, resource: { buffer: bufDPI } }, { binding: 5, resource: { buffer: bufState } },
            { binding: 6, resource: { buffer: bufOutputKeys } }, { binding: 7, resource: { buffer: bufOutputValues } },
            { binding: 8, resource: { buffer: bufTotalCount } }, { binding: 9, resource: { buffer: bufALen } },
            { binding: 10, resource: { buffer: bufBLen } }, { binding: 11, resource: { buffer: bufNumWg } },
        ]});

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);
        const dpiDX = Math.min(dpiBlocks, MAXWORKGROUP), dpiDY = Math.ceil(dpiBlocks / MAXWORKGROUP);
        const lbDX = Math.min(numWg, MAXWORKGROUP), lbDY = Math.ceil(numWg / MAXWORKGROUP);

        const stateZeros = new Uint32Array(numWg).fill(0);
        await device.queue.onSubmittedWorkDone();

        for (let w = 0; w < warmup; w++) {
            device.queue.writeBuffer(bufState, 0, stateZeros);
            device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
            const enc = device.createCommandEncoder();
            let p = enc.beginComputePass(); p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup); p.dispatchWorkgroups(dpiDX, dpiDY); p.end();
            p = enc.beginComputePass(); p.setPipeline(this.lookbackPipeline); p.setBindGroup(0, lookbackBindGroup); p.dispatchWorkgroups(lbDX, lbDY); p.end();
            device.queue.submit([enc.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        const dpiTimes: number[] = [], lbTimes: number[] = [], totalTimes: number[] = [];
        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufState, 0, stateZeros);
            device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
            const enc = device.createCommandEncoder();
            let p = enc.beginComputePass(this.tsm.createComputePassDescriptor(0, 1)); p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup); p.dispatchWorkgroups(dpiDX, dpiDY); p.end();
            p = enc.beginComputePass(this.tsm.createComputePassDescriptor(2, 3)); p.setPipeline(this.lookbackPipeline); p.setBindGroup(0, lookbackBindGroup); p.dispatchWorkgroups(lbDX, lbDY); p.end();
            this.tsm.resolve(enc); device.queue.submit([enc.finish()]); await device.queue.onSubmittedWorkDone();
            const ts = await this.tsm.downloadTimestampResult();
            if (ts.length >= 4) { dpiTimes.push((ts[1]-ts[0])/1e6); lbTimes.push((ts[3]-ts[2])/1e6); totalTimes.push((ts[3]-ts[0])/1e6); }
        }

        const avg = (a: number[]) => a.length > 0 ? a.reduce((s,v) => s+v, 0) / a.length : 0;

        // Final run for readback
        device.queue.writeBuffer(bufState, 0, stateZeros);
        device.queue.writeBuffer(bufTotalCount, 0, new Uint32Array([0]));
        const fe = device.createCommandEncoder();
        let fp = fe.beginComputePass(); fp.setPipeline(this.dpiPipeline); fp.setBindGroup(0, dpiBindGroup); fp.dispatchWorkgroups(dpiDX, dpiDY); fp.end();
        fp = fe.beginComputePass(); fp.setPipeline(this.lookbackPipeline); fp.setBindGroup(0, lookbackBindGroup); fp.dispatchWorkgroups(lbDX, lbDY); fp.end();
        const rb = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        fe.copyBufferToBuffer(bufTotalCount, 0, rb, 0, 4);
        device.queue.submit([fe.finish()]); await device.queue.onSubmittedWorkDone();
        await rb.mapAsync(GPUMapMode.READ);
        const totalCount = new Uint32Array(rb.getMappedRange().slice(0))[0];
        rb.unmap();

        bufAKeys.destroy(); bufAValues.destroy(); bufBKeys.destroy(); bufBValues.destroy();
        bufALen.destroy(); bufBLen.destroy(); bufNumWg.destroy();
        bufDPI.destroy(); bufState.destroy(); bufOutputKeys.destroy(); bufOutputValues.destroy();
        bufTotalCount.destroy(); rb.destroy();

        return { resultKeys: new Uint32Array(0), resultValues: new Uint32Array(0), totalCount, timing: { dpiMs: avg(dpiTimes), lookbackMs: avg(lbTimes), totalMs: avg(totalTimes) } };
    }
}

export async function runSemiJoinTest(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  SEMI-JOIN via GPU Set Intersection By Key');
    console.log('  SELECT * FROM A WHERE A.key IN (SELECT key FROM B)');
    console.log('  DPI (keys only) + Decoupled Lookback By Key (OP_MODE=0)');
    console.log('========================================================================\n');

    const NUM_ITERATIONS = 10, NUM_WARMUP = 3;
    const tsm = new TimestampQueryManager(device, 8);
    if (!tsm.timestampSupported) { console.log('  ERROR: timestamp queries not supported.\n'); return; }
    const joiner = new SemiJoin(device, tsm);

    console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations\n`);
    console.log('  Dataset                | |A|          | |B|          | Semi-Join    | DPI(ms)   | Lookback(ms) | Total(ms) | Match');
    console.log('  -----------------------|--------------|--------------|--------------|-----------|--------------|-----------|------');

    const tpchDatasets = [
        { name: 'lineitem×orders SF1', aKeys: 'tpch/lineitem_orders_sf1_A_keys.bin', aValues: 'tpch/lineitem_orders_sf1_A_values.bin', bKeys: 'tpch/lineitem_orders_sf1_B_keys.bin' },
        { name: 'lineitem×part SF1', aKeys: 'tpch/lineitem_part_sf1_A_keys.bin', aValues: 'tpch/lineitem_part_sf1_A_values.bin', bKeys: 'tpch/lineitem_part_sf1_B_keys.bin' },
        { name: 'lineitem×orders SF5', aKeys: 'tpch/lineitem_orders_sf5_A_keys.bin', aValues: 'tpch/lineitem_orders_sf5_A_values.bin', bKeys: 'tpch/lineitem_orders_sf5_B_keys.bin' },
        { name: 'lineitem×part SF5', aKeys: 'tpch/lineitem_part_sf5_A_keys.bin', aValues: 'tpch/lineitem_part_sf5_A_values.bin', bKeys: 'tpch/lineitem_part_sf5_B_keys.bin' },
        { name: 'lineitem×orders SF10', aKeys: 'tpch/lineitem_orders_sf10_A_keys.bin', aValues: 'tpch/lineitem_orders_sf10_A_values.bin', bKeys: 'tpch/lineitem_orders_sf10_B_keys.bin' },
        { name: 'lineitem×part SF10', aKeys: 'tpch/lineitem_part_sf10_A_keys.bin', aValues: 'tpch/lineitem_part_sf10_A_values.bin', bKeys: 'tpch/lineitem_part_sf10_B_keys.bin' },
    ];

    for (const ds of tpchDatasets) {
        try {
            const aKeys = await loadRawUint32Array(`./${ds.aKeys}`);
            const aValues = await loadRawUint32Array(`./${ds.aValues}`);
            const bKeys = await loadRawUint32Array(`./${ds.bKeys}`);

            const result = await joiner.run(aKeys, aValues, bKeys, NUM_ITERATIONS, NUM_WARMUP);

            let matchStr = 'skip';
            if (aKeys.length <= 64_000_000) {
                const aInterleaved = new Uint32Array(aKeys.length * 2);
                for (let i = 0; i < aKeys.length; i++) { aInterleaved[i * 2] = aKeys[i]; aInterleaved[i * 2 + 1] = aValues[i]; }
                const cpuResult = setIntersectionCPUByKey(aInterleaved, bKeys);
                matchStr = result.totalCount === cpuResult.length / 2 ? 'OK' : 'FAIL';
            }

            const nc = ds.name.padEnd(22), ac = aKeys.length.toLocaleString().padStart(12), bc = bKeys.length.toLocaleString().padStart(12);
            const jc = result.totalCount.toLocaleString().padStart(12);
            console.log(`  ${nc} |${ac} |${bc} |${jc} |${result.timing.dpiMs.toFixed(3).padStart(9)} |${result.timing.lookbackMs.toFixed(3).padStart(12)} |${result.timing.totalMs.toFixed(3).padStart(9)} | ${matchStr}`);
        } catch { console.log(`  ${ds.name.padEnd(22)} | not found`); }
    }
    console.log('');
}
