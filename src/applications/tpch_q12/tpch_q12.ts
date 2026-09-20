/**
 * TPC-H Q12: Filter → Join → Group-By Aggregation (Single Encoder)
 *
 * All 3 GPU primitives in one queue.submit:
 *   Pass 0-1: Filter Mark (predicate evaluation)
 *   Buffer copy: flags → flagsCopy
 *   Prefix sum passes (ExclusiveScanPipeline)
 *   Pass 2-3: Filter Scatter (stream compaction)
 *   Pass 4-5: DPI (join merge path boundaries)
 *   Pass 6-7: Lookback By Key (join intersection)
 *   Pass 8-9: GroupBy Aggregation
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from '../../balanced_path/common/prefix_sum/exclusive_scan';
import { setOpMode, setIntersectionCPUByKey } from '../../utils';

import filterMarkShader from './q12_filter_mark.wgsl';
import filterScatterShader from './q12_filter_scatter.wgsl';
import groupByShader from './q12_groupby.wgsl';
import computeDiagonalsShader from '../../balanced_path/common/balanced_path_biased.wgsl';
import lookbackByKeyShaderBase from '../../balanced_path/common/by_key/set_availability_decoupled_lookback_by_key.wgsl';

const WORKGROUP_SIZE = 256;
const NV = 256 * 12; // 3072
const DPI_WG_SIZE = 256;
const MAX_DISPATCH_X = 65535;

async function loadRaw(url: string): Promise<Uint32Array> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}`);
    return new Uint32Array(await resp.arrayBuffer());
}

// "mean ms (std s, min a, max b, n)" over per-run times.
function formatStats(values: number[]): string {
    const n = values.length;
    const mean = values.reduce((s, v) => s + v, 0) / n;
    const std = n > 1 ? Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
    return `${mean.toFixed(3)} ms (std ${std.toFixed(3)}, min ${Math.min(...values).toFixed(3)}, max ${Math.max(...values).toFixed(3)}, n=${n})`;
}

export async function runTPCHQ12Test(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  TPC-H Q12: Filter → Join → Group-By (Single Encoder)');
    console.log('  Three composable GPU primitives in one queue.submit');
    console.log('========================================================================\n');

    const NUM_WARMUP = 10;
    const NUM_ITERATIONS = 100;

    // 10 timestamp slots: filter mark[0,1], filter scatter[2,3], DPI[4,5], lookback[6,7], groupby[8,9]
    const tsm = new TimestampQueryManager(device, 10);
    if (!tsm.timestampSupported) { console.log('  ERROR: timestamp queries not supported.\n'); return; }

    const scan = new ExclusiveScanPipeline(device);

    // ---- Create all pipelines once ----

    // Filter Mark (7 bindings: 4 input + flags + atomic counter + params)
    const markBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]});
    const markPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [markBGL] }),
        compute: { module: device.createShaderModule({ code: filterMarkShader }), entryPoint: 'main' }
    });

    // Filter Scatter
    const scatterBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]});
    const scatterPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [scatterBGL] }),
        compute: { module: device.createShaderModule({ code: filterScatterShader }), entryPoint: 'main' }
    });

    // Join DPI (non-batched, keys only)
    const dpiBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    ]});
    const dpiPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [dpiBGL] }),
        compute: { module: device.createShaderModule({ code: computeDiagonalsShader }), entryPoint: 'compute_diagonals' }
    });

    // Join Lookback By Key (OP_MODE=0)
    const lbBGL = device.createBindGroupLayout({ entries: [
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
    ]});
    const lbPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [lbBGL] }),
        compute: { module: device.createShaderModule({ code: setOpMode(lookbackByKeyShaderBase, 0) }), entryPoint: 'decoupled_lookback_by_key_kernel' }
    });

    // GroupBy
    const gbBGL = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ]});
    const gbPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [gbBGL] }),
        compute: { module: device.createShaderModule({ code: groupByShader }), entryPoint: 'main' }
    });

    // ---- Datasets ----
    const datasets = [
        { name: 'SF1', sf: 1 },
        { name: 'SF5', sf: 5 },
        { name: 'SF10', sf: 10 },
    ];

    for (const ds of datasets) {
        try {
            const sf = ds.sf;
            const prefix = `tpch/q12_sf${sf}`;

            const metaResp = await fetch(`./${prefix}_meta.json`);
            if (!metaResp.ok) { console.log(`  ${ds.name}: data not found`); continue; }
            const meta = await metaResp.json();

            console.log(`  === ${ds.name} ===`);
            console.log(`  lineitem: ${meta.lineitem_total.toLocaleString()} rows`);

            // Load all data
            const [fShipdate, fCommitdate, fReceiptdate, fShipmode, fOrderkey,
                   jAKeys, jAVals, jBKeys, jBVals,
                   gbShipmode, gbIsHigh] = await Promise.all([
                loadRaw(`./${prefix}_filter_shipdate.bin`),
                loadRaw(`./${prefix}_filter_commitdate.bin`),
                loadRaw(`./${prefix}_filter_receiptdate.bin`),
                loadRaw(`./${prefix}_filter_shipmode.bin`),
                loadRaw(`./${prefix}_filter_orderkey.bin`),
                loadRaw(`./${prefix}_join_A_keys.bin`),
                loadRaw(`./${prefix}_join_A_values.bin`),
                loadRaw(`./${prefix}_join_B_keys.bin`),
                loadRaw(`./${prefix}_join_B_values.bin`),
                loadRaw(`./${prefix}_groupby_shipmode.bin`),
                loadRaw(`./${prefix}_groupby_ishigh.bin`),
            ]);

            // ---- Create all GPU buffers ----
            const N = fShipdate.length;
            const alignedN = scan.getAlignedSize(N);
            const filterWg = Math.ceil(N / WORKGROUP_SIZE);
            const filterDX = Math.min(filterWg, MAX_DISPATCH_X), filterDY = Math.ceil(filterWg / MAX_DISPATCH_X);

            const jALen = jAKeys.length, jBLen = jBKeys.length;
            const joinNumWg = Math.ceil((jALen + jBLen) / NV);
            const joinMaxOut = Math.min(jALen, jBLen);
            const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
            const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
            const dpiBlocks = Math.ceil(joinNumWg / subgroupsPerWg);
            const dpiDX = Math.min(dpiBlocks, MAX_DISPATCH_X), dpiDY = Math.ceil(dpiBlocks / MAX_DISPATCH_X);
            const lbDX = Math.min(joinNumWg, MAX_DISPATCH_X), lbDY = Math.ceil(joinNumWg / MAX_DISPATCH_X);

            const gbN = gbShipmode.length;
            const gbWg = Math.ceil(gbN / WORKGROUP_SIZE);
            const gbDX = Math.min(gbWg, MAX_DISPATCH_X), gbDY = Math.ceil(gbWg / MAX_DISPATCH_X);

            // Filter buffers
            const bfSD = device.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bfCD = device.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bfRD = device.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bfSM = device.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bfOK = device.createBuffer({ size: N*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bfFlags = device.createBuffer({ size: alignedN*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
            const bfFlagsCopy = device.createBuffer({ size: alignedN*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
            const bfParams = device.createBuffer({ size: 12, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bfScatterP = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bfFilterCnt = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });
            const bfOutOK = device.createBuffer({ size: Math.max(N,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
            const bfOutSM = device.createBuffer({ size: Math.max(N,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });

            // Join buffers
            const bjAK = device.createBuffer({ size: Math.max(4,jALen*4), usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bjAV = device.createBuffer({ size: Math.max(4,jALen*4), usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bjBK = device.createBuffer({ size: Math.max(4,jBLen*4), usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bjBV = device.createBuffer({ size: Math.max(4,jBLen*4), usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bjALenU = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bjBLenU = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bjNumWgU = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bjDPI = device.createBuffer({ size: 2*(joinNumWg+1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
            const bjState = device.createBuffer({ size: Math.max(4,joinNumWg*4), usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bjOutK = device.createBuffer({ size: Math.max(joinMaxOut,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
            const bjOutV = device.createBuffer({ size: Math.max(joinMaxOut,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC });
            const bjTotalCnt = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });

            // GroupBy buffers
            const bgSM = device.createBuffer({ size: Math.max(gbN,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bgIH = device.createBuffer({ size: Math.max(gbN,1)*4, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
            const bgParams = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
            const bgResult = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST });

            // Upload static data (once)
            device.queue.writeBuffer(bfSD, 0, new Uint32Array(fShipdate));
            device.queue.writeBuffer(bfCD, 0, new Uint32Array(fCommitdate));
            device.queue.writeBuffer(bfRD, 0, new Uint32Array(fReceiptdate));
            device.queue.writeBuffer(bfSM, 0, new Uint32Array(fShipmode));
            device.queue.writeBuffer(bfOK, 0, new Uint32Array(fOrderkey));
            device.queue.writeBuffer(bfParams, 0, new Uint32Array([N, meta.date_low, meta.date_high]));
            device.queue.writeBuffer(bfScatterP, 0, new Uint32Array([N]));
            device.queue.writeBuffer(bjAK, 0, new Uint32Array(jAKeys));
            device.queue.writeBuffer(bjAV, 0, new Uint32Array(jAVals));
            device.queue.writeBuffer(bjBK, 0, new Uint32Array(jBKeys));
            device.queue.writeBuffer(bjBV, 0, new Uint32Array(jBVals));
            device.queue.writeBuffer(bjALenU, 0, new Uint32Array([jALen]));
            device.queue.writeBuffer(bjBLenU, 0, new Uint32Array([jBLen]));
            device.queue.writeBuffer(bjNumWgU, 0, new Uint32Array([joinNumWg]));
            device.queue.writeBuffer(bgSM, 0, new Uint32Array(gbShipmode));
            device.queue.writeBuffer(bgIH, 0, new Uint32Array(gbIsHigh));
            device.queue.writeBuffer(bgParams, 0, new Uint32Array([gbN]));

            // Prepare scanner
            const scanner = scan.prepareGPUInput(bfFlags, alignedN);

            // ---- Create bind groups ----
            const markBG = device.createBindGroup({ layout: markBGL, entries: [
                { binding: 0, resource: { buffer: bfSD } }, { binding: 1, resource: { buffer: bfCD } },
                { binding: 2, resource: { buffer: bfRD } }, { binding: 3, resource: { buffer: bfSM } },
                { binding: 4, resource: { buffer: bfFlags } }, { binding: 5, resource: { buffer: bfFilterCnt } },
                { binding: 6, resource: { buffer: bfParams } },
            ]});
            const scatterBG = device.createBindGroup({ layout: scatterBGL, entries: [
                { binding: 0, resource: { buffer: bfOK } }, { binding: 1, resource: { buffer: bfSM } },
                { binding: 2, resource: { buffer: bfFlags } }, { binding: 3, resource: { buffer: bfFlagsCopy } },
                { binding: 4, resource: { buffer: bfOutOK } }, { binding: 5, resource: { buffer: bfOutSM } },
                { binding: 6, resource: { buffer: bfScatterP } },
            ]});
            const dpiBG = device.createBindGroup({ layout: dpiBGL, entries: [
                { binding: 0, resource: { buffer: bjAK } }, { binding: 1, resource: { buffer: bjBK } },
                { binding: 2, resource: { buffer: bjDPI } }, { binding: 3, resource: { buffer: bjALenU } },
                { binding: 4, resource: { buffer: bjBLenU } }, { binding: 5, resource: { buffer: bjNumWgU } },
            ]});
            const lbBG = device.createBindGroup({ layout: lbBGL, entries: [
                { binding: 0, resource: { buffer: bjAK } }, { binding: 1, resource: { buffer: bjAV } },
                { binding: 2, resource: { buffer: bjBK } }, { binding: 3, resource: { buffer: bjBV } },
                { binding: 4, resource: { buffer: bjDPI } }, { binding: 5, resource: { buffer: bjState } },
                { binding: 6, resource: { buffer: bjOutK } }, { binding: 7, resource: { buffer: bjOutV } },
                { binding: 8, resource: { buffer: bjTotalCnt } }, { binding: 9, resource: { buffer: bjALenU } },
                { binding: 10, resource: { buffer: bjBLenU } }, { binding: 11, resource: { buffer: bjNumWgU } },
            ]});
            const gbBG = device.createBindGroup({ layout: gbBGL, entries: [
                { binding: 0, resource: { buffer: bgSM } }, { binding: 1, resource: { buffer: bgIH } },
                { binding: 2, resource: { buffer: bgParams } }, { binding: 3, resource: { buffer: bgResult } },
            ]});

            const joinStateZeros = new Uint32Array(joinNumWg).fill(0);
            // Only need to zero the padding region (N to alignedN), mark kernel writes [0..N)
            const flagsPaddingZeros = new Uint32Array(alignedN - N).fill(0);

            await device.queue.onSubmittedWorkDone();
            console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations\n`);

            // ---- Run function: all 3 steps in one encoder ----
            const runOnce = async (timed: boolean) => {
                // Reset per-iteration state
                if (flagsPaddingZeros.length > 0) {
                    device.queue.writeBuffer(bfFlags, N * 4, flagsPaddingZeros);
                }
                device.queue.writeBuffer(bfFilterCnt, 0, new Uint32Array([0]));
                device.queue.writeBuffer(bjState, 0, joinStateZeros);
                device.queue.writeBuffer(bjTotalCnt, 0, new Uint32Array([0]));
                device.queue.writeBuffer(bgResult, 0, new Uint32Array([0, 0, 0, 0]));

                const enc = device.createCommandEncoder();

                // Pass 0-1: Filter Mark
                let p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(0, 1) : undefined);
                p.setPipeline(markPipeline); p.setBindGroup(0, markBG);
                p.dispatchWorkgroups(filterDX, filterDY); p.end();

                // Copy flags before scan
                enc.copyBufferToBuffer(bfFlags, 0, bfFlagsCopy, 0, alignedN * 4);

                // Prefix sum (records multiple passes internally)
                scanner.recordScanCommands(enc, N);

                // Pass 2-3: Filter Scatter
                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(2, 3) : undefined);
                p.setPipeline(scatterPipeline); p.setBindGroup(0, scatterBG);
                p.dispatchWorkgroups(filterDX, filterDY); p.end();

                // Pass 4-5: Join DPI
                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(4, 5) : undefined);
                p.setPipeline(dpiPipeline); p.setBindGroup(0, dpiBG);
                p.dispatchWorkgroups(dpiDX, dpiDY); p.end();

                // Pass 6-7: Join Lookback
                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(6, 7) : undefined);
                p.setPipeline(lbPipeline); p.setBindGroup(0, lbBG);
                p.dispatchWorkgroups(lbDX, lbDY); p.end();

                // Pass 8-9: GroupBy
                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(8, 9) : undefined);
                p.setPipeline(gbPipeline); p.setBindGroup(0, gbBG);
                p.dispatchWorkgroups(gbDX, gbDY); p.end();

                // Readback buffers
                const rbFilter = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
                const rbJoin = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
                const rbGB = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });

                enc.copyBufferToBuffer(bfFilterCnt, 0, rbFilter, 0, 4);
                enc.copyBufferToBuffer(bjTotalCnt, 0, rbJoin, 0, 4);
                enc.copyBufferToBuffer(bgResult, 0, rbGB, 0, 16);

                if (timed) tsm.resolve(enc);
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();

                // Read timestamps
                let filterMs = 0, joinMs = 0, groupByMs = 0, totalMs = 0;
                if (timed) {
                    const ts = await tsm.downloadTimestampResult();
                    if (ts.length >= 10) {
                        filterMs = (ts[3] - ts[0]) / 1e6;   // mark start → scatter end
                        joinMs = (ts[7] - ts[4]) / 1e6;     // DPI start → lookback end
                        groupByMs = (ts[9] - ts[8]) / 1e6;  // groupby start → end
                        totalMs = (ts[9] - ts[0]) / 1e6;    // everything
                    }
                }

                // Read filter count (from atomic counter in mark kernel)
                await rbFilter.mapAsync(GPUMapMode.READ);
                const filteredCount = new Uint32Array(rbFilter.getMappedRange().slice(0))[0];
                rbFilter.unmap(); rbFilter.destroy();

                // Read join count
                await rbJoin.mapAsync(GPUMapMode.READ);
                const joinCount = new Uint32Array(rbJoin.getMappedRange().slice(0))[0];
                rbJoin.unmap(); rbJoin.destroy();

                // Read groupby result
                await rbGB.mapAsync(GPUMapMode.READ);
                const gr = new Uint32Array(rbGB.getMappedRange().slice(0));
                const [mailHigh, mailLow, shipHigh, shipLow] = [gr[0], gr[1], gr[2], gr[3]];
                rbGB.unmap(); rbGB.destroy();

                return { filteredCount, joinCount, mailHigh, mailLow, shipHigh, shipLow,
                         filterMs, joinMs, groupByMs, totalMs };
            };

            // ---- Non-composed baseline: each step in its own submit ----
            const runOnceNonComposed = async (timed: boolean) => {
                // Reset buffers
                if (flagsPaddingZeros.length > 0) {
                    device.queue.writeBuffer(bfFlags, N * 4, flagsPaddingZeros);
                }
                device.queue.writeBuffer(bfFilterCnt, 0, new Uint32Array([0]));
                device.queue.writeBuffer(bjState, 0, joinStateZeros);
                device.queue.writeBuffer(bjTotalCnt, 0, new Uint32Array([0]));
                device.queue.writeBuffer(bgResult, 0, new Uint32Array([0, 0, 0, 0]));

                let filterMs = 0, joinMs = 0, groupByMs = 0;
                const wallT0 = performance.now();

                // Step 1: Filter (mark + scan + scatter) — separate submit
                {
                    const enc = device.createCommandEncoder();
                    let p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(0, 1) : undefined);
                    p.setPipeline(markPipeline); p.setBindGroup(0, markBG);
                    p.dispatchWorkgroups(filterDX, filterDY); p.end();
                    enc.copyBufferToBuffer(bfFlags, 0, bfFlagsCopy, 0, alignedN * 4);
                    scanner.recordScanCommands(enc, N);
                    p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(2, 3) : undefined);
                    p.setPipeline(scatterPipeline); p.setBindGroup(0, scatterBG);
                    p.dispatchWorkgroups(filterDX, filterDY); p.end();
                    if (timed) tsm.resolve(enc);
                    device.queue.submit([enc.finish()]);
                    await device.queue.onSubmittedWorkDone();
                    if (timed) {
                        const ts = await tsm.downloadTimestampResult();
                        if (ts.length >= 4) filterMs = (ts[3] - ts[0]) / 1e6;
                    }
                }

                // Step 2: Join (DPI + Lookback) — separate submit
                {
                    const enc = device.createCommandEncoder();
                    let p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(4, 5) : undefined);
                    p.setPipeline(dpiPipeline); p.setBindGroup(0, dpiBG);
                    p.dispatchWorkgroups(dpiDX, dpiDY); p.end();
                    p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(6, 7) : undefined);
                    p.setPipeline(lbPipeline); p.setBindGroup(0, lbBG);
                    p.dispatchWorkgroups(lbDX, lbDY); p.end();
                    if (timed) tsm.resolve(enc);
                    device.queue.submit([enc.finish()]);
                    await device.queue.onSubmittedWorkDone();
                    if (timed) {
                        const ts = await tsm.downloadTimestampResult();
                        if (ts.length >= 8) joinMs = (ts[7] - ts[4]) / 1e6;
                    }
                }

                // Step 3: GroupBy — separate submit
                {
                    const enc = device.createCommandEncoder();
                    let p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(8, 9) : undefined);
                    p.setPipeline(gbPipeline); p.setBindGroup(0, gbBG);
                    p.dispatchWorkgroups(gbDX, gbDY); p.end();
                    const rbFilter = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
                    const rbJoin = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
                    const rbGB = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ });
                    enc.copyBufferToBuffer(bfFilterCnt, 0, rbFilter, 0, 4);
                    enc.copyBufferToBuffer(bjTotalCnt, 0, rbJoin, 0, 4);
                    enc.copyBufferToBuffer(bgResult, 0, rbGB, 0, 16);
                    if (timed) tsm.resolve(enc);
                    device.queue.submit([enc.finish()]);
                    await device.queue.onSubmittedWorkDone();
                    if (timed) {
                        const ts = await tsm.downloadTimestampResult();
                        if (ts.length >= 10) groupByMs = (ts[9] - ts[8]) / 1e6;
                    }
                    await rbFilter.mapAsync(GPUMapMode.READ);
                    const filteredCount = new Uint32Array(rbFilter.getMappedRange().slice(0))[0];
                    rbFilter.unmap(); rbFilter.destroy();
                    await rbJoin.mapAsync(GPUMapMode.READ);
                    const joinCount = new Uint32Array(rbJoin.getMappedRange().slice(0))[0];
                    rbJoin.unmap(); rbJoin.destroy();
                    await rbGB.mapAsync(GPUMapMode.READ);
                    const gr = new Uint32Array(rbGB.getMappedRange().slice(0));
                    const [mailHigh, mailLow, shipHigh, shipLow] = [gr[0], gr[1], gr[2], gr[3]];
                    rbGB.unmap(); rbGB.destroy();
                    const gpuTotalMs = filterMs + joinMs + groupByMs;
                    const wallMs = performance.now() - wallT0;
                    return { filteredCount, joinCount, mailHigh, mailLow, shipHigh, shipLow,
                             filterMs, joinMs, groupByMs, gpuTotalMs, wallMs };
                }
            };

            // ========== Run composed (single encoder) ==========
            // Warmup
            for (let w = 0; w < NUM_WARMUP; w++) await runOnce(false);

            // Timed runs
            let sumFilter = 0, sumJoin = 0, sumGB = 0, sumTotal = 0;
            const runsFilter: number[] = [], runsJoin: number[] = [], runsGB: number[] = [], runsTotal: number[] = [];
            let last: any;
            for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
                last = await runOnce(true);
                sumFilter += last.filterMs; sumJoin += last.joinMs;
                sumGB += last.groupByMs; sumTotal += last.totalMs;
                runsFilter.push(last.filterMs); runsJoin.push(last.joinMs);
                runsGB.push(last.groupByMs); runsTotal.push(last.totalMs);
            }

            // Validate
            const filterOK = last.filteredCount === meta.filter_count;
            const jAI = new Uint32Array(jAKeys.length * 2);
            for (let i = 0; i < jAKeys.length; i++) { jAI[i*2] = jAKeys[i]; jAI[i*2+1] = jAVals[i]; }
            const cpuJC = setIntersectionCPUByKey(jAI, jBKeys).length / 2;
            const joinOK = last.joinCount === cpuJC;
            const exp = meta.q12_result;
            const gbOK = exp.MAIL && exp.SHIP &&
                last.mailHigh === exp.MAIL.high && last.mailLow === exp.MAIL.low &&
                last.shipHigh === exp.SHIP.high && last.shipLow === exp.SHIP.low;

            console.log(`  Step 1 (Filter):   ${last.filteredCount.toLocaleString()} rows (expected: ${meta.filter_count.toLocaleString()})  ${filterOK ? 'OK' : 'FAIL'}`);
            console.log(`  Step 2 (Join):     ${last.joinCount.toLocaleString()} rows (expected: ${cpuJC.toLocaleString()})  ${joinOK ? 'OK' : 'FAIL'}`);
            console.log(`  Step 3 (GroupBy):  MAIL(high=${last.mailHigh}, low=${last.mailLow})  SHIP(high=${last.shipHigh}, low=${last.shipLow})  ${gbOK ? 'OK' : 'FAIL'}`);
            console.log(`\n  --- GPU Timing (avg of ${NUM_ITERATIONS} runs, COMPOSED single encoder) ---`);
            console.log(`  Filter:   ${(sumFilter/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  Join:     ${(sumJoin/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  GroupBy:  ${(sumGB/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  Total:    ${(sumTotal/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  [tpch-stats] SF${sf} composed total: ${formatStats(runsTotal)}`);
            console.log(`  [tpch-stats] SF${sf} composed filter: ${formatStats(runsFilter)}`);
            console.log(`  [tpch-stats] SF${sf} composed join: ${formatStats(runsJoin)}`);
            console.log(`  [tpch-stats] SF${sf} composed groupby: ${formatStats(runsGB)}`);

            // ========== Run non-composed (separate submits) ==========
            for (let w = 0; w < NUM_WARMUP; w++) await runOnceNonComposed(false);

            let ncSumFilter = 0, ncSumJoin = 0, ncSumGB = 0, ncSumGpuTotal = 0, ncSumWall = 0;
            const ncRunsGpuTotal: number[] = [], ncRunsWall: number[] = [];
            let ncLast: any;
            for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
                ncLast = await runOnceNonComposed(true);
                ncSumFilter += ncLast.filterMs; ncSumJoin += ncLast.joinMs;
                ncSumGB += ncLast.groupByMs; ncSumGpuTotal += ncLast.gpuTotalMs;
                ncSumWall += ncLast.wallMs;
                ncRunsGpuTotal.push(ncLast.gpuTotalMs); ncRunsWall.push(ncLast.wallMs);
            }
            console.log(`  [tpch-stats] SF${sf} non-composed gpu total: ${formatStats(ncRunsGpuTotal)}`);
            console.log(`  [tpch-stats] SF${sf} non-composed wall: ${formatStats(ncRunsWall)}`);

            console.log(`\n  --- GPU Timing (avg of ${NUM_ITERATIONS} runs, NON-COMPOSED separate submits) ---`);
            console.log(`  Filter (GPU):   ${(ncSumFilter/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  Join (GPU):     ${(ncSumJoin/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  GroupBy (GPU):  ${(ncSumGB/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  GPU Total:      ${(ncSumGpuTotal/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`  Wall-clock:     ${(ncSumWall/NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`\n  --- Composition Comparison ---`);
            const composedGpu = sumTotal/NUM_ITERATIONS;
            const nonComposedGpu = ncSumGpuTotal/NUM_ITERATIONS;
            const nonComposedWall = ncSumWall/NUM_ITERATIONS;
            console.log(`  Composed (GPU timestamp):      ${composedGpu.toFixed(3)} ms`);
            console.log(`  Non-composed (GPU timestamp):  ${nonComposedGpu.toFixed(3)} ms`);
            console.log(`  Non-composed (wall-clock):     ${nonComposedWall.toFixed(3)} ms`);
            console.log(`  GPU speedup:  ${(nonComposedGpu/composedGpu).toFixed(2)}x`);
            console.log(`  Wall speedup: ${(nonComposedWall/composedGpu).toFixed(2)}x`);
            console.log('');

            // Cleanup
            bfSD.destroy();bfCD.destroy();bfRD.destroy();bfSM.destroy();bfOK.destroy();
            bfFlags.destroy();bfFlagsCopy.destroy();bfParams.destroy();bfScatterP.destroy();
            bfFilterCnt.destroy();bfOutOK.destroy();bfOutSM.destroy();
            bjAK.destroy();bjAV.destroy();bjBK.destroy();bjBV.destroy();
            bjALenU.destroy();bjBLenU.destroy();bjNumWgU.destroy();
            bjDPI.destroy();bjState.destroy();bjOutK.destroy();bjOutV.destroy();bjTotalCnt.destroy();
            bgSM.destroy();bgIH.destroy();bgParams.destroy();bgResult.destroy();

        } catch (e) { console.log(`  ${ds.name}: error: ${e}`); }
    }
}
