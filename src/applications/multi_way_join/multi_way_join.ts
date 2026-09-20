/**
 * GPU-accelerated Multi-way Join (3-table chain join) — Single Encoder
 *
 * lineitem JOIN orders ON orderkey JOIN customer ON custkey
 * Both joins in one queue.submit:
 *   Pass 0-1: Join 1 DPI
 *   Pass 2-3: Join 1 Lookback
 *   Pass 4-5: Join 2 DPI
 *   Pass 6-7: Join 2 Lookback
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

export async function runMultiWayJoinTest(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  MULTI-WAY JOIN (3-table chain, single encoder)');
    console.log('  lineitem JOIN orders ON orderkey JOIN customer ON custkey');
    console.log('========================================================================\n');

    const NUM_ITERATIONS = 100, NUM_WARMUP = 10;
    // 8 timestamp slots: j1 DPI[0,1], j1 LB[2,3], j2 DPI[4,5], j2 LB[6,7]
    const tsm = new TimestampQueryManager(device, 8);
    if (!tsm.timestampSupported) { console.log('  ERROR: timestamp queries not supported.\n'); return; }

    // ---- Create pipelines once ----
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

    console.log(`  ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations\n`);

    const datasets = [
        { name: 'SF1', sf: 1 },
        { name: 'SF5', sf: 5 },
        { name: 'SF10', sf: 10 },
    ];

    for (const ds of datasets) {
        try {
            const sf = ds.sf;
            console.log(`  === ${ds.name}: lineitem JOIN orders JOIN customer ===`);

            const [j1aKeys, j1aVals, j1bKeys, j1bVals, j2aKeys, j2aVals, j2bKeys, j2bVals] = await Promise.all([
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join1_A_keys.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join1_A_values.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join1_B_keys.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join1_B_values.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join2_A_keys.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join2_A_values.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join2_B_keys.bin`),
                loadRawUint32Array(`./tpch/multiway_sf${sf}_join2_B_values.bin`),
            ]);

            console.log(`    Join 1: lineitem(${j1aKeys.length.toLocaleString()}) x orders(${j1bKeys.length.toLocaleString()})`);
            console.log(`    Join 2: R1(${j2aKeys.length.toLocaleString()}) x customer(${j2bKeys.length.toLocaleString()})`);

            // Helper to create buffers for one join
            const setupJoin = (aKeys: Uint32Array, aVals: Uint32Array, bKeys: Uint32Array, bVals: Uint32Array) => {
                const aLen = aKeys.length, bLen = bKeys.length;
                const numWg = Math.ceil((aLen + bLen) / NV);
                const maxOut = Math.min(aLen, bLen);
                const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
                const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
                const dpiBlocks = Math.ceil(numWg / subgroupsPerWg);

                const bAK = device.createBuffer({ size: Math.max(4, aLen*4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const bAV = device.createBuffer({ size: Math.max(4, aLen*4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const bBK = device.createBuffer({ size: Math.max(4, bLen*4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const bBV = device.createBuffer({ size: Math.max(4, bLen*4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const bAL = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                const bBL = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                const bNW = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
                const bDPI = device.createBuffer({ size: 2*(numWg+1)*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                const bSt = device.createBuffer({ size: Math.max(4, numWg*4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
                const bOK = device.createBuffer({ size: Math.max(maxOut,1)*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                const bOV = device.createBuffer({ size: Math.max(maxOut,1)*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                const bTC = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

                device.queue.writeBuffer(bAK, 0, new Uint32Array(aKeys));
                device.queue.writeBuffer(bAV, 0, new Uint32Array(aVals));
                device.queue.writeBuffer(bBK, 0, new Uint32Array(bKeys));
                device.queue.writeBuffer(bBV, 0, new Uint32Array(bVals));
                device.queue.writeBuffer(bAL, 0, new Uint32Array([aLen]));
                device.queue.writeBuffer(bBL, 0, new Uint32Array([bLen]));
                device.queue.writeBuffer(bNW, 0, new Uint32Array([numWg]));

                const dpiBG = device.createBindGroup({ layout: dpiBGL, entries: [
                    { binding: 0, resource: { buffer: bAK } }, { binding: 1, resource: { buffer: bBK } },
                    { binding: 2, resource: { buffer: bDPI } }, { binding: 3, resource: { buffer: bAL } },
                    { binding: 4, resource: { buffer: bBL } }, { binding: 5, resource: { buffer: bNW } },
                ]});
                const lbBG = device.createBindGroup({ layout: lbBGL, entries: [
                    { binding: 0, resource: { buffer: bAK } }, { binding: 1, resource: { buffer: bAV } },
                    { binding: 2, resource: { buffer: bBK } }, { binding: 3, resource: { buffer: bBV } },
                    { binding: 4, resource: { buffer: bDPI } }, { binding: 5, resource: { buffer: bSt } },
                    { binding: 6, resource: { buffer: bOK } }, { binding: 7, resource: { buffer: bOV } },
                    { binding: 8, resource: { buffer: bTC } }, { binding: 9, resource: { buffer: bAL } },
                    { binding: 10, resource: { buffer: bBL } }, { binding: 11, resource: { buffer: bNW } },
                ]});

                const dpiDX = Math.min(dpiBlocks, MAXWORKGROUP), dpiDY = Math.ceil(dpiBlocks / MAXWORKGROUP);
                const lbDX = Math.min(numWg, MAXWORKGROUP), lbDY = Math.ceil(numWg / MAXWORKGROUP);
                const stateZeros = new Uint32Array(numWg).fill(0);

                return { dpiBG, lbBG, bSt, bTC, dpiDX, dpiDY, lbDX, lbDY, stateZeros, numWg,
                         bufs: [bAK,bAV,bBK,bBV,bAL,bBL,bNW,bDPI,bSt,bOK,bOV,bTC] };
            };

            const j1 = setupJoin(j1aKeys, j1aVals, j1bKeys, j1bVals);
            const j2 = setupJoin(j2aKeys, j2aVals, j2bKeys, j2bVals);

            await device.queue.onSubmittedWorkDone();

            // ---- Run both joins in one encoder ----
            const runOnce = async (timed: boolean) => {
                // Reset state
                device.queue.writeBuffer(j1.bSt, 0, j1.stateZeros);
                device.queue.writeBuffer(j1.bTC, 0, new Uint32Array([0]));
                device.queue.writeBuffer(j2.bSt, 0, j2.stateZeros);
                device.queue.writeBuffer(j2.bTC, 0, new Uint32Array([0]));

                const enc = device.createCommandEncoder();

                // Join 1: DPI [0,1] + Lookback [2,3]
                let p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(0, 1) : undefined);
                p.setPipeline(dpiPipeline); p.setBindGroup(0, j1.dpiBG);
                p.dispatchWorkgroups(j1.dpiDX, j1.dpiDY); p.end();

                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(2, 3) : undefined);
                p.setPipeline(lbPipeline); p.setBindGroup(0, j1.lbBG);
                p.dispatchWorkgroups(j1.lbDX, j1.lbDY); p.end();

                // Join 2: DPI [4,5] + Lookback [6,7]
                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(4, 5) : undefined);
                p.setPipeline(dpiPipeline); p.setBindGroup(0, j2.dpiBG);
                p.dispatchWorkgroups(j2.dpiDX, j2.dpiDY); p.end();

                p = enc.beginComputePass(timed ? tsm.createComputePassDescriptor(6, 7) : undefined);
                p.setPipeline(lbPipeline); p.setBindGroup(0, j2.lbBG);
                p.dispatchWorkgroups(j2.lbDX, j2.lbDY); p.end();

                // Readback
                const rb1 = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                const rb2 = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
                enc.copyBufferToBuffer(j1.bTC, 0, rb1, 0, 4);
                enc.copyBufferToBuffer(j2.bTC, 0, rb2, 0, 4);

                if (timed) tsm.resolve(enc);
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();

                let dpiMs = 0, lbMs = 0, totalMs = 0;
                if (timed) {
                    const ts = await tsm.downloadTimestampResult();
                    if (ts.length >= 8) {
                        dpiMs = (ts[1]-ts[0])/1e6 + (ts[5]-ts[4])/1e6;
                        lbMs = (ts[3]-ts[2])/1e6 + (ts[7]-ts[6])/1e6;
                        totalMs = (ts[7]-ts[0])/1e6;
                    }
                }

                await rb1.mapAsync(GPUMapMode.READ);
                const j1Count = new Uint32Array(rb1.getMappedRange().slice(0))[0];
                rb1.unmap(); rb1.destroy();

                await rb2.mapAsync(GPUMapMode.READ);
                const j2Count = new Uint32Array(rb2.getMappedRange().slice(0))[0];
                rb2.unmap(); rb2.destroy();

                return { j1Count, j2Count, dpiMs, lbMs, totalMs };
            };

            // Warmup
            for (let w = 0; w < NUM_WARMUP; w++) await runOnce(false);

            // Timed
            let sumDpi = 0, sumLb = 0, sumTotal = 0;
            let last: any;
            for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
                last = await runOnce(true);
                sumDpi += last.dpiMs; sumLb += last.lbMs; sumTotal += last.totalMs;
            }

            // CPU validation
            let matchStr = 'skip';
            if (j1aKeys.length <= 64_000_000) {
                const j1aI = new Uint32Array(j1aKeys.length * 2);
                for (let i = 0; i < j1aKeys.length; i++) { j1aI[i*2] = j1aKeys[i]; j1aI[i*2+1] = j1aVals[i]; }
                const cpuJ1 = setIntersectionCPUByKey(j1aI, j1bKeys).length / 2;
                const j2aI = new Uint32Array(j2aKeys.length * 2);
                for (let i = 0; i < j2aKeys.length; i++) { j2aI[i*2] = j2aKeys[i]; j2aI[i*2+1] = j2aVals[i]; }
                const cpuJ2 = setIntersectionCPUByKey(j2aI, j2bKeys).length / 2;
                matchStr = (last.j1Count === cpuJ1 && last.j2Count === cpuJ2) ? 'OK' : 'FAIL';
            }

            console.log(`    Join 1 result: ${last.j1Count.toLocaleString()} rows`);
            console.log(`    Join 2 result: ${last.j2Count.toLocaleString()} rows  ${matchStr}`);
            console.log(`    --- GPU Timing (avg of ${NUM_ITERATIONS} runs, single encoder) ---`);
            console.log(`    DPI total:      ${(sumDpi / NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`    Lookback total: ${(sumLb / NUM_ITERATIONS).toFixed(3)} ms`);
            console.log(`    GPU Total:      ${(sumTotal / NUM_ITERATIONS).toFixed(3)} ms`);
            console.log('');

            // Cleanup
            j1.bufs.forEach(b => b.destroy());
            j2.bufs.forEach(b => b.destroy());

        } catch (e) {
            console.log(`  ${ds.name}: error or data not found (${e})\n`);
        }
    }
}
