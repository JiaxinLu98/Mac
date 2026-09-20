// ============================================================================
// gather_bench.ts  (rebuttal experiment — standalone, touches no library code)
// ----------------------------------------------------------------------------
// Question this answers: is WebGPU's per-pair packing (copyBufferToBuffer per
// pair, as in webgpu_batched_sets.ts) the bottleneck, and does a single gather
// SHADER speed it up the way the CUDA pack_pairs_kernel did?
//
// It packs P pairs (each a run of T u32) from ONE consolidated source buffer
// into a flat dst buffer, two ways, and times each (wall-clock incl. submit +
// onSubmittedWorkDone):
//   (A) per-pair copyBufferToBuffer  (P copies recorded into one encoder, 1 submit)
//   (B) one gather compute dispatch  (gather.wgsl)
//
// Configs mirror the ECLAT datasets: many-small-pairs (chess-like) vs
// few-large-pairs (webdocs-like). If gather wins big at high P and barely at
// low P, WebGPU's packing is the bottleneck (Option A feasible). If gather
// barely helps even at high P, WebGPU's e2e overhead is elsewhere (Chrome
// submission/readback), and the per-pair comparison was already fair.
//
// To run: add to src/app.ts (after `device` is created):
//     import { runGatherBenchTest } from './applications/gather_bench/gather_bench';
//     await runGatherBenchTest(device);
// then `npm run serve`, open http://localhost:8080, read the console (F12).
// ============================================================================

import gatherShaderCode from './gather.wgsl';

export async function runGatherBenchTest(device: GPUDevice) {
    console.log('===== WebGPU packing: per-pair copyBufferToBuffer vs gather shader =====');
    console.log(`maxStorageBufferBindingSize = ${(device.limits.maxStorageBufferBindingSize / 1e6).toFixed(0)} MB`);

    // (P pairs, T elements each). Kept < ~85 MB/buffer so it runs on any device.
    const configs = [
        { name: 'chess-like   (many small pairs)', P: 166000, T: 128 },
        { name: 'mid                              ', P: 50000,  T: 128 },
        { name: 'webdocs-like (few large pairs)  ', P: 1400,   T: 15000 },
    ];
    const WARMUP = 3, ITERS = 10;

    // Build the gather pipeline once.
    const bgl = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
    });
    const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        compute: { module: device.createShaderModule({ code: gatherShaderCode }), entryPoint: 'main' },
    });

    for (const { name, P, T } of configs) {
        const total = P * T;

        // Consolidated source data + per-pair (srcOff, dstOff, len) metadata.
        const srcData = new Uint32Array(total);
        for (let i = 0; i < total; i++) srcData[i] = i;
        const meta = new Uint32Array(P * 4);
        for (let p = 0; p < P; p++) {
            meta[p * 4 + 0] = p * T; // srcOff
            meta[p * 4 + 1] = p * T; // dstOff
            meta[p * 4 + 2] = T;     // len
            meta[p * 4 + 3] = 0;
        }

        const srcBuf = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(srcBuf, 0, srcData);
        const dstBuf = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const metaBuf = device.createBuffer({ size: P * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(metaBuf, 0, meta);
        const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([P, 0, 0, 0]));

        const bg = device.createBindGroup({
            layout: bgl, entries: [
                { binding: 0, resource: { buffer: srcBuf } },
                { binding: 1, resource: { buffer: dstBuf } },
                { binding: 2, resource: { buffer: metaBuf } },
                { binding: 3, resource: { buffer: paramsBuf } },
            ],
        });
        const wgX = Math.min(P, 65535);
        const wgY = Math.ceil(P / 65535);

        // (A) per-pair copyBufferToBuffer — exactly what webgpu_batched_sets.ts does.
        const runCopy = async () => {
            const enc = device.createCommandEncoder();
            for (let p = 0; p < P; p++)
                enc.copyBufferToBuffer(srcBuf, meta[p * 4 + 0] * 4, dstBuf, meta[p * 4 + 1] * 4, meta[p * 4 + 2] * 4);
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
        };
        // (B) one gather compute dispatch.
        const runGather = async () => {
            const enc = device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(wgX, wgY);
            pass.end();
            device.queue.submit([enc.finish()]);
            await device.queue.onSubmittedWorkDone();
        };

        for (let w = 0; w < WARMUP; w++) { await runCopy(); await runGather(); }

        let tCopy = 0, cMin = 1e30, cMax = 0;
        for (let it = 0; it < ITERS; it++) { const t0 = performance.now(); await runCopy(); const dt = performance.now() - t0; tCopy += dt; cMin = Math.min(cMin, dt); cMax = Math.max(cMax, dt); }
        tCopy /= ITERS;

        let tGather = 0, gMin = 1e30, gMax = 0;
        for (let it = 0; it < ITERS; it++) { const t0 = performance.now(); await runGather(); const dt = performance.now() - t0; tGather += dt; gMin = Math.min(gMin, dt); gMax = Math.max(gMax, dt); }
        tGather /= ITERS;

        // Spot-check gather correctness (dstBuf holds the last gather result).
        let valid = 'skip';
        if (total <= 8_000_000) {
            const stg = device.createBuffer({ size: total * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            const enc = device.createCommandEncoder();
            enc.copyBufferToBuffer(dstBuf, 0, stg, 0, total * 4);
            device.queue.submit([enc.finish()]);
            await stg.mapAsync(GPUMapMode.READ);
            const got = new Uint32Array(stg.getMappedRange().slice(0));
            stg.unmap(); stg.destroy();
            valid = (got[0] === srcData[0] && got[total - 1] === srcData[total - 1]) ? 'PASS' : 'FAIL';
        }

        console.log(
            `${name} | P=${P} T=${T} (${(total * 4 / 1e6).toFixed(0)}MB) | ` +
            `copyBufferToBuffer=${tCopy.toFixed(2)}ms [${cMin.toFixed(1)}-${cMax.toFixed(1)}] | ` +
            `gather=${tGather.toFixed(2)}ms [${gMin.toFixed(1)}-${gMax.toFixed(1)}] | ` +
            `speedup=${(tCopy / tGather).toFixed(1)}x | ${valid}`
        );

        srcBuf.destroy(); dstBuf.destroy(); metaBuf.destroy(); paramsBuf.destroy();
    }
    console.log('===== done =====');
}
