/**
 * BatchedJaccardCounter
 *
 * Computes Jaccard similarity for multiple pairs in a single batch.
 * DPI once + lookback twice (OP_MODE=0 for intersection, OP_MODE=2 for union).
 * Returns per-pair intersection counts, union counts, and Jaccard values.
 */

import TimestampQueryManager from '../../TimestampQueryManager';
import { setOpMode } from '../../utils';
import batchedDpiShader from '../../balanced_path/common/balanced_path_batched.wgsl';
import lookbackShaderBase from './set_availability_decoupled_lookback_batched_perpair_opmode.wgsl';

const MAXWORKGROUP = 65535;
const DPI_WG_SIZE = 256;

export class BatchedJaccardCounter {
    private device: GPUDevice;
    private tsm: TimestampQueryManager;

    private dpiPipeline: GPUComputePipeline;
    private dpiBindGroupLayout: GPUBindGroupLayout;

    private intersectionPipeline: GPUComputePipeline;
    private unionPipeline: GPUComputePipeline;
    private lookbackBindGroupLayout: GPUBindGroupLayout;

    constructor(device: GPUDevice, tsm: TimestampQueryManager) {
        this.device = device;
        this.tsm = tsm;

        // DPI (shared, operation-independent)
        this.dpiBindGroupLayout = device.createBindGroupLayout({
            label: 'BatchedJaccard DPI layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ]
        });

        this.dpiPipeline = device.createComputePipeline({
            label: 'BatchedJaccard DPI pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.dpiBindGroupLayout] }),
            compute: {
                module: device.createShaderModule({ code: batchedDpiShader }),
                entryPoint: 'compute_diagonals_batched'
            }
        });

        // Lookback bind group layout (8 bindings — per-pair counts + pairIdPerWg)
        this.lookbackBindGroupLayout = device.createBindGroupLayout({
            label: 'BatchedJaccard lookback layout',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // keysA
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // keysB
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // dpi
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // state
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },             // pairCounts
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // wgInfo
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },  // pairIdPerWg
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },             // totalWg
            ]
        });

        const lookbackLayout = device.createPipelineLayout({ bindGroupLayouts: [this.lookbackBindGroupLayout] });

        // Intersection (OP_MODE=0)
        this.intersectionPipeline = device.createComputePipeline({
            label: 'BatchedJaccard intersection pipeline',
            layout: lookbackLayout,
            compute: {
                module: device.createShaderModule({ code: setOpMode(lookbackShaderBase, 0) }),
                entryPoint: 'decoupled_lookback_batched_kernel'
            }
        });

        // Union (OP_MODE=2)
        this.unionPipeline = device.createComputePipeline({
            label: 'BatchedJaccard union pipeline',
            layout: lookbackLayout,
            compute: {
                module: device.createShaderModule({ code: setOpMode(lookbackShaderBase, 2) }),
                entryPoint: 'decoupled_lookback_batched_kernel'
            }
        });
    }

    async run(
        keysA: Uint32Array,
        keysB: Uint32Array,
        wgInfo: Uint32Array,
        pairIdPerWg: Uint32Array,
        totalWg: number,
        totalDpiEntries: number,
        numPairs: number,
        iterations: number = 1,
        warmup: number = 0,
        noReuse: boolean = false,
    ): Promise<{
        intersectionCounts: Uint32Array;
        unionCounts: Uint32Array;
        jaccardValues: Float64Array;
        timing: { dpiMs: number; intersectionMs: number; unionMs: number; totalMs: number; totalRunsMs: number[] };
    }> {
        const device = this.device;

        if (totalWg === 0 || numPairs === 0) {
            return {
                intersectionCounts: new Uint32Array(numPairs),
                unionCounts: new Uint32Array(numPairs),
                jaccardValues: new Float64Array(numPairs),
                timing: { dpiMs: 0, intersectionMs: 0, unionMs: 0, totalMs: 0, totalRunsMs: [] }
            };
        }

        // Shared buffers
        const bufKeysA = device.createBuffer({ size: Math.max(4, keysA.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufKeysA, 0, new Uint32Array(keysA));

        const bufKeysB = device.createBuffer({ size: Math.max(4, keysB.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufKeysB, 0, new Uint32Array(keysB));

        const bufDpi = device.createBuffer({ size: Math.max(4, totalDpiEntries * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const bufWgInfo = device.createBuffer({ size: Math.max(4, wgInfo.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufWgInfo, 0, new Uint32Array(wgInfo));

        const bufPairIdPerWg = device.createBuffer({ size: Math.max(4, pairIdPerWg.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufPairIdPerWg, 0, new Uint32Array(pairIdPerWg));

        const bufTotalWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(bufTotalWg, 0, new Uint32Array([totalWg]));

        // Separate state + counts for intersection and union
        const bufStateInter = device.createBuffer({ size: Math.max(4, totalWg * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufCountsInter = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        const bufStateUnion = device.createBuffer({ size: Math.max(4, totalWg * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const bufCountsUnion = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });

        // DPI bind group (shared)
        const dpiBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufWgInfo } },
                { binding: 4, resource: { buffer: bufTotalWg } },
            ]
        });

        // Intersection lookback bind group
        const interBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufStateInter } },
                { binding: 4, resource: { buffer: bufCountsInter } },
                { binding: 5, resource: { buffer: bufWgInfo } },
                { binding: 6, resource: { buffer: bufPairIdPerWg } },
                { binding: 7, resource: { buffer: bufTotalWg } },
            ]
        });

        // Union lookback bind group
        const unionBindGroup = device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufStateUnion } },
                { binding: 4, resource: { buffer: bufCountsUnion } },
                { binding: 5, resource: { buffer: bufWgInfo } },
                { binding: 6, resource: { buffer: bufPairIdPerWg } },
                { binding: 7, resource: { buffer: bufTotalWg } },
            ]
        });

        // Dispatch sizes
        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const subgroupsPerWg = DPI_WG_SIZE / subgroupSize;
        const dpiBlocks = Math.ceil(totalWg / subgroupsPerWg);
        const dpiDispatchX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiDispatchY = Math.ceil(dpiBlocks / MAXWORKGROUP);
        const lbDispatchX = Math.min(totalWg, MAXWORKGROUP);
        const lbDispatchY = Math.ceil(totalWg / MAXWORKGROUP);

        const stateZeros = new Uint32Array(totalWg).fill(0);
        const countsZeros = new Uint32Array(numPairs).fill(0);

        await device.queue.onSubmittedWorkDone();

        // Warmup
        for (let w = 0; w < warmup; w++) {
            device.queue.writeBuffer(bufStateInter, 0, stateZeros);
            device.queue.writeBuffer(bufCountsInter, 0, countsZeros);
            device.queue.writeBuffer(bufStateUnion, 0, stateZeros);
            device.queue.writeBuffer(bufCountsUnion, 0, countsZeros);

            const enc = device.createCommandEncoder();
            let p = enc.beginComputePass();
            p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup);
            p.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); p.end();

            p = enc.beginComputePass();
            p.setPipeline(this.intersectionPipeline); p.setBindGroup(0, interBindGroup);
            p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

            if (noReuse) {
                // DPI again before union (no reuse)
                p = enc.beginComputePass();
                p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup);
                p.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); p.end();
            }

            p = enc.beginComputePass();
            p.setPipeline(this.unionPipeline); p.setBindGroup(0, unionBindGroup);
            p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

            device.queue.submit([enc.finish()]);
        }
        await device.queue.onSubmittedWorkDone();

        // Timed runs
        const dpiTimes: number[] = [], interTimes: number[] = [], unionTimes: number[] = [], totalTimes: number[] = [];

        for (let iter = 0; iter < iterations; iter++) {
            device.queue.writeBuffer(bufStateInter, 0, stateZeros);
            device.queue.writeBuffer(bufCountsInter, 0, countsZeros);
            device.queue.writeBuffer(bufStateUnion, 0, stateZeros);
            device.queue.writeBuffer(bufCountsUnion, 0, countsZeros);

            const enc = device.createCommandEncoder();

            if (noReuse) {
                // Without DPI reuse: DPI1 → Intersection → DPI2 → Union (4 dispatches)
                let p = enc.beginComputePass(this.tsm.createComputePassDescriptor(0, 1));
                p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup);
                p.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); p.end();

                p = enc.beginComputePass(this.tsm.createComputePassDescriptor(2, 3));
                p.setPipeline(this.intersectionPipeline); p.setBindGroup(0, interBindGroup);
                p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

                p = enc.beginComputePass(this.tsm.createComputePassDescriptor(4, 5));
                p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup);
                p.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); p.end();

                p = enc.beginComputePass(this.tsm.createComputePassDescriptor(6, 7));
                p.setPipeline(this.unionPipeline); p.setBindGroup(0, unionBindGroup);
                p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

                this.tsm.resolve(enc);
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();

                const ts = await this.tsm.downloadTimestampResult();
                if (ts.length >= 8) {
                    const dpi1 = (ts[1] - ts[0]) / 1e6;
                    const dpi2 = (ts[5] - ts[4]) / 1e6;
                    dpiTimes.push(dpi1 + dpi2);
                    interTimes.push((ts[3] - ts[2]) / 1e6);
                    unionTimes.push((ts[7] - ts[6]) / 1e6);
                    totalTimes.push((ts[7] - ts[0]) / 1e6);
                }
            } else {
                // With DPI reuse: DPI → Intersection → Union (3 dispatches)
                let p = enc.beginComputePass(this.tsm.createComputePassDescriptor(0, 1));
                p.setPipeline(this.dpiPipeline); p.setBindGroup(0, dpiBindGroup);
                p.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); p.end();

                p = enc.beginComputePass(this.tsm.createComputePassDescriptor(2, 3));
                p.setPipeline(this.intersectionPipeline); p.setBindGroup(0, interBindGroup);
                p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

                p = enc.beginComputePass(this.tsm.createComputePassDescriptor(4, 5));
                p.setPipeline(this.unionPipeline); p.setBindGroup(0, unionBindGroup);
                p.dispatchWorkgroups(lbDispatchX, lbDispatchY); p.end();

                this.tsm.resolve(enc);
                device.queue.submit([enc.finish()]);
                await device.queue.onSubmittedWorkDone();

                const ts = await this.tsm.downloadTimestampResult();
                if (ts.length >= 6) {
                    dpiTimes.push((ts[1] - ts[0]) / 1e6);
                    interTimes.push((ts[3] - ts[2]) / 1e6);
                    unionTimes.push((ts[5] - ts[4]) / 1e6);
                    totalTimes.push((ts[5] - ts[0]) / 1e6);
                }
            }
        }

        const avg = (a: number[]) => a.length > 0 ? a.reduce((s, v) => s + v, 0) / a.length : 0;

        // Final run for readback
        device.queue.writeBuffer(bufStateInter, 0, stateZeros);
        device.queue.writeBuffer(bufCountsInter, 0, countsZeros);
        device.queue.writeBuffer(bufStateUnion, 0, stateZeros);
        device.queue.writeBuffer(bufCountsUnion, 0, countsZeros);

        const finalEnc = device.createCommandEncoder();
        let fp = finalEnc.beginComputePass();
        fp.setPipeline(this.dpiPipeline); fp.setBindGroup(0, dpiBindGroup);
        fp.dispatchWorkgroups(dpiDispatchX, dpiDispatchY); fp.end();

        fp = finalEnc.beginComputePass();
        fp.setPipeline(this.intersectionPipeline); fp.setBindGroup(0, interBindGroup);
        fp.dispatchWorkgroups(lbDispatchX, lbDispatchY); fp.end();

        fp = finalEnc.beginComputePass();
        fp.setPipeline(this.unionPipeline); fp.setBindGroup(0, unionBindGroup);
        fp.dispatchWorkgroups(lbDispatchX, lbDispatchY); fp.end();

        const rbInter = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const rbUnion = device.createBuffer({ size: Math.max(4, numPairs * 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        finalEnc.copyBufferToBuffer(bufCountsInter, 0, rbInter, 0, numPairs * 4);
        finalEnc.copyBufferToBuffer(bufCountsUnion, 0, rbUnion, 0, numPairs * 4);
        device.queue.submit([finalEnc.finish()]);
        await device.queue.onSubmittedWorkDone();

        await rbInter.mapAsync(GPUMapMode.READ);
        const intersectionCounts = new Uint32Array(rbInter.getMappedRange().slice(0));
        rbInter.unmap();

        await rbUnion.mapAsync(GPUMapMode.READ);
        const unionCounts = new Uint32Array(rbUnion.getMappedRange().slice(0));
        rbUnion.unmap();

        // Compute Jaccard values
        const jaccardValues = new Float64Array(numPairs);
        for (let i = 0; i < numPairs; i++) {
            jaccardValues[i] = unionCounts[i] > 0 ? intersectionCounts[i] / unionCounts[i] : 0;
        }

        // Cleanup
        bufKeysA.destroy(); bufKeysB.destroy(); bufDpi.destroy();
        bufWgInfo.destroy(); bufPairIdPerWg.destroy(); bufTotalWg.destroy();
        bufStateInter.destroy(); bufCountsInter.destroy();
        bufStateUnion.destroy(); bufCountsUnion.destroy();
        rbInter.destroy(); rbUnion.destroy();

        return {
            intersectionCounts, unionCounts, jaccardValues,
            timing: { dpiMs: avg(dpiTimes), intersectionMs: avg(interTimes), unionMs: avg(unionTimes), totalMs: avg(totalTimes), totalRunsMs: totalTimes }
        };
    }

    /**
     * One run over sets that already live on the GPU. bufKeysA and bufKeysB are the sets'
     * own buffers (they may be the same buffer) and wgInfo addresses every set by its
     * element offset, so no set is copied. Creates this run's metadata, state, and count
     * buffers, records DPI once (twice with noReuse) and both lookbacks in one submission,
     * and reads the counts and pass timestamps with a single wait.
     */
    async runInPlace(
        bufKeysA: GPUBuffer,
        bufKeysB: GPUBuffer,
        wgInfo: Uint32Array,
        pairIdPerWg: Uint32Array,
        totalWg: number,
        totalDpiEntries: number,
        numPairs: number,
        noReuse: boolean = false,
    ): Promise<{
        intersectionCounts: Uint32Array;
        unionCounts: Uint32Array;
        /**
         * GPU pass times (timestamps) and host segments: createMs (buffers, metadata upload
         * calls, bind groups, encoding), waitMs (submit until counts and timestamps are
         * mapped), readMs (copy out, unmap, destroy), runMs (the whole call).
         */
        timing: {
            dpiMs: number; intersectionMs: number; unionMs: number; kernelMs: number;
            createMs: number; waitMs: number; readMs: number; runMs: number;
        };
    }> {
        const device = this.device;
        if (totalWg === 0 || numPairs === 0) {
            return {
                intersectionCounts: new Uint32Array(numPairs),
                unionCounts: new Uint32Array(numPairs),
                timing: { dpiMs: 0, intersectionMs: 0, unionMs: 0, kernelMs: 0, createMs: 0, waitMs: 0, readMs: 0, runMs: 0 }
            };
        }
        const runT0 = performance.now();

        const upload = (data: Uint32Array, usage: number): GPUBuffer => {
            const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
            return buffer;
        };
        const bufWgInfo = upload(wgInfo, GPUBufferUsage.STORAGE);
        const bufPairIdPerWg = upload(pairIdPerWg, GPUBufferUsage.STORAGE);
        const bufTotalWg = upload(new Uint32Array([totalWg]), GPUBufferUsage.UNIFORM);
        // New buffers are zero-initialized, so the lookback state and counts need no clearing.
        const bufDpi = device.createBuffer({ size: Math.max(4, totalDpiEntries * 4), usage: GPUBufferUsage.STORAGE });
        const bufStateInter = device.createBuffer({ size: totalWg * 4, usage: GPUBufferUsage.STORAGE });
        const bufStateUnion = device.createBuffer({ size: totalWg * 4, usage: GPUBufferUsage.STORAGE });
        const bufCountsInter = device.createBuffer({ size: numPairs * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufCountsUnion = device.createBuffer({ size: numPairs * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const rbInter = device.createBuffer({ size: numPairs * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const rbUnion = device.createBuffer({ size: numPairs * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const dpiBindGroup = device.createBindGroup({
            layout: this.dpiBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: bufWgInfo } },
                { binding: 4, resource: { buffer: bufTotalWg } },
            ]
        });
        const lookbackBindGroup = (state: GPUBuffer, counts: GPUBuffer) => device.createBindGroup({
            layout: this.lookbackBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufKeysA } },
                { binding: 1, resource: { buffer: bufKeysB } },
                { binding: 2, resource: { buffer: bufDpi } },
                { binding: 3, resource: { buffer: state } },
                { binding: 4, resource: { buffer: counts } },
                { binding: 5, resource: { buffer: bufWgInfo } },
                { binding: 6, resource: { buffer: bufPairIdPerWg } },
                { binding: 7, resource: { buffer: bufTotalWg } },
            ]
        });
        const interBindGroup = lookbackBindGroup(bufStateInter, bufCountsInter);
        const unionBindGroup = lookbackBindGroup(bufStateUnion, bufCountsUnion);

        const subgroupSize = (device.adapterInfo as any)?.subgroupSize || 32;
        const dpiBlocks = Math.ceil(totalWg / (DPI_WG_SIZE / subgroupSize));
        const dpiX = Math.min(dpiBlocks, MAXWORKGROUP);
        const dpiY = Math.ceil(dpiBlocks / MAXWORKGROUP);
        const lbX = Math.min(totalWg, MAXWORKGROUP);
        const lbY = Math.ceil(totalWg / MAXWORKGROUP);

        const encoder = device.createCommandEncoder();
        const pass = (pipeline: GPUComputePipeline, bindGroup: GPUBindGroup, x: number, y: number, ts: number) => {
            const p = encoder.beginComputePass(this.tsm.createComputePassDescriptor(ts, ts + 1));
            p.setPipeline(pipeline);
            p.setBindGroup(0, bindGroup);
            p.dispatchWorkgroups(x, y);
            p.end();
        };
        pass(this.dpiPipeline, dpiBindGroup, dpiX, dpiY, 0);
        pass(this.intersectionPipeline, interBindGroup, lbX, lbY, 2);
        if (noReuse) pass(this.dpiPipeline, dpiBindGroup, dpiX, dpiY, 4);
        const unionTs = noReuse ? 6 : 4;
        pass(this.unionPipeline, unionBindGroup, lbX, lbY, unionTs);
        this.tsm.resolve(encoder);
        encoder.copyBufferToBuffer(bufCountsInter, 0, rbInter, 0, numPairs * 4);
        encoder.copyBufferToBuffer(bufCountsUnion, 0, rbUnion, 0, numPairs * 4);
        const commandBuffer = encoder.finish();
        const createMs = performance.now() - runT0;

        const waitT0 = performance.now();
        device.queue.submit([commandBuffer]);
        const [, , ts] = await Promise.all([
            rbInter.mapAsync(GPUMapMode.READ),
            rbUnion.mapAsync(GPUMapMode.READ),
            this.tsm.downloadTimestampResult(),
        ]);
        const waitMs = performance.now() - waitT0;
        const readT0 = performance.now();
        const intersectionCounts = new Uint32Array(rbInter.getMappedRange().slice(0));
        const unionCounts = new Uint32Array(rbUnion.getMappedRange().slice(0));
        rbInter.unmap();
        rbUnion.unmap();

        for (const b of [bufWgInfo, bufPairIdPerWg, bufTotalWg, bufDpi, bufStateInter, bufStateUnion,
                         bufCountsInter, bufCountsUnion, rbInter, rbUnion]) {
            b.destroy();
        }
        const readMs = performance.now() - readT0;

        const span = (begin: number) => ts.length > begin + 1 ? (ts[begin + 1] - ts[begin]) / 1e6 : 0;
        const dpiMs = span(0) + (noReuse ? span(4) : 0);
        const intersectionMs = span(2);
        const unionMs = span(unionTs);
        return {
            intersectionCounts,
            unionCounts,
            timing: {
                dpiMs, intersectionMs, unionMs, kernelMs: dpiMs + intersectionMs + unionMs,
                createMs, waitMs, readMs, runMs: performance.now() - runT0
            }
        };
    }
}
