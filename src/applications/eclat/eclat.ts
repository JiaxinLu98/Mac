/**
 * ECLAT v2 — GPU intersection with full output (no CPU intersection)
 *
 * Improvement over v1: uses WebGPUBatchedSets which writes intersection
 * elements to GPU output buffers while hiding batch preparation and chunking.
 *
 * Still reads back pairCounts (small) for min_support filtering on CPU.
 */

import { GPUSetHandle, WebGPUBatchedSets } from '../../webgpu_batched_sets';

const ECLAT_MAX_CHUNK_GPU_BYTES = 64 * 1024 * 1024;

// Reuse data loading and CPU validation from eclat.ts
// (These are duplicated here to keep eclat.ts untouched)

function parseTransactionDB(text: string): number[][] {
    const transactions: number[][] = [];
    for (const line of text.split('\n')) {
        const trimmed = line.trim();
        if (trimmed === '' || trimmed.startsWith('#') || trimmed.startsWith('//')) continue;
        const items = trimmed.split(/\s+/).map(s => parseInt(s, 10)).filter(n => !isNaN(n));
        if (items.length > 0) transactions.push(items);
    }
    return transactions;
}

async function loadBinaryTidsets(prefix: string): Promise<Map<number, Uint32Array>> {
    const [metaResp, itemsResp, offsetsResp, tidsetsResp] = await Promise.all([
        fetch(`./${prefix}.tidsets.meta.json`),
        fetch(`./${prefix}.items.bin`),
        fetch(`./${prefix}.offsets.bin`),
        fetch(`./${prefix}.tidsets.bin`),
    ]);
    if (!metaResp.ok || !itemsResp.ok || !offsetsResp.ok || !tidsetsResp.ok) {
        throw new Error(`Binary tidset files not found for ${prefix}. Run: node scripts/convert_fimi_to_tidsets.js`);
    }
    const meta = await metaResp.json();
    const items = new Uint32Array(await itemsResp.arrayBuffer());
    const offsets = new Uint32Array(await offsetsResp.arrayBuffer());
    const tidsetData = new Uint32Array(await tidsetsResp.arrayBuffer());

    const result = new Map<number, Uint32Array>();
    for (let i = 0; i < items.length; i++) {
        result.set(items[i], tidsetData.subarray(offsets[i], offsets[i + 1]));
    }
    console.log(`        binary tidsets loaded: ${meta.numItems.toLocaleString()} items, ${meta.totalTidsetElements.toLocaleString()} total elements`);
    return result;
}

function buildVerticalFormat(transactions: number[][]): Map<number, Uint32Array> {
    const tidSets = new Map<number, number[]>();
    for (let tid = 0; tid < transactions.length; tid++) {
        for (const item of transactions[tid]) {
            if (!tidSets.has(item)) tidSets.set(item, []);
            tidSets.get(item)!.push(tid);
        }
    }
    const result = new Map<number, Uint32Array>();
    for (const [item, tids] of tidSets) {
        result.set(item, new Uint32Array(tids.sort((a, b) => a - b)));
    }
    return result;
}

function intersectSorted(a: Uint32Array, b: Uint32Array): Uint32Array {
    const result: number[] = [];
    let i = 0, j = 0;
    while (i < a.length && j < b.length) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { result.push(a[i]); i++; j++; }
    }
    return new Uint32Array(result);
}

function eclatCPU(tidsets: Map<number, Uint32Array>, minSupport: number): Map<string, number> {
    const frequent = new Map<string, number>();
    const frequentItems: number[] = [];
    for (const [item, tidset] of tidsets) {
        if (tidset.length >= minSupport) {
            frequent.set(JSON.stringify([item]), tidset.length);
            frequentItems.push(item);
        }
    }
    frequentItems.sort((a, b) => a - b);
    function mine(prefix: number[], prefixTidset: Uint32Array, candidates: number[]) {
        for (let i = 0; i < candidates.length; i++) {
            const item = candidates[i];
            const newTidset = intersectSorted(prefixTidset, tidsets.get(item)!);
            if (newTidset.length >= minSupport) {
                const newItemset = [...prefix, item];
                frequent.set(JSON.stringify(newItemset), newTidset.length);
                mine(newItemset, newTidset, candidates.slice(i + 1));
            }
        }
    }
    for (let i = 0; i < frequentItems.length; i++) {
        mine([frequentItems[i]], tidsets.get(frequentItems[i])!, frequentItems.slice(i + 1));
    }
    return frequent;
}

// --- ECLAT v2 GPU miner ---

interface LevelItem {
    itemset: number[];
    tidset: GPUSetHandle;
}

export interface EclatV2Timing {
    totalDpiMs: number;
    totalLookbackMs: number;
    totalGpuKernelMs: number;
    totalBatchPrepMs: number;
    totalCandidateGenMs: number;
    endToEndMs: number;
}

async function eclatGPUv2(
    batchedSets: WebGPUBatchedSets,
    tidsets: Map<number, Uint32Array>,
    minSupport: number
): Promise<{ frequent: Map<string, number>; timing: EclatV2Timing }> {
    const frequent = new Map<string, number>();
    const baseTidsetHandles = new Map<number, GPUSetHandle>();

    let totalDpiMs = 0, totalLookbackMs = 0, totalGpuKernelMs = 0;
    let totalBatchPrepMs = 0, totalCandidateGenMs = 0;
    const endToEndT0 = performance.now();

    // Level-1
    const frequentItems: number[] = [];
    for (const [item, tidset] of tidsets) {
        if (tidset.length >= minSupport) {
            frequent.set(JSON.stringify([item]), tidset.length);
            frequentItems.push(item);
            baseTidsetHandles.set(item, batchedSets.uploadSet(tidset));
        }
    }
    frequentItems.sort((a, b) => a - b);
    console.log(`    Level 1: ${frequentItems.length} frequent items`);

    let currentLevel: LevelItem[] = frequentItems.map(item => ({
        itemset: [item],
        tidset: baseTidsetHandles.get(item)!
    }));

    let level = 2;

    try {
        while (currentLevel.length > 0) {
            const candidateT0 = performance.now();
            const pairs: Array<{ tidsetA: GPUSetHandle; tidsetB: GPUSetHandle; parentIdx: number; childItem: number }> = [];

            for (let i = 0; i < currentLevel.length; i++) {
                const parent = currentLevel[i];
                const maxItem = parent.itemset[parent.itemset.length - 1];
                for (const item of frequentItems) {
                    if (item <= maxItem) continue;
                    pairs.push({ tidsetA: parent.tidset, tidsetB: baseTidsetHandles.get(item)!, parentIdx: i, childItem: item });
                }
            }
            totalCandidateGenMs += performance.now() - candidateT0;

            if (pairs.length === 0) break;

            console.log(`    Level ${level}: ${pairs.length} candidate pairs from ${currentLevel.length} parents`);

            const result = await batchedSets.batchedIntersectionWithOutputGPU(
                pairs.map(p => ({ a: p.tidsetA, b: p.tidsetB })),
                { maxChunkGpuBytes: ECLAT_MAX_CHUNK_GPU_BYTES }
            );

            totalBatchPrepMs += result.timing.batchPrepMs;
            totalDpiMs += result.timing.dpiMs;
            totalLookbackMs += result.timing.lookbackMs;
            totalGpuKernelMs += result.timing.totalMs;

            const nextLevel: LevelItem[] = [];
            let frequentCount = 0;

            for (let pairId = 0; pairId < pairs.length; pairId++) {
                const support = result.pairCounts[pairId];
                if (support >= minSupport) {
                    const parent = currentLevel[pairs[pairId].parentIdx];
                    const newItemset = [...parent.itemset, pairs[pairId].childItem];
                    frequent.set(JSON.stringify(newItemset), support);
                    frequentCount++;
                    nextLevel.push({ itemset: newItemset, tidset: result.outputHandles[pairId] });
                } else {
                    result.outputHandles[pairId].dispose();
                }
            }

            for (const item of currentLevel) {
                if (item.itemset.length > 1) {
                    item.tidset.dispose();
                }
            }

            console.log(`    Level ${level}: ${frequentCount} frequent itemsets, ${result.chunkCount} chunks`);
            currentLevel = nextLevel;
            level++;
        }
    } finally {
        for (const item of currentLevel) {
            if (item.itemset.length > 1) {
                item.tidset.dispose();
            }
        }

        for (const handle of baseTidsetHandles.values()) {
            handle.dispose();
        }
    }

    const endToEndMs = performance.now() - endToEndT0;

    return {
        frequent,
        timing: { totalDpiMs, totalLookbackMs, totalGpuKernelMs, totalBatchPrepMs, totalCandidateGenMs, endToEndMs }
    };
}

// --- ECLAT on the in-place batched path ---

export interface EclatInPlaceTiming extends EclatV2Timing {
    chunkCount: number;
    submitCount: number;
    // Diagnostic breakdown of the end-to-end time (in-place path only).
    setupMs?: number;
    batchedCallMs?: number;
    encodeMs?: number;
    waitMs?: number;
    handlesMs?: number;
    resultsMs?: number;
    pairMapMs?: number;
    readbackMs?: number;
    errorScopeMs?: number;
    loopOtherMs?: number;
}

// Itemsets with the same classId share every item but the last (an equivalence class).
interface InPlaceLevelItem extends LevelItem {
    classId: number;
}

async function eclatGPUInPlace(
    batchedSets: WebGPUBatchedSets,
    tidsets: Map<number, Uint32Array>,
    minSupport: number,
    verbose: boolean
): Promise<{ frequent: Map<string, number>; timing: EclatInPlaceTiming }> {
    const frequent = new Map<string, number>();
    const baseTidsetHandles = new Map<number, GPUSetHandle>();

    let totalDpiMs = 0, totalLookbackMs = 0, totalGpuKernelMs = 0;
    let totalBatchPrepMs = 0, totalCandidateGenMs = 0;
    let chunkCount = 0, submitCount = 0;
    let setupMs = 0, batchedCallMs = 0, encodeMs = 0, waitMs = 0, handlesMs = 0, resultsMs = 0;
    let pairMapMs = 0, readbackMs = 0, errorScopeMs = 0, loopOtherMs = 0;
    const endToEndT0 = performance.now();

    // Level-1: all frequent base tidsets share arena buffers and are read in place.
    const frequentItems: number[] = [];
    const frequentTidsets: Uint32Array[] = [];
    for (const [item, tidset] of tidsets) {
        if (tidset.length >= minSupport) {
            frequent.set(JSON.stringify([item]), tidset.length);
            frequentItems.push(item);
            frequentTidsets.push(tidset);
        }
    }
    const uploaded = batchedSets.uploadSets(frequentTidsets);
    frequentItems.forEach((item, i) => baseTidsetHandles.set(item, uploaded[i]));
    frequentItems.sort((a, b) => a - b);
    if (verbose) console.log(`    Level 1: ${frequentItems.length} frequent items`);

    let currentLevel: InPlaceLevelItem[] = frequentItems.map(item => ({
        itemset: [item],
        tidset: baseTidsetHandles.get(item)!,
        classId: 0
    }));
    setupMs = performance.now() - endToEndT0;

    let level = 2;

    try {
        while (currentLevel.length > 0) {
            const candidateT0 = performance.now();
            const pairs: Array<{ tidsetA: GPUSetHandle; tidsetB: GPUSetHandle; parentIdx: number; childItem: number }> = [];

            // Textbook ECLAT: intersect every two itemsets of one equivalence class. Members of a
            // class are contiguous and sorted by their last item.
            for (let i = 0; i < currentLevel.length; i++) {
                const parent = currentLevel[i];
                for (let j = i + 1; j < currentLevel.length && currentLevel[j].classId === parent.classId; j++) {
                    const sibling = currentLevel[j];
                    pairs.push({
                        tidsetA: parent.tidset,
                        tidsetB: sibling.tidset,
                        parentIdx: i,
                        childItem: sibling.itemset[sibling.itemset.length - 1]
                    });
                }
            }
            totalCandidateGenMs += performance.now() - candidateT0;

            if (pairs.length === 0) break;

            if (verbose) console.log(`    Level ${level}: ${pairs.length} candidate pairs from ${currentLevel.length} parents`);

            const pairMapT0 = performance.now();
            const handlePairs = pairs.map(p => ({ a: p.tidsetA, b: p.tidsetB }));
            pairMapMs += performance.now() - pairMapT0;
            const batchedT0 = performance.now();
            const result = await batchedSets.batchedIntersectionWithOutputGPUInPlace(handlePairs);
            batchedCallMs += performance.now() - batchedT0;
            encodeMs += result.timing.encodeMs ?? 0;
            waitMs += result.timing.waitMs ?? 0;
            handlesMs += result.timing.handlesMs ?? 0;
            readbackMs += result.timing.readbackMs ?? 0;
            errorScopeMs += result.timing.errorScopeMs ?? 0;
            loopOtherMs += result.timing.loopOtherMs ?? 0;

            totalBatchPrepMs += result.timing.batchPrepMs;
            totalDpiMs += result.timing.dpiMs;
            totalLookbackMs += result.timing.lookbackMs;
            totalGpuKernelMs += result.timing.totalMs;
            chunkCount += result.chunkCount;
            submitCount += result.submitCount;
            const resultsT0 = performance.now();

            const nextLevel: InPlaceLevelItem[] = [];
            let frequentCount = 0;

            for (let pairId = 0; pairId < pairs.length; pairId++) {
                const support = result.pairCounts[pairId];
                if (support >= minSupport) {
                    const parent = currentLevel[pairs[pairId].parentIdx];
                    const newItemset = [...parent.itemset, pairs[pairId].childItem];
                    frequent.set(JSON.stringify(newItemset), support);
                    frequentCount++;
                    nextLevel.push({ itemset: newItemset, tidset: result.outputHandles[pairId], classId: pairs[pairId].parentIdx });
                } else {
                    result.outputHandles[pairId].dispose();
                }
            }

            for (const item of currentLevel) {
                if (item.itemset.length > 1) {
                    item.tidset.dispose();
                }
            }

            resultsMs += performance.now() - resultsT0;
            if (verbose) {
                console.log(`    Level ${level}: ${frequentCount} frequent itemsets, ${result.chunkCount} chunks, ${result.submitCount} submits`);
            }
            currentLevel = nextLevel;
            level++;
        }
    } finally {
        for (const item of currentLevel) {
            if (item.itemset.length > 1) {
                item.tidset.dispose();
            }
        }

        for (const handle of baseTidsetHandles.values()) {
            handle.dispose();
        }
    }

    const endToEndMs = performance.now() - endToEndT0;
    // Like the CUDA and Thrust baselines, pooled buffers are freed after the timed region.
    batchedSets.releaseOutputPool();

    return {
        frequent,
        timing: {
            totalDpiMs, totalLookbackMs, totalGpuKernelMs, totalBatchPrepMs, totalCandidateGenMs, endToEndMs,
            chunkCount, submitCount, setupMs, batchedCallMs, encodeMs, waitMs, handlesMs, resultsMs,
            pairMapMs, readbackMs, errorScopeMs, loopOtherMs
        }
    };
}

const ECLAT_DATASETS: { [name: string]: { file: string; binaryPrefix?: string; minSupport: number } } = {
    mushroom: { file: 'fimi/mushroom.dat', minSupport: 1000 },
    chess: { file: 'fimi/chess.dat', minSupport: 2000 },
    kosarak: { file: 'fimi/kosarak.dat', minSupport: 5000 },
    accidents: { file: 'fimi/accidents.dat', minSupport: 300000 },
    webdocs: { file: 'fimi/webdocs.dat', binaryPrefix: 'fimi/webdocs', minSupport: 500000 },
};

async function loadEclatDataset(ds: { file: string; binaryPrefix?: string }): Promise<Map<number, Uint32Array>> {
    if (ds.binaryPrefix) {
        return loadBinaryTidsets(ds.binaryPrefix);
    }
    const resp = await fetch(`./${ds.file}`);
    if (!resp.ok) throw new Error(`${ds.file} not found`);
    return buildVerticalFormat(parseTransactionDB(await resp.text()));
}

function summarize(values: number[]): { mean: number; std: number; min: number; max: number } {
    const mean = values.reduce((s, v) => s + v, 0) / values.length;
    const variance = values.length > 1
        ? values.reduce((s, v) => s + (v - mean) * (v - mean), 0) / (values.length - 1)
        : 0;
    return { mean, std: Math.sqrt(variance), min: Math.min(...values), max: Math.max(...values) };
}

function sameFrequentItemsets(a: Map<string, number>, b: Map<string, number>): boolean {
    if (a.size !== b.size) return false;
    for (const [itemset, support] of a) {
        if (b.get(itemset) !== support) return false;
    }
    return true;
}

/**
 * ECLAT benchmark, selected with ?app=eclat&ds=chess,kosarak,webdocs&impl=inplace,v2&w=3&n=10.
 * Every implementation is validated against the CPU miner. Results are logged as
 * "[eclat-result] {json}" lines, followed by "[eclat-done]".
 */
export async function runEclatInPlaceTest(device: GPUDevice, params: URLSearchParams): Promise<void> {
    const datasetNames = (params.get('ds') || 'chess,kosarak,webdocs').split(',');
    const impls = (params.get('impl') || 'inplace').split(',');
    const numWarmup = parseInt(params.get('w') || '3', 10);
    const numIterations = Math.max(1, parseInt(params.get('n') || '10', 10));

    const adapterInfo = (device as any).adapterInfo;
    console.log(`[eclat] adapter: ${adapterInfo?.vendor} / ${adapterInfo?.architecture} / ${adapterInfo?.description}`);
    console.log(`[eclat] maxBufferSize=${device.limits.maxBufferSize}, maxStorageBufferBindingSize=${device.limits.maxStorageBufferBindingSize}, timestamp-query=${device.features.has('timestamp-query')}`);
    device.lost.then(info => console.error('GPU device lost:', info.message, info.reason));
    // Per-submission error scopes are on by default (paper results). ?scopes=0 turns them off (diagnostic A/B only).
    const inPlaceErrorScopes = params.get('scopes') !== '0';
    console.log(`[eclat] per-submission error scopes: ${inPlaceErrorScopes}`);
    const batchedSets = new WebGPUBatchedSets(device, { inPlaceErrorScopes });

    for (const name of datasetNames) {
        const ds = ECLAT_DATASETS[name];
        if (ds === undefined) {
            console.log(`[eclat] unknown dataset: ${name}`);
            continue;
        }
        try {
            const loadT0 = performance.now();
            const tidsets = await loadEclatDataset(ds);
            const preprocessMs = performance.now() - loadT0;

            const cpuT0 = performance.now();
            const cpuFrequent = eclatCPU(tidsets, ds.minSupport);
            const cpuMs = performance.now() - cpuT0;
            console.log(`[eclat] ${name}: ${tidsets.size} items, min_support=${ds.minSupport}, preprocess ${preprocessMs.toFixed(0)} ms, CPU reference ${cpuFrequent.size} itemsets in ${cpuMs.toFixed(0)} ms`);

            for (const impl of impls) {
                if (impl !== 'inplace' && impl !== 'v2') {
                    console.log(`[eclat] unknown impl: ${impl}`);
                    continue;
                }
                const runOnce = async (verbose: boolean): Promise<{ frequent: Map<string, number>; timing: EclatInPlaceTiming }> => {
                    if (impl === 'inplace') {
                        return eclatGPUInPlace(batchedSets, tidsets, ds.minSupport, verbose);
                    }
                    const r = await eclatGPUv2(batchedSets, tidsets, ds.minSupport);
                    return { frequent: r.frequent, timing: { ...r.timing, chunkCount: -1, submitCount: -1 } };
                };

                for (let w = 0; w < numWarmup; w++) {
                    await runOnce(w === 0);
                }
                const timings: EclatInPlaceTiming[] = [];
                let lastFrequent = new Map<string, number>();
                for (let i = 0; i < numIterations; i++) {
                    const r = await runOnce(numWarmup === 0 && i === 0);
                    timings.push(r.timing);
                    lastFrequent = r.frequent;
                }

                const match = sameFrequentItemsets(lastFrequent, cpuFrequent);
                const e2e = summarize(timings.map(t => t.endToEndMs));
                const kernel = summarize(timings.map(t => t.totalGpuKernelMs));
                const mean = (f: (t: EclatInPlaceTiming) => number) => summarize(timings.map(f)).mean;
                const last = timings[timings.length - 1];

                console.log(`  ${name} [${impl}] (${numIterations} runs): match=${match ? 'PASS' : 'FAIL'}, chunks=${last.chunkCount}, submits=${last.submitCount}`);
                console.log(`    DPI ${mean(t => t.totalDpiMs).toFixed(2)} ms, Lookback ${mean(t => t.totalLookbackMs).toFixed(2)} ms, kernel ${kernel.mean.toFixed(2)} ms (std ${kernel.std.toFixed(2)})`);
                console.log(`    batch prep ${mean(t => t.totalBatchPrepMs).toFixed(1)} ms, candidate gen ${mean(t => t.totalCandidateGenMs).toFixed(1)} ms`);
                console.log(`    breakdown: setup ${mean(t => t.setupMs ?? 0).toFixed(1)} ms, batched call ${mean(t => t.batchedCallMs ?? 0).toFixed(1)} ms ` +
                    `(encode ${mean(t => t.encodeMs ?? 0).toFixed(1)}, submit+wait ${mean(t => t.waitMs ?? 0).toFixed(1)}, handles ${mean(t => t.handlesMs ?? 0).toFixed(1)}), ` +
                    `results ${mean(t => t.resultsMs ?? 0).toFixed(1)} ms`);
                console.log(`    inside batched call: prep ${mean(t => t.totalBatchPrepMs).toFixed(1)}, encode ${mean(t => t.encodeMs ?? 0).toFixed(1)}, ` +
                    `submit+wait ${mean(t => t.waitMs ?? 0).toFixed(1)}, readback ${mean(t => t.readbackMs ?? 0).toFixed(1)}, ` +
                    `error scopes ${mean(t => t.errorScopeMs ?? 0).toFixed(1)}, handles ${mean(t => t.handlesMs ?? 0).toFixed(1)}, ` +
                    `loop other ${mean(t => t.loopOtherMs ?? 0).toFixed(1)}, ` +
                    `unattributed ${mean(t => (t.batchedCallMs ?? 0) - t.totalBatchPrepMs - (t.encodeMs ?? 0) - (t.waitMs ?? 0) - (t.readbackMs ?? 0) - (t.errorScopeMs ?? 0) - (t.handlesMs ?? 0) - (t.loopOtherMs ?? 0)).toFixed(1)} ms` +
                    ` | pair list ${mean(t => t.pairMapMs ?? 0).toFixed(1)} ms (outside the call)`);
                console.log(`    end-to-end ${e2e.mean.toFixed(1)} ms (std ${e2e.std.toFixed(1)}, min ${e2e.min.toFixed(1)}, max ${e2e.max.toFixed(1)})`);
                console.log(`[eclat-result] ${JSON.stringify({
                    dataset: name, impl, match, itemsets: lastFrequent.size, cpuItemsets: cpuFrequent.size,
                    chunks: last.chunkCount, submits: last.submitCount,
                    dpiMs: mean(t => t.totalDpiMs), lookbackMs: mean(t => t.totalLookbackMs),
                    kernelMs: kernel, e2eMs: e2e,
                    batchPrepMs: mean(t => t.totalBatchPrepMs), candidateGenMs: mean(t => t.totalCandidateGenMs),
                    breakdownMs: {
                        setup: mean(t => t.setupMs ?? 0), batchedCall: mean(t => t.batchedCallMs ?? 0),
                        encode: mean(t => t.encodeMs ?? 0), submitWait: mean(t => t.waitMs ?? 0),
                        handles: mean(t => t.handlesMs ?? 0), results: mean(t => t.resultsMs ?? 0),
                    },
                    e2eRuns: timings.map(t => t.endToEndMs),
                })}`);
            }
        } catch (error) {
            console.log(`[eclat] ${name}: error: ${error}`);
        }
    }
    console.log('[eclat-done]');
}

// --- Test runner ---

export async function runEclatV2Test(device: GPUDevice): Promise<void> {
    console.log('\n========================================================================');
    console.log('  ECLAT v2 — GPU Full Output (no CPU intersection)');
    console.log('  Batched DPI + Lookback with element scatter');
    console.log('========================================================================\n');

    device.lost.then(info => console.error('GPU device lost:', info.message, info.reason));
    const batchedSets = new WebGPUBatchedSets(device);

    const datasets: Array<{ name: string; file: string; binaryPrefix?: string; minSupport: number }> = [
        // { name: 'mushroom', file: 'fimi/mushroom.dat', minSupport: 1000 },
        // { name: 'chess', file: 'fimi/chess.dat', minSupport: 2000 },
        // { name: 'kosarak', file: 'fimi/kosarak.dat', minSupport: 5000 },
        // { name: 'accidents', file: 'fimi/accidents.dat', minSupport: 300000 },
        { name: 'webdocs', file: 'fimi/webdocs.dat', binaryPrefix: 'fimi/webdocs', minSupport: 500000 },
    ];

    for (const ds of datasets) {
        try {
            let tidsets: Map<number, Uint32Array>;
            let preprocessMs: number;

            if (ds.binaryPrefix) {
                console.log(`  ${ds.name}: loading binary tidsets (${ds.binaryPrefix})...`);
                const loadT0 = performance.now();
                tidsets = await loadBinaryTidsets(ds.binaryPrefix);
                preprocessMs = performance.now() - loadT0;
            } else {
                const resp = await fetch(`./${ds.file}`);
                if (!resp.ok) { console.log(`  ${ds.name}: not found`); continue; }

                console.log(`  ${ds.name}: loading...`);
                const text = await resp.text();

                const parseT0 = performance.now();
                const transactions = parseTransactionDB(text);
                const parseMs = performance.now() - parseT0;

                const vertT0 = performance.now();
                tidsets = buildVerticalFormat(transactions);
                const vertMs = performance.now() - vertT0;
                console.log(`  ${ds.name}: ${transactions.length.toLocaleString()} transactions, ${tidsets.size.toLocaleString()} items`);
                preprocessMs = parseMs + vertMs;
            }

            // Warmup + timed runs
            const NUM_WARMUP = 0;
            const NUM_ITERATIONS = 1;
            console.log(`  ${ds.name}: GPU ECLAT v2 (min_support=${ds.minSupport}, ${NUM_WARMUP} warmup + ${NUM_ITERATIONS} iterations)...`);

            for (let w = 0; w < NUM_WARMUP; w++) {
                await eclatGPUv2(batchedSets, tidsets, ds.minSupport);
            }

            const timings: EclatV2Timing[] = [];
            let gpuResult: { frequent: Map<string, number>; timing: EclatV2Timing } | null = null;
            for (let iter = 0; iter < NUM_ITERATIONS; iter++) {
                gpuResult = await eclatGPUv2(batchedSets, tidsets, ds.minSupport);
                timings.push(gpuResult.timing);
            }

            const avg = (fn: (t: EclatV2Timing) => number) => timings.reduce((s, t) => s + fn(t), 0) / NUM_ITERATIONS;

            console.log(`  ${ds.name}: RESULTS (avg of ${NUM_ITERATIONS} runs)`);
            console.log(`    Frequent itemsets: ${gpuResult!.frequent.size.toLocaleString()}`);
            console.log('    --- Preprocessing ---');
            console.log(`    Parse + Vertical:  ${preprocessMs.toFixed(1)} ms`);
            console.log('    --- GPU Kernel (TimestampQuery, avg) ---');
            console.log(`    DPI total:         ${avg(t => t.totalDpiMs).toFixed(3)} ms`);
            console.log(`    Lookback total:    ${avg(t => t.totalLookbackMs).toFixed(3)} ms`);
            console.log(`    GPU Kernel total:  ${avg(t => t.totalGpuKernelMs).toFixed(3)} ms`);
            console.log('    --- CPU Overhead (avg) ---');
            console.log(`    Candidate gen:     ${avg(t => t.totalCandidateGenMs).toFixed(1)} ms`);
            console.log(`    Batch preparation: ${avg(t => t.totalBatchPrepMs).toFixed(1)} ms`);
            console.log('    --- Summary ---');
            console.log(`    End-to-end (v2, avg): ${avg(t => t.endToEndMs).toFixed(1)} ms`);
            console.log('');

        } catch (error) {
            console.log(`  ${ds.name}: error: ${error}`);
        }
    }
    console.log('  ECLAT v2 tests complete.\n');
}
