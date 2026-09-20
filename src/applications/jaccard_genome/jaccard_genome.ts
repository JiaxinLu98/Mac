/**
 * Batched Jaccard Similarity on Genome k-mer Sets
 *
 * Computes pairwise Jaccard similarity between bacterial genomes represented as sorted
 * k-mer hash sets (Uint32Array).
 *
 * Every run uploads the genomes once (uploadSets) and computes the intersection and union
 * sizes of all pairs in place with WebGPUBatchedSets, so a run's end-to-end time covers the
 * upload, the GPU work, and the count readback, the same unit as the native CUDA, Thrust,
 * and C++17 stdpar versions. The DPI-reuse and no-reuse variants alternate every run so that
 * both see the same machine state. Every timed run is printed for median/percentile/std.
 *
 * Host segments of a run: upload (uploadSets), prep (workgroup metadata), create (buffers,
 * bind groups, encoding), errorScope (error-scope round trips), wait (submit until the
 * results are mapped), read (copy out, unmap, destroy), dispose (release the set handles).
 * uploadSets only queues the writes, so the transfer itself lands in wait. With bd=1 the
 * run waits for the queue after uploadSets, which moves the transfer into upload. bd=1 is
 * for the breakdown only and adds a synchronization, so formal runs use bd=0.
 *
 * URL: ?app=jaccard&w=3&n=10[&bd=1]
 */

import { WebGPUBatchedSets } from '../../webgpu_batched_sets';

// --- Load k-mer binary files ---

async function loadKmerSet(url: string): Promise<Uint32Array> {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}`);
    const buf = await resp.arrayBuffer();
    return new Uint32Array(buf);
}

// --- CPU reference sizes ---

function countsCPU(a: Uint32Array, b: Uint32Array): { intersection: number; union: number } {
    let i = 0, j = 0, inter = 0;
    while (i < a.length && j < b.length) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { inter++; i++; j++; }
    }
    return { intersection: inter, union: a.length + b.length - inter };
}

// --- Genome definitions ---

const GENOMES = [
    { name: 'E. coli K-12', file: 'genomes/ecoli_k12.kmers.bin' },
    { name: 'Salmonella LT2', file: 'genomes/salmonella_lt2.kmers.bin' },
    { name: 'B. subtilis', file: 'genomes/bacillus_subtilis.kmers.bin' },
    { name: 'P. aeruginosa', file: 'genomes/pseudomonas_aeruginosa.kmers.bin' },
    { name: 'S. aureus', file: 'genomes/staphylococcus_aureus.kmers.bin' },
    { name: 'M. tuberculosis', file: 'genomes/mycobacterium_tb.kmers.bin' },
    { name: 'S. pneumoniae', file: 'genomes/streptococcus_pneumoniae.kmers.bin' },
    { name: 'H. pylori', file: 'genomes/helicobacter_pylori.kmers.bin' },
    { name: 'V. cholerae', file: 'genomes/vibrio_cholerae.kmers.bin' },
    { name: 'C. difficile', file: 'genomes/clostridioides_difficile.kmers.bin' },
];

interface RunRecord {
    e2eMs: number;
    kernelMs: number;
    dpiMs: number;
    intersectionMs: number;
    unionMs: number;
    uploadMs: number;
    prepMs: number;
    createMs: number;
    errorScopeMs: number;
    waitMs: number;
    readMs: number;
    disposeMs: number;
}

// "median, mean (std, min, max, n)" over per-run times.
function formatStats(values: number[]): string {
    const n = values.length;
    if (n === 0) return 'no timed runs';
    const sorted = [...values].sort((x, y) => x - y);
    const median = n % 2 ? sorted[(n - 1) / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    const mean = values.reduce((s, v) => s + v, 0) / n;
    const std = n > 1 ? Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n - 1)) : 0;
    return `median ${median.toFixed(3)} ms, mean ${mean.toFixed(3)} ms ` +
        `(std ${std.toFixed(3)}, min ${sorted[0].toFixed(3)}, max ${sorted[n - 1].toFixed(3)}, n=${n})`;
}

export async function runJaccardGenomeTest(device: GPUDevice): Promise<void> {
    const params = new URLSearchParams(typeof location === 'undefined' ? '' : location.search);
    const numWarmup = parseInt(params.get('w') ?? '3', 10);
    const numRuns = parseInt(params.get('n') ?? '10', 10);
    const breakdown = params.get('bd') === '1';

    console.log('\n========================================================================');
    console.log('  JACCARD SIMILARITY on Genome k-mer Sets (Batched, sets read in place)');
    console.log('  DPI once + Lookback twice (intersection + union) per batch');
    console.log('========================================================================\n');

    const batchedSets = new WebGPUBatchedSets(device);

    console.log('  Loading genome k-mer sets...');
    const datas: Uint32Array[] = [];
    for (const g of GENOMES) {
        try {
            const data = await loadKmerSet(`./${g.file}`);
            datas.push(data);
            console.log(`    ${g.name}: ${data.length.toLocaleString()} k-mers`);
        } catch {
            console.log(`    ${g.name}: not found (${g.file})`);
        }
    }
    if (datas.length < 2) {
        console.log('  Need at least 2 genomes. Run: bash scripts/download_genomes.sh\n');
        return;
    }

    const pairIdx: Array<[number, number]> = [];
    for (let i = 0; i < datas.length; i++) {
        for (let j = i + 1; j < datas.length; j++) pairIdx.push([i, j]);
    }
    const reference = pairIdx.map(([i, j]) => countsCPU(datas[i], datas[j]));
    const arenaMB = datas.reduce((s, d) => s + d.byteLength, 0) / 1048576;
    console.log(`\n  ${datas.length} genomes, ${pairIdx.length} pairs, ${arenaMB.toFixed(1)} MB of sets`);
    console.log(`  ${numWarmup} warmup + ${numRuns} runs per variant\n`);

    let chunkCount = 0;
    const runOnce = async (noReuse: boolean): Promise<{ record: RunRecord; pass: boolean }> => {
        const t0 = performance.now();
        const handles = batchedSets.uploadSets(datas);
        if (breakdown) await device.queue.onSubmittedWorkDone();
        const uploadMs = performance.now() - t0;
        const result = await batchedSets.batchedIntersectionUnionCountsInPlace(
            pairIdx.map(([i, j]) => ({ a: handles[i], b: handles[j] })), { noReuse });
        const disposeT0 = performance.now();
        for (const h of handles) h.dispose();
        const disposeMs = performance.now() - disposeT0;
        const e2eMs = performance.now() - t0;

        chunkCount = result.chunkCount;
        const pass = reference.every((r, p) =>
            result.intersectionCounts[p] === r.intersection && result.unionCounts[p] === r.union);
        const tm = result.timing;
        return {
            record: {
                e2eMs,
                kernelMs: tm.kernelMs,
                dpiMs: tm.dpiMs,
                intersectionMs: tm.intersectionMs,
                unionMs: tm.unionMs,
                uploadMs,
                prepMs: tm.batchPrepMs,
                createMs: tm.createMs,
                errorScopeMs: tm.errorScopeMs,
                waitMs: tm.waitMs,
                readMs: tm.readMs,
                disposeMs,
            },
            pass
        };
    };

    const variants = ['reuse', 'noreuse'] as const;
    const timed: Record<string, RunRecord[]> = { reuse: [], noreuse: [] };
    const passed: Record<string, boolean> = { reuse: true, noreuse: true };
    for (let r = 0; r < numWarmup + numRuns; r++) {
        for (let k = 0; k < 2; k++) {
            const variant = variants[(r + k) % 2];  // alternate which variant runs first
            const { record, pass } = await runOnce(variant === 'noreuse');
            passed[variant] = passed[variant] && pass;
            if (r >= numWarmup) timed[variant].push(record);
        }
    }

    for (const variant of variants) {
        const runs = timed[variant];
        const column = (f: keyof RunRecord) => runs.map(x => x[f]);
        console.log(`\n  webgpu [${variant}]: ${passed[variant] ? 'ALL PASS' : 'MISMATCH'}, chunks=${chunkCount}`);
        console.log(`    e2e     ${formatStats(column('e2eMs'))}`);
        console.log(`    kernel  ${formatStats(column('kernelMs'))}`);
        console.log(`    dpi     ${formatStats(column('dpiMs'))}`);
        console.log(`    inter   ${formatStats(column('intersectionMs'))}`);
        console.log(`    union   ${formatStats(column('unionMs'))}`);
        const segments: Array<[string, keyof RunRecord]> = [
            ['upload', 'uploadMs'], ['prep', 'prepMs'], ['create', 'createMs'], ['errscope', 'errorScopeMs'],
            ['wait', 'waitMs'], ['read', 'readMs'], ['dispose', 'disposeMs'],
        ];
        for (const [label, key] of segments) console.log(`    ${label.padEnd(9)}${formatStats(column(key))}`);
        console.log(`[jaccard-result] ${JSON.stringify({
            impl: 'webgpu', variant, warmup: numWarmup, runs: runs.length, pass: passed[variant], chunks: chunkCount,
            breakdown_sync: breakdown,
            e2e_ms: column('e2eMs'), kernel_ms: column('kernelMs'), dpi_ms: column('dpiMs'),
            intersection_ms: column('intersectionMs'), union_ms: column('unionMs'),
            upload_ms: column('uploadMs'), prep_ms: column('prepMs'), create_ms: column('createMs'),
            errscope_ms: column('errorScopeMs'), wait_ms: column('waitMs'), read_ms: column('readMs'),
            dispose_ms: column('disposeMs'),
        })}`);
    }
    console.log('');
}
