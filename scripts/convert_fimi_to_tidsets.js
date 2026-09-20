#!/usr/bin/env node
/**
 * Convert FIMI transaction database (.dat) to binary tidset format.
 *
 * Usage: node --max-old-space-size=8192 scripts/convert_fimi_to_tidsets.js fimi/webdocs.dat
 *
 * Output (same directory as input):
 *   <name>.tidsets.bin    — concatenated sorted Uint32Array tidsets
 *   <name>.offsets.bin    — Uint32Array [start0, start1, ..., startN, totalLen] per item
 *   <name>.items.bin      — Uint32Array of item IDs (sorted)
 *   <name>.tidsets.meta.json — { numTransactions, numItems, totalTidsetElements }
 *
 * The browser loads these binary files directly, bypassing text parsing.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

async function main() {
    const inputPath = process.argv[2];
    if (!inputPath) {
        console.error('Usage: node --max-old-space-size=8192 scripts/convert_fimi_to_tidsets.js <file.dat>');
        process.exit(1);
    }

    const baseName = path.basename(inputPath, '.dat');
    const outDir = path.dirname(inputPath);

    console.log(`Processing: ${inputPath}`);

    // --- Pass 1: Count transactions and collect item→tid mappings ---
    console.log('Pass 1: building tidsets...');

    const tidSets = new Map(); // item → number[]
    let numTransactions = 0;

    const rl = readline.createInterface({
        input: fs.createReadStream(inputPath, { encoding: 'utf8', highWaterMark: 64 * 1024 }),
        crlfDelay: Infinity,
    });

    for await (const line of rl) {
        const trimmed = line.trim();
        if (trimmed === '' || trimmed[0] === '#' || trimmed.startsWith('//')) continue;

        const items = trimmed.split(/\s+/);
        for (const s of items) {
            const item = parseInt(s, 10);
            if (isNaN(item)) continue;
            if (!tidSets.has(item)) tidSets.set(item, []);
            tidSets.get(item).push(numTransactions);
        }

        numTransactions++;
        if (numTransactions % 500000 === 0) {
            console.log(`  ${(numTransactions / 1000000).toFixed(1)}M transactions, ${tidSets.size.toLocaleString()} items...`);
        }
    }

    console.log(`  ${numTransactions.toLocaleString()} transactions, ${tidSets.size.toLocaleString()} unique items`);

    // --- Sort items and compute offsets ---
    console.log('Sorting items and computing offsets...');

    const sortedItems = Array.from(tidSets.keys()).sort((a, b) => a - b);
    const numItems = sortedItems.length;

    // Compute total elements and offsets
    const offsets = new Uint32Array(numItems + 1);
    let totalElements = 0;
    for (let i = 0; i < numItems; i++) {
        offsets[i] = totalElements;
        totalElements += tidSets.get(sortedItems[i]).length;
    }
    offsets[numItems] = totalElements;

    console.log(`  Total tidset elements: ${totalElements.toLocaleString()}`);

    // --- Build concatenated tidsets ---
    console.log('Building concatenated tidsets...');

    const tidsetData = new Uint32Array(totalElements);
    for (let i = 0; i < numItems; i++) {
        const tids = tidSets.get(sortedItems[i]);
        // tids are already in insertion order = sorted (transactions processed sequentially)
        const start = offsets[i];
        for (let j = 0; j < tids.length; j++) {
            tidsetData[start + j] = tids[j];
        }
    }

    // Free memory
    tidSets.clear();

    // --- Write binary files ---
    const itemsArray = new Uint32Array(sortedItems);

    const tidsetPath = path.join(outDir, `${baseName}.tidsets.bin`);
    const offsetsPath = path.join(outDir, `${baseName}.offsets.bin`);
    const itemsPath = path.join(outDir, `${baseName}.items.bin`);
    const metaPath = path.join(outDir, `${baseName}.tidsets.meta.json`);

    console.log(`Writing ${tidsetPath} (${(tidsetData.byteLength / 1024 / 1024).toFixed(1)} MB)...`);
    fs.writeFileSync(tidsetPath, Buffer.from(tidsetData.buffer));

    console.log(`Writing ${offsetsPath} (${(offsets.byteLength / 1024 / 1024).toFixed(1)} MB)...`);
    fs.writeFileSync(offsetsPath, Buffer.from(offsets.buffer));

    console.log(`Writing ${itemsPath} (${(itemsArray.byteLength / 1024 / 1024).toFixed(1)} MB)...`);
    fs.writeFileSync(itemsPath, Buffer.from(itemsArray.buffer));

    const meta = { numTransactions, numItems, totalTidsetElements: totalElements };
    fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
    console.log(`Writing ${metaPath}`);
    console.log(meta);

    console.log('Done!');
}

main().catch(err => { console.error(err); process.exit(1); });
