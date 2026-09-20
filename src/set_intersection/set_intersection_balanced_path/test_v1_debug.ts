/**
 * Debug test for V1 Set Intersection
 *
 * Test with small arrays to verify each step:
 * 1. DPI computation (balanced path)
 * 2. Count phase
 */

import computeDiagonalsShader from './balanced_path_biased.wgsl';
import countShader from './set_availability_intersection_count_v1.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan';

const STAR_MASK = 0x80000000;
const INDEX_MASK = 0x7FFFFFFF;

export async function debugV1Intersection(device: GPUDevice) {
    console.log('\n========== DEBUG V1 INTERSECTION ==========\n');

    // Small test case
    // A = [1, 3, 3, 5, 7, 9]
    // B = [2, 3, 3, 6, 7, 8]
    // Expected intersection: [3, 3, 7] -> count = 3
    const A = new Uint32Array([1, 3, 3, 5, 7, 9]);
    const B = new Uint32Array([2, 3, 3, 6, 7, 8]);

    console.log('A:', Array.from(A));
    console.log('B:', Array.from(B));
    console.log('Expected intersection: [3, 3, 7], count = 3');

    // CPU reference
    let cpuCount = 0;
    let ai = 0, bi = 0;
    while (ai < A.length && bi < B.length) {
        if (A[ai] < B[bi]) {
            ai++;
        } else if (A[ai] > B[bi]) {
            bi++;
        } else {
            cpuCount++;
            ai++;
            bi++;
        }
    }
    console.log('CPU merge-path count:', cpuCount);

    // Create buffers
    const bufferA = device.createBuffer({
        size: A.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferA, 0, A);

    const bufferB = device.createBuffer({
        size: B.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferB, 0, B);

    // Use partition size that creates just 1 workgroup
    const partitionSize = 1792;  // NV
    const total = A.length + B.length;
    const numWg = Math.ceil(total / partitionSize);
    console.log(`\nPartition size: ${partitionSize}, Total: ${total}, NumWg: ${numWg}`);

    // ============ Step 1: Compute Diagonals ============
    console.log('\n--- Step 1: Compute Diagonals (DPI) ---');

    const dpiSize = 2 * (numWg + 1);
    const bufferDPI = device.createBuffer({
        size: dpiSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayoutDiag = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
    });

    const diagPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutDiag] }),
        compute: {
            module: device.createShaderModule({ code: computeDiagonalsShader }),
            entryPoint: 'compute_diagonals'
        }
    });

    const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferALen, 0, new Uint32Array([A.length]));
    device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([B.length]));
    device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

    const diagBindGroup = device.createBindGroup({
        layout: bindGroupLayoutDiag,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferALen } },
            { binding: 4, resource: { buffer: bufferBLen } },
            { binding: 5, resource: { buffer: bufferNumWg } },
        ]
    });

    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(diagPipeline);
    pass.setBindGroup(0, diagBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read DPI
    const dpiReadback = device.createBuffer({ size: dpiSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferDPI, 0, dpiReadback, 0, dpiSize * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await dpiReadback.mapAsync(GPUMapMode.READ);
    const dpiData = new Uint32Array(dpiReadback.getMappedRange().slice(0));
    dpiReadback.unmap();

    console.log('DPI raw data:', Array.from(dpiData));

    for (let i = 0; i <= numWg; i++) {
        const packedA = dpiData[i];
        const aIdx = packedA & INDEX_MASK;
        const star = (packedA & STAR_MASK) !== 0;
        const bIdx = dpiData[numWg + 1 + i];
        console.log(`  Partition ${i}: aIdx=${aIdx}, bIdx=${bIdx}, star=${star}`);
    }

    // ============ Step 2: Count Phase ============
    console.log('\n--- Step 2: Count Phase ---');

    const scan = new ExclusiveScanPipeline(device);
    const alignedSize = scan.getAlignedSize(numWg);

    const bufferCounts = device.createBuffer({
        size: alignedSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayoutCount = device.createBindGroupLayout({
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

    const countPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutCount] }),
        compute: {
            module: device.createShaderModule({ code: countShader }),
            entryPoint: 'count_availability'
        }
    });

    const bufferNumWgTotal = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWg]));

    const countBindGroup = device.createBindGroup({
        layout: bindGroupLayoutCount,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferCounts } },
            { binding: 4, resource: { buffer: bufferALen } },
            { binding: 5, resource: { buffer: bufferBLen } },
            { binding: 6, resource: { buffer: bufferNumWgTotal } },
        ]
    });

    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(countPipeline);
    pass.setBindGroup(0, countBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read counts
    const countsReadback = device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferCounts, 0, countsReadback, 0, numWg * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await countsReadback.mapAsync(GPUMapMode.READ);
    const countsData = new Uint32Array(countsReadback.getMappedRange().slice(0));
    countsReadback.unmap();

    let totalGpuCount = 0;
    for (let i = 0; i < numWg; i++) {
        console.log(`  Workgroup ${i} count: ${countsData[i]}`);
        totalGpuCount += countsData[i];
    }

    console.log(`\nTotal GPU count: ${totalGpuCount}`);
    console.log(`Expected count: ${cpuCount}`);
    console.log(`Match: ${totalGpuCount === cpuCount ? '✔ YES' : '✗ NO'}`);

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferDPI.destroy();
    bufferCounts.destroy();
    bufferALen.destroy();
    bufferBLen.destroy();
    bufferNumWg.destroy();
    bufferNumWgTotal.destroy();
    dpiReadback.destroy();
    countsReadback.destroy();

    console.log('\n========== END DEBUG ==========\n');

    return totalGpuCount === cpuCount;
}

/**
 * Debug with larger dataset that requires multiple workgroups
 */
export async function debugV1IntersectionLarge(device: GPUDevice) {
    console.log('\n========== DEBUG V1 INTERSECTION (LARGE) ==========\n');

    // Create larger arrays that span multiple workgroups
    // NV = 1792, so we need more than 1792 elements total
    const size = 2000;  // This will create ~2 workgroups
    const A = new Uint32Array(size);
    const B = new Uint32Array(size);

    // Fill with values that have some overlap
    // A: [0, 2, 4, 6, 8, ...] (even numbers)
    // B: [0, 3, 6, 9, 12, ...] (multiples of 3)
    // Intersection: multiples of 6 within range
    for (let i = 0; i < size; i++) {
        A[i] = i * 2;  // 0, 2, 4, 6, 8, ...
        B[i] = i * 3;  // 0, 3, 6, 9, 12, ...
    }

    console.log(`A: [${A[0]}, ${A[1]}, ${A[2]}, ... ${A[size-1]}] (size=${size})`);
    console.log(`B: [${B[0]}, ${B[1]}, ${B[2]}, ... ${B[size-1]}] (size=${size})`);

    // CPU reference - merge path count
    let cpuCount = 0;
    let ai = 0, bi = 0;
    while (ai < A.length && bi < B.length) {
        if (A[ai] < B[bi]) {
            ai++;
        } else if (A[ai] > B[bi]) {
            bi++;
        } else {
            cpuCount++;
            ai++;
            bi++;
        }
    }
    console.log('CPU merge-path count:', cpuCount);

    // Create buffers
    const bufferA = device.createBuffer({
        size: A.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferA, 0, A);

    const bufferB = device.createBuffer({
        size: B.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferB, 0, B);

    // Use default partition size
    const partitionSize = 1792;  // NV
    const total = A.length + B.length;
    const numWg = Math.ceil(total / partitionSize);
    console.log(`\nPartition size: ${partitionSize}, Total: ${total}, NumWg: ${numWg}`);

    // ============ Step 1: Compute Diagonals ============
    console.log('\n--- Step 1: Compute Diagonals (DPI) ---');

    const dpiSize = 2 * (numWg + 1);
    const bufferDPI = device.createBuffer({
        size: dpiSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayoutDiag = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
    });

    const diagPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutDiag] }),
        compute: {
            module: device.createShaderModule({ code: computeDiagonalsShader }),
            entryPoint: 'compute_diagonals'
        }
    });

    const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferALen, 0, new Uint32Array([A.length]));
    device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([B.length]));
    device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

    const diagBindGroup = device.createBindGroup({
        layout: bindGroupLayoutDiag,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferALen } },
            { binding: 4, resource: { buffer: bufferBLen } },
            { binding: 5, resource: { buffer: bufferNumWg } },
        ]
    });

    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(diagPipeline);
    pass.setBindGroup(0, diagBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read DPI
    const dpiReadback = device.createBuffer({ size: dpiSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferDPI, 0, dpiReadback, 0, dpiSize * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await dpiReadback.mapAsync(GPUMapMode.READ);
    const dpiData = new Uint32Array(dpiReadback.getMappedRange().slice(0));
    dpiReadback.unmap();

    for (let i = 0; i <= numWg; i++) {
        const packedA = dpiData[i];
        const aIdx = packedA & INDEX_MASK;
        const star = (packedA & STAR_MASK) !== 0;
        const bIdx = dpiData[numWg + 1 + i];
        const diag = Math.floor((i * total) / numWg);
        console.log(`  Partition ${i}: diag=${diag}, aIdx=${aIdx}, bIdx=${bIdx}, star=${star}, A[aIdx]=${aIdx < A.length ? A[aIdx] : 'END'}, B[bIdx]=${bIdx < B.length ? B[bIdx] : 'END'}`);
    }

    // Verify partitions cover all elements
    let totalACount = 0, totalBCount = 0;
    for (let i = 0; i < numWg; i++) {
        const a0 = dpiData[i] & INDEX_MASK;
        const a1 = dpiData[i + 1] & INDEX_MASK;
        const b0 = dpiData[numWg + 1 + i];
        const b1 = dpiData[numWg + 1 + i + 1];
        totalACount += (a1 - a0);
        totalBCount += (b1 - b0);
        console.log(`  WG ${i}: A[${a0}..${a1}) (${a1-a0} elements), B[${b0}..${b1}) (${b1-b0} elements)`);
    }
    console.log(`  Total A elements in partitions: ${totalACount} (expected ${A.length})`);
    console.log(`  Total B elements in partitions: ${totalBCount} (expected ${B.length})`);

    // ============ Step 2: Count Phase ============
    console.log('\n--- Step 2: Count Phase ---');

    const scan = new ExclusiveScanPipeline(device);
    const alignedSize = scan.getAlignedSize(numWg);

    const bufferCounts = device.createBuffer({
        size: alignedSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayoutCount = device.createBindGroupLayout({
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

    const countPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutCount] }),
        compute: {
            module: device.createShaderModule({ code: countShader }),
            entryPoint: 'count_availability'
        }
    });

    const bufferNumWgTotal = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWg]));

    const countBindGroup = device.createBindGroup({
        layout: bindGroupLayoutCount,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferCounts } },
            { binding: 4, resource: { buffer: bufferALen } },
            { binding: 5, resource: { buffer: bufferBLen } },
            { binding: 6, resource: { buffer: bufferNumWgTotal } },
        ]
    });

    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(countPipeline);
    pass.setBindGroup(0, countBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read counts
    const countsReadback = device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferCounts, 0, countsReadback, 0, numWg * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await countsReadback.mapAsync(GPUMapMode.READ);
    const countsData = new Uint32Array(countsReadback.getMappedRange().slice(0));
    countsReadback.unmap();

    // Calculate expected count per partition using CPU
    const expectedCounts: number[] = [];
    for (let wg = 0; wg < numWg; wg++) {
        const a0 = dpiData[wg] & INDEX_MASK;
        const a1 = dpiData[wg + 1] & INDEX_MASK;
        const star0 = (dpiData[wg] & STAR_MASK) !== 0;
        const star1 = (dpiData[wg + 1] & STAR_MASK) !== 0;
        const b0 = dpiData[numWg + 1 + wg] + (star0 ? 1 : 0);
        const b1 = dpiData[numWg + 1 + wg + 1] + (star1 ? 1 : 0);

        // CPU count for this partition
        let partitionCount = 0;
        let ai = a0, bi = b0;
        while (ai < a1 && bi < b1) {
            if (A[ai] < B[bi]) {
                ai++;
            } else if (A[ai] > B[bi]) {
                bi++;
            } else {
                partitionCount++;
                ai++;
                bi++;
            }
        }
        expectedCounts.push(partitionCount);
    }

    let totalGpuCount = 0;
    let totalExpected = 0;
    for (let i = 0; i < numWg; i++) {
        const match = countsData[i] === expectedCounts[i] ? '✔' : '✗';
        console.log(`  Workgroup ${i}: GPU count=${countsData[i]}, Expected=${expectedCounts[i]} ${match}`);
        totalGpuCount += countsData[i];
        totalExpected += expectedCounts[i];
    }

    console.log(`\nTotal GPU count: ${totalGpuCount}`);
    console.log(`Total expected (per-partition sum): ${totalExpected}`);
    console.log(`CPU merge-path count: ${cpuCount}`);
    console.log(`Match: ${totalGpuCount === cpuCount ? '✔ YES' : '✗ NO'}`);

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferDPI.destroy();
    bufferCounts.destroy();
    bufferALen.destroy();
    bufferBLen.destroy();
    bufferNumWg.destroy();
    bufferNumWgTotal.destroy();
    dpiReadback.destroy();
    countsReadback.destroy();

    console.log('\n========== END DEBUG (LARGE) ==========\n');

    return totalGpuCount === cpuCount;
}

/**
 * Debug with dataset that triggers star bit (duplicates at partition boundaries)
 */
export async function debugV1IntersectionStarBit(device: GPUDevice) {
    console.log('\n========== DEBUG V1 INTERSECTION (STAR BIT) ==========\n');

    // Create arrays with many duplicates to trigger star bit
    // Star bit is triggered when a value spans across partition boundaries
    // We need total > 1792 and duplicates at the boundary

    const size = 2000;
    const A = new Uint32Array(size);
    const B = new Uint32Array(size);

    // Fill with values that have lots of duplicates
    // A: many repeated values - each value repeats 10 times
    // B: similar pattern
    // This ensures duplicates will span partition boundaries
    for (let i = 0; i < size; i++) {
        A[i] = Math.floor(i / 10) * 3;  // 0,0,0,0,0,0,0,0,0,0, 3,3,3,..., 6,6,6,...
        B[i] = Math.floor(i / 8) * 3;   // 0,0,0,0,0,0,0,0, 3,3,3,3,3,3,3,3, 6,6,...
    }

    // Sort to ensure sorted order (should already be sorted but just in case)
    A.sort((a, b) => a - b);
    B.sort((a, b) => a - b);

    console.log(`A first 30: [${Array.from(A.slice(0, 30)).join(', ')}]`);
    console.log(`A last 10: [${Array.from(A.slice(-10)).join(', ')}]`);
    console.log(`B first 30: [${Array.from(B.slice(0, 30)).join(', ')}]`);
    console.log(`B last 10: [${Array.from(B.slice(-10)).join(', ')}]`);

    // CPU reference - merge path count
    let cpuCount = 0;
    let ai = 0, bi = 0;
    const cpuMatches: number[] = [];
    while (ai < A.length && bi < B.length) {
        if (A[ai] < B[bi]) {
            ai++;
        } else if (A[ai] > B[bi]) {
            bi++;
        } else {
            cpuMatches.push(A[ai]);
            cpuCount++;
            ai++;
            bi++;
        }
    }
    console.log(`CPU merge-path count: ${cpuCount}`);
    console.log(`First 20 matches: [${cpuMatches.slice(0, 20).join(', ')}]`);

    // Create buffers
    const bufferA = device.createBuffer({
        size: A.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferA, 0, A);

    const bufferB = device.createBuffer({
        size: B.length * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(bufferB, 0, B);

    // Use default partition size
    const partitionSize = 1792;  // NV
    const total = A.length + B.length;
    const numWg = Math.ceil(total / partitionSize);
    console.log(`\nPartition size: ${partitionSize}, Total: ${total}, NumWg: ${numWg}`);

    // ============ Step 1: Compute Diagonals ============
    console.log('\n--- Step 1: Compute Diagonals (DPI) ---');

    const dpiSize = 2 * (numWg + 1);
    const bufferDPI = device.createBuffer({
        size: dpiSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const bindGroupLayoutDiag = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ]
    });

    const diagPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutDiag] }),
        compute: {
            module: device.createShaderModule({ code: computeDiagonalsShader }),
            entryPoint: 'compute_diagonals'
        }
    });

    const bufferALen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferBLen = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bufferNumWg = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferALen, 0, new Uint32Array([A.length]));
    device.queue.writeBuffer(bufferBLen, 0, new Uint32Array([B.length]));
    device.queue.writeBuffer(bufferNumWg, 0, new Uint32Array([numWg]));

    const diagBindGroup = device.createBindGroup({
        layout: bindGroupLayoutDiag,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferALen } },
            { binding: 4, resource: { buffer: bufferBLen } },
            { binding: 5, resource: { buffer: bufferNumWg } },
        ]
    });

    let encoder = device.createCommandEncoder();
    let pass = encoder.beginComputePass();
    pass.setPipeline(diagPipeline);
    pass.setBindGroup(0, diagBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read DPI
    const dpiReadback = device.createBuffer({ size: dpiSize * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferDPI, 0, dpiReadback, 0, dpiSize * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await dpiReadback.mapAsync(GPUMapMode.READ);
    const dpiData = new Uint32Array(dpiReadback.getMappedRange().slice(0));
    dpiReadback.unmap();

    let starCount = 0;
    for (let i = 0; i <= numWg; i++) {
        const packedA = dpiData[i];
        const aIdx = packedA & INDEX_MASK;
        const star = (packedA & STAR_MASK) !== 0;
        const bIdx = dpiData[numWg + 1 + i];
        const diag = Math.floor((i * total) / numWg);
        if (star) starCount++;
        console.log(`  Partition ${i}: diag=${diag}, aIdx=${aIdx}, bIdx=${bIdx}, star=${star}, A[aIdx]=${aIdx < A.length ? A[aIdx] : 'END'}, B[bIdx]=${bIdx < B.length ? B[bIdx] : 'END'}`);
    }
    console.log(`  Star bits triggered: ${starCount}`);

    // Show partition details
    for (let i = 0; i < numWg; i++) {
        const a0 = dpiData[i] & INDEX_MASK;
        const a1 = dpiData[i + 1] & INDEX_MASK;
        const star0 = (dpiData[i] & STAR_MASK) !== 0;
        const star1 = (dpiData[i + 1] & STAR_MASK) !== 0;
        const b0_raw = dpiData[numWg + 1 + i];
        const b1_raw = dpiData[numWg + 1 + i + 1];
        const b0 = b0_raw + (star0 ? 1 : 0);
        const b1 = b1_raw + (star1 ? 1 : 0);

        console.log(`  WG ${i}: A[${a0}..${a1}) (${a1-a0} els), B[${b0_raw}+${star0?1:0}..${b1_raw}+${star1?1:0}) = B[${b0}..${b1}) (${b1-b0} els)`);
        if (a0 < A.length && a1 > 0) {
            console.log(`         A values: [${A[a0]}, ..., ${A[Math.min(a1-1, A.length-1)]}]`);
        }
        if (b0 < B.length && b1 > 0) {
            console.log(`         B values: [${B[b0]}, ..., ${B[Math.min(b1-1, B.length-1)]}]`);
        }
    }

    // ============ Step 2: Count Phase ============
    console.log('\n--- Step 2: Count Phase ---');

    const scan = new ExclusiveScanPipeline(device);
    const alignedSize = scan.getAlignedSize(numWg);

    const bufferCounts = device.createBuffer({
        size: alignedSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });

    const bindGroupLayoutCount = device.createBindGroupLayout({
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

    const countPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayoutCount] }),
        compute: {
            module: device.createShaderModule({ code: countShader }),
            entryPoint: 'count_availability'
        }
    });

    const bufferNumWgTotal = device.createBuffer({ size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bufferNumWgTotal, 0, new Uint32Array([numWg]));

    const countBindGroup = device.createBindGroup({
        layout: bindGroupLayoutCount,
        entries: [
            { binding: 0, resource: { buffer: bufferA } },
            { binding: 1, resource: { buffer: bufferB } },
            { binding: 2, resource: { buffer: bufferDPI } },
            { binding: 3, resource: { buffer: bufferCounts } },
            { binding: 4, resource: { buffer: bufferALen } },
            { binding: 5, resource: { buffer: bufferBLen } },
            { binding: 6, resource: { buffer: bufferNumWgTotal } },
        ]
    });

    encoder = device.createCommandEncoder();
    pass = encoder.beginComputePass();
    pass.setPipeline(countPipeline);
    pass.setBindGroup(0, countBindGroup);
    pass.dispatchWorkgroups(numWg);
    pass.end();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Read counts
    const countsReadback = device.createBuffer({ size: numWg * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(bufferCounts, 0, countsReadback, 0, numWg * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await countsReadback.mapAsync(GPUMapMode.READ);
    const countsData = new Uint32Array(countsReadback.getMappedRange().slice(0));
    countsReadback.unmap();

    // Calculate expected count per partition using CPU
    const expectedCounts: number[] = [];
    for (let wg = 0; wg < numWg; wg++) {
        const a0 = dpiData[wg] & INDEX_MASK;
        const a1 = dpiData[wg + 1] & INDEX_MASK;
        const star0 = (dpiData[wg] & STAR_MASK) !== 0;
        const star1 = (dpiData[wg + 1] & STAR_MASK) !== 0;
        const b0 = dpiData[numWg + 1 + wg] + (star0 ? 1 : 0);
        const b1 = dpiData[numWg + 1 + wg + 1] + (star1 ? 1 : 0);

        // CPU count for this partition
        let partitionCount = 0;
        let ai = a0, bi = b0;
        while (ai < a1 && bi < b1) {
            if (A[ai] < B[bi]) {
                ai++;
            } else if (A[ai] > B[bi]) {
                bi++;
            } else {
                partitionCount++;
                ai++;
                bi++;
            }
        }
        expectedCounts.push(partitionCount);
    }

    let totalGpuCount = 0;
    let totalExpected = 0;
    for (let i = 0; i < numWg; i++) {
        const match = countsData[i] === expectedCounts[i] ? '✔' : '✗';
        console.log(`  Workgroup ${i}: GPU count=${countsData[i]}, Expected=${expectedCounts[i]} ${match}`);
        totalGpuCount += countsData[i];
        totalExpected += expectedCounts[i];
    }

    console.log(`\nTotal GPU count: ${totalGpuCount}`);
    console.log(`Total expected (per-partition sum): ${totalExpected}`);
    console.log(`CPU merge-path count: ${cpuCount}`);
    console.log(`Match: ${totalGpuCount === cpuCount ? '✔ YES' : '✗ NO'}`);

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferDPI.destroy();
    bufferCounts.destroy();
    bufferALen.destroy();
    bufferBLen.destroy();
    bufferNumWg.destroy();
    bufferNumWgTotal.destroy();
    dpiReadback.destroy();
    countsReadback.destroy();

    console.log('\n========== END DEBUG (STAR BIT) ==========\n');

    return totalGpuCount === cpuCount;
}
