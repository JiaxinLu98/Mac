import merge_path from './merge.wgsl';
import TimestampQueryManager from '../TimestampQueryManager';

export class GPUMerger {
    PADDING: number = 1024;
    NUM_WORKGROUPS: number = 128;
    computeDiagonalsPipeline: GPUComputePipeline;
    mergeSinglePathPipeline: GPUComputePipeline;
    device: GPUDevice;
    timestampQueryManager: TimestampQueryManager;
    bindGroupLayout: GPUBindGroupLayout;

    // Reusable GPU buffers
    uniformBufferALength: GPUBuffer;
    uniformBufferBLength: GPUBuffer;
    uniformBufferCLength: GPUBuffer;

    public lastAvgSecondsTotal: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;
        this.bindGroupLayout = this.device.createBindGroupLayout({
            label: 'merge path bind group layout',
            entries: [
                { 
                    binding: 0, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'read-only-storage' } 
                },
                { 
                    binding: 1, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'read-only-storage' } 
                },
                { 
                    binding: 2, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'storage' } 
                },
                { 
                    binding: 3, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'storage' } 
                },
                { 
                    binding: 4, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'uniform' } 
                },
                { 
                    binding: 5, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'uniform' } 
                },
                { 
                    binding: 6, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'uniform' } 
                },
            ]
        });

        const pipelineLayout = this.device.createPipelineLayout({
            label: 'merge path pipeline layout',
            bindGroupLayouts: [this.bindGroupLayout]
        });

        let shader_code = `${merge_path}`;
        const shader = this.device.createShaderModule({
            label: 'merge path shader',
            code: shader_code
        });

        this.computeDiagonalsPipeline = this.device.createComputePipeline({
            label: 'merge path compute diagonals pipeline',
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint: 'compute_diagonals'
            }
        });

        this.mergeSinglePathPipeline = this.device.createComputePipeline({
            label: 'merge path merge single path pipeline',
            layout: pipelineLayout,
            compute: {
                module: shader,
                entryPoint:'merge_single_path'
            }
        });

        this.uniformBufferALength = device.createBuffer({
            label: 'Uniform A Length',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.uniformBufferBLength = device.createBuffer({
            label: 'Uniform B Length',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.uniformBufferCLength = device.createBuffer({
            label: 'Uniform C Length',
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    public async merge(a_in: Uint32Array, a_len: number, b_in: Uint32Array, b_len: number, iters: number) {
        const device = this.device;
        const QUERIES_PER_ITER = 4; 

        // Inputs are expected to be pre-sorted and flattened arrays.
        const a_sorted = a_in;
        const b_sorted = b_in;

        const a_in_len = a_sorted.length;
        if(a_len > a_in_len) {
            throw new Error(
            `Merge input error: Requested merge length for A (${a_len}) ` +
            `exceeds the actual input array size (${a_in_len}). ` +
            `'a_len' must be less than or equal to 'a_in.length'.`
            );
        }

        const b_in_len = b_sorted.length;
        if (b_len > b_in_len) {
            throw new Error(
                `Merge input error: Requested merge length for B (${b_len}) ` +
                `exceeds the actual input array size (${b_in_len}). ` +
                `'b_len' must be less than or equal to 'b_in.length'.`
            );
        }

        const c_len = a_len + b_len;
        const POSITIVE_INFINITY = 0xFFFFFFFF;

        // 2. Pad the arrays with positive infinity.
        // This is a requirement of the merge_path algorithm to handle edge cases.
        const a_padded_len = a_len + this.PADDING;
        const b_padded_len = b_len + this.PADDING;

        const a_padded_data = new Uint32Array(a_padded_len);
        a_padded_data.set(a_sorted);
        for (let i = a_len; i < a_padded_data.length; i++) {
            a_padded_data[i] = POSITIVE_INFINITY;
        }

        const b_padded_data = new Uint32Array(b_padded_len);
        b_padded_data.set(b_sorted);
        for (let i = b_len; i < b_padded_data.length; i++) {
            b_padded_data[i] = POSITIVE_INFINITY;
        }

        // 3. Create GPU Buffers for data and uniforms.
        const bufferA = device.createBuffer({
            label: 'Buffer A',
            size: a_padded_data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferA, 0, a_padded_data);

        const bufferB = device.createBuffer({
            label: 'Buffer B',
            size: b_padded_data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferB, 0, b_padded_data);

        const bufferC = device.createBuffer({
            label: 'Buffer C (Output)',
            size: c_len * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferC, 0, new Uint32Array(c_len));

        const dpi_size = 2 * (this.NUM_WORKGROUPS + 1);
        const bufferDPI = device.createBuffer({
            label: 'Buffer DPI',
            size: dpi_size * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferDPI, 0, new Uint32Array(dpi_size));

        device.queue.writeBuffer(this.uniformBufferALength, 0, new Uint32Array([a_len]));
        device.queue.writeBuffer(this.uniformBufferBLength, 0, new Uint32Array([b_len]));
        device.queue.writeBuffer(this.uniformBufferCLength, 0, new Uint32Array([c_len]));

        // 4. Create Bind Group to link resources to the shader.
        const bindGroup = device.createBindGroup({
            layout: this.bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: bufferA } },
                { binding: 1, resource: { buffer: bufferB } },
                { binding: 2, resource: { buffer: bufferC } },
                { binding: 3, resource: { buffer: bufferDPI } },
                { binding: 4, resource: { buffer: this.uniformBufferALength } },
                { binding: 5, resource: { buffer: this.uniformBufferBLength } },
                { binding: 6, resource: { buffer: this.uniformBufferCLength } },
            ],
        });

        let wallTotalMs = 0;

        for(let i = 0; i < iters; i++) {
            // 5. Encode and dispatch GPU commands.
            // Pass 1: Compute diagonal intersections to partition the work.

            const base = i * QUERIES_PER_ITER;

            const wallStart = performance.now();

            const commandEncoder1 = device.createCommandEncoder();
            const pass1 = commandEncoder1.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1));
            pass1.setPipeline(this.computeDiagonalsPipeline);
            pass1.setBindGroup(0, bindGroup);
            pass1.dispatchWorkgroups(this.NUM_WORKGROUPS);
            pass1.end();

            // Pass 2: Merge the partitions in parallel.
            const pass2 = commandEncoder1.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 2, base + 3));
            pass2.setPipeline(this.mergeSinglePathPipeline);
            pass2.setBindGroup(0, bindGroup);
            pass2.dispatchWorkgroups(this.NUM_WORKGROUPS);
            pass2.end();

            this.timestampQueryManager.resolve(commandEncoder1);
            device.queue.submit([commandEncoder1.finish()]);
            await device.queue.onSubmittedWorkDone();

            const wallEnd = performance.now();
            wallTotalMs += (wallEnd - wallStart);
        }  
        
        const wallAvgMs = wallTotalMs / iters;
        console.log(
            `MergePath x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(6)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(6)} s/iter`
        );

        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumPartitionTicks = 0;
        let sumMergeTicks = 0;

        for (let i = 0; i < iters; i++) {
            const base = i * QUERIES_PER_ITER;

            const partTicks  = timestamps[base + 1] - timestamps[base + 0];
            const mergeTicks = timestamps[base + 3] - timestamps[base + 2];

            sumPartitionTicks += partTicks;
            sumMergeTicks     += mergeTicks;
        }

        const avgPartitionTicks = sumPartitionTicks / iters;
        const avgMergeTicks     = sumMergeTicks     / iters;

        const timestampPeriod = 1e-9;

        const avgPartitionSeconds = avgPartitionTicks * timestampPeriod;
        const avgMergeSeconds     = avgMergeTicks     * timestampPeriod;
        const avgTotalSeconds     = avgPartitionSeconds + avgMergeSeconds;

        this.lastAvgSecondsTotal = avgTotalSeconds;

        console.log(
            `GPU timestamps avg over ${iters} iterations:`,
            `partition = ${avgPartitionSeconds.toFixed(9)} s,`,
            `merge = ${avgMergeSeconds.toFixed(9)} s,`,
            `total = ${avgTotalSeconds.toFixed(9)} s`
        );

        // 6. Copy the result from the GPU back to a readable buffer on the CPU.
        const commandEncoder = device.createCommandEncoder();
        const readbackBuffer = device.createBuffer({
            size: bufferC.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        commandEncoder.copyBufferToBuffer(bufferC, 0, readbackBuffer, 0, bufferC.size);
        device.queue.submit([commandEncoder.finish()]);
        await device.queue.onSubmittedWorkDone();

        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const resultData = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();

        // Release GPU resources to allow re-entry into the run function.
        bufferA.destroy();
        bufferB.destroy();
        bufferC.destroy();
        bufferDPI.destroy();
        readbackBuffer.destroy();

        // Return result as a standard JavaScript array.
        return resultData;
    }
}