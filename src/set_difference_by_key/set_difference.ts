import mark from './mark.wgsl';
import scatter from './scatter.wgsl';
import { ExclusiveScanPipeline } from './prefix_sum/exclusive_scan'; 
import { readSize } from '../utils';
import TimestampQueryManager from '../TimestampQueryManager';

const WORKGROUP_SIZE = 256;
const MAXTUPLES = 128000000;
const MAXWORKGROUP = 65535;

export class GPUSetDifferenceByKey {
    markPipeline: GPUComputePipeline;
    scatterPipeline: GPUComputePipeline;
    device: GPUDevice;
    timestampQueryManager: TimestampQueryManager;
	bindGroupLayoutMark: GPUBindGroupLayout;
    bindGroupLayoutScatter: GPUBindGroupLayout;

    private iterationIndex: number = 0;
    private queriesPerIter: number = 0;
    private numScanChunks: number = 0;
    public lastAvgSecondsTotal: number = 0;

    constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
        this.device = device;
        this.timestampQueryManager = timestampQueryManager;

        this.bindGroupLayoutMark = this.device.createBindGroupLayout({
			label: 'mark bind group layout',
			entries: [
				{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}, // setA
				{binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: {type: "read-only-storage"}}, // setB
				{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},           // flags
				{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: {type: "uniform"}},           // params
			]
		});

        const pipelineLayoutMark = this.device.createPipelineLayout({
			label: 'mark pipeline layout',
			bindGroupLayouts: [this.bindGroupLayoutMark]
		});

		let shader_code_mark = `${mark}`;
		const shader_mark = this.device.createShaderModule({
			label: 'mark shader',
			code: shader_code_mark
		});

		this.markPipeline = this.device.createComputePipeline({
			label: 'mark pipeline',
			layout: pipelineLayoutMark,
			compute: {
				module: shader_mark,
				entryPoint: 'main'
			}
		});

        this.bindGroupLayoutScatter = device.createBindGroupLayout({
        label: 'scatter bind group layout',
        entries: [
            {
            // data
            binding: 0, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "storage"}
            },
            {
            // data_size
            binding: 1, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "uniform"}
            },
            {
            // valid_flags
            binding: 2, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "storage"}
            },
            {
            // new_data
            binding: 3, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "storage"}
            },
            {
            // new_data_size
            binding: 4, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "storage"}
            },
            {
            // is_valid
            binding: 5, 
            visibility: GPUShaderStage.COMPUTE, 
            buffer: {type: "read-only-storage"}
            },
        ]
        });

        const pipelineLayoutScatter = device.createPipelineLayout({
        label: 'scatter pipeline layout',
        bindGroupLayouts: [this.bindGroupLayoutScatter]
        });

        let shader_code_scatter = `${scatter}`;
        const shader_scatter = this.device.createShaderModule({
        label: 'scatter shader',
        code: shader_code_scatter
        });

        this.scatterPipeline = device.createComputePipeline({
        label: 'scatter pipeline',
        layout: pipelineLayoutScatter,
        compute: {
            module: shader_scatter,
            entryPoint: 'main'
        }
        });
    }

    public computeQueries(lenA: number) {
        const scan = new ExclusiveScanPipeline(this.device);
        const aligned = scan.getAlignedSize(lenA);
        const maxScanSize = scan.maxScanSize;
        const numChunks = Math.ceil(aligned / maxScanSize);
        this.numScanChunks = numChunks;
        // mark(2) + scan(2) + scatter(2)
        return this.queriesPerIter = 6;
    }

    public setIterationIndex(i: number) {
        this.iterationIndex = i;
    }

    private getQueryBaseOffset(): number {
        if (this.queriesPerIter === 0) {
            throw new Error("computeChunks(lenA) must be called before using timestamps.");
        }
        return this.iterationIndex * this.queriesPerIter;
    }

    public async computeDifference(
        setA: Uint32Array,
        setB: Uint32Array,
        iters: number
    ) {
        const device = this.device;

        this.computeQueries(setA.length);
        if (this.queriesPerIter === 0) {
            throw new Error("queriesPerIter is 0; make sure computeQueries(lenA) was called.");
        }
        const QUERIES_PER_ITER = this.queriesPerIter;

        // Create GPU buffers.
		const bufferA = device.createBuffer({
			label: 'A buffer',
			size:  setA.length * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
        device.queue.writeBuffer(bufferA, 0, new Uint32Array(setA));

		const bufferB = device.createBuffer({
			label: 'B buffer',
			size:  setB.length * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(bufferB, 0, new Uint32Array(setB));

        const bufferFlags = this.device.createBuffer({
            label: 'Flags buffer',
            size:  MAXTUPLES * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(bufferFlags, 0, new Uint32Array(0));

        let lastResult: Uint32Array;
        let wallTotalMs = 0;

        let set_difference_result;

        for(let i = 0; i < iters; i++) {
            this.setIterationIndex(i);
            const t0 = performance.now();

            // API CALL: MARK
            const recordedFlags = await this.runMarkPhase(bufferA, bufferB, bufferFlags, setA.length/2, setB.length/2);
            // API CALL: SCAN
            await this.runExclusivePhase(bufferFlags, setA.length/2);
            // API CALL: SCATTER
            set_difference_result = await this.runScatterPhase(bufferA, setA.length/2, bufferFlags, recordedFlags);

            const t1 = performance.now();
            wallTotalMs += (t1 - t0);

            lastResult = set_difference_result;
            recordedFlags.destroy();
        }

       const wallAvgMs = wallTotalMs / iters;
        console.log(
            `SetDifference x${iters}:`,
            `wall total = ${(wallTotalMs / 1000).toFixed(9)} s,`,
            `wall avg = ${(wallAvgMs / 1000).toFixed(9)} s/iter`
        );

        const timestamps = await this.timestampQueryManager.downloadTimestampResult();

        let sumMark = 0;
        let sumScan = 0;
        let sumScatter = 0;

        for (let i = 0; i < iters; ++i) {
            const base = i * QUERIES_PER_ITER;

            // mark: [base + 0, base + 1]
            const markTicks = timestamps[base + 1] - timestamps[base + 0];

            // scan
            const scanFirstChunkTicks = timestamps[base + 3] - timestamps[base + 2];
            const scanTicksApprox = scanFirstChunkTicks * this.numScanChunks;

            // scatter
            const scatterTicks = timestamps[base + 5] - timestamps[base + 4];

            sumMark    += markTicks;
            sumScan    += scanTicksApprox;
            sumScatter += scatterTicks;
        }

        const avgTicks = {
            mark:    sumMark    / iters,
            scan:    sumScan    / iters,
            scatter: sumScatter / iters,
        };

        const timestampPeriod = 1e-9;

        const avgSeconds = {
            mark:    avgTicks.mark    * timestampPeriod,
            scan:    avgTicks.scan    * timestampPeriod,
            scatter: avgTicks.scatter * timestampPeriod,
        };

        const totalAvgSeconds = avgSeconds.mark + avgSeconds.scan + avgSeconds.scatter;
        this.lastAvgSecondsTotal = totalAvgSeconds;

        console.log(
            `GPU timestamps avg over ${iters} iterations:`,
            `mark = ${avgSeconds.mark.toFixed(9)} s,`,
            `scan = ${avgSeconds.scan.toFixed(9)} s,`,
            `scatter = ${avgSeconds.scatter.toFixed(9)} s,`,
            `total = ${(avgSeconds.mark + avgSeconds.scan + avgSeconds.scatter).toFixed(9)} s`
        );

        bufferA.destroy();
        bufferB.destroy();
        bufferFlags.destroy();
        
        return lastResult;
    }

    private async runMarkPhase(
        setABuffer: GPUBuffer,
        setBBuffer: GPUBuffer,
        flagsBuffer: GPUBuffer,
        lenA: number,
        lenB: number
    ) {
        const bufferRecordFlags = this.device.createBuffer({
            label: 'Buffer Record Flags',
            size:  lenA * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        })

        const params = this.device.createBuffer({
            label: 'params buffer',
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(params.getMappedRange()).set([lenA, lenB]);
        params.unmap();

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayoutMark,
            entries: [
                { binding: 0, resource: { buffer: setABuffer } },
                { binding: 1, resource: { buffer: setBBuffer } },
                { binding: 2, resource: { buffer: flagsBuffer } },
                { binding: 3, resource: { buffer: params } },
            ],
        });

        const workGroup = Math.ceil(lenA / WORKGROUP_SIZE);
        const groupX = Math.min(workGroup, MAXWORKGROUP);
        const groupY = Math.ceil(workGroup / groupX);

        const commandEncoder = this.device.createCommandEncoder();
        const base = this.getQueryBaseOffset();
		const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1));
		pass.setPipeline(this.markPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(groupX, groupY);
		pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        commandEncoder.copyBufferToBuffer(flagsBuffer, 0, bufferRecordFlags, 0, lenA * Uint32Array.BYTES_PER_ELEMENT);

		this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();
        params.destroy();

        return bufferRecordFlags;
    }

    private async runExclusivePhase(flagsBuffer: GPUBuffer, lenA: number,) {
        const scan = new ExclusiveScanPipeline(this.device);
        const aligned = scan.getAlignedSize(lenA);
        if(aligned > lenA) {
            const padCount = aligned - lenA;
            this.device.queue.writeBuffer(flagsBuffer, lenA * 4, new Uint32Array(padCount));
        }

        const numChunks = Math.ceil(aligned / scan.maxScanSize);
        this.numScanChunks = numChunks;

        const base = this.getQueryBaseOffset();
        const scanBaseIndex = base + 2;

        const scanner = scan.prepareGPUInput(flagsBuffer, aligned);
        await scanner.scan(lenA, this.timestampQueryManager, scanBaseIndex);
    }

    public async runScatterPhase(bufferASet: GPUBuffer, lenA: number, bufferFlags: GPUBuffer, recordedFlags: GPUBuffer) {
        const bufferDataSize = this.device.createBuffer({
            label: 'data size buffer',
            size:   Uint32Array.BYTES_PER_ELEMENT,
            usage:  GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(bufferDataSize.getMappedRange()).set([lenA]);
        bufferDataSize.unmap();

        const bufferNewDataSize = this.device.createBuffer({
            label: 'new data size buffer',
            size:   Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        const bufferResult = this.device.createBuffer({
            label: 'Buffer Result',
            size:  lenA * 2 * Uint32Array.BYTES_PER_ELEMENT,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        })

        const bindGroup = this.device.createBindGroup({
            layout: this.bindGroupLayoutScatter,
            entries: [
            {binding: 0, resource: {buffer: bufferASet}},
            {binding: 1, resource: {buffer: bufferDataSize}},
            {binding: 2, resource: {buffer: bufferFlags}},
            {binding: 3, resource: {buffer: bufferResult}},
            {binding: 4, resource: {buffer: bufferNewDataSize}},
            {binding: 5, resource: {buffer: recordedFlags}},
            ]
        });

        const workGroup = Math.ceil(lenA / WORKGROUP_SIZE);
        const groupX = Math.min(workGroup, MAXWORKGROUP);
        const groupY = Math.ceil(workGroup / groupX);

        const commandEncoder = this.device.createCommandEncoder();

        const base = this.getQueryBaseOffset();
        const scatterBase = base + 4;
        
		const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(scatterBase, scatterBase + 1));
		pass.setPipeline(this.scatterPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(groupX, groupY);
		pass.end();
        this.timestampQueryManager.resolve(commandEncoder);

        this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();

        // Read Back Set Difference Result
        const result_size = await readSize(this.device, bufferNewDataSize);
        const commandEncoder1 = this.device.createCommandEncoder();
        const readbackBuffer = this.device.createBuffer({
        label: 'read back set difference result',
        size:  result_size * 2 * Uint32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        commandEncoder1.copyBufferToBuffer(bufferResult, 0, readbackBuffer, 0, result_size * 2 * Uint32Array.BYTES_PER_ELEMENT);
        this.device.queue.submit([commandEncoder1.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        await readbackBuffer.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readbackBuffer.getMappedRange().slice(0));
        readbackBuffer.unmap();
        readbackBuffer.destroy();
        
        console.log(result.length/2);

        return result;
    }
    
}