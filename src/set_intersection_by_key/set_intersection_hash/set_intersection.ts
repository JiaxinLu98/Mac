import { readSize } from '../../utils';
import build_hash_table from './build_hash_table.wgsl';
import hash_join from './hash_join.wgsl';
import { GPUSorter } from './radix_sort/sort';
import TimestampQueryManager from '../../TimestampQueryManager';

const LOAD_FACTOR = 0.6;
const MAXWORKGROUP = 65535;

export class GPUSetIntersection {
	buildPipeline: GPUComputePipeline;
	joinPipeline: GPUComputePipeline;
	device: GPUDevice;
	timestampQueryManager: TimestampQueryManager;
	bindGroupLayoutBuild: GPUBindGroupLayout;
	bindGroupLayoutJoin: GPUBindGroupLayout;

	uniformBufferArraySize: GPUBuffer;
  uniformBufferHashSize: GPUBuffer;

	private iterationIndex: number = 0;

	constructor(device: GPUDevice, timestampQueryManager: TimestampQueryManager) {
		this.device = device;
		this.timestampQueryManager = timestampQueryManager;

		this.bindGroupLayoutBuild = this.device.createBindGroupLayout({
			label: 'build hash table bind group layout',
			entries: [
				{
				// a_in
				binding: 0, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
				{
				// a_in_size
				binding: 1, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "uniform"}
				},
				{
				// a_in_hash_table
				binding: 2, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
				{
				// a_in_hash_table_size
				binding: 3, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "uniform"}
				},
			]
		});

		const pipelineLayoutBuild = this.device.createPipelineLayout({
			label: 'build pipeline layout',
			bindGroupLayouts: [this.bindGroupLayoutBuild]
		});

		let shader_code_build = `${build_hash_table}`;
		const shader_build = this.device.createShaderModule({
			label: 'build shader',
			code: shader_code_build
		});

		this.buildPipeline = this.device.createComputePipeline({
			label: 'build pipeline',
			layout: pipelineLayoutBuild,
			compute: {
				module: shader_build,
				entryPoint: 'build_hash_table'
			}
		});

		this.bindGroupLayoutJoin = this.device.createBindGroupLayout({
			label: 'join bind group layout',
			entries: [
				{
				// a_in_hash_table
				binding: 0, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
				{
				// a_in_hash_table_size
				binding: 1, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "read-only-storage"}
				},
				{
				// b_in
				binding: 2, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
				{
				// b_in_size
				binding: 3, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "uniform"}
				},
				{
				// result
				binding: 4, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
				{
				// result_size
				binding: 5, 
				visibility: GPUShaderStage.COMPUTE, 
				buffer: {type: "storage"}
				},
			]
		});

		const pipelineLayoutJoin = this.device.createPipelineLayout({
		label: 'join pipeline layout',
		bindGroupLayouts: [this.bindGroupLayoutJoin]
		});

		let shader_code_join = `${hash_join}`;
		const shader_join = this.device.createShaderModule({
			label: 'join shader',
			code: shader_code_join
		});

		this.joinPipeline = this.device.createComputePipeline({
			label: 'join pipeline',
			layout: pipelineLayoutJoin,
			compute: {
				module: shader_join,
				entryPoint: 'get_join_result'
			}
		});

    this.uniformBufferArraySize = device.createBuffer({
      label: 'Uniform Array Size',
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.uniformBufferHashSize = device.createBuffer({
      label: 'Uniform Hash Size',
      size: Uint32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
	}

	public setIterationIndex(i: number) {
    this.iterationIndex = i;
  }

	private getQueryBaseOffset(): number {
    const QUERIES_PER_ITER = 12;
    return this.iterationIndex * QUERIES_PER_ITER;
  }

	public async buildHashTable(
		arraySize: number,
		bufferArray: GPUBuffer): 
		Promise<{
		bufferHashTable: GPUBuffer,
		bufferHashTableSize: GPUBuffer
	}> {
		const device = this.device;
		let hash_table_size = Math.ceil(arraySize / LOAD_FACTOR);
		let exponent = Math.ceil(Math.log2(hash_table_size));
		hash_table_size = 1 << exponent;
		device.queue.writeBuffer(this.uniformBufferArraySize, 0, new Uint32Array([arraySize]));
		device.queue.writeBuffer(this.uniformBufferHashSize, 0, new Uint32Array([hash_table_size]));

		// Create GPU buffers.
		const bufferHashTable = device.createBuffer({
			label: 'hash table buffer',
			size: hash_table_size * 2 * Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		const hashInit = new Uint32Array(hash_table_size * 2).fill(0xffffffff);
		device.queue.writeBuffer(bufferHashTable, 0, hashInit);

		const bufferHashTableSize = device.createBuffer({
			label: 'hash table size buffer',
			size: Uint32Array.BYTES_PER_ELEMENT,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
		});
		device.queue.writeBuffer(bufferHashTableSize, 0, new Uint32Array([hash_table_size]));

		// Create Bind Group to link resources to the shader.
		const bindGroup = device.createBindGroup({
			layout: this.bindGroupLayoutBuild,
			entries: [
				{binding: 0, resource: {buffer: bufferArray}},
				{binding: 1, resource: {buffer: this.uniformBufferArraySize}},
				{binding: 2, resource: {buffer: bufferHashTable}},
				{binding: 3, resource: {buffer: this.uniformBufferHashSize}},
			]
		});

		// Encode and dispatch GPU command.
		const workGroup = Math.ceil(arraySize / 256);
    const groupX = Math.min(workGroup, MAXWORKGROUP);
    const groupY = Math.ceil(workGroup / groupX);

		const commandEncoder = device.createCommandEncoder();
		const base = this.getQueryBaseOffset();
		const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 0, base + 1));
		pass.setPipeline(this.buildPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(groupX, groupY);
		pass.end();
		this.timestampQueryManager.resolve(commandEncoder);
		device.queue.submit([commandEncoder.finish()]);
		await device.queue.onSubmittedWorkDone();

		// Return results.
		return {bufferHashTable, bufferHashTableSize};
	}

	public async getJoinResult(
		bufferA: GPUBuffer, 
		bufferASize: GPUBuffer, 
		bufferB: GPUBuffer, 
		bufferBSize: GPUBuffer, 
		bufferResult: GPUBuffer, 
		bufferResultSize: GPUBuffer,
		bSize: number
	) 
	{
		const device = this.device;

		// Create Bind Group to link resources to the shader.s
		const bindGroup = device.createBindGroup({
		layout: this.bindGroupLayoutJoin,
		entries: [
			{binding: 0, resource: {buffer: bufferA}},
			{binding: 1, resource: {buffer: bufferASize}},
			{binding: 2, resource: {buffer: bufferB}},
			{binding: 3, resource: {buffer: bufferBSize}},
			{binding: 4, resource: {buffer: bufferResult}},
			{binding: 5, resource: {buffer: bufferResultSize}},
		]
		});

		// Encode and dispatch GPU command.
		const workGroup = Math.ceil(bSize / 256);
    const groupX = Math.min(workGroup, MAXWORKGROUP);
    const groupY = Math.ceil(workGroup / groupX);

    const commandEncoder = device.createCommandEncoder();
		const base = this.getQueryBaseOffset();
		const pass = commandEncoder.beginComputePass(this.timestampQueryManager.createComputePassDescriptor(base + 2, base + 3));
		pass.setPipeline(this.joinPipeline);
		pass.setBindGroup(0, bindGroup);
		pass.dispatchWorkgroups(groupX, groupY);
		pass.end();
		this.timestampQueryManager.resolve(commandEncoder);
		device.queue.submit([commandEncoder.finish()]);
		await device.queue.onSubmittedWorkDone();
	} 

	public async runSortPhase( // TODO: Need to revise, sort is not correct now
		bufferResultSize: GPUBuffer
	) {
		const device = this.device;

		const sortBufferSize = await readSize(device, bufferResultSize);
		const sorter = new GPUSorter(device, 16, this.timestampQueryManager);
		const sortBuffer = sorter.createSortBuffers(sortBufferSize);
		await sorter.sort(device.queue, sortBuffer, sortBufferSize);
	}
}