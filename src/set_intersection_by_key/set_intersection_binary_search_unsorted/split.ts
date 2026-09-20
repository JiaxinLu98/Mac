import split_pairs from './split_pairs.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const MAXWORKGROUP = 65535;

export class GPUSplit {
  splitPipeline: GPUComputePipeline;
  device: GPUDevice;
  bindGroupLayout: GPUBindGroupLayout;
  timestampQueryManager: TimestampQueryManager | null;

  constructor(device: GPUDevice, timestampQueryManager?: TimestampQueryManager) {
    this.device = device;
    this.timestampQueryManager = timestampQueryManager ?? null;

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'split pairs bind group layout',
      entries: [
        {
          // pairs
          binding: 0, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "read-only-storage"}
        },
        {
          // keys
          binding: 1, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "storage"}
        },
        {
          // vals
          binding: 2, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "storage"}
        },
        {
          // params
          binding: 3, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "uniform"}
        },
      ]
    });

    const pipelineLayout = device.createPipelineLayout({
      label: 'split pairs pipeline layout',
      bindGroupLayouts: [this.bindGroupLayout]
    });

    const shader_code = `${split_pairs}`;
    const shader = device.createShaderModule({
      label: 'split pairs shader',
      code: shader_code
    });

    this.splitPipeline = device.createComputePipeline({
      label: 'split pairs pipeline',
      layout: pipelineLayout,
      compute: {
        module: shader,
        entryPoint: 'split_pairs'
      }
    });
  }

  public async splitPairs(
    bufferPairs: GPUBuffer,
    count: number,
    bufferKeys: GPUBuffer,
    bufferVals: GPUBuffer,
    keyIndex: 0 | 1,
    baseIndex?: number,
  ) {
    const device = this.device;
    if(count === 0) return;

    // Create GPU buffers.
    const params = device.createBuffer({
      label: 'split pairs params',
      size: 8,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(params, 0, new Uint32Array([count, keyIndex]));

    // Create Bind Group to link resources to the shader.
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: bufferPairs } },
        { binding: 1, resource: { buffer: bufferKeys } },
        { binding: 2, resource: { buffer: bufferVals } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder({label: "Split"});

    // Reset keys buffer and vals buffer.
    commandEncoder.clearBuffer(bufferKeys);
    commandEncoder.clearBuffer(bufferVals);

    // Encode and dispatch GPU command.
    const workGroupSplit = Math.ceil(count / 256);
    const splitGroupX = Math.min(workGroupSplit, MAXWORKGROUP);
    const splitGroupY = Math.ceil(workGroupSplit / splitGroupX);

    let passEncoderSplit: GPUComputePassEncoder;

    if (this.timestampQueryManager && baseIndex !== undefined) {
      const desc = this.timestampQueryManager.createComputePassDescriptor(
        baseIndex,
        baseIndex + 1
      );
      passEncoderSplit = commandEncoder.beginComputePass(desc);
    } else {
      passEncoderSplit = commandEncoder.beginComputePass();
    }

    passEncoderSplit.setPipeline(this.splitPipeline);
    passEncoderSplit.setBindGroup(0, bindGroup);
    passEncoderSplit.dispatchWorkgroups(splitGroupX, splitGroupY);
    passEncoderSplit.end();

    if (this.timestampQueryManager && baseIndex !== undefined) {
      this.timestampQueryManager.resolve(commandEncoder);
    }

    this.device.queue.submit([commandEncoder.finish()]);
		await this.device.queue.onSubmittedWorkDone();
  }
}