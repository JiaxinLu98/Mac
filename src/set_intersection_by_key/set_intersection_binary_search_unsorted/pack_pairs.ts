import pack_pairs from './pack_pairs.wgsl';
import TimestampQueryManager from '../../TimestampQueryManager';

const MAXWORKGROUP = 65535;

export class GPUPack {
  packPipeline: GPUComputePipeline;
  device: GPUDevice;
  bindGroupLayout: GPUBindGroupLayout;
  timestampQueryManager: TimestampQueryManager | null;

  constructor(device: GPUDevice, timestampQueryManager?: TimestampQueryManager) {
    this.device = device;
    this.timestampQueryManager = timestampQueryManager ?? null;

    this.bindGroupLayout = device.createBindGroupLayout({
      label: 'pack pairs bind group layout',
      entries: [
        {
          // xs
          binding: 0, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "read-only-storage"}
        },
        {
          // ys
          binding: 1, 
          visibility: GPUShaderStage.COMPUTE, 
          buffer: {type: "read-only-storage"}
        },
        {
          // pairs
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
      label: 'pack pairs pipeline layout',
      bindGroupLayouts: [this.bindGroupLayout]
    });

    const shader_code = `${pack_pairs}`;
    const shader = device.createShaderModule({
      label: 'pack pairs shader',
      code: shader_code
    });

    this.packPipeline = device.createComputePipeline({
      label: 'pack pairs pipeline',
      layout: pipelineLayout,
      compute: {
        module: shader,
        entryPoint: 'pack_pairs'
      }
    });
  }

  public async packPairs(
    bufferXS: GPUBuffer, 
    bufferYS: GPUBuffer, 
    bufferPairs: GPUBuffer, 
    count: number,
    baseIndex: number,
  ) {
    const device = this.device;
    if(count === 0) return;

    // Create GPU buffers.
    const params = device.createBuffer({
      label: 'pack pairs params',
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(params, 0, new Uint32Array([count]));

    // Create Bind Group to link resources to the shader.
    const bindGroup = device.createBindGroup({
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: bufferXS } },
        { binding: 1, resource: { buffer: bufferYS } },
        { binding: 2, resource: { buffer: bufferPairs } },
        { binding: 3, resource: { buffer: params } },
      ],
    });

    // Encode and dispatch GPU command.
    const workGroupPack = Math.ceil(count / 256);
    const packGroupX = Math.min(workGroupPack, MAXWORKGROUP);
    const packGroupY = Math.ceil(workGroupPack / packGroupX);

    const commandEncoder = this.device.createCommandEncoder({label: "Pack"});
    let passEncoderPack: GPUComputePassEncoder;
    if (this.timestampQueryManager) {
      passEncoderPack = commandEncoder.beginComputePass(
        this.timestampQueryManager.createComputePassDescriptor(baseIndex, baseIndex + 1),
      );
    } else {
      passEncoderPack = commandEncoder.beginComputePass();
    }
    passEncoderPack.setPipeline(this.packPipeline);
    passEncoderPack.setBindGroup(0, bindGroup);
    passEncoderPack.dispatchWorkgroups(packGroupX, packGroupY);
    passEncoderPack.end();

    if (this.timestampQueryManager) {
      this.timestampQueryManager.resolve(commandEncoder);
    }

    // Reset xs buffer and ys buffer.
    commandEncoder.clearBuffer(bufferXS);
    commandEncoder.clearBuffer(bufferYS);

    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
}