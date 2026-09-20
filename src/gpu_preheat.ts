import preheatShader from './gpu_preheat.wgsl';

// Number of elements the preheat workload updates per pass (16 MB of u32).
const PREHEAT_ELEMENTS = 1 << 22;

/**
 * Keeps the GPU busy with unrelated work (gpu_preheat.wgsl) for `durationMs`, so that it
 * reaches its steady performance state before a benchmark's warmup runs. The native CUDA,
 * C++17 stdpar, and wgpu-native harnesses run the same workload. Returns the number of
 * submissions.
 */
export async function gpuPreheat(device: GPUDevice, durationMs: number): Promise<number> {
    if (durationMs <= 0) return 0;
    const buffer = device.createBuffer({ size: PREHEAT_ELEMENTS * 4, usage: GPUBufferUsage.STORAGE });
    const module = device.createShaderModule({ code: preheatShader });
    const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer } }],
    });
    const workgroups = Math.ceil(PREHEAT_ELEMENTS / 256);

    const t0 = performance.now();
    let submissions = 0;
    while (performance.now() - t0 < durationMs) {
        const encoder = device.createCommandEncoder();
        for (let p = 0; p < 4; p++) {
            const pass = encoder.beginComputePass();
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(workgroups);
            pass.end();
        }
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        submissions++;
    }
    buffer.destroy();
    return submissions;
}
