/**
 * Set Intersection using Balanced Path + Decoupled Lookback (C++ / wgpu-native)
 *
 * Two-phase GPU pipeline:
 *   1. DPI (Diagonal Path Intersection) - compute merge path partition boundaries
 *   2. Decoupled Lookback - single-pass count + scan + write intersection results
 *
 * Ported from TypeScript host code. WGSL shaders copied directly.
 * Requires wgpu-native v24+ for subgroup support.
 */

#include <webgpu/webgpu.h>
#ifndef WEBGPU_BACKEND_DAWN
#include <webgpu/wgpu.h>   // wgpu-native extras (wgpuDevicePoll, native subgroup feature)
#endif

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: WGPUStringView from C string
// ---------------------------------------------------------------------------
static WGPUStringView wgpuStr(const char* s) {
    return {s, s ? strlen(s) : 0};
}

// ---------------------------------------------------------------------------
// Constants (must match WGSL shaders)
// ---------------------------------------------------------------------------
static constexpr uint32_t NT = 256;     // threads per workgroup
static constexpr uint32_t VT = 12;      // values per thread
static constexpr uint32_t NV = NT * VT; // 3072 elements per workgroup
static constexpr uint32_t MAX_DISPATCH_X = 65535;
static constexpr uint32_t DPI_WORKGROUP_SIZE = 256; // DPI shader workgroup size (subgroup version)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open file: %s\n", path);
        exit(1);
    }
    auto sz = f.tellg();
    f.seekg(0);
    std::string buf(static_cast<size_t>(sz), '\0');
    f.read(buf.data(), sz);
    return buf;
}

static std::vector<uint32_t> loadBinaryData(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "ERROR: cannot open binary file: %s\n", path);
        exit(1);
    }
    uint64_t count = 0;
    f.read(reinterpret_cast<char*>(&count), 8);
    std::vector<uint32_t> data(static_cast<size_t>(count));
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * 4));
    return data;
}

// CPU reference: sorted merge intersection
static std::vector<uint32_t> cpuSetIntersection(const std::vector<uint32_t>& a,
                                                  const std::vector<uint32_t>& b) {
    std::vector<uint32_t> result;
    size_t ai = 0, bi = 0;
    while (ai < a.size() && bi < b.size()) {
        if (a[ai] < b[bi])      ++ai;
        else if (a[ai] > b[bi]) ++bi;
        else { result.push_back(a[ai]); ++ai; ++bi; }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Global state for async callbacks
// ---------------------------------------------------------------------------
static WGPUInstance g_instance = nullptr;
static WGPUAdapter g_adapter = nullptr;
static WGPUDevice  g_device  = nullptr;
static bool        g_requestDone = false;
static uint32_t    g_subgroupSize = 32;  // queried at runtime; 32 for NVIDIA, 64 for AMD

// ---------------------------------------------------------------------------
// Backend differences. The host code is shared by the wgpu-native build
// (set_intersection_cpp) and the Dawn build (set_intersection_dawn, which
// defines WEBGPU_BACKEND_DAWN). Everything else below is identical.
// ---------------------------------------------------------------------------
#ifdef WEBGPU_BACKEND_DAWN
using TimestampWrites = WGPUPassTimestampWrites;
static const WGPUFeatureName kSubgroupsFeature = WGPUFeatureName_Subgroups;
static const bool kStripSubgroupsDirective = false;  // Dawn requires "enable subgroups;"
static const char* const kImplName = "dawn";
#else
using TimestampWrites = WGPUComputePassTimestampWrites;
static const WGPUFeatureName kSubgroupsFeature = (WGPUFeatureName)WGPUNativeFeature_Subgroup;
static const bool kStripSubgroupsDirective = true;   // wgpu-native v24 rejects the directive
static const char* const kImplName = "wgpu-native";
#endif

// Processes pending callbacks and lets queued GPU work progress (used while waiting for a map).
static void pollGpu(WGPUDevice device) {
#ifdef WEBGPU_BACKEND_DAWN
    (void)device;
    wgpuInstanceProcessEvents(g_instance);
#else
    wgpuDevicePoll(device, true, nullptr);
#endif
}

// Blocks until all work submitted to the queue has finished.
static bool g_workDone = false;
static void onWorkDone(WGPUQueueWorkDoneStatus /*status*/, void* /*userdata1*/, void* /*userdata2*/) {
    g_workDone = true;
}
static void waitQueueIdle(WGPUDevice device, WGPUQueue queue) {
    WGPUQueueWorkDoneCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = onWorkDone;
    g_workDone = false;
    wgpuQueueOnSubmittedWorkDone(queue, info);
    while (!g_workDone) pollGpu(device);
}

// v24 callback: adapter ready
static void onAdapterReady(WGPURequestAdapterStatus status,
                            WGPUAdapter adapter,
                            WGPUStringView message,
                            void* /*userdata1*/, void* /*userdata2*/) {
    if (status != WGPURequestAdapterStatus_Success) {
        fprintf(stderr, "Adapter request failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_adapter = adapter;
    g_requestDone = true;
}

// v24 callback: device ready
static void onDeviceReady(WGPURequestDeviceStatus status,
                           WGPUDevice device,
                           WGPUStringView message,
                           void* /*userdata1*/, void* /*userdata2*/) {
    if (status != WGPURequestDeviceStatus_Success) {
        fprintf(stderr, "Device request failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_device = device;
    g_requestDone = true;
}

// v24 callback: uncaptured error
static void onDeviceError(WGPUDevice const* /*device*/, WGPUErrorType type,
                           WGPUStringView message,
                           void* /*userdata1*/, void* /*userdata2*/) {
    fprintf(stderr, "[WebGPU Error %u] %.*s\n", (unsigned)type,
            (int)message.length, message.data ? message.data : "");
}

// v24 callback: buffer map done
static bool g_mapDone = false;
static void onMapDone(WGPUMapAsyncStatus status, WGPUStringView message,
                       void* /*userdata1*/, void* /*userdata2*/) {
    if (status != WGPUMapAsyncStatus_Success) {
        fprintf(stderr, "Buffer map failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
        exit(1);
    }
    g_mapDone = true;
}

// Helper to create WGPUBufferMapCallbackInfo
static WGPUBufferMapCallbackInfo mapCallbackInfo() {
    WGPUBufferMapCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = onMapDone;
    return info;
}

// ---------------------------------------------------------------------------
// Run a single test case
// ---------------------------------------------------------------------------
struct TestResult {
    bool passed;
    uint32_t gpuCount;
    uint32_t cpuCount;
};

static TestResult runIntersection(
    WGPUDevice device, WGPUQueue queue,
    WGPUComputePipeline diagPipeline, WGPUBindGroupLayout diagBGL,
    WGPUComputePipeline lookbackPipeline, WGPUBindGroupLayout lookbackBGL,
    const std::vector<uint32_t>& setA,
    const std::vector<uint32_t>& setB,
    bool verbose)
{
    const uint32_t a_len = (uint32_t)setA.size();
    const uint32_t b_len = (uint32_t)setB.size();
    const uint32_t total = a_len + b_len;

    if (total == 0) {
        auto cpuRes = cpuSetIntersection(setA, setB);
        return {cpuRes.empty(), 0, (uint32_t)cpuRes.size()};
    }

    const uint32_t numWg = (total + NV - 1) / NV;
    const uint32_t maxOutput = std::min(a_len, b_len);

    // DPI dispatch: each workgroup has multiple subgroups, each handling one diagonal
    const uint32_t subgroupsPerWg = DPI_WORKGROUP_SIZE / g_subgroupSize;
    const uint32_t dpiBlocks = (numWg + subgroupsPerWg - 1) / subgroupsPerWg;
    const uint32_t dpiDispatchX = std::min(dpiBlocks, MAX_DISPATCH_X);
    const uint32_t dpiDispatchY = (dpiBlocks + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    // Lookback dispatch: unchanged, one workgroup per partition
    const uint32_t lbDispatchX = std::min(numWg, MAX_DISPATCH_X);
    const uint32_t lbDispatchY = (numWg + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    if (verbose) {
        printf("  |A|=%u  |B|=%u  numWg=%u  dpiDispatch=(%u,%u)  lbDispatch=(%u,%u)\n",
               a_len, b_len, numWg, dpiDispatchX, dpiDispatchY, lbDispatchX, lbDispatchY);
    }

    // ---- Create buffers ----
    auto makeBuf = [&](const char* label, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc{};
        desc.label = wgpuStr(label);
        desc.size = std::max(size, (uint64_t)4);
        desc.usage = usage;
        return wgpuDeviceCreateBuffer(device, &desc);
    };

    WGPUBuffer bufA         = makeBuf("A",          (uint64_t)a_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufB         = makeBuf("B",          (uint64_t)b_len * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufALen      = makeBuf("aLen",       4,                   WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen      = makeBuf("bLen",       4,                   WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg     = makeBuf("numWg",      4,                   WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI       = makeBuf("DPI",        (uint64_t)(2*(numWg+1))*4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState     = makeBuf("State",      (uint64_t)numWg * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufOutput    = makeBuf("Output",     (uint64_t)std::max(maxOutput, 1u) * 4, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt  = makeBuf("TotalCount", 4,                   WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);
    WGPUBuffer mapTotalCnt  = makeBuf("mapTotal",   4,                   WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    WGPUBuffer mapOutput    = makeBuf("mapOutput",  (uint64_t)std::max(maxOutput, 1u) * 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    // ---- Write data ----
    wgpuQueueWriteBuffer(queue, bufA,    0, setA.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufB,    0, setB.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen, 0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen, 0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg,0, &numWg, 4);

    // Zero-fill state and totalCount
    std::vector<uint32_t> zeros(numWg, 0);
    wgpuQueueWriteBuffer(queue, bufState,    0, zeros.data(), numWg * 4);
    uint32_t zero = 0;
    wgpuQueueWriteBuffer(queue, bufTotalCnt, 0, &zero, 4);

    // ---- Create bind groups ----
    auto bufEntry = [](uint32_t binding, WGPUBuffer buf) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = wgpuBufferGetSize(buf);
        return e;
    };

    // DPI bind group: 6 entries
    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.label = wgpuStr("DPI BG");
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    // Lookback bind group: 9 entries
    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufState), bufEntry(4, bufOutput), bufEntry(5, bufTotalCnt),
        bufEntry(6, bufALen), bufEntry(7, bufBLen), bufEntry(8, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.label = wgpuStr("LB BG");
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 9;
    lbBGD.entries = lbEntries;
    WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(device, &lbBGD);

    // ---- Encode & dispatch ----
    WGPUCommandEncoderDescriptor encDesc{};
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

    // Pass 1: DPI (subgroup-optimized, fewer workgroups)
    {
        WGPUComputePassDescriptor cpd{};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
        wgpuComputePassEncoderSetPipeline(pass, diagPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, diagBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, dpiDispatchX, dpiDispatchY, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Pass 2: Decoupled Lookback (unchanged)
    {
        WGPUComputePassDescriptor cpd{};
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
        wgpuComputePassEncoderSetPipeline(pass, lookbackPipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, lbBG, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, lbDispatchX, lbDispatchY, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
    }

    // Copy totalCount to staging
    wgpuCommandEncoderCopyBufferToBuffer(encoder, bufTotalCnt, 0, mapTotalCnt, 0, 4);

    WGPUCommandBufferDescriptor cbDesc{};
    WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
    wgpuQueueSubmit(queue, 1, &cb);
    wgpuCommandBufferRelease(cb);
    wgpuCommandEncoderRelease(encoder);

    // ---- Read back total count ----
    g_mapDone = false;
    wgpuBufferMapAsync(mapTotalCnt, WGPUMapMode_Read, 0, 4, mapCallbackInfo());
    while (!g_mapDone) {
        pollGpu(device);
    }

    const uint32_t* pCount = (const uint32_t*)wgpuBufferGetConstMappedRange(mapTotalCnt, 0, 4);
    uint32_t totalCount = pCount ? *pCount : 0;
    wgpuBufferUnmap(mapTotalCnt);

    if (verbose) {
        printf("  GPU intersection count: %u\n", totalCount);
    }

    // ---- Read back output if needed ----
    std::vector<uint32_t> gpuResult;
    if (totalCount > 0) {
        WGPUCommandEncoderDescriptor ed2{};
        WGPUCommandEncoder enc2 = wgpuDeviceCreateCommandEncoder(device, &ed2);
        wgpuCommandEncoderCopyBufferToBuffer(enc2, bufOutput, 0, mapOutput, 0, (uint64_t)totalCount * 4);
        WGPUCommandBufferDescriptor cbd2{};
        WGPUCommandBuffer cb2 = wgpuCommandEncoderFinish(enc2, &cbd2);
        wgpuQueueSubmit(queue, 1, &cb2);
        wgpuCommandBufferRelease(cb2);
        wgpuCommandEncoderRelease(enc2);

        g_mapDone = false;
        wgpuBufferMapAsync(mapOutput, WGPUMapMode_Read, 0, (size_t)totalCount * 4, mapCallbackInfo());
        while (!g_mapDone) {
            pollGpu(device);
        }

        const uint32_t* pData = (const uint32_t*)wgpuBufferGetConstMappedRange(mapOutput, 0, (size_t)totalCount * 4);
        if (pData) {
            gpuResult.assign(pData, pData + totalCount);
        }
        wgpuBufferUnmap(mapOutput);
    }

    // ---- CPU validation ----
    auto cpuResult = cpuSetIntersection(setA, setB);
    bool passed = (totalCount == (uint32_t)cpuResult.size());

    if (passed && totalCount > 0) {
        for (uint32_t i = 0; i < totalCount; i++) {
            if (gpuResult[i] != cpuResult[i]) {
                passed = false;
                if (verbose) {
                    printf("  MISMATCH at index %u: GPU=%u, CPU=%u\n",
                           i, gpuResult[i], cpuResult[i]);
                }
                break;
            }
        }
    }

    if (verbose) {
        printf("  CPU intersection count: %zu\n", cpuResult.size());
        if (totalCount > 0 && totalCount <= 20) {
            printf("  GPU result: [");
            for (uint32_t i = 0; i < totalCount; i++)
                printf("%s%u", i ? ", " : "", gpuResult[i]);
            printf("]\n");
        } else if (totalCount > 20) {
            printf("  GPU result: [%u, %u, %u, ..., %u, %u, %u] (%u elements)\n",
                   gpuResult[0], gpuResult[1], gpuResult[2],
                   gpuResult[totalCount-3], gpuResult[totalCount-2], gpuResult[totalCount-1],
                   totalCount);
        }
        printf("  %s\n", passed ? "PASSED" : "FAILED");
    }

    // ---- Cleanup ----
    wgpuBindGroupRelease(diagBG);
    wgpuBindGroupRelease(lbBG);
    wgpuBufferRelease(bufA);
    wgpuBufferRelease(bufB);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutput);
    wgpuBufferRelease(bufTotalCnt);
    wgpuBufferRelease(mapTotalCnt);
    wgpuBufferRelease(mapOutput);

    return {passed, totalCount, (uint32_t)cpuResult.size()};
}

// ---------------------------------------------------------------------------
// GPU preheat: unrelated integer work (the same workload as src/gpu_preheat.wgsl) that keeps
// the GPU busy so it reaches its steady performance state before the warmup runs.
// ---------------------------------------------------------------------------
static const char* PREHEAT_WGSL = R"(
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) {
        return;
    }
    var x = data[i];
    for (var k = 0u; k < 16u; k++) {
        x = x * 1664525u + 1013904223u;
    }
    data[i] = x;
}
)";

static int gpuPreheat(WGPUDevice device, WGPUQueue queue, double durationMs) {
    if (durationMs <= 0) return 0;
    const uint32_t elements = 1u << 22;

    WGPUShaderSourceWGSL wgslDesc{};
    wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgslDesc.code = {PREHEAT_WGSL, strlen(PREHEAT_WGSL)};
    WGPUShaderModuleDescriptor smDesc{};
    smDesc.nextInChain = &wgslDesc.chain;
    smDesc.label = wgpuStr("preheat shader");
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &smDesc);

    WGPUComputePipelineDescriptor cpDesc{};
    cpDesc.label = wgpuStr("preheat pipeline");
    cpDesc.layout = nullptr;  // layout inferred from the shader
    cpDesc.compute.module = module;
    cpDesc.compute.entryPoint = wgpuStr("main");
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &cpDesc);
    WGPUBindGroupLayout bgl = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);

    WGPUBufferDescriptor bufDesc{};
    bufDesc.label = wgpuStr("preheat data");
    bufDesc.size = (uint64_t)elements * 4;
    bufDesc.usage = WGPUBufferUsage_Storage;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(device, &bufDesc);

    WGPUBindGroupEntry entry{};
    entry.binding = 0;
    entry.buffer = buffer;
    entry.size = bufDesc.size;
    WGPUBindGroupDescriptor bgDesc{};
    bgDesc.layout = bgl;
    bgDesc.entryCount = 1;
    bgDesc.entries = &entry;
    WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(device, &bgDesc);

    const uint32_t workgroups = (elements + 255) / 256;
    auto t0 = std::chrono::steady_clock::now();
    int submissions = 0;
    while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() < durationMs) {
        WGPUCommandEncoderDescriptor encDesc{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        for (int p = 0; p < 4; p++) {
            WGPUComputePassDescriptor cpd{};
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
            wgpuComputePassEncoderSetPipeline(pass, pipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bindGroup, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        WGPUCommandBufferDescriptor cbDesc{};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);
        waitQueueIdle(device, queue);  // wait for the submission to finish
        submissions++;
    }

    wgpuBindGroupRelease(bindGroup);
    wgpuBufferRelease(buffer);
    wgpuBindGroupLayoutRelease(bgl);
    wgpuComputePipelineRelease(pipeline);
    wgpuShaderModuleRelease(module);
    return submissions;
}

// ---------------------------------------------------------------------------
// GPU Timestamp Benchmark
// ---------------------------------------------------------------------------
struct BenchmarkTiming {
    double dpiMs;
    double lookbackMs;
    double totalMs;
    // Every timed run. Kernel time is DPI + Lookback. Span runs from the start of the
    // DPI pass to the end of the Lookback pass.
    std::vector<double> dpi, lookback, kernel, span;
    uint32_t count = 0;  // total_count of the last run, checked against the CPU reference
};

static BenchmarkTiming runBenchmark(
    WGPUDevice device, WGPUQueue queue,
    WGPUComputePipeline diagPipeline, WGPUBindGroupLayout diagBGL,
    WGPUComputePipeline lookbackPipeline, WGPUBindGroupLayout lookbackBGL,
    const std::vector<uint32_t>& setA,
    const std::vector<uint32_t>& setB,
    double preheatMs, int warmupIters, int measuredIters)
{
    const uint32_t a_len = (uint32_t)setA.size();
    const uint32_t b_len = (uint32_t)setB.size();
    const uint32_t total = a_len + b_len;
    const uint32_t numWg = (total + NV - 1) / NV;
    const uint32_t maxOutput = std::min(a_len, b_len);

    // DPI dispatch (subgroup-optimized)
    const uint32_t subgroupsPerWg = DPI_WORKGROUP_SIZE / g_subgroupSize;
    const uint32_t dpiBlocks = (numWg + subgroupsPerWg - 1) / subgroupsPerWg;
    const uint32_t dpiDispatchX = std::min(dpiBlocks, MAX_DISPATCH_X);
    const uint32_t dpiDispatchY = (dpiBlocks + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    // Lookback dispatch (unchanged)
    const uint32_t lbDispatchX = std::min(numWg, MAX_DISPATCH_X);
    const uint32_t lbDispatchY = (numWg + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X;

    // ---- Create buffers ----
    auto makeBuf = [&](const char* label, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor desc{};
        desc.label = wgpuStr(label);
        desc.size = std::max(size, (uint64_t)4);
        desc.usage = usage;
        return wgpuDeviceCreateBuffer(device, &desc);
    };

    WGPUBuffer bufA        = makeBuf("A",          (uint64_t)a_len * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufB        = makeBuf("B",          (uint64_t)b_len * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufALen     = makeBuf("aLen",       4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufBLen     = makeBuf("bLen",       4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufNumWg    = makeBuf("numWg",      4,                                    WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufDPI      = makeBuf("DPI",        (uint64_t)(2*(numWg+1))*4,            WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufState    = makeBuf("State",      (uint64_t)numWg * 4,                  WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst);
    WGPUBuffer bufOutput   = makeBuf("Output",     (uint64_t)std::max(maxOutput, 1u) * 4,WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
    WGPUBuffer bufTotalCnt = makeBuf("TotalCount", 4,                                    WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst);

    // Upload input data (once)
    wgpuQueueWriteBuffer(queue, bufA,     0, setA.data(), a_len * 4);
    wgpuQueueWriteBuffer(queue, bufB,     0, setB.data(), b_len * 4);
    wgpuQueueWriteBuffer(queue, bufALen,  0, &a_len, 4);
    wgpuQueueWriteBuffer(queue, bufBLen,  0, &b_len, 4);
    wgpuQueueWriteBuffer(queue, bufNumWg, 0, &numWg, 4);

    // ---- Create bind groups ----
    auto bufEntry = [](uint32_t binding, WGPUBuffer buf) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = wgpuBufferGetSize(buf);
        return e;
    };

    WGPUBindGroupEntry diagEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufALen), bufEntry(4, bufBLen), bufEntry(5, bufNumWg),
    };
    WGPUBindGroupDescriptor diagBGD{};
    diagBGD.label = wgpuStr("DPI BG");
    diagBGD.layout = diagBGL;
    diagBGD.entryCount = 6;
    diagBGD.entries = diagEntries;
    WGPUBindGroup diagBG = wgpuDeviceCreateBindGroup(device, &diagBGD);

    WGPUBindGroupEntry lbEntries[] = {
        bufEntry(0, bufA), bufEntry(1, bufB), bufEntry(2, bufDPI),
        bufEntry(3, bufState), bufEntry(4, bufOutput), bufEntry(5, bufTotalCnt),
        bufEntry(6, bufALen), bufEntry(7, bufBLen), bufEntry(8, bufNumWg),
    };
    WGPUBindGroupDescriptor lbBGD{};
    lbBGD.label = wgpuStr("LB BG");
    lbBGD.layout = lookbackBGL;
    lbBGD.entryCount = 9;
    lbBGD.entries = lbEntries;
    WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(device, &lbBGD);

    // ---- Timestamp query resources ----
    WGPUQuerySetDescriptor qsDesc{};
    qsDesc.type = WGPUQueryType_Timestamp;
    qsDesc.count = 4;
    WGPUQuerySet querySet = wgpuDeviceCreateQuerySet(device, &qsDesc);

    const uint64_t tsBufferSize = 4 * sizeof(uint64_t);
    WGPUBuffer tsResolveBuf = makeBuf("tsResolve", tsBufferSize,
                                       WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
    WGPUBuffer tsMapBuf     = makeBuf("tsMap",     tsBufferSize,
                                       WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);

    // Zeros for per-iteration state reset
    std::vector<uint32_t> zeros(numWg, 0);
    uint32_t zero = 0;

    // Unrelated GPU work first, so the GPU reaches its steady performance state.
    gpuPreheat(device, queue, preheatMs);

    // ---- Run iterations ----
    double dpiSum = 0, lookbackSum = 0, totalSum = 0;
    std::vector<double> dpiRuns, lookbackRuns, kernelRuns, spanRuns;
    const int totalIters = warmupIters + measuredIters;
    // Host wall-clock window of the timed iterations, printed as [timing-window] after the loop (outside the timed
    // passes) so the GPU clock can be checked against nsys GPU metrics.
    auto wallNs = [] {
        return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    };
    long long timedStartNs = 0;

    for (int iter = 0; iter < totalIters; iter++) {
        if (iter == warmupIters) timedStartNs = wallNs();
        wgpuQueueWriteBuffer(queue, bufState,    0, zeros.data(), numWg * 4);
        wgpuQueueWriteBuffer(queue, bufTotalCnt, 0, &zero, 4);

        WGPUCommandEncoderDescriptor encDesc{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);

        // Pass 1: DPI (subgroup-optimized, timestamps 0, 1)
        {
            TimestampWrites tsWrites{};
            tsWrites.querySet = querySet;
            tsWrites.beginningOfPassWriteIndex = 0;
            tsWrites.endOfPassWriteIndex = 1;

            WGPUComputePassDescriptor cpd{};
            cpd.timestampWrites = &tsWrites;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
            wgpuComputePassEncoderSetPipeline(pass, diagPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, diagBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, dpiDispatchX, dpiDispatchY, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        // Pass 2: Decoupled Lookback (unchanged, timestamps 2, 3)
        {
            TimestampWrites tsWrites{};
            tsWrites.querySet = querySet;
            tsWrites.beginningOfPassWriteIndex = 2;
            tsWrites.endOfPassWriteIndex = 3;

            WGPUComputePassDescriptor cpd{};
            cpd.timestampWrites = &tsWrites;
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &cpd);
            wgpuComputePassEncoderSetPipeline(pass, lookbackPipeline);
            wgpuComputePassEncoderSetBindGroup(pass, 0, lbBG, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, lbDispatchX, lbDispatchY, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }

        wgpuCommandEncoderResolveQuerySet(encoder, querySet, 0, 4, tsResolveBuf, 0);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, tsResolveBuf, 0, tsMapBuf, 0, tsBufferSize);

        WGPUCommandBufferDescriptor cbDesc{};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);

        // Read back timestamps
        g_mapDone = false;
        wgpuBufferMapAsync(tsMapBuf, WGPUMapMode_Read, 0, tsBufferSize, mapCallbackInfo());
        while (!g_mapDone) {
            pollGpu(device);
        }

        const uint64_t* ts = (const uint64_t*)wgpuBufferGetConstMappedRange(tsMapBuf, 0, tsBufferSize);
        if (ts && iter >= warmupIters) {
            double dpiNs      = (double)(ts[1] - ts[0]);
            double lookbackNs = (double)(ts[3] - ts[2]);
            double totalNs    = (double)(ts[3] - ts[0]);

            dpiSum      += dpiNs      / 1e6;
            lookbackSum += lookbackNs / 1e6;
            totalSum    += totalNs    / 1e6;
            dpiRuns.push_back(dpiNs / 1e6);
            lookbackRuns.push_back(lookbackNs / 1e6);
            kernelRuns.push_back((dpiNs + lookbackNs) / 1e6);
            spanRuns.push_back(totalNs / 1e6);
        }
        wgpuBufferUnmap(tsMapBuf);
    }
    printf("[timing-window] {\"n\":%u,\"start_ns\":%lld,\"end_ns\":%lld}\n", a_len, timedStartNs, wallNs());
    fflush(stdout);

    // ---- Read back the result count of the last run ----
    uint32_t resultCount = 0;
    {
        WGPUBuffer countMap = makeBuf("countMap", 4, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
        WGPUCommandEncoderDescriptor encDesc{};
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encDesc);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, bufTotalCnt, 0, countMap, 0, 4);
        WGPUCommandBufferDescriptor cbDesc{};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(encoder, &cbDesc);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(encoder);
        g_mapDone = false;
        wgpuBufferMapAsync(countMap, WGPUMapMode_Read, 0, 4, mapCallbackInfo());
        while (!g_mapDone) {
            pollGpu(device);
        }
        const uint32_t* c = (const uint32_t*)wgpuBufferGetConstMappedRange(countMap, 0, 4);
        if (c) resultCount = c[0];
        wgpuBufferUnmap(countMap);
        wgpuBufferRelease(countMap);
    }

    // ---- Cleanup ----
    wgpuQuerySetDestroy(querySet);
    wgpuQuerySetRelease(querySet);
    wgpuBindGroupRelease(diagBG);
    wgpuBindGroupRelease(lbBG);
    wgpuBufferRelease(bufA);
    wgpuBufferRelease(bufB);
    wgpuBufferRelease(bufALen);
    wgpuBufferRelease(bufBLen);
    wgpuBufferRelease(bufNumWg);
    wgpuBufferRelease(bufDPI);
    wgpuBufferRelease(bufState);
    wgpuBufferRelease(bufOutput);
    wgpuBufferRelease(bufTotalCnt);
    wgpuBufferRelease(tsResolveBuf);
    wgpuBufferRelease(tsMapBuf);

    return {
        dpiSum      / measuredIters,
        lookbackSum / measuredIters,
        totalSum    / measuredIters,
        dpiRuns, lookbackRuns, kernelRuns, spanRuns,
        resultCount
    };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
// Usage: App [data dir] [ranges, e.g. e6 or e2,e6]
// Every timed run is printed in a [micro-result] JSON line.
int main(int argc, char** argv) {
    printf("=== Set Intersection (Balanced Path + Decoupled Lookback) ===\n");
    printf("=== C++ host, backend: %s (subgroup-optimized DPI) ===\n\n", kImplName);

    // ---- Instance ----
    WGPUInstanceDescriptor instDesc{};
    WGPUInstance instance = wgpuCreateInstance(&instDesc);
    g_instance = instance;
    if (!instance) {
        fprintf(stderr, "Failed to create WebGPU instance\n");
        return 1;
    }

    // ---- Adapter ----
    WGPURequestAdapterOptions adapterOpts{};
    adapterOpts.powerPreference = WGPUPowerPreference_HighPerformance;

    g_requestDone = false;
    WGPURequestAdapterCallbackInfo adapterCbInfo{};
    adapterCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    adapterCbInfo.callback = onAdapterReady;
    wgpuInstanceRequestAdapter(instance, &adapterOpts, adapterCbInfo);
    while (!g_requestDone) {
        wgpuInstanceProcessEvents(instance);
    }
    assert(g_adapter);

    // Print adapter info
    WGPUAdapterInfo adapterInfo{};
    wgpuAdapterGetInfo(g_adapter, &adapterInfo);
    printf("Adapter: %.*s\n", (int)adapterInfo.device.length,
           adapterInfo.device.data ? adapterInfo.device.data : "unknown");
    printf("Driver:  %.*s\n", (int)adapterInfo.description.length,
           adapterInfo.description.data ? adapterInfo.description.data : "unknown");
    printf("Backend: ");
    switch (adapterInfo.backendType) {
        case WGPUBackendType_Vulkan: printf("Vulkan\n"); break;
        case WGPUBackendType_D3D12:  printf("D3D12\n");  break;
        case WGPUBackendType_D3D11:  printf("D3D11\n");  break;
        case WGPUBackendType_Metal:  printf("Metal\n");  break;
        default: printf("Other (%d)\n", adapterInfo.backendType);
    }
    printf("\n");

    // ---- Device ----
    WGPULimits adapterLimits{};
    wgpuAdapterGetLimits(g_adapter, &adapterLimits);
#ifndef WEBGPU_BACKEND_DAWN
    // wgpu-native v24 leaves the adapter's maxBufferSize undefined on Vulkan, and the device then falls
    // back to the 256 MB default, which skips the 128M dataset. Request the storage binding size instead.
    if (adapterLimits.maxBufferSize == WGPU_LIMIT_U64_UNDEFINED)
        adapterLimits.maxBufferSize = adapterLimits.maxStorageBufferBindingSize;
#endif

    WGPUDeviceDescriptor devDesc{};
    devDesc.label = wgpuStr("GPU Device");
    devDesc.requiredLimits = &adapterLimits;
    devDesc.defaultQueue.label = wgpuStr("Default Queue");

    // Error callback (UncapturedErrorCallbackInfo has no mode field)
    devDesc.uncapturedErrorCallbackInfo.callback = onDeviceError;
#ifdef WEBGPU_BACKEND_DAWN
    // Full-resolution timestamps, as in the Chrome runs (--enable-unsafe-webgpu).
    static const char* const kEnabledToggles[] = {"allow_unsafe_apis"};
    static const char* const kDisabledToggles[] = {"timestamp_quantization"};
    WGPUDawnTogglesDescriptor toggles{};
    toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    toggles.enabledToggleCount = 1;
    toggles.enabledToggles = kEnabledToggles;
    toggles.disabledToggleCount = 1;
    toggles.disabledToggles = kDisabledToggles;
    devDesc.nextInChain = &toggles.chain;
#endif

    // Request features: timestamp-query + subgroups
    WGPUBool hasTimestampQuery = wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery);
    WGPUBool hasSubgroups = wgpuAdapterHasFeature(g_adapter,
        kSubgroupsFeature);

    std::vector<WGPUFeatureName> requiredFeatures;
    if (hasTimestampQuery) requiredFeatures.push_back(WGPUFeatureName_TimestampQuery);
    if (hasSubgroups)      requiredFeatures.push_back(kSubgroupsFeature);

    if (!requiredFeatures.empty()) {
        devDesc.requiredFeatureCount = requiredFeatures.size();
        devDesc.requiredFeatures = requiredFeatures.data();
    }

    g_requestDone = false;
    WGPURequestDeviceCallbackInfo deviceCbInfo{};
    deviceCbInfo.mode = WGPUCallbackMode_AllowProcessEvents;
    deviceCbInfo.callback = onDeviceReady;
    wgpuAdapterRequestDevice(g_adapter, &devDesc, deviceCbInfo);
    while (!g_requestDone) {
        wgpuInstanceProcessEvents(instance);
    }
    assert(g_device);

    WGPUQueue queue = wgpuDeviceGetQueue(g_device);

    // Query actual device limits
    WGPULimits deviceLimits{};
    wgpuDeviceGetLimits(g_device, &deviceLimits);
    uint64_t maxBufSize = deviceLimits.maxBufferSize;

    // Subgroup size heuristic (wgpu v24 limits don't expose minSubgroupSize)
    g_subgroupSize = 32; // NVIDIA=32, AMD=64

    printf("Device ready. maxBufferSize = %llu MB\n", (unsigned long long)(maxBufSize / (1024*1024)));
    printf("Timestamp queries: %s\n", hasTimestampQuery ? "supported" : "NOT supported");
    printf("Subgroups: %s (assumed size=%u, DPI subgroups/wg=%u)\n\n",
           hasSubgroups ? "supported" : "NOT supported",
           g_subgroupSize, DPI_WORKGROUP_SIZE / g_subgroupSize);

    // ---- Load shaders ----
    // The shaders are the browser library's own files from src/balanced_path/common, copied
    // into shaders/ by CMake: balanced_path_biased.wgsl (subgroup DPI) and
    // not_by_key/set_availability_decoupled_lookback.wgsl (OP_MODE 0 = intersection).
    const char* shaderDirs[] = { "shaders", "../shaders", "../../shaders" };
    const char* shaderDir = nullptr;
    for (auto d : shaderDirs) {
        char probe[256];
        snprintf(probe, sizeof(probe), "%s/balanced_path_biased.wgsl", d);
        std::ifstream test(probe);
        if (test.good()) { shaderDir = d; break; }
    }
    if (!shaderDir) {
        fprintf(stderr, "ERROR: shader directory not found. Run from the build directory.\n");
        return 1;
    }
    char dpiPath[256], lbPath[256];
    snprintf(dpiPath, sizeof(dpiPath), "%s/balanced_path_biased.wgsl", shaderDir);
    snprintf(lbPath,  sizeof(lbPath),  "%s/set_availability_decoupled_lookback.wgsl", shaderDir);
    std::string dpiCode = readFile(dpiPath);
    std::string lbCode  = readFile(lbPath);
    // wgpu-native v24 enables subgroups through the device feature and rejects the directive.
    const std::string subgroupsDirective = "enable subgroups;";
    if (kStripSubgroupsDirective) {
        for (size_t pos; (pos = dpiCode.find(subgroupsDirective)) != std::string::npos;)
            dpiCode.erase(pos, subgroupsDirective.size());
    }
    if (lbCode.find("const OP_MODE: u32 = 0u;") == std::string::npos) {
        fprintf(stderr, "ERROR: lookback shader does not declare OP_MODE 0 (intersection)\n");
        return 1;
    }

    auto createShaderModule = [&](const char* label, const std::string& code) {
        WGPUShaderSourceWGSL wgslDesc{};
        wgslDesc.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgslDesc.code = {code.c_str(), code.size()};

        WGPUShaderModuleDescriptor smDesc{};
        smDesc.nextInChain = &wgslDesc.chain;
        smDesc.label = wgpuStr(label);
        return wgpuDeviceCreateShaderModule(g_device, &smDesc);
    };

    WGPUShaderModule dpiShaderModule = createShaderModule("DPI Shader", dpiCode);
    WGPUShaderModule lbShaderModule  = createShaderModule("Lookback Shader", lbCode);

    // ---- Create bind group layouts ----
    auto bglEntry = [](uint32_t binding, WGPUBufferBindingType type) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = type;
        return e;
    };

    // DPI: 2 read-only-storage, 1 storage, 3 uniform
    WGPUBindGroupLayoutEntry diagLayoutEntries[] = {
        bglEntry(0, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(2, WGPUBufferBindingType_Storage),
        bglEntry(3, WGPUBufferBindingType_Uniform),
        bglEntry(4, WGPUBufferBindingType_Uniform),
        bglEntry(5, WGPUBufferBindingType_Uniform),
    };
    WGPUBindGroupLayoutDescriptor diagBGLD{};
    diagBGLD.label = wgpuStr("DPI BGL");
    diagBGLD.entryCount = 6;
    diagBGLD.entries = diagLayoutEntries;
    WGPUBindGroupLayout diagBGL = wgpuDeviceCreateBindGroupLayout(g_device, &diagBGLD);

    // Lookback: 3 read-only-storage, 3 storage, 3 uniform
    WGPUBindGroupLayoutEntry lbLayoutEntries[] = {
        bglEntry(0, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(1, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(2, WGPUBufferBindingType_ReadOnlyStorage),
        bglEntry(3, WGPUBufferBindingType_Storage),
        bglEntry(4, WGPUBufferBindingType_Storage),
        bglEntry(5, WGPUBufferBindingType_Storage),
        bglEntry(6, WGPUBufferBindingType_Uniform),
        bglEntry(7, WGPUBufferBindingType_Uniform),
        bglEntry(8, WGPUBufferBindingType_Uniform),
    };
    WGPUBindGroupLayoutDescriptor lbBGLD{};
    lbBGLD.label = wgpuStr("Lookback BGL");
    lbBGLD.entryCount = 9;
    lbBGLD.entries = lbLayoutEntries;
    WGPUBindGroupLayout lookbackBGL = wgpuDeviceCreateBindGroupLayout(g_device, &lbBGLD);

    // ---- Create pipelines ----
    auto makePipeline = [&](const char* label, WGPUBindGroupLayout bgl,
                            WGPUShaderModule sm, const char* entryPoint) {
        WGPUPipelineLayoutDescriptor plDesc{};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &bgl;
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g_device, &plDesc);

        WGPUComputePipelineDescriptor cpDesc{};
        cpDesc.label = wgpuStr(label);
        cpDesc.layout = pl;
        cpDesc.compute.module = sm;
        cpDesc.compute.entryPoint = wgpuStr(entryPoint);

        WGPUComputePipeline pipe = wgpuDeviceCreateComputePipeline(g_device, &cpDesc);
        wgpuPipelineLayoutRelease(pl);
        return pipe;
    };

    WGPUComputePipeline diagPipeline = makePipeline("DPI Pipeline", diagBGL,
                                                      dpiShaderModule, "compute_diagonals");
    WGPUComputePipeline lbPipeline   = makePipeline("Lookback Pipeline", lookbackBGL,
                                                      lbShaderModule, "decoupled_lookback_kernel");

    printf("Pipelines created.\n\n");

    // ======================================================================
    // Locate data directory
    // ======================================================================
    const char* dataDirs[] = {
        "../../../public/data",
        "../public/data",
        "public/data",
    };
    const char* dataDir = argc > 1 ? argv[1] : nullptr;
    for (auto d : dataDirs) {
        if (dataDir) break;
        char probe[256];
        snprintf(probe, sizeof(probe), "%s/A_1e6.bin", d);
        std::ifstream test(probe);
        if (test.good()) { dataDir = d; break; }
    }
    if (!dataDir) fprintf(stderr, "ERROR: data directory not found. Pass it as the first argument.\n");

    // Ranges to run, from the second argument (default: e2,e6).
    std::vector<std::string> rangeNames;
    {
        std::string list = argc > 2 ? argv[2] : "e2,e6";
        size_t pos = 0;
        while (pos <= list.size()) {
            size_t comma = list.find(',', pos);
            if (comma == std::string::npos) comma = list.size();
            if (comma > pos) rangeNames.push_back(list.substr(pos, comma - pos));
            pos = comma + 1;
        }
    }

    // ======================================================================
    // GPU Timestamp Benchmark
    // ======================================================================
    if (hasTimestampQuery && dataDir) {
        printf("\n=== GPU Timestamp Benchmark ===\n");
        printf("=== (Pure GPU time, no CPU overhead) ===\n\n");

        // Unrelated GPU work before each size, from MICRO_PREHEAT_MS (default 0, off). The preheat protocol sets
        // 1000 so every size is measured at full GPU clock, like --preheat-ms of the CUDA and stdpar programs.
        const double PREHEAT_MS = getenv("MICRO_PREHEAT_MS") ? atof(getenv("MICRO_PREHEAT_MS")) : 0;
        const int WARMUP = 10;
        const int MEASURED = 100;
        printf("Warmup: %d iterations, Measured: %d iterations (averaged)\n\n", WARMUP, MEASURED);

        printf("%-10s %12s  %10s  %10s  %10s\n",
               "Dataset", "Input Size", "DPI(ms)", "Lookback(ms)", "Total(ms)");
        printf("%-10s %12s  %10s  %10s  %10s\n",
               "-------", "----------", "-------", "------------", "--------");

        const char* bmSizes[]  = {"1", "2", "4", "8", "16", "32", "64", "128"};
        std::vector<const char*> bmRanges;
        for (const auto& r : rangeNames) bmRanges.push_back(r.c_str());

        for (auto sz : bmSizes) {
            for (auto rng : bmRanges) {
                char pathA[256], pathB[256], dsName[64];
                snprintf(pathA, sizeof(pathA), "%s/A_%s%s.bin", dataDir, sz, rng);
                snprintf(pathB, sizeof(pathB), "%s/B_%s%s.bin", dataDir, sz, rng);
                snprintf(dsName, sizeof(dsName), "%s%s", sz, rng);

                std::ifstream fA(pathA), fB(pathB);
                if (!fA.good() || !fB.good()) continue;
                fA.close(); fB.close();

                auto A = loadBinaryData(pathA);
                auto B = loadBinaryData(pathB);

                uint64_t largestBuf = (uint64_t)std::max(A.size(), B.size()) * 4;
                if (largestBuf > maxBufSize) {
                    printf("%-10s %10zuM  %s\n", dsName, A.size()/1000000, "(skipped: buffer exceeds maxBufferSize)");
                    continue;
                }

                char inputStr[32];
                snprintf(inputStr, sizeof(inputStr), "%zuM+%zuM",
                         A.size()/1000000, B.size()/1000000);

                auto timing = runBenchmark(g_device, queue,
                                            diagPipeline, diagBGL,
                                            lbPipeline, lookbackBGL,
                                            A, B, PREHEAT_MS, WARMUP, MEASURED);

                const char* check = "skip";
                if (A.size() <= 64000000 && B.size() <= 64000000) {
                    check = cpuSetIntersection(A, B).size() == timing.count ? "PASS" : "FAIL";
                }
                printf("%-10s %12s  %10.3f  %10.3f  %10.3f  count %u %s\n",
                       dsName, inputStr,
                       timing.dpiMs, timing.lookbackMs, timing.totalMs, timing.count, check);

                auto jsonArray = [](const std::vector<double>& v) {
                    std::string out = "[";
                    char buf[32];
                    for (size_t i = 0; i < v.size(); i++) {
                        snprintf(buf, sizeof(buf), "%s%.5f", i ? "," : "", v[i]);
                        out += buf;
                    }
                    return out + "]";
                };
                printf("[micro-result] {\"impl\":\"%s\",\"op\":\"intersection\",\"dataset\":\"%s\",\"preheat_ms\":%.0f,"
                       "\"warmup\":%d,\"runs\":%zu,\"count\":%u,\"check\":\"%s\",\"dpi_ms\":%s,\"lookback_ms\":%s,\"kernel_ms\":%s,\"span_ms\":%s}\n",
                       kImplName, dsName, PREHEAT_MS, WARMUP, timing.kernel.size(), timing.count, check,
                       jsonArray(timing.dpi).c_str(), jsonArray(timing.lookback).c_str(),
                       jsonArray(timing.kernel).c_str(), jsonArray(timing.span).c_str());
            }
        }
        printf("\n");
    } else if (!hasTimestampQuery) {
        printf("\nBenchmark skipped: timestamp queries not supported on this device.\n\n");
    }

    // ---- Cleanup ----
    wgpuComputePipelineRelease(diagPipeline);
    wgpuComputePipelineRelease(lbPipeline);
    wgpuBindGroupLayoutRelease(diagBGL);
    wgpuBindGroupLayoutRelease(lookbackBGL);
    wgpuShaderModuleRelease(dpiShaderModule);
    wgpuShaderModuleRelease(lbShaderModule);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(g_device);
    wgpuAdapterRelease(g_adapter);
    wgpuInstanceRelease(instance);

    return 0;
}
