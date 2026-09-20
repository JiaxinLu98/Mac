/**
 * ECLAT Frequent Itemset Mining (C++ / wgpu-native) — optimized batched pipeline
 *
 * Same 2-phase batched pipeline as main.cpp, with the host-side design of the
 * TypeScript library's in-place path (src/webgpu_batched_sets.ts):
 *   - base tidsets live in shared arena buffers and every tidset is read in place,
 *     so a chunk binds one A buffer and one B buffer (no per-pair packing copy)
 *   - compact metadata: 8 u32 per pair (pairInfo) + 1 u32 per workgroup
 *   - consecutive chunks are encoded into one command buffer that shares pooled
 *     scratch buffers through aligned binding windows, with a single wait for the
 *     per-pair counts and timestamps
 *   - chunks hold at most chunkMB of input, and every buffer respects the
 *     device-level (not adapter-level) storage limits
 *
 * Usage: ./AppOpt <dataset.dat> <min_support> [warmup=3] [iterations=10] [chunkMB=512]
 */

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

static WGPUStringView wgpuStr(const char* s) { return {s, s ? strlen(s) : 0}; }

static constexpr uint32_t NT = 256;
static constexpr uint32_t VT = 12;
static constexpr uint32_t NV = NT * VT;
static constexpr uint32_t MAX_DISPATCH_X = 65535;
static constexpr uint32_t DPI_WG_SIZE = 256;
static constexpr uint32_t SUBGROUP_SIZE = 32;
static constexpr uint32_t PAIR_INFO_STRIDE = 8;
// A timestamp query set holds at most 4096 queries; each chunk uses 4.
static constexpr uint32_t MAX_CHUNKS_PER_SUBMIT = 1024;
static constexpr uint32_t QUERY_COUNT = 4 * MAX_CHUNKS_PER_SUBMIT;

static uint64_t alignUp(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

static double msSince(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

static std::string readFile(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    auto size = f.tellg();
    f.seekg(0);
    std::string buf((size_t)size, '\0');
    f.read(buf.data(), size);
    return buf;
}

// ============================================================================
// Data loading & tidset construction (same as main.cpp)
// ============================================================================
struct TransactionDB { std::vector<std::vector<uint32_t>> transactions; };

TransactionDB parseTransactionDB(const std::string& path) {
    TransactionDB db;
    std::ifstream file(path);
    if (!file.is_open()) { fprintf(stderr, "Cannot open %s\n", path.c_str()); exit(1); }
    std::string line;
    while (std::getline(file, line)) {
        size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        if (line[s] == '#' || (line[s] == '/' && s + 1 < line.size() && line[s + 1] == '/')) continue;
        std::vector<uint32_t> items;
        std::istringstream iss(line);
        std::string tok;
        while (iss >> tok) {
            char* e;
            long v = strtol(tok.c_str(), &e, 10);
            if (e != tok.c_str()) items.push_back((uint32_t)v);
        }
        if (!items.empty()) db.transactions.push_back(std::move(items));
    }
    return db;
}

struct Tidsets {
    std::vector<uint32_t> items;
    std::vector<std::vector<uint32_t>> tids;
    std::map<uint32_t, size_t> itemIndex;
};

Tidsets buildVerticalFormat(const TransactionDB& db) {
    std::map<uint32_t, std::vector<uint32_t>> tm;
    for (uint32_t tid = 0; tid < (uint32_t)db.transactions.size(); tid++)
        for (uint32_t item : db.transactions[tid]) tm[item].push_back(tid);
    Tidsets ts;
    for (auto& [item, tids] : tm) {
        ts.itemIndex[item] = ts.items.size();
        ts.items.push_back(item);
        ts.tids.push_back(std::move(tids));
    }
    return ts;
}

// ============================================================================
// CPU ECLAT for validation (same as main.cpp)
// ============================================================================
struct FrequentItemset { std::vector<uint32_t> itemset; uint32_t support; };

static std::vector<uint32_t> intersectCPU(const uint32_t* a, uint32_t al, const uint32_t* b, uint32_t bl) {
    std::vector<uint32_t> r;
    uint32_t i = 0, j = 0;
    while (i < al && j < bl) {
        if (a[i] < b[j]) i++;
        else if (a[i] > b[j]) j++;
        else { r.push_back(a[i]); i++; j++; }
    }
    return r;
}

void eclatCPURec(const Tidsets& ts, const std::vector<uint32_t>& pfx, const std::vector<uint32_t>& pt,
                 const std::vector<uint32_t>& cands, uint32_t ms, std::vector<FrequentItemset>& res) {
    for (size_t ci = 0; ci < cands.size(); ci++) {
        size_t idx = ts.itemIndex.at(cands[ci]);
        auto nt = intersectCPU(pt.data(), (uint32_t)pt.size(), ts.tids[idx].data(), (uint32_t)ts.tids[idx].size());
        if ((uint32_t)nt.size() >= ms) {
            auto ni = pfx;
            ni.push_back(cands[ci]);
            res.push_back({ni, (uint32_t)nt.size()});
            std::vector<uint32_t> rem(cands.begin() + ci + 1, cands.end());
            eclatCPURec(ts, ni, nt, rem, ms, res);
        }
    }
}

std::vector<FrequentItemset> eclatCPU(const Tidsets& ts, uint32_t ms) {
    std::vector<FrequentItemset> res;
    std::vector<uint32_t> fi;
    for (size_t i = 0; i < ts.items.size(); i++)
        if ((uint32_t)ts.tids[i].size() >= ms) {
            fi.push_back(ts.items[i]);
            res.push_back({{ts.items[i]}, (uint32_t)ts.tids[i].size()});
        }
    std::sort(fi.begin(), fi.end());
    for (size_t i = 0; i < fi.size(); i++) {
        size_t idx = ts.itemIndex.at(fi[i]);
        std::vector<uint32_t> rem(fi.begin() + i + 1, fi.end());
        eclatCPURec(ts, {fi[i]}, {ts.tids[idx].begin(), ts.tids[idx].end()}, rem, ms, res);
    }
    return res;
}

// ============================================================================
// GPU-resident tidsets: a handle addresses its elements inside a shared buffer
// ============================================================================
// Output buffers of one run. A released buffer is kept and handed to a later chunk that
// fits in it, so later levels reuse the memory of earlier ones instead of creating buffers.
struct WgpuOutputPool {
    std::multimap<uint64_t, WGPUBuffer> freeBuffers;  // size in bytes -> buffer
    ~WgpuOutputPool() {
        for (auto& b : freeBuffers) { wgpuBufferDestroy(b.second); wgpuBufferRelease(b.second); }
    }
};

struct WgpuSetBuffer {
    WGPUBuffer buf;
    WgpuOutputPool* pool;  // null: released on destruction
    explicit WgpuSetBuffer(WGPUBuffer b, WgpuOutputPool* p = nullptr) : buf(b), pool(p) {}
    ~WgpuSetBuffer() {
        if (!buf) return;
        if (pool) pool->freeBuffers.emplace(wgpuBufferGetSize(buf), buf);
        else wgpuBufferRelease(buf);
        buf = nullptr;
    }
    WgpuSetBuffer(const WgpuSetBuffer&) = delete;
    WgpuSetBuffer& operator=(const WgpuSetBuffer&) = delete;
};

struct WgpuSetHandle {
    std::shared_ptr<WgpuSetBuffer> buffer;
    uint32_t offset = 0;  // element offset
    uint32_t length = 0;  // element count
    uint64_t byteLen() const { return (uint64_t)length * 4; }
};

// ============================================================================
// Chunks and submissions
// ============================================================================
// Pairs that share one A source buffer and one B source buffer. aStart/bStart in
// pairInfo are the tidsets' element offsets inside those buffers; the state, DPI
// and output offsets are chunk-local.
struct Chunk {
    size_t pairStart = 0;
    uint32_t numPairs = 0, totalWg = 0, totalDpiEntries = 0, maxOutputSize = 0;
    std::shared_ptr<WgpuSetBuffer> bufA, bufB;
    std::vector<uint32_t> pairInfo, pairIdPerWg, pairOutputOffsets;
};

static Chunk prepareChunk(const std::vector<WgpuSetHandle>& aH, const std::vector<WgpuSetHandle>& bH, size_t pairStart) {
    Chunk c;
    c.pairStart = pairStart;
    c.numPairs = (uint32_t)aH.size();
    c.bufA = aH[0].buffer;
    c.bufB = bH[0].buffer;
    c.pairInfo.assign((size_t)c.numPairs * PAIR_INFO_STRIDE, 0);
    c.pairOutputOffsets.resize(c.numPairs);

    for (uint32_t i = 0; i < c.numPairs; i++) {
        uint32_t al = aH[i].length, bl = bH[i].length;
        c.pairOutputOffsets[i] = c.maxOutputSize;
        if (al > 0 && bl > 0) {
            uint32_t nw = (al + bl + NV - 1) / NV;
            uint32_t* p = &c.pairInfo[(size_t)i * PAIR_INFO_STRIDE];
            p[0] = aH[i].offset; p[1] = al;
            p[2] = bH[i].offset; p[3] = bl;
            p[4] = nw;
            p[5] = c.totalWg;
            p[6] = c.totalDpiEntries;
            p[7] = c.maxOutputSize;
            c.totalWg += nw;
            c.totalDpiEntries += 2 * (nw + 1);
        }
        c.maxOutputSize += std::min(al, bl);
    }

    c.pairIdPerWg.reserve(c.totalWg);
    for (uint32_t i = 0; i < c.numPairs; i++)
        c.pairIdPerWg.insert(c.pairIdPerWg.end(), c.pairInfo[(size_t)i * PAIR_INFO_STRIDE + 4], i);
    return c;
}

// Aligned windows of one submission's shared scratch buffers, one per chunk.
class SubmitLayout {
public:
    std::vector<uint64_t> info, pairId, dpi, state, counts;
    uint64_t infoBytes = 0, pairIdBytes = 0, dpiBytes = 0, stateBytes = 0, countsBytes = 0;

    explicit SubmitLayout(uint64_t alignment) : align_(alignment) {}

    size_t chunkCount() const { return info.size(); }

    bool fits(const Chunk& c, uint64_t limit) const {
        auto next = [&](uint64_t total, uint64_t bytes) { return alignUp(total, align_) + bytes; };
        return chunkCount() < MAX_CHUNKS_PER_SUBMIT &&
               next(infoBytes, (uint64_t)c.numPairs * PAIR_INFO_STRIDE * 4) <= limit &&
               next(pairIdBytes, (uint64_t)c.totalWg * 4) <= limit &&
               next(dpiBytes, (uint64_t)c.totalDpiEntries * 4) <= limit &&
               next(stateBytes, (uint64_t)c.totalWg * 4) <= limit &&
               next(countsBytes, (uint64_t)c.numPairs * 4) <= limit;
    }

    void add(const Chunk& c) {
        place(info, infoBytes, (uint64_t)c.numPairs * PAIR_INFO_STRIDE * 4);
        place(pairId, pairIdBytes, (uint64_t)c.totalWg * 4);
        place(dpi, dpiBytes, (uint64_t)c.totalDpiEntries * 4);
        place(state, stateBytes, (uint64_t)c.totalWg * 4);
        place(counts, countsBytes, (uint64_t)c.numPairs * 4);
    }

private:
    uint64_t align_;

    void place(std::vector<uint64_t>& offsets, uint64_t& total, uint64_t bytes) {
        offsets.push_back(alignUp(total, align_));
        total = offsets.back() + bytes;
    }
};

// Scratch buffer reused across submissions; recreated only when it must grow.
struct PooledBuffer {
    const char* label;
    WGPUBufferUsage usage;
    WGPUBuffer buf = nullptr;
    uint64_t size = 0;

    WGPUBuffer ensure(WGPUDevice device, uint64_t bytes) {
        bytes = std::max<uint64_t>(alignUp(bytes, 4), 4);
        if (!buf || size < bytes) {
            release();
            WGPUBufferDescriptor d{};
            d.label = wgpuStr(label);
            d.size = bytes;
            d.usage = usage;
            buf = wgpuDeviceCreateBuffer(device, &d);
            size = bytes;
        }
        return buf;
    }

    void release() {
        if (buf) { wgpuBufferDestroy(buf); wgpuBufferRelease(buf); buf = nullptr; size = 0; }
    }
};

// ============================================================================
// Global wgpu state and callbacks
// ============================================================================
static WGPUAdapter g_adapter = nullptr;
static WGPUDevice g_device = nullptr;
static bool g_done = false;

static void onAdapter(WGPURequestAdapterStatus s, WGPUAdapter a, WGPUStringView m, void*, void*) {
    if (s != WGPURequestAdapterStatus_Success) { fprintf(stderr, "Adapter: %.*s\n", (int)m.length, m.data); exit(1); }
    g_adapter = a;
    g_done = true;
}
static void onDevice(WGPURequestDeviceStatus s, WGPUDevice d, WGPUStringView m, void*, void*) {
    if (s != WGPURequestDeviceStatus_Success) { fprintf(stderr, "Device: %.*s\n", (int)m.length, m.data); exit(1); }
    g_device = d;
    g_done = true;
}
static void onError(WGPUDevice const*, WGPUErrorType t, WGPUStringView m, void*, void*) {
    fprintf(stderr, "[Err %u] %.*s\n", (unsigned)t, (int)m.length, m.data);
}
static void onMapped(WGPUMapAsyncStatus s, WGPUStringView m, void* done, void*) {
    if (s != WGPUMapAsyncStatus_Success) { fprintf(stderr, "Map: %.*s\n", (int)m.length, m.data); exit(1); }
    *static_cast<bool*>(done) = true;
}
static WGPUBufferMapCallbackInfo mapCallback(bool* done) {
    WGPUBufferMapCallbackInfo info{};
    info.mode = WGPUCallbackMode_AllowProcessEvents;
    info.callback = onMapped;
    info.userdata1 = done;
    return info;
}

static void meanStd(const std::vector<double>& v, double& mean, double& sd) {
    mean = 0;
    for (double x : v) mean += x;
    mean /= v.size();
    double var = 0;
    for (double x : v) var += (x - mean) * (x - mean);
    sd = v.size() > 1 ? std::sqrt(var / (v.size() - 1)) : 0.0;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <dataset.dat> <min_support> [warmup=3] [iterations=10] [chunkMB=512]\n", argv[0]);
        return 1;
    }
    const char* dsPath = argv[1];
    uint32_t minSup = (uint32_t)atoi(argv[2]);
    int nWarm = (argc > 3) ? atoi(argv[3]) : 3;
    int nIter = (argc > 4) ? atoi(argv[4]) : 10;
    uint64_t maxChunkBytes = (uint64_t)((argc > 5) ? atoll(argv[5]) : 512) * 1024 * 1024;
    if (nIter < 1) { fprintf(stderr, "iterations must be >= 1\n"); return 1; }

    printf("=== ECLAT Frequent Itemset Mining ===\n");
    printf("=== C++ / wgpu-native (in-place batched DPI + lookback, one submission per level) ===\n\n");

    // ---- wgpu init ----
    WGPUInstanceDescriptor id{};
    WGPUInstance inst = wgpuCreateInstance(&id);
    WGPURequestAdapterOptions ao{};
    ao.powerPreference = WGPUPowerPreference_HighPerformance;
    g_done = false;
    WGPURequestAdapterCallbackInfo ac{};
    ac.mode = WGPUCallbackMode_AllowProcessEvents;
    ac.callback = onAdapter;
    wgpuInstanceRequestAdapter(inst, &ao, ac);
    while (!g_done) wgpuInstanceProcessEvents(inst);

    WGPUAdapterInfo ai{};
    wgpuAdapterGetInfo(g_adapter, &ai);
    printf("Adapter: %.*s\n", (int)ai.device.length, ai.device.data ? ai.device.data : "?");

    WGPULimits adapterLimits{};
    wgpuAdapterGetLimits(g_adapter, &adapterLimits);
    WGPUBool hasTQ = wgpuAdapterHasFeature(g_adapter, WGPUFeatureName_TimestampQuery);
    WGPUBool hasSG = wgpuAdapterHasFeature(g_adapter, (WGPUFeatureName)WGPUNativeFeature_Subgroup);
    if (!hasSG) { fprintf(stderr, "The batched DPI shader requires the subgroup feature, which this adapter lacks\n"); return 1; }
    std::vector<WGPUFeatureName> feats{(WGPUFeatureName)WGPUNativeFeature_Subgroup};
    if (hasTQ) feats.push_back(WGPUFeatureName_TimestampQuery);

    WGPUDeviceDescriptor dd{};
    dd.label = wgpuStr("Dev");
    dd.requiredLimits = &adapterLimits;
    dd.defaultQueue.label = wgpuStr("Q");
    dd.uncapturedErrorCallbackInfo.callback = onError;
    dd.requiredFeatureCount = feats.size();
    dd.requiredFeatures = feats.data();
    g_done = false;
    WGPURequestDeviceCallbackInfo dc{};
    dc.mode = WGPUCallbackMode_AllowProcessEvents;
    dc.callback = onDevice;
    wgpuAdapterRequestDevice(g_adapter, &dd, dc);
    while (!g_done) wgpuInstanceProcessEvents(inst);

    WGPUQueue queue = wgpuDeviceGetQueue(g_device);

    // Buffers must respect what the device actually enforces, which can be below
    // what the adapter reports (wgpu-native allows at most a 2 GB binding).
    WGPULimits devLimits{};
    wgpuDeviceGetLimits(g_device, &devLimits);
    const uint64_t storageLimit = std::min<uint64_t>(devLimits.maxStorageBufferBindingSize, devLimits.maxBufferSize);
    const uint64_t storageAlign = devLimits.minStorageBufferOffsetAlignment;
    const uint64_t uniformAlign = devLimits.minUniformBufferOffsetAlignment;
    printf("Timestamp: %s, Subgroups: yes\n", hasTQ ? "yes" : "no");
    printf("maxBufferSize: adapter %llu, device %llu\n",
           (unsigned long long)adapterLimits.maxBufferSize, (unsigned long long)devLimits.maxBufferSize);
    printf("maxStorageBufferBindingSize: adapter %llu, device %llu\n",
           (unsigned long long)adapterLimits.maxStorageBufferBindingSize, (unsigned long long)devLimits.maxStorageBufferBindingSize);
    printf("chunk input limit: %llu MB, storage limit used: %llu bytes\n\n",
           (unsigned long long)(maxChunkBytes >> 20), (unsigned long long)storageLimit);

    // ---- Shaders ----
    const char* sdirs[] = {"shaders", "../shaders", "../../shaders"};
    const char* sdir = nullptr;
    for (auto d : sdirs) {
        char p[256];
        snprintf(p, sizeof(p), "%s/balanced_path_batched_pairinfo.wgsl", d);
        if (std::ifstream(p).good()) { sdir = d; break; }
    }
    if (!sdir) { fprintf(stderr, "Shader dir not found\n"); return 1; }
    char dpiPath[256], lbPath[256];
    snprintf(dpiPath, sizeof(dpiPath), "%s/balanced_path_batched_pairinfo.wgsl", sdir);
    snprintf(lbPath, sizeof(lbPath), "%s/set_availability_decoupled_lookback_batched_fulloutput_pairinfo.wgsl", sdir);
    auto createShader = [&](const char* label, const std::string& code) {
        WGPUShaderSourceWGSL w{};
        w.chain.sType = WGPUSType_ShaderSourceWGSL;
        w.code = {code.c_str(), code.size()};
        WGPUShaderModuleDescriptor d{};
        d.nextInChain = &w.chain;
        d.label = wgpuStr(label);
        return wgpuDeviceCreateShaderModule(g_device, &d);
    };
    // The shader is shared with the browser build, which needs `enable subgroups;`.
    // wgpu-native v24 enables subgroups through the device feature and rejects the directive.
    std::string dpiCode = readFile(dpiPath);
    const std::string enableSubgroups = "enable subgroups;";
    size_t enablePos = dpiCode.find(enableSubgroups);
    if (enablePos != std::string::npos) dpiCode.erase(enablePos, enableSubgroups.size());
    WGPUShaderModule dpiSM = createShader("DPI", dpiCode);
    WGPUShaderModule lbSM = createShader("LB", readFile(lbPath));

    // ---- Bind group layouts and pipelines ----
    auto layoutEntry = [](uint32_t binding, WGPUBufferBindingType type) {
        WGPUBindGroupLayoutEntry e{};
        e.binding = binding;
        e.visibility = WGPUShaderStage_Compute;
        e.buffer.type = type;
        return e;
    };
    WGPUBindGroupLayoutEntry dpiEntries[] = {
        layoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage),  // keysA
        layoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage),  // keysB
        layoutEntry(2, WGPUBufferBindingType_Storage),          // dpi
        layoutEntry(3, WGPUBufferBindingType_ReadOnlyStorage),  // pairInfo
        layoutEntry(4, WGPUBufferBindingType_ReadOnlyStorage),  // pairIdPerWg
        layoutEntry(5, WGPUBufferBindingType_Uniform),          // totalWg
    };
    WGPUBindGroupLayoutDescriptor dpiLayoutDesc{};
    dpiLayoutDesc.entryCount = 6;
    dpiLayoutDesc.entries = dpiEntries;
    WGPUBindGroupLayout dpiBGL = wgpuDeviceCreateBindGroupLayout(g_device, &dpiLayoutDesc);

    WGPUBindGroupLayoutEntry lbEntries[] = {
        layoutEntry(0, WGPUBufferBindingType_ReadOnlyStorage),  // keysA
        layoutEntry(1, WGPUBufferBindingType_ReadOnlyStorage),  // keysB
        layoutEntry(2, WGPUBufferBindingType_ReadOnlyStorage),  // dpi
        layoutEntry(3, WGPUBufferBindingType_Storage),          // state
        layoutEntry(4, WGPUBufferBindingType_Storage),          // pairCounts
        layoutEntry(5, WGPUBufferBindingType_Storage),          // output
        layoutEntry(6, WGPUBufferBindingType_ReadOnlyStorage),  // pairInfo
        layoutEntry(7, WGPUBufferBindingType_ReadOnlyStorage),  // pairIdPerWg
        layoutEntry(8, WGPUBufferBindingType_Uniform),          // totalWg
    };
    WGPUBindGroupLayoutDescriptor lbLayoutDesc{};
    lbLayoutDesc.entryCount = 9;
    lbLayoutDesc.entries = lbEntries;
    WGPUBindGroupLayout lbBGL = wgpuDeviceCreateBindGroupLayout(g_device, &lbLayoutDesc);

    auto makePipeline = [&](const char* label, WGPUBindGroupLayout bgl, WGPUShaderModule sm, const char* entry) {
        WGPUPipelineLayoutDescriptor pld{};
        pld.bindGroupLayoutCount = 1;
        pld.bindGroupLayouts = &bgl;
        WGPUPipelineLayout pl = wgpuDeviceCreatePipelineLayout(g_device, &pld);
        WGPUComputePipelineDescriptor cpd{};
        cpd.label = wgpuStr(label);
        cpd.layout = pl;
        cpd.compute.module = sm;
        cpd.compute.entryPoint = wgpuStr(entry);
        WGPUComputePipeline p = wgpuDeviceCreateComputePipeline(g_device, &cpd);
        wgpuPipelineLayoutRelease(pl);
        return p;
    };
    WGPUComputePipeline dpiPipe = makePipeline("DPI", dpiBGL, dpiSM, "compute_diagonals_batched");
    WGPUComputePipeline lbPipe = makePipeline("LB", lbBGL, lbSM, "decoupled_lookback_batched_kernel");
    printf("Pipelines created.\n\n");

    // ---- Parse dataset ----
    auto pt0 = std::chrono::high_resolution_clock::now();
    auto db = parseTransactionDB(dsPath);
    double parseMs = msSince(pt0);
    auto vt0 = std::chrono::high_resolution_clock::now();
    auto tidsets = buildVerticalFormat(db);
    double vertMs = msSince(vt0);
    printf("%zu txns, %zu items (parse: %.1f ms, vert: %.1f ms)\n", db.transactions.size(), tidsets.items.size(), parseMs, vertMs);
    printf("min_support=%u, %d warmup + %d iters\n\n", minSup, nWarm, nIter);

    // ---- Buffers, pools, timestamps ----
    auto makeBuf = [&](const char* label, uint64_t size, WGPUBufferUsage usage) {
        WGPUBufferDescriptor d{};
        d.label = wgpuStr(label);
        d.size = std::max<uint64_t>(size, 4);
        d.usage = usage;
        return wgpuDeviceCreateBuffer(g_device, &d);
    };
    auto bindWhole = [](uint32_t binding, WGPUBuffer buf) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = 0;
        e.size = wgpuBufferGetSize(buf);
        return e;
    };
    auto bindWindow = [](uint32_t binding, WGPUBuffer buf, uint64_t offset, uint64_t size) {
        WGPUBindGroupEntry e{};
        e.binding = binding;
        e.buffer = buf;
        e.offset = offset;
        e.size = size;
        return e;
    };

    PooledBuffer poolInfo{"pairInfo", WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst};
    PooledBuffer poolPairId{"pairIdPerWg", WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst};
    PooledBuffer poolDpi{"dpi", WGPUBufferUsage_Storage};
    PooledBuffer poolState{"state", WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst};
    PooledBuffer poolCounts{"pairCounts", WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst};
    PooledBuffer poolCountsRead{"pairCountsRead", WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead};
    PooledBuffer poolTotalWg{"totalWg", WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst};

    WGPUQuerySet querySet = nullptr;
    WGPUBuffer tsResolve = nullptr, tsRead = nullptr;
    if (hasTQ) {
        WGPUQuerySetDescriptor qsd{};
        qsd.type = WGPUQueryType_Timestamp;
        qsd.count = QUERY_COUNT;
        querySet = wgpuDeviceCreateQuerySet(g_device, &qsd);
        tsResolve = makeBuf("tsResolve", (uint64_t)QUERY_COUNT * 8, WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc);
        tsRead = makeBuf("tsRead", (uint64_t)QUERY_COUNT * 8, WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead);
    }

    // ---- Upload sets into shared arena buffers, each within the storage limit ----
    auto uploadArenas = [&](const std::vector<std::pair<const uint32_t*, uint32_t>>& sets) {
        std::vector<WgpuSetHandle> handles(sets.size());
        size_t start = 0;
        while (start < sets.size()) {
            size_t end = start;
            uint64_t bytes = 0;
            while (end < sets.size() && (end == start || bytes + (uint64_t)sets[end].second * 4 <= storageLimit)) {
                bytes += (uint64_t)sets[end].second * 4;
                end++;
            }
            if (bytes > storageLimit) { fprintf(stderr, "Tidset %zu exceeds the storage limit\n", start); exit(1); }

            WGPUBuffer buf = makeBuf("arena", bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc);
            auto owner = std::make_shared<WgpuSetBuffer>(buf);
            uint64_t offset = 0;
            for (size_t i = start; i < end; i++) {
                if (sets[i].second > 0)
                    wgpuQueueWriteBuffer(queue, buf, offset, sets[i].first, (size_t)sets[i].second * 4);
                handles[i] = {owner, (uint32_t)(offset / 4), sets[i].second};
                offset += (uint64_t)sets[i].second * 4;
            }
            start = end;
        }
        return handles;
    };

    // ---- Encode all chunks of one submission into one command buffer, wait once ----
    struct SubmitTiming { double dpiMs = 0, lbMs = 0, totalMs = 0; };
    auto submitChunks = [&](const std::vector<Chunk>& chunks, const SubmitLayout& lay,
                            std::vector<std::vector<uint32_t>>& counts,
                            std::vector<std::shared_ptr<WgpuSetBuffer>>& outputs,
                            WgpuOutputPool& outPool) {
        size_t n = chunks.size();
        std::vector<uint32_t> infoHost(lay.infoBytes / 4, 0), pairIdHost(lay.pairIdBytes / 4, 0);
        std::vector<uint32_t> totalWgHost((size_t)(n * uniformAlign / 4), 0);
        for (size_t c = 0; c < n; c++) {
            std::copy(chunks[c].pairInfo.begin(), chunks[c].pairInfo.end(), infoHost.begin() + lay.info[c] / 4);
            std::copy(chunks[c].pairIdPerWg.begin(), chunks[c].pairIdPerWg.end(), pairIdHost.begin() + lay.pairId[c] / 4);
            totalWgHost[c * uniformAlign / 4] = chunks[c].totalWg;
        }

        WGPUBuffer bInfo = poolInfo.ensure(g_device, lay.infoBytes);
        WGPUBuffer bPairId = poolPairId.ensure(g_device, lay.pairIdBytes);
        WGPUBuffer bDpi = poolDpi.ensure(g_device, lay.dpiBytes);
        WGPUBuffer bState = poolState.ensure(g_device, lay.stateBytes);
        WGPUBuffer bCounts = poolCounts.ensure(g_device, lay.countsBytes);
        WGPUBuffer bCountsRead = poolCountsRead.ensure(g_device, lay.countsBytes);
        WGPUBuffer bTotalWg = poolTotalWg.ensure(g_device, n * uniformAlign);
        wgpuQueueWriteBuffer(queue, bInfo, 0, infoHost.data(), infoHost.size() * 4);
        wgpuQueueWriteBuffer(queue, bPairId, 0, pairIdHost.data(), pairIdHost.size() * 4);
        wgpuQueueWriteBuffer(queue, bTotalWg, 0, totalWgHost.data(), totalWgHost.size() * 4);

        WGPUCommandEncoderDescriptor ed{};
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, &ed);
        wgpuCommandEncoderClearBuffer(enc, bState, 0, lay.stateBytes);
        wgpuCommandEncoderClearBuffer(enc, bCounts, 0, lay.countsBytes);

        std::vector<WGPUBindGroup> bindGroups;
        bindGroups.reserve(2 * n);
        counts.assign(n, {});
        outputs.assign(n, nullptr);

        auto runPass = [&](WGPUComputePipeline pipe, WGPUBindGroup bg, uint32_t gx, uint32_t gy, uint32_t tsBegin) {
            WGPUComputePassTimestampWrites tw{};
            WGPUComputePassDescriptor cd{};
            if (querySet) {
                tw.querySet = querySet;
                tw.beginningOfPassWriteIndex = tsBegin;
                tw.endOfPassWriteIndex = tsBegin + 1;
                cd.timestampWrites = &tw;
            }
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &cd);
            wgpuComputePassEncoderSetPipeline(pass, pipe);
            wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        };

        for (size_t c = 0; c < n; c++) {
            const Chunk& ch = chunks[c];
            uint64_t outBytes = (uint64_t)std::max(ch.maxOutputSize, 1u) * 4;
            WGPUBuffer bOut;
            auto freeIt = outPool.freeBuffers.lower_bound(outBytes);
            if (freeIt != outPool.freeBuffers.end()) {
                bOut = freeIt->second;
                outPool.freeBuffers.erase(freeIt);
            } else {
                bOut = makeBuf("out", outBytes, WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc);
            }
            outputs[c] = std::make_shared<WgpuSetBuffer>(bOut, &outPool);

            WGPUBindGroupEntry dpiBind[] = {
                bindWhole(0, ch.bufA->buf),
                bindWhole(1, ch.bufB->buf),
                bindWindow(2, bDpi, lay.dpi[c], (uint64_t)ch.totalDpiEntries * 4),
                bindWindow(3, bInfo, lay.info[c], (uint64_t)ch.numPairs * PAIR_INFO_STRIDE * 4),
                bindWindow(4, bPairId, lay.pairId[c], (uint64_t)ch.totalWg * 4),
                bindWindow(5, bTotalWg, c * uniformAlign, 4),
            };
            WGPUBindGroupDescriptor dpiBGD{};
            dpiBGD.layout = dpiBGL;
            dpiBGD.entryCount = 6;
            dpiBGD.entries = dpiBind;
            WGPUBindGroup dpiBG = wgpuDeviceCreateBindGroup(g_device, &dpiBGD);

            WGPUBindGroupEntry lbBind[] = {
                bindWhole(0, ch.bufA->buf),
                bindWhole(1, ch.bufB->buf),
                bindWindow(2, bDpi, lay.dpi[c], (uint64_t)ch.totalDpiEntries * 4),
                bindWindow(3, bState, lay.state[c], (uint64_t)ch.totalWg * 4),
                bindWindow(4, bCounts, lay.counts[c], (uint64_t)ch.numPairs * 4),
                bindWhole(5, bOut),
                bindWindow(6, bInfo, lay.info[c], (uint64_t)ch.numPairs * PAIR_INFO_STRIDE * 4),
                bindWindow(7, bPairId, lay.pairId[c], (uint64_t)ch.totalWg * 4),
                bindWindow(8, bTotalWg, c * uniformAlign, 4),
            };
            WGPUBindGroupDescriptor lbBGD{};
            lbBGD.layout = lbBGL;
            lbBGD.entryCount = 9;
            lbBGD.entries = lbBind;
            WGPUBindGroup lbBG = wgpuDeviceCreateBindGroup(g_device, &lbBGD);
            bindGroups.push_back(dpiBG);
            bindGroups.push_back(lbBG);

            uint32_t spw = DPI_WG_SIZE / SUBGROUP_SIZE;
            uint32_t dpiBlocks = (ch.totalWg + spw - 1) / spw;
            runPass(dpiPipe, dpiBG, std::min(dpiBlocks, MAX_DISPATCH_X), (dpiBlocks + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X,
                    (uint32_t)(4 * c));
            runPass(lbPipe, lbBG, std::min(ch.totalWg, MAX_DISPATCH_X), (ch.totalWg + MAX_DISPATCH_X - 1) / MAX_DISPATCH_X,
                    (uint32_t)(4 * c + 2));
        }

        uint64_t tsBytes = 4 * n * 8;
        if (querySet) {
            wgpuCommandEncoderResolveQuerySet(enc, querySet, 0, (uint32_t)(4 * n), tsResolve, 0);
            wgpuCommandEncoderCopyBufferToBuffer(enc, tsResolve, 0, tsRead, 0, tsBytes);
        }
        wgpuCommandEncoderCopyBufferToBuffer(enc, bCounts, 0, bCountsRead, 0, lay.countsBytes);
        WGPUCommandBufferDescriptor cbd{};
        WGPUCommandBuffer cb = wgpuCommandEncoderFinish(enc, &cbd);
        wgpuQueueSubmit(queue, 1, &cb);
        wgpuCommandBufferRelease(cb);
        wgpuCommandEncoderRelease(enc);
        for (WGPUBindGroup bg : bindGroups) wgpuBindGroupRelease(bg);

        // The only wait of this submission.
        bool countsMapped = false, tsMapped = !querySet;
        wgpuBufferMapAsync(bCountsRead, WGPUMapMode_Read, 0, lay.countsBytes, mapCallback(&countsMapped));
        if (querySet) wgpuBufferMapAsync(tsRead, WGPUMapMode_Read, 0, tsBytes, mapCallback(&tsMapped));
        while (!countsMapped || !tsMapped) wgpuDevicePoll(g_device, true, nullptr);

        auto pc = (const uint32_t*)wgpuBufferGetConstMappedRange(bCountsRead, 0, lay.countsBytes);
        for (size_t c = 0; c < n; c++)
            counts[c].assign(pc + lay.counts[c] / 4, pc + lay.counts[c] / 4 + chunks[c].numPairs);
        wgpuBufferUnmap(bCountsRead);

        SubmitTiming t;
        if (querySet) {
            auto ts = (const uint64_t*)wgpuBufferGetConstMappedRange(tsRead, 0, tsBytes);
            for (size_t c = 0; c < n; c++) {
                t.dpiMs += (double)(ts[4 * c + 1] - ts[4 * c]) / 1e6;
                t.lbMs += (double)(ts[4 * c + 3] - ts[4 * c + 2]) / 1e6;
                t.totalMs += (double)(ts[4 * c + 3] - ts[4 * c]) / 1e6;
            }
            wgpuBufferUnmap(tsRead);
        }
        return t;
    };

    // ---- GPU ECLAT ----
    struct RunResult {
        std::vector<FrequentItemset> frequent;
        double dpiMs = 0, lbMs = 0, kernelMs = 0, e2eMs = 0;
        uint32_t chunks = 0, submits = 0;
    };
    auto eclatGPU = [&](bool verbose) {
        RunResult r;
        auto t0 = std::chrono::high_resolution_clock::now();
        // Declared before every tidset handle, so all output buffers are back in the pool when it is destroyed.
        WgpuOutputPool outPool;

        // Level-1: all frequent base tidsets share arena buffers and are read in place.
        std::vector<uint32_t> freqItems;
        std::vector<std::pair<const uint32_t*, uint32_t>> baseSets;
        for (size_t i = 0; i < tidsets.items.size(); i++) {
            uint32_t len = (uint32_t)tidsets.tids[i].size();
            if (len >= minSup) {
                freqItems.push_back(tidsets.items[i]);
                r.frequent.push_back({{tidsets.items[i]}, len});
                baseSets.push_back({tidsets.tids[i].data(), len});
            }
        }
        std::vector<WgpuSetHandle> uploaded = uploadArenas(baseSets);
        std::map<uint32_t, WgpuSetHandle> baseHandles;
        for (size_t k = 0; k < freqItems.size(); k++) baseHandles.emplace(freqItems[k], uploaded[k]);
        std::sort(freqItems.begin(), freqItems.end());
        if (verbose) printf("  Level 1: %zu frequent items\n", freqItems.size());

        // Itemsets with the same classId share every item but the last (an equivalence class).
        struct LevelItem { std::vector<uint32_t> itemset; WgpuSetHandle tidset; size_t classId; };
        std::vector<LevelItem> curLevel;
        for (uint32_t item : freqItems) curLevel.push_back({{item}, baseHandles.at(item), 0});

        int level = 2;
        while (!curLevel.empty()) {
            struct CandPair { WgpuSetHandle tidA, tidB; size_t parentIdx; uint32_t childItem; };
            // Textbook ECLAT: intersect every two itemsets of one equivalence class. Members of a
            // class are contiguous and sorted by their last item.
            std::vector<CandPair> pairs;
            for (size_t i = 0; i < curLevel.size(); i++) {
                for (size_t j = i + 1; j < curLevel.size() && curLevel[j].classId == curLevel[i].classId; j++)
                    pairs.push_back({curLevel[i].tidset, curLevel[j].tidset, i, curLevel[j].itemset.back()});
            }
            if (pairs.empty()) break;
            if (verbose) printf("  Level %d: %zu pairs from %zu parents\n", level, pairs.size(), curLevel.size());

            std::vector<uint32_t> allCounts(pairs.size(), 0);
            std::vector<WgpuSetHandle> survivors(pairs.size());
            std::vector<Chunk> pending;
            SubmitLayout layout(storageAlign);
            uint32_t levelChunks = 0, levelSubmits = 0;

            auto submitPending = [&]() {
                if (pending.empty()) return;
                std::vector<std::vector<uint32_t>> counts;
                std::vector<std::shared_ptr<WgpuSetBuffer>> outputs;
                SubmitTiming t = submitChunks(pending, layout, counts, outputs, outPool);
                r.dpiMs += t.dpiMs;
                r.lbMs += t.lbMs;
                r.kernelMs += t.totalMs;
                for (size_t c = 0; c < pending.size(); c++) {
                    const Chunk& ch = pending[c];
                    for (uint32_t i = 0; i < ch.numPairs; i++) {
                        allCounts[ch.pairStart + i] = counts[c][i];
                        if (counts[c][i] >= minSup)
                            survivors[ch.pairStart + i] = {outputs[c], ch.pairOutputOffsets[i], counts[c][i]};
                    }
                }
                pending.clear();
                layout = SubmitLayout(storageAlign);
                levelSubmits++;
            };

            // A chunk binds one A buffer and one B buffer, so it also ends where either
            // source buffer changes; its input stays within maxChunkBytes and its output
            // buffer within the storage limit.
            for (size_t cStart = 0; cStart < pairs.size();) {
                const WgpuSetBuffer* bufA = pairs[cStart].tidA.buffer.get();
                const WgpuSetBuffer* bufB = pairs[cStart].tidB.buffer.get();
                size_t cEnd = cStart;
                uint64_t inBytes = 0, outBytes = 0;
                while (cEnd < pairs.size()) {
                    const CandPair& p = pairs[cEnd];
                    if (p.tidA.buffer.get() != bufA || p.tidB.buffer.get() != bufB) break;
                    uint64_t pairIn = p.tidA.byteLen() + p.tidB.byteLen();
                    uint64_t pairOut = (uint64_t)std::min(p.tidA.length, p.tidB.length) * 4;
                    if (cEnd > cStart && (inBytes + pairIn > maxChunkBytes || outBytes + pairOut > storageLimit)) break;
                    inBytes += pairIn;
                    outBytes += pairOut;
                    cEnd++;
                }
                if (outBytes > storageLimit) { fprintf(stderr, "Pair %zu output exceeds the storage limit\n", cStart); exit(1); }

                std::vector<WgpuSetHandle> aH(cEnd - cStart), bH(cEnd - cStart);
                for (size_t i = cStart; i < cEnd; i++) {
                    aH[i - cStart] = pairs[i].tidA;
                    bH[i - cStart] = pairs[i].tidB;
                }
                Chunk ch = prepareChunk(aH, bH, cStart);
                levelChunks++;

                if (ch.totalWg > 0) {
                    if (!layout.fits(ch, storageLimit)) submitPending();
                    if (!layout.fits(ch, storageLimit)) { fprintf(stderr, "Chunk at pair %zu exceeds the storage limit\n", cStart); exit(1); }
                    layout.add(ch);
                    pending.push_back(std::move(ch));
                }
                cStart = cEnd;
            }
            submitPending();
            r.chunks += levelChunks;
            r.submits += levelSubmits;

            std::vector<LevelItem> nextLevel;
            uint32_t frequentCount = 0;
            for (size_t pid = 0; pid < pairs.size(); pid++) {
                if (allCounts[pid] >= minSup) {
                    auto ni = curLevel[pairs[pid].parentIdx].itemset;
                    ni.push_back(pairs[pid].childItem);
                    r.frequent.push_back({ni, allCounts[pid]});
                    frequentCount++;
                    if (survivors[pid].buffer) nextLevel.push_back({ni, survivors[pid], pairs[pid].parentIdx});
                }
            }
            if (verbose) printf("  Level %d: %u frequent, %u chunks, %u submits\n", level, frequentCount, levelChunks, levelSubmits);
            curLevel = std::move(nextLevel);
            level++;
        }

        r.e2eMs = msSince(t0);
        return r;
    };

    // ---- Warmup + timed runs ----
    for (int w = 0; w < nWarm; w++) eclatGPU(w == 0);

    std::vector<double> dpiTimes, lbTimes, kernelTimes, e2eTimes;
    RunResult last;
    for (int i = 0; i < nIter; i++) {
        last = eclatGPU(nWarm == 0 && i == 0);
        dpiTimes.push_back(last.dpiMs);
        lbTimes.push_back(last.lbMs);
        kernelTimes.push_back(last.kernelMs);
        e2eTimes.push_back(last.e2eMs);
    }

    // ---- CPU validation ----
    printf("\n  CPU ECLAT for validation...\n");
    auto ct0 = std::chrono::high_resolution_clock::now();
    auto cpuRes = eclatCPU(tidsets, minSup);
    double cpuMs = msSince(ct0);

    bool match = last.frequent.size() == cpuRes.size();
    if (match) {
        auto cmp = [](const FrequentItemset& a, const FrequentItemset& b) { return a.itemset < b.itemset; };
        auto gs = last.frequent;
        auto cs = cpuRes;
        std::sort(gs.begin(), gs.end(), cmp);
        std::sort(cs.begin(), cs.end(), cmp);
        for (size_t i = 0; i < gs.size() && match; i++)
            if (gs[i].itemset != cs[i].itemset || gs[i].support != cs[i].support) match = false;
    }

    double dpiMean, dpiSd, lbMean, lbSd, kMean, kSd, eMean, eSd;
    meanStd(dpiTimes, dpiMean, dpiSd);
    meanStd(lbTimes, lbMean, lbSd);
    meanStd(kernelTimes, kMean, kSd);
    meanStd(e2eTimes, eMean, eSd);

    printf("\n  RESULTS (%d runs)\n", nIter);
    printf("    Frequent itemsets: %zu (GPU) vs %zu (CPU)\n", last.frequent.size(), cpuRes.size());
    printf("    Match:             %s\n", match ? "PASS" : "FAIL");
    printf("    Chunks per run:    %u\n", last.chunks);
    printf("    Submits per run:   %u\n", last.submits);
    printf("    Parse + Vertical:  %.1f ms\n", parseMs + vertMs);
    printf("    DPI total:         %.3f ms\n", dpiMean);
    printf("    Lookback total:    %.3f ms\n", lbMean);
    printf("    GPU Kernel total:  %.3f ms (std %.3f)\n", kMean, kSd);
    printf("    End-to-end (avg):  %.1f ms (std %.1f, min %.1f, max %.1f)\n", eMean, eSd,
           *std::min_element(e2eTimes.begin(), e2eTimes.end()), *std::max_element(e2eTimes.begin(), e2eTimes.end()));
    printf("    CPU baseline:      %.1f ms\n\n", cpuMs);

    // ---- Cleanup ----
    for (PooledBuffer* p : {&poolInfo, &poolPairId, &poolDpi, &poolState, &poolCounts, &poolCountsRead, &poolTotalWg})
        p->release();
    if (querySet) {
        wgpuQuerySetDestroy(querySet);
        wgpuQuerySetRelease(querySet);
        wgpuBufferRelease(tsResolve);
        wgpuBufferRelease(tsRead);
    }
    wgpuComputePipelineRelease(dpiPipe);
    wgpuComputePipelineRelease(lbPipe);
    wgpuBindGroupLayoutRelease(dpiBGL);
    wgpuBindGroupLayoutRelease(lbBGL);
    wgpuShaderModuleRelease(dpiSM);
    wgpuShaderModuleRelease(lbSM);
    wgpuQueueRelease(queue);
    wgpuDeviceRelease(g_device);
    wgpuAdapterRelease(g_adapter);
    wgpuInstanceRelease(inst);
    return match ? 0 : 2;
}
