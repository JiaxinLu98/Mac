import { runUnified4PhaseVs2PhaseTest, runUnifiedSingleOpTest, run2PhaseOnlyTest, runFusionBenchmark } from './balanced_path/common/not_by_key/test_unified_4phase_vs_2phase';
import { runUnifiedByKey4PhaseVs2PhaseTest, runUnifiedByKeySingleOpTest } from './balanced_path/common/by_key/test_unified_4phase_vs_2phase_by_key';
import { runEquiJoinTest } from './applications/equi_join/equi_join';
import { runSemiJoinTest } from './applications/semi_join/semi_join';
import { runAntiJoinTest } from './applications/anti_join/anti_join';
import { runMultiWayJoinTest } from './applications/multi_way_join/multi_way_join';
import { runTPCHQ12Test } from './applications/tpch_q12/tpch_q12';
import { runEclatV2Test, runEclatInPlaceTest } from './applications/eclat/eclat';
import { runJaccardGenomeTest } from './applications/jaccard_genome/jaccard_genome';
import { runGatherBenchTest } from './applications/gather_bench/gather_bench'; // rebuttal experiment
// (not in the M5 Pro package) import { runReachingDefinitionsV0Test } from '../p3hpc/reaching_definitions/test_reaching_definitions'; // P3HPC
// (not in the M5 Pro package) import { runReachingDefinitionsRealTest } from '../p3hpc/reaching_definitions/test_reaching_definitions_real'; // P3HPC
// (not in the M5 Pro package) import { runReachingDefinitionsBatchedTest } from '../p3hpc/reaching_definitions/test_reaching_definitions_batched'; // P3HPC v1
// (not in the M5 Pro package) import { runRDBenchmark } from '../p3hpc/reaching_definitions/test_rd_benchmark'; // P3HPC benchmark + logging (v1/v2a/v2b)
// (not in the M5 Pro package) import { runRDV2Test } from '../p3hpc/reaching_definitions/test_rd_v2'; // P3HPC v2a/v2b validation
// (not in the M5 Pro package) import { runRDBitvectorTest } from '../p3hpc/reaching_definitions/test_rd_bitvector'; // P3HPC bitvector baseline validation
// (not in the M5 Pro package) import { runRDUniverseSweep } from '../p3hpc/reaching_definitions/test_rd_universe_sweep'; // P3HPC universe crossover sweep

(async () => {
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    // Get a GPU device to render with. ?gpu=low-power or ?gpu=high-performance selects the adapter
    // on machines with two GPUs (e.g. the laptop's Intel iGPU or its RTX 3060).
    const gpuPreference = new URLSearchParams(window.location.search).get('gpu');
    let adapter = await navigator.gpu.requestAdapter(
        gpuPreference === 'low-power' || gpuPreference === 'high-performance'
            ? { powerPreference: gpuPreference }
            : undefined);
    const adapterInfo = (adapter as any)?.info;
    console.log('[adapter]', JSON.stringify({
        requested: gpuPreference ?? 'default',
        vendor: adapterInfo?.vendor,
        architecture: adapterInfo?.architecture,
        device: adapterInfo?.device,
        description: adapterInfo?.description,
    }));
	const supportsTimestampQueries = adapter?.features.has('timestamp-query');
    const supportsSubgroups = adapter?.features.has('subgroups' as GPUFeatureName);
    const requiredFeatures: GPUFeatureName[] = [];
    if (supportsTimestampQueries) requiredFeatures.push('timestamp-query');
    if (supportsSubgroups) requiredFeatures.push('subgroups' as GPUFeatureName);
    let gpuDeviceDesc = {
        requiredLimits: {
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        },
        requiredFeatures,
    };
    let device = await adapter.requestDevice(gpuDeviceDesc);

    console.log("maxBufferSize =", adapter.limits.maxBufferSize);
    console.log("maxStorageBufferBindingSize =", adapter.limits.maxStorageBufferBindingSize);
    console.log("subgroups =", supportsSubgroups ? "supported" : "NOT supported");

    // ?app=eclat runs the ECLAT benchmark instead of the default experiment below.
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('app') === 'eclat') {
        await runEclatInPlaceTest(device, urlParams);
        return;
    }
    // ?app=micro&sizes=1,2,...,128 (2-step intersection, e6), ?app=tpch, ?app=jaccard.
    // Each prints [bench-done] so a headless runner can exit.
    if (urlParams.get('app') === 'micro') {
        await run2PhaseOnlyTest(device, 0, urlParams.get('sizes')?.split(','));
        console.log('[bench-done]');
        return;
    }
    // ?app=fusion[&ops=0,2&range=e2&sizes=1,...,128&w=10&n=100]: 4-step vs 2-step pipeline.
    if (urlParams.get('app') === 'fusion') {
        await runFusionBenchmark(device);
        console.log('[bench-done]');
        return;
    }
    if (urlParams.get('app') === 'tpch') {
        await runTPCHQ12Test(device);
        console.log('[bench-done]');
        return;
    }
    if (urlParams.get('app') === 'jaccard') {
        await runJaccardGenomeTest(device);
        console.log('[bench-done]');
        return;
    }

    // =================================
    // P3HPC: REACHING DEFINITIONS v0 (correctness validation, GPU vs CPU)
    // =================================
    // await runReachingDefinitionsV0Test(device);  // v0: synthetic CFGs
    // await runReachingDefinitionsRealTest(device);  // v0: real-program CFGs (needs dev server)
    // await runReachingDefinitionsBatchedTest(device);  // v1-gterm + v1 correctness, all datasets (needs dev server)
    // await runRDV2Test(device);  // v2a/v2b correctness, all datasets (needs dev server)
    // await runRDBitvectorTest(device);  // bitvector gterm + plain correctness (needs dev server)
    // P3HPC is not in the M5 Pro package, so there is no default experiment here: pass ?app=fusion|micro|eclat.
    console.log('no ?app= given: use ?app=fusion, ?app=micro or ?app=eclat');
    // await runRDUniverseSweep(device);  // universe crossover sweep (?ds=sqlite3&groups=1,8,64,512,0&n=10&w=2)
    return;

    // ===== Rebuttal: WebGPU 2-phase intersection timing + variance (CV / median / p95) =====
    // Reads the [stats] lines printed by two_phase_pipeline.ts. (Remove to restore.)
    // await run2PhaseOnlyTest(device, 0);  // 0 = intersection
    // await runGatherBenchTest(device); return;  // earlier packing benchmark

    // =================================
    // UNIFIED: All 4 Ops 4-Phase vs 2-Phase
    // =================================
    // await runUnifiedSingleOpTest(device, 0);  // intersection only
    // await runUnifiedSingleOpTest(device, 1);  // difference only
    // await runUnifiedSingleOpTest(device, 2);  // union only
    // await runUnifiedSingleOpTest(device, 3);  // sym_difference only

    // =================================
    // 2-PHASE ONLY (single op benchmark)
    // =================================
    // await run2PhaseOnlyTest(device, 0);  // intersection only
    // await run2PhaseOnlyTest(device, 1);  // difference only
    // await run2PhaseOnlyTest(device, 2);  // union only
    // await run2PhaseOnlyTest(device, 3);  // sym_difference only

    // =================================
    // UNIFIED BY KEY: All 4 Ops 4-Phase vs 2-Phase
    // =================================
    // await runUnifiedByKeySingleOpTest(device, 0);  // intersection by key only
    // await runUnifiedByKeySingleOpTest(device, 1);  // difference by key only
    // await runUnifiedByKeySingleOpTest(device, 2);  // union by key only
    // await runUnifiedByKeySingleOpTest(device, 3);  // sym_difference by key only


    // =================================
    // MULTI-WAY JOIN (Application)
    // =================================
    // await runMultiWayJoinTest(device);

    // =================================
    // TPC-H Q12 (Application)
    // =================================
    // await runTPCHQ12Test(device);

    // =================================
    // ECLAT Frequent Itemset Mining (Application)
    // =================================
    // await runEclatV2Test(device);

    // =================================
    // JACCARD GENOME (Application)
    // =================================
    await runJaccardGenomeTest(device);
})();

