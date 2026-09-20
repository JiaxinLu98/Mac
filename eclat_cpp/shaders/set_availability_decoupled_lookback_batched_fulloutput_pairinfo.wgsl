// ============================================================================
// Unified Set Availability - Decoupled Lookback - BATCHED FULL OUTPUT
//
// Variant of batched lookback that writes BOTH per-pair counts AND
// intersection elements to output buffers. Used by ECLAT where
// tidsets must stay on GPU for the next level.
//
// COMPACT METADATA: pair-level fields come from pairInfo (8 u32 per pair, same
// layout as balanced_path_batched_pairinfo.wgsl); each workgroup only stores
// its pair id. keysA/keysB are the tidsets' own buffers (read in place).
//
// Output elements are written to a flat output[] array using global
// prefix sum offsets. Per-pair counts go to pairCounts[pairId].
// ============================================================================

const OP_MODE: u32 = 0u;

// ============================================================================
// Bindings (8 storage + 1 uniform)
// ============================================================================
@group(0) @binding(0) var<storage, read> keysA: array<u32>;
@group(0) @binding(1) var<storage, read> keysB: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> pairCounts: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<u32>;
@group(0) @binding(6) var<storage, read> pairInfo: array<u32>;
@group(0) @binding(7) var<storage, read> pairIdPerWg: array<u32>;
@group(0) @binding(8) var<uniform> totalWg_uniform: u32;

// ============================================================================
// Constants
// ============================================================================
const NT: u32 = 256u;
const VT: u32 = 12u;
const NV: u32 = NT * VT;

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;
const PAIR_INFO_STRIDE: u32 = 8u;  // same as batched DPI shader

const STATUS_NOT_READY: u32 = 0u;
const STATUS_PARTIAL: u32 = 1u;
const STATUS_INCLUSIVE: u32 = 2u;
const STATUS_SHIFT: u32 = 30u;
const VALUE_MASK: u32 = 0x3FFFFFFFu;

const POS_INF: u32 = 0xFFFFFFFFu;
const NEG_INF: u32 = 0u;

// ============================================================================
// Shared Memory
// ============================================================================
var<workgroup> keys_shared: array<u32, 3090>;
var<workgroup> wg_a0: u32;
var<workgroup> wg_a1: u32;
var<workgroup> wg_b0: u32;
var<workgroup> wg_b1: u32;
var<workgroup> wg_a_count: u32;
var<workgroup> wg_b_count: u32;
var<workgroup> wg_b_start: u32;
var<workgroup> wg_extended: bool;
var<workgroup> wg_bit0: u32;
var<workgroup> shared_scan: array<u32, NT>;
var<workgroup> wg_local_total: u32;
var<workgroup> wg_exclusive_prefix: u32;

// Per-pair info loaded from pairInfo
var<workgroup> wg_pairAStart: u32;
var<workgroup> wg_pairALen: u32;
var<workgroup> wg_pairBStart: u32;
var<workgroup> wg_pairBLen: u32;
var<workgroup> wg_localWgIdx: u32;
var<workgroup> wg_pairNumWg: u32;
var<workgroup> wg_globalWgOffset: u32;
var<workgroup> wg_dpiOffset: u32;
var<workgroup> wg_pairOutOffset: u32;

// ============================================================================
// State Pack/Unpack
// ============================================================================
fn pack_state(flag: u32, value: u32) -> u32 {
    return (flag << STATUS_SHIFT) | (value & VALUE_MASK);
}
fn unpack_flag(packed: u32) -> u32 {
    return packed >> STATUS_SHIFT;
}
fn unpack_value(packed: u32) -> u32 {
    return packed & VALUE_MASK;
}

fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad2ToShared - reads from keysA/keysB with pair offsets
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    if (tid == 0u) {
        keys_shared[0] = NEG_INF;
        keys_shared[a_load_count + 1u] = POS_INF;
        keys_shared[b_shared_start] = NEG_INF;
        keys_shared[b_shared_start + b_load_count + 1u] = POS_INF;
    }

    // Load A elements (keysA with pair offset)
    if (tid < a_load_count) { keys_shared[tid + 1u] = keysA[a_global_offset + tid]; }
    if (tid + 256u < a_load_count) { keys_shared[tid + 257u] = keysA[a_global_offset + tid + 256u]; }
    if (tid + 512u < a_load_count) { keys_shared[tid + 513u] = keysA[a_global_offset + tid + 512u]; }
    if (tid + 768u < a_load_count) { keys_shared[tid + 769u] = keysA[a_global_offset + tid + 768u]; }
    if (tid + 1024u < a_load_count) { keys_shared[tid + 1025u] = keysA[a_global_offset + tid + 1024u]; }
    if (tid + 1280u < a_load_count) { keys_shared[tid + 1281u] = keysA[a_global_offset + tid + 1280u]; }
    if (tid + 1536u < a_load_count) { keys_shared[tid + 1537u] = keysA[a_global_offset + tid + 1536u]; }
    if (tid + 1792u < a_load_count) { keys_shared[tid + 1793u] = keysA[a_global_offset + tid + 1792u]; }
    if (tid + 2048u < a_load_count) { keys_shared[tid + 2049u] = keysA[a_global_offset + tid + 2048u]; }
    if (tid + 2304u < a_load_count) { keys_shared[tid + 2305u] = keysA[a_global_offset + tid + 2304u]; }
    if (tid + 2560u < a_load_count) { keys_shared[tid + 2561u] = keysA[a_global_offset + tid + 2560u]; }
    if (tid + 2816u < a_load_count) { keys_shared[tid + 2817u] = keysA[a_global_offset + tid + 2816u]; }

    // Load B elements (keysB with pair offset)
    if (tid < b_load_count) { keys_shared[b_shared_start + tid + 1u] = keysB[b_global_offset + tid]; }
    if (tid + 256u < b_load_count) { keys_shared[b_shared_start + tid + 257u] = keysB[b_global_offset + tid + 256u]; }
    if (tid + 512u < b_load_count) { keys_shared[b_shared_start + tid + 513u] = keysB[b_global_offset + tid + 512u]; }
    if (tid + 768u < b_load_count) { keys_shared[b_shared_start + tid + 769u] = keysB[b_global_offset + tid + 768u]; }
    if (tid + 1024u < b_load_count) { keys_shared[b_shared_start + tid + 1025u] = keysB[b_global_offset + tid + 1024u]; }
    if (tid + 1280u < b_load_count) { keys_shared[b_shared_start + tid + 1281u] = keysB[b_global_offset + tid + 1280u]; }
    if (tid + 1536u < b_load_count) { keys_shared[b_shared_start + tid + 1537u] = keysB[b_global_offset + tid + 1536u]; }
    if (tid + 1792u < b_load_count) { keys_shared[b_shared_start + tid + 1793u] = keysB[b_global_offset + tid + 1792u]; }
    if (tid + 2048u < b_load_count) { keys_shared[b_shared_start + tid + 2049u] = keysB[b_global_offset + tid + 2048u]; }
    if (tid + 2304u < b_load_count) { keys_shared[b_shared_start + tid + 2305u] = keysB[b_global_offset + tid + 2304u]; }
    if (tid + 2560u < b_load_count) { keys_shared[b_shared_start + tid + 2561u] = keysB[b_global_offset + tid + 2560u]; }
    if (tid + 2816u < b_load_count) { keys_shared[b_shared_start + tid + 2817u] = keysB[b_global_offset + tid + 2816u]; }
}

// ============================================================================
// Local Balanced Path functions (identical to original - operate on shared memory)
// ============================================================================
fn lower_bound_local_a(end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn lower_bound_local_b(b_start: u32, end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid + 1u] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn upper_bound_local_b(b_start: u32, range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid + 1u] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn merge_path_local(a_count: u32, b_start: u32, b_count: u32, diag: u32) -> u32 {
    var lo: u32 = select(0u, diag - b_count, diag > b_count);
    var hi: u32 = min(diag, a_count);
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let a_key = keys_shared[mid + 1u];
        let b_key = keys_shared[b_start + diag - mid];
        if (a_key <= b_key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn balanced_path_local(a_count: u32, b_start: u32, b_count: u32, diag: i32) -> vec2<u32> {
    let diag_u = u32(max(0, diag));
    if (diag_u == 0u) { return vec2<u32>(0u, 0u); }
    if (diag_u >= a_count + b_count) { return vec2<u32>(a_count, 0u); }

    let p = merge_path_local(a_count, b_start, b_count, diag_u);
    var a_index = p;
    var b_index = diag_u - p;
    var star: u32 = 0u;

    if (b_index < b_count) {
        let x = keys_shared[b_start + b_index + 1u];
        let a_start = lower_bound_local_a(a_index, x);
        let b_start_run = lower_bound_local_b(b_start, b_index, x);
        let a_run = a_index - a_start;
        let b_run = b_index - b_start_run;
        let x_count = a_run + b_run;
        var b_advance = max(x_count >> 1u, x_count - a_run);
        var b_end_hint = min(b_count, b_start_run + b_advance + 1u);
        b_end_hint = max(b_end_hint, min(b_count, b_index + 1u));
        let b_run_end = upper_bound_local_b(b_start, b_index, b_end_hint, x);
        let actual_b_run = b_run_end - b_start_run;
        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        star = select(0u, 1u, round_up);
        a_index = a_start + a_advance;
    }
    return vec2<u32>(a_index, star);
}

// ============================================================================
// Serial Set Operation - intersection, with result storage for scatter
// Returns commit bitmask. Stores results in arrays for later scatter.
// ============================================================================
fn serial_intersection(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    star: u32,
    b_adjust: u32,
    extended: bool,
    results: ptr<function, array<u32, 12>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;
    var a_key = keys_shared[a_idx + 1u];
    var b_key = keys_shared[b_idx + 1u];

    for (var i: u32 = 0u; i < VT; i++) {
        var test: bool;
        if (extended) {
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else {
            test = (i32(a_idx) + i32(b_idx) < end_diag);
        }
        if (!test) { break; }

        let pA = a_key < b_key;
        let pB = b_key < a_key;

        // Intersection: always output a_key
        (*results)[i] = a_key;

        // Commit when equal (pA == pB, both false)
        if (pA == pB) {
            commit |= (1u << i);
        }

        a_idx += select(0u, 1u, !pB);
        b_idx += select(0u, 1u, !pA);
        a_key = select(a_key, keys_shared[a_idx + 1u], !pB);
        b_key = select(b_key, keys_shared[b_idx + 1u], !pA);
    }

    return commit;
}

// ============================================================================
// Workgroup Exclusive Scan
// ============================================================================
fn workgroup_exclusive_scan_with_total(tid: u32, value: u32, total_ptr: ptr<function, u32>) -> u32 {
    shared_scan[tid] = value;
    workgroupBarrier();
    for (var offset: u32 = 1u; offset < NT; offset *= 2u) {
        var temp: u32 = 0u;
        if (tid >= offset) { temp = shared_scan[tid - offset]; }
        workgroupBarrier();
        shared_scan[tid] += temp;
        workgroupBarrier();
    }
    *total_ptr = shared_scan[NT - 1u];
    if (tid == 0u) { return 0u; } else { return shared_scan[tid - 1u]; }
}

// ============================================================================
// Decoupled Lookback - bounded by pair's globalWgOffset
// ============================================================================
fn decoupled_lookback_batched(stateIdx: u32, globalWgOffset: u32, local_total: u32) -> u32 {
    if (stateIdx == globalWgOffset) {
        // First workgroup in this pair
        atomicStore(&state[stateIdx], pack_state(STATUS_INCLUSIVE, local_total));
        return 0u;
    }

    atomicStore(&state[stateIdx], pack_state(STATUS_PARTIAL, local_total));

    var lookback_id: i32 = i32(stateIdx) - 1;
    var running_sum: u32 = 0u;

    // Don't look back past this pair's boundary
    while (lookback_id >= i32(globalWgOffset)) {
        var predecessor_state: u32;
        loop {
            predecessor_state = atomicLoad(&state[u32(lookback_id)]);
            let flag = unpack_flag(predecessor_state);
            if (flag != STATUS_NOT_READY) { break; }
        }

        let flag = unpack_flag(predecessor_state);
        let value = unpack_value(predecessor_state);

        if (flag == STATUS_INCLUSIVE) {
            running_sum += value;
            break;
        } else {
            running_sum += value;
            lookback_id -= 1;
        }
    }

    let exclusive_prefix = running_sum;
    let inclusive_value = exclusive_prefix + local_total;
    atomicStore(&state[stateIdx], pack_state(STATUS_INCLUSIVE, inclusive_value));

    return exclusive_prefix;
}

// ============================================================================
// Main Kernel
// ============================================================================
@compute @workgroup_size(256)
fn decoupled_lookback_batched_kernel(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block = get_workgroup_index(wg_id);
    let totalWg = totalWg_uniform;

    if (block >= totalWg) { return; }

    // ========================================================================
    // Step 0: Load pair info from pairInfo (thread 0)
    // ========================================================================
    if (tid == 0u) {
        let infoBase = pairIdPerWg[block] * PAIR_INFO_STRIDE;
        wg_pairAStart     = pairInfo[infoBase + 0u];
        wg_pairALen       = pairInfo[infoBase + 1u];
        wg_pairBStart     = pairInfo[infoBase + 2u];
        wg_pairBLen       = pairInfo[infoBase + 3u];
        wg_pairNumWg      = pairInfo[infoBase + 4u];
        wg_globalWgOffset = pairInfo[infoBase + 5u];
        wg_localWgIdx     = block - wg_globalWgOffset;
        wg_dpiOffset      = pairInfo[infoBase + 6u];
        wg_pairOutOffset  = pairInfo[infoBase + 7u];
    }
    workgroupBarrier();

    let pairAStart     = wg_pairAStart;
    let pairALen       = wg_pairALen;
    let pairBStart     = wg_pairBStart;
    let pairBLen       = wg_pairBLen;
    let localWgIdx     = wg_localWgIdx;
    let pairNumWg      = wg_pairNumWg;
    let globalWgOffset = wg_globalWgOffset;
    let dpiOff         = wg_dpiOffset;

    // ========================================================================
    // Step 1: Read partition boundaries from DPI (pair-local addressing)
    // ========================================================================
    if (tid == 0u) {
        // DPI layout for this pair:
        //   A indices: dpi[dpiOff + 0 .. dpiOff + pairNumWg]
        //   B indices: dpi[dpiOff + pairNumWg + 1 .. dpiOff + 2*pairNumWg + 1]
        let bp0 = dpi[dpiOff + localWgIdx];
        let bp1 = dpi[dpiOff + localWgIdx + 1u];

        let a0 = bp0 & INDEX_MASK;
        let a1 = bp1 & INDEX_MASK;

        let bit0 = select(0u, 1u, (bp0 & STAR_MASK) != 0u);
        let bit1 = select(0u, 1u, (bp1 & STAR_MASK) != 0u);

        let b0 = dpi[dpiOff + pairNumWg + 1u + localWgIdx] + bit0;
        let b1 = dpi[dpiOff + pairNumWg + 1u + localWgIdx + 1u] + bit1;

        let a_count2 = a1 - a0;
        let b_count2 = b1 - b0;
        let extended = (a1 < pairALen) && (b1 < pairBLen);
        let b_start = a_count2 + 2u + select(0u, 1u, extended);

        wg_a0 = a0;
        wg_a1 = a1;
        wg_b0 = b0;
        wg_b1 = b1;
        wg_a_count = a_count2;
        wg_b_count = b_count2;
        wg_b_start = b_start;
        wg_extended = extended;
        wg_bit0 = bit0;
    }
    workgroupBarrier();

    let a0 = wg_a0;
    let b0 = wg_b0;
    let a_count2 = wg_a_count;
    let b_count2 = wg_b_count;
    let b_start = wg_b_start;
    let extended = wg_extended;
    let bit0 = wg_bit0;

    // ========================================================================
    // Step 2: Load data with pair offsets
    // ========================================================================
    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    // Global offsets into keysA/keysB
    device_load_2_to_shared(tid, pairAStart + a0, a_load_count, pairBStart + b0, b_load_count, b_start);
    workgroupBarrier();

    // ========================================================================
    // Step 3: Local Balanced Path
    // ========================================================================
    let partition_size = a_count2 + b_count2;
    let diag_start = i32(VT * tid) - i32(bit0);
    let diag = min(diag_start, i32(partition_size));

    let bp = balanced_path_local(a_count2, b_start, b_count2, diag);

    let a0tid = bp.x;
    let star = bp.y;
    let b0tid_true = i32(VT * tid) + i32(star) - i32(a0tid) - i32(bit0);
    let b_adjust = u32(max(0, -b0tid_true));
    let b0tid = u32(max(0, b0tid_true));

    // ========================================================================
    // Step 4: Serial intersection (with result storage)
    // ========================================================================
    var results: array<u32, 12>;

    let commit = serial_intersection(
        a0tid, a_count2,
        b_start + b0tid, b_start + b_count2,
        star, b_adjust, extended,
        &results
    );

    // ========================================================================
    // Step 5: Workgroup exclusive scan
    // ========================================================================
    let local_count = countOneBits(commit);
    var wg_total: u32 = 0u;
    let local_offset = workgroup_exclusive_scan_with_total(tid, local_count, &wg_total);

    // ========================================================================
    // Step 6: Decoupled Lookback (pair-bounded)
    // ========================================================================
    let myStateIdx = globalWgOffset + localWgIdx;

    if (tid == 0u) {
        wg_local_total = wg_total;
        wg_exclusive_prefix = decoupled_lookback_batched(myStateIdx, globalWgOffset, wg_total);

        // Last workgroup of this pair: write count to per-pair output
        if (localWgIdx == pairNumWg - 1u) {
            let pairTotal = wg_exclusive_prefix + wg_total;
            pairCounts[pairIdPerWg[block]] = pairTotal;
        }
    }
    workgroupBarrier();

    // ========================================================================
    // Step 7: Scatter results to output - Unrolled (VT=12)
    // pairOutputOffset shifts each pair's output to its reserved region
    // ========================================================================
    let pair_output_offset = wg_pairOutOffset;
    let global_offset = pair_output_offset + wg_exclusive_prefix + local_offset;
    var write_pos = global_offset;

    if ((commit & 0x01u) != 0u) { output[write_pos] = results[0]; write_pos++; }
    if ((commit & 0x02u) != 0u) { output[write_pos] = results[1]; write_pos++; }
    if ((commit & 0x04u) != 0u) { output[write_pos] = results[2]; write_pos++; }
    if ((commit & 0x08u) != 0u) { output[write_pos] = results[3]; write_pos++; }
    if ((commit & 0x10u) != 0u) { output[write_pos] = results[4]; write_pos++; }
    if ((commit & 0x20u) != 0u) { output[write_pos] = results[5]; write_pos++; }
    if ((commit & 0x40u) != 0u) { output[write_pos] = results[6]; write_pos++; }
    if ((commit & 0x80u) != 0u) { output[write_pos] = results[7]; write_pos++; }
    if ((commit & 0x100u) != 0u) { output[write_pos] = results[8]; write_pos++; }
    if ((commit & 0x200u) != 0u) { output[write_pos] = results[9]; write_pos++; }
    if ((commit & 0x400u) != 0u) { output[write_pos] = results[10]; write_pos++; }
    if ((commit & 0x800u) != 0u) { output[write_pos] = results[11]; write_pos++; }
}
