// ============================================================================
// Set Availability - Union By Key with Decoupled Lookback - SENTINEL
//
// Single-pass approach using Decoupled Lookback for prefix sum computation.
// Combines Count + Scan + Write in one kernel.
//
// Sentinel optimization applied:
// - Add sentinel values at boundaries to eliminate bounds checking in binary search
// - NEG_INF (0) at start, POS_INF (0xFFFFFFFF) at end of each key array section
// - All data indices shifted by +1 to accommodate leading sentinel
//
// For multiset union by key (A U B):
// - Compare keys only
// - If A_key < B_key: emit A's key-value, advance A
// - If B_key < A_key: emit B's key-value, advance B
// - If A_key == B_key: emit A's key-value (tie goes to A), advance both
//
// Thrust semantics: if A has m equivalent keys and B has n equivalent keys,
// output all m from A, then max(n-m, 0) from B. Total: max(m, n).
//
// With sentinels, boundary handling is automatic:
// - A exhausted -> a_key = POS_INF -> pB=true -> emit B, advance B
// - B exhausted -> b_key = POS_INF -> pA=true -> emit A, advance A
//
// NOTE: A values are stored in shared memory, B values are read from global
// memory to stay within shared memory budget.
//
// Flow:
// 1. Read partition boundaries from DPI
// 2. Load A keys+values and B keys into shared memory with sentinels
// 3. Each thread runs Local BalancedPath to find its starting position
// 4. Each thread runs SerialSetUnionByKey to get results
// 5. Workgroup exclusive scan to compute local offsets and total
// 6. Thread 0 performs Decoupled Lookback to get global offset
// 7. All threads scatter result keys and values to output
// ============================================================================

// ============================================================================
// Bindings (12 bindings - union needs b_values unlike intersection/difference)
// ============================================================================
@group(0) @binding(0) var<storage, read> a_keys: array<u32>;
@group(0) @binding(1) var<storage, read> a_values: array<u32>;
@group(0) @binding(2) var<storage, read> b_keys: array<u32>;
@group(0) @binding(3) var<storage, read> b_values: array<u32>;
@group(0) @binding(4) var<storage, read> dpi: array<u32>;
@group(0) @binding(5) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> output_keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> output_values: array<u32>;
@group(0) @binding(8) var<storage, read_write> total_count: array<atomic<u32>, 1>;
@group(0) @binding(9) var<uniform> a_length: u32;
@group(0) @binding(10) var<uniform> b_length: u32;
@group(0) @binding(11) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants
// ============================================================================
const NT: u32 = 256u;           // Threads per workgroup
const VT: u32 = 12u;            // Values per thread
const NV: u32 = NT * VT;        // Total elements per workgroup = 3072

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// Decoupled Lookback state flags
const STATUS_NOT_READY: u32 = 0u;
const STATUS_PARTIAL: u32 = 1u;
const STATUS_INCLUSIVE: u32 = 2u;

// State encoding: bits 31-30 = flag, bits 29-0 = value
const STATUS_SHIFT: u32 = 30u;
const VALUE_MASK: u32 = 0x3FFFFFFFu;

// Sentinel values
const POS_INF: u32 = 0xFFFFFFFFu;
const NEG_INF: u32 = 0u;

// ============================================================================
// Shared Memory Layout with Sentinels
// ============================================================================
// keys_shared: A keys + B keys with sentinels
// a_values_shared: A values with +1 offset (no sentinels needed for values)
// B values: read from global memory (to stay within shared memory budget)
//
// Total key size: NV + VT + 2 (original) + 4 (sentinels) = 3090
var<workgroup> keys_shared: array<u32, 3090>;
var<workgroup> a_values_shared: array<u32, 3090>;

// Workgroup-level variables
var<workgroup> wg_a0: u32;
var<workgroup> wg_a1: u32;
var<workgroup> wg_b0: u32;
var<workgroup> wg_b1: u32;
var<workgroup> wg_a_count: u32;
var<workgroup> wg_b_count: u32;
var<workgroup> wg_b_start: u32;
var<workgroup> wg_extended: bool;
var<workgroup> wg_bit0: u32;

// For workgroup-level scan
var<workgroup> shared_scan: array<u32, NT>;
var<workgroup> wg_local_total: u32;
var<workgroup> wg_exclusive_prefix: u32;

// ============================================================================
// State Pack/Unpack Functions
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

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad: Cooperative loading with sentinels for keys and A values
// Keys: offset by +1 with NEG_INF/POS_INF sentinels
// A Values: offset by +1 (no sentinels, just matching layout)
// B Values: NOT loaded to shared memory (read from global memory)
// ============================================================================
fn device_load_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Write key sentinels
    if (tid == 0u) {
        keys_shared[0] = NEG_INF;
        keys_shared[a_load_count + 1u] = POS_INF;
        keys_shared[b_shared_start] = NEG_INF;
        keys_shared[b_shared_start + b_load_count + 1u] = POS_INF;
    }

    // Load A keys and A values: unrolled 12 iterations, with +1 offset
    if (tid < a_load_count) {
        keys_shared[tid + 1u] = a_keys[a_global_offset + tid];
        a_values_shared[tid + 1u] = a_values[a_global_offset + tid];
    }
    if (tid + 256u < a_load_count) {
        keys_shared[tid + 257u] = a_keys[a_global_offset + tid + 256u];
        a_values_shared[tid + 257u] = a_values[a_global_offset + tid + 256u];
    }
    if (tid + 512u < a_load_count) {
        keys_shared[tid + 513u] = a_keys[a_global_offset + tid + 512u];
        a_values_shared[tid + 513u] = a_values[a_global_offset + tid + 512u];
    }
    if (tid + 768u < a_load_count) {
        keys_shared[tid + 769u] = a_keys[a_global_offset + tid + 768u];
        a_values_shared[tid + 769u] = a_values[a_global_offset + tid + 768u];
    }
    if (tid + 1024u < a_load_count) {
        keys_shared[tid + 1025u] = a_keys[a_global_offset + tid + 1024u];
        a_values_shared[tid + 1025u] = a_values[a_global_offset + tid + 1024u];
    }
    if (tid + 1280u < a_load_count) {
        keys_shared[tid + 1281u] = a_keys[a_global_offset + tid + 1280u];
        a_values_shared[tid + 1281u] = a_values[a_global_offset + tid + 1280u];
    }
    if (tid + 1536u < a_load_count) {
        keys_shared[tid + 1537u] = a_keys[a_global_offset + tid + 1536u];
        a_values_shared[tid + 1537u] = a_values[a_global_offset + tid + 1536u];
    }
    if (tid + 1792u < a_load_count) {
        keys_shared[tid + 1793u] = a_keys[a_global_offset + tid + 1792u];
        a_values_shared[tid + 1793u] = a_values[a_global_offset + tid + 1792u];
    }
    if (tid + 2048u < a_load_count) {
        keys_shared[tid + 2049u] = a_keys[a_global_offset + tid + 2048u];
        a_values_shared[tid + 2049u] = a_values[a_global_offset + tid + 2048u];
    }
    if (tid + 2304u < a_load_count) {
        keys_shared[tid + 2305u] = a_keys[a_global_offset + tid + 2304u];
        a_values_shared[tid + 2305u] = a_values[a_global_offset + tid + 2304u];
    }
    if (tid + 2560u < a_load_count) {
        keys_shared[tid + 2561u] = a_keys[a_global_offset + tid + 2560u];
        a_values_shared[tid + 2561u] = a_values[a_global_offset + tid + 2560u];
    }
    if (tid + 2816u < a_load_count) {
        keys_shared[tid + 2817u] = a_keys[a_global_offset + tid + 2816u];
        a_values_shared[tid + 2817u] = a_values[a_global_offset + tid + 2816u];
    }

    // Load B keys only: unrolled 12 iterations, with +1 offset
    // (B values read from global memory to save shared memory)
    if (tid < b_load_count) {
        keys_shared[b_shared_start + tid + 1u] = b_keys[b_global_offset + tid];
    }
    if (tid + 256u < b_load_count) {
        keys_shared[b_shared_start + tid + 257u] = b_keys[b_global_offset + tid + 256u];
    }
    if (tid + 512u < b_load_count) {
        keys_shared[b_shared_start + tid + 513u] = b_keys[b_global_offset + tid + 512u];
    }
    if (tid + 768u < b_load_count) {
        keys_shared[b_shared_start + tid + 769u] = b_keys[b_global_offset + tid + 768u];
    }
    if (tid + 1024u < b_load_count) {
        keys_shared[b_shared_start + tid + 1025u] = b_keys[b_global_offset + tid + 1024u];
    }
    if (tid + 1280u < b_load_count) {
        keys_shared[b_shared_start + tid + 1281u] = b_keys[b_global_offset + tid + 1280u];
    }
    if (tid + 1536u < b_load_count) {
        keys_shared[b_shared_start + tid + 1537u] = b_keys[b_global_offset + tid + 1536u];
    }
    if (tid + 1792u < b_load_count) {
        keys_shared[b_shared_start + tid + 1793u] = b_keys[b_global_offset + tid + 1792u];
    }
    if (tid + 2048u < b_load_count) {
        keys_shared[b_shared_start + tid + 2049u] = b_keys[b_global_offset + tid + 2048u];
    }
    if (tid + 2304u < b_load_count) {
        keys_shared[b_shared_start + tid + 2305u] = b_keys[b_global_offset + tid + 2304u];
    }
    if (tid + 2560u < b_load_count) {
        keys_shared[b_shared_start + tid + 2561u] = b_keys[b_global_offset + tid + 2560u];
    }
    if (tid + 2816u < b_load_count) {
        keys_shared[b_shared_start + tid + 2817u] = b_keys[b_global_offset + tid + 2816u];
    }
}

// ============================================================================
// Local Balanced Path - Simplified Binary Search with Sentinel Offset
//
// Since VT=12 < 16, get_local_biased_levels(VT) always returns 0.
// Simplified to standard binary search for better performance.
// Physical indices are +1 from logical indices.
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
        if (a_key <= b_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn balanced_path_local(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    let diag_u = u32(max(0, diag));

    if (diag_u == 0u) {
        return vec2<u32>(0u, 0u);
    }
    if (diag_u >= a_count + b_count) {
        return vec2<u32>(a_count, 0u);
    }

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
// Serial Set Union By Key - with sentinel offset
// Branchless select() for pointer advancement + pre-fetch pattern.
// Physical indices are +1 from logical indices.
//
// Union always emits every element:
// - pA (A < B): emit A's key + A's value from shared memory
// - pB (B < A): emit B's key from shared memory + B's value from global memory
// - equal (!pA && !pB): emit A's key + A's value (tie goes to A)
//
// With sentinels, boundary handling is automatic:
// - A exhausted -> a_key = POS_INF -> pB=true -> emit B, advance B
// - B exhausted -> b_key = POS_INF -> pA=true -> emit A, advance A
//
// NOTE: A values from shared memory, B values from global memory
// ============================================================================
fn serial_set_union_by_key(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    b_start_base: u32,
    b_global_offset: u32,
    star: u32,
    b_adjust: u32,
    extended: bool,
    result_keys: ptr<function, array<u32, 12>>,
    result_values: ptr<function, array<u32, 12>>,
    indices: ptr<function, array<u32, 12>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

    // Pre-fetch first keys and A value (physical = logical + 1 for sentinel)
    var a_key = keys_shared[a_idx + 1u];
    var b_key = keys_shared[b_idx + 1u];
    var a_val = a_values_shared[a_idx + 1u];

    for (var i: u32 = 0u; i < VT; i++) {
        var test: bool;
        if (extended) {
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else {
            test = (i32(a_idx) + i32(b_idx) < end_diag);
        }

        if (!test) {
            break;
        }

        let pA = a_key < b_key;  // A < B
        let pB = b_key < a_key;  // B < A

        // Union output: pB -> emit B; otherwise -> emit A (includes tie case)
        (*result_keys)[i] = select(a_key, b_key, pB);
        if (pB) {
            // B values read from global memory
            // b_idx is in shared memory coordinates (b_start_base + local_b_index)
            (*result_values)[i] = b_values[b_global_offset + (b_idx - b_start_base)];
        } else {
            // A values from shared memory (pre-fetched)
            (*result_values)[i] = a_val;
        }
        (*indices)[i] = select(a_idx, b_idx, pB);

        // Union: always commit (every iteration produces output)
        commit |= (1u << i);

        // Branchless pointer advancement using select()
        // A < B: advance A only   (!pB=true -> a++, !pA=false -> b stays)
        // B < A: advance B only   (!pB=false -> a stays, !pA=true -> b++)
        // A == B: advance both    (!pB=true -> a++, !pA=true -> b++)
        a_idx += select(0u, 1u, !pB);
        b_idx += select(0u, 1u, !pA);

        // Pre-fetch next keys and A value (with +1 physical offset)
        a_key = select(a_key, keys_shared[a_idx + 1u], !pB);
        a_val = select(a_val, a_values_shared[a_idx + 1u], !pB);
        b_key = select(b_key, keys_shared[b_idx + 1u], !pA);
    }

    return commit;
}

// ============================================================================
// Workgroup Exclusive Scan with Total
// ============================================================================
fn workgroup_exclusive_scan_with_total(tid: u32, value: u32, total_ptr: ptr<function, u32>) -> u32 {
    shared_scan[tid] = value;
    workgroupBarrier();

    for (var offset: u32 = 1u; offset < NT; offset *= 2u) {
        var temp: u32 = 0u;
        if (tid >= offset) {
            temp = shared_scan[tid - offset];
        }
        workgroupBarrier();
        shared_scan[tid] += temp;
        workgroupBarrier();
    }

    *total_ptr = shared_scan[NT - 1u];

    if (tid == 0u) {
        return 0u;
    } else {
        return shared_scan[tid - 1u];
    }
}

// ============================================================================
// Decoupled Lookback
// ============================================================================
fn decoupled_lookback(wg_id: u32, local_total: u32) -> u32 {
    var exclusive_prefix: u32 = 0u;

    if (wg_id == 0u) {
        atomicStore(&state[0u], pack_state(STATUS_INCLUSIVE, local_total));
        return 0u;
    }

    atomicStore(&state[wg_id], pack_state(STATUS_PARTIAL, local_total));

    var lookback_id: i32 = i32(wg_id) - 1;
    var running_sum: u32 = 0u;

    while (lookback_id >= 0) {
        var predecessor_state: u32;
        loop {
            predecessor_state = atomicLoad(&state[u32(lookback_id)]);
            let flag = unpack_flag(predecessor_state);
            if (flag != STATUS_NOT_READY) {
                break;
            }
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

    exclusive_prefix = running_sum;

    let inclusive_value = exclusive_prefix + local_total;
    atomicStore(&state[wg_id], pack_state(STATUS_INCLUSIVE, inclusive_value));

    return exclusive_prefix;
}

// ============================================================================
// Main Kernel
// ============================================================================
@compute @workgroup_size(256)
fn union_by_key_decoupled_lookback(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block = get_workgroup_index(wg_id);

    if (block >= num_wg_total) {
        return;
    }

    // ========================================================================
    // Step 1: Read partition boundaries from DPI (thread 0)
    // ========================================================================
    if (tid == 0u) {
        let bp0 = dpi[block];
        let bp1 = dpi[block + 1u];

        let a0 = bp0 & INDEX_MASK;
        let a1 = bp1 & INDEX_MASK;

        let bit0 = select(0u, 1u, (bp0 & STAR_MASK) != 0u);
        let bit1 = select(0u, 1u, (bp1 & STAR_MASK) != 0u);

        let b0 = dpi[num_wg_total + 1u + block] + bit0;
        let b1 = dpi[num_wg_total + 1u + block + 1u] + bit1;

        let a_count2 = a1 - a0;
        let b_count2 = b1 - b0;
        let extended = (a1 < a_length) && (b1 < b_length);
        // b_start needs +2 for sentinel space: +1 for A trailing sentinel, +1 for B leading sentinel
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
    // Step 2: Load data into shared memory with sentinels
    // ========================================================================
    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    device_load_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    // ========================================================================
    // Step 3: Each thread finds its starting position using Local BalancedPath
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
    // Step 4: Serial set union by key
    // ========================================================================
    var result_keys_arr: array<u32, 12>;
    var result_values_arr: array<u32, 12>;
    var indices: array<u32, 12>;

    let commit = serial_set_union_by_key(
        a0tid,
        a_count2,
        b_start + b0tid,
        b_start + b_count2,
        b_start,        // b_start_base for computing local B index
        b0,             // b_global_offset for reading B values from global memory
        star,
        b_adjust,
        extended,
        &result_keys_arr,
        &result_values_arr,
        &indices
    );

    // ========================================================================
    // Step 5: Workgroup-level exclusive scan to get local offsets and total
    // ========================================================================
    let local_count = countOneBits(commit);
    var wg_total: u32 = 0u;
    let local_offset = workgroup_exclusive_scan_with_total(tid, local_count, &wg_total);

    // ========================================================================
    // Step 6: Thread 0 performs Decoupled Lookback
    // ========================================================================
    if (tid == 0u) {
        wg_local_total = wg_total;
        wg_exclusive_prefix = decoupled_lookback(block, wg_total);

        if (block == num_wg_total - 1u) {
            atomicStore(&total_count[0], wg_exclusive_prefix + wg_total);
        }
    }
    workgroupBarrier();

    // ========================================================================
    // Step 7: Scatter result keys and values - Unrolled (VT=12 iterations)
    // ========================================================================
    let global_offset = wg_exclusive_prefix + local_offset;

    var write_pos = global_offset;

    if ((commit & 0x001u) != 0u) { output_keys[write_pos] = result_keys_arr[0]; output_values[write_pos] = result_values_arr[0]; write_pos++; }
    if ((commit & 0x002u) != 0u) { output_keys[write_pos] = result_keys_arr[1]; output_values[write_pos] = result_values_arr[1]; write_pos++; }
    if ((commit & 0x004u) != 0u) { output_keys[write_pos] = result_keys_arr[2]; output_values[write_pos] = result_values_arr[2]; write_pos++; }
    if ((commit & 0x008u) != 0u) { output_keys[write_pos] = result_keys_arr[3]; output_values[write_pos] = result_values_arr[3]; write_pos++; }
    if ((commit & 0x010u) != 0u) { output_keys[write_pos] = result_keys_arr[4]; output_values[write_pos] = result_values_arr[4]; write_pos++; }
    if ((commit & 0x020u) != 0u) { output_keys[write_pos] = result_keys_arr[5]; output_values[write_pos] = result_values_arr[5]; write_pos++; }
    if ((commit & 0x040u) != 0u) { output_keys[write_pos] = result_keys_arr[6]; output_values[write_pos] = result_values_arr[6]; write_pos++; }
    if ((commit & 0x080u) != 0u) { output_keys[write_pos] = result_keys_arr[7]; output_values[write_pos] = result_values_arr[7]; write_pos++; }
    if ((commit & 0x100u) != 0u) { output_keys[write_pos] = result_keys_arr[8]; output_values[write_pos] = result_values_arr[8]; write_pos++; }
    if ((commit & 0x200u) != 0u) { output_keys[write_pos] = result_keys_arr[9]; output_values[write_pos] = result_values_arr[9]; write_pos++; }
    if ((commit & 0x400u) != 0u) { output_keys[write_pos] = result_keys_arr[10]; output_values[write_pos] = result_values_arr[10]; write_pos++; }
    if ((commit & 0x800u) != 0u) { output_keys[write_pos] = result_keys_arr[11]; output_values[write_pos] = result_values_arr[11]; write_pos++; }
}
