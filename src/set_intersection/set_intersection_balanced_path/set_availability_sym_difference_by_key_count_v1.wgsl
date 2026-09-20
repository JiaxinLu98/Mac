// ============================================================================
// Set Availability - Symmetric Difference By Key Count Kernel (Strategy A / v1)
//
// ModernGPU-style DeviceComputeSetAvailability implementation
// Two-pass approach: Count phase
//
// For multiset symmetric difference by key (Thrust semantics):
// - Compare keys only
// - If A has m copies of key x and B has n copies:
//   - Output |m - n| copies
//   - If m > n: last (m-n) from A
//   - If m < n: last (n-m) from B
//   - If m == n: no output (cancel out)
//
// Merge logic:
// - pA (A_key < B_key): emit from A, advance A
// - pB (B_key < A_key): emit from B, advance B
// - equal (!pA && !pB): advance both, no emit (one cancels one)
// ============================================================================

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a_keys: array<u32>;
@group(0) @binding(1) var<storage, read> b_keys: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<uniform> a_length: u32;
@group(0) @binding(5) var<uniform> b_length: u32;
@group(0) @binding(6) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants (ModernGPU terminology)
// ============================================================================
const NT: u32 = 256u;
const VT: u32 = 7u;
const NV: u32 = NT * VT;

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// ============================================================================
// Shared Memory
// ============================================================================
var<workgroup> keys_shared: array<u32, 1801>;

var<workgroup> wg_a0: u32;
var<workgroup> wg_a1: u32;
var<workgroup> wg_b0: u32;
var<workgroup> wg_b1: u32;
var<workgroup> wg_a_count: u32;
var<workgroup> wg_b_count: u32;
var<workgroup> wg_b_start: u32;
var<workgroup> wg_extended: bool;
var<workgroup> wg_bit0: u32;

var<workgroup> shared_counts: array<u32, NT>;

// ============================================================================
// Helper Functions
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

fn device_load_keys_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    var i = tid;
    while (i < a_load_count) {
        keys_shared[i] = a_keys[a_global_offset + i];
        i += NT;
    }
    i = tid;
    while (i < b_load_count) {
        keys_shared[b_shared_start + i] = b_keys[b_global_offset + i];
        i += NT;
    }
}

// ============================================================================
// Local Balanced Path with Biased Binary Search (ModernGPU Style)
// ============================================================================
fn get_local_biased_levels(partition_size: u32) -> u32 {
    if (partition_size >= 512u) { return 4u; }
    if (partition_size >= 128u) { return 3u; }
    if (partition_size >= 32u)  { return 2u; }
    if (partition_size >= 16u)  { return 1u; }
    return 0u;
}

fn lower_bound_local_a(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    if (levels >= 4u && lo < hi) {
        let scale = (1u << 9u) - 1u;
        let mid = (lo + scale * hi) >> 9u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 3u && lo < hi) {
        let scale = (1u << 7u) - 1u;
        let mid = (lo + scale * hi) >> 7u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 2u && lo < hi) {
        let scale = (1u << 5u) - 1u;
        let mid = (lo + scale * hi) >> 5u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 1u && lo < hi) {
        let scale = (1u << 4u) - 1u;
        let mid = (lo + scale * hi) >> 4u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn lower_bound_local_b(b_start: u32, end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    if (levels >= 4u && lo < hi) {
        let scale = (1u << 9u) - 1u;
        let mid = (lo + scale * hi) >> 9u;
        if (keys_shared[b_start + mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 3u && lo < hi) {
        let scale = (1u << 7u) - 1u;
        let mid = (lo + scale * hi) >> 7u;
        if (keys_shared[b_start + mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 2u && lo < hi) {
        let scale = (1u << 5u) - 1u;
        let mid = (lo + scale * hi) >> 5u;
        if (keys_shared[b_start + mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    if (levels >= 1u && lo < hi) {
        let scale = (1u << 4u) - 1u;
        let mid = (lo + scale * hi) >> 4u;
        if (keys_shared[b_start + mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn upper_bound_local_b(b_start: u32, range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn merge_path_local(a_count: u32, b_start: u32, b_count: u32, diag: u32) -> u32 {
    var lo: u32 = select(0u, diag - b_count, diag > b_count);
    var hi: u32 = min(diag, a_count);

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let a_key = keys_shared[mid];
        let b_key = keys_shared[b_start + diag - 1u - mid];
        if (a_key <= b_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

fn balanced_path_local_biased(
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
        let x = keys_shared[b_start + b_index];
        let levels = get_local_biased_levels(VT);
        let a_start = lower_bound_local_a(a_index, x, levels);
        let b_start_run = lower_bound_local_b(b_start, b_index, x, levels);
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
// Serial Set Symmetric Difference By Key (ModernGPU Style)
//
// For symmetric difference: commit when pA || pB (keys not equal)
// - pA (A < B): emit from A, advance A
// - pB (B < A): emit from B, advance B
// - equal (!pA && !pB): advance both, no emit (one cancels one)
//
// This implements Thrust's multiset symmetric difference semantics:
// If A has m copies and B has n copies, outputs |m-n| copies.
// ============================================================================
fn serial_set_sym_difference_by_key(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    star: u32,
    b_adjust: u32,
    extended: bool,
    results: ptr<function, array<u32, 7>>,
    indices: ptr<function, array<u32, 7>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

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

        // Handle boundary cases
        if (!extended) {
            if (a_idx >= a_end) {
                // A exhausted: emit remaining B elements
                (*results)[i] = keys_shared[b_idx];
                (*indices)[i] = b_idx;
                commit |= (1u << i);
                b_idx++;
                continue;
            }
            if (b_idx >= b_end) {
                // B exhausted: emit remaining A elements
                (*results)[i] = keys_shared[a_idx];
                (*indices)[i] = a_idx;
                commit |= (1u << i);
                a_idx++;
                continue;
            }
        }

        let a_key = keys_shared[a_idx];
        let b_key = keys_shared[b_idx];

        let pA = a_key < b_key;
        let pB = b_key < a_key;

        // Symmetric difference: commit when keys are not equal (pA || pB)
        if (pA) {
            // A < B: emit from A
            (*results)[i] = a_key;
            (*indices)[i] = a_idx;
            commit |= (1u << i);
        } else if (pB) {
            // B < A: emit from B
            (*results)[i] = b_key;
            (*indices)[i] = b_idx;
            commit |= (1u << i);
        }
        // else: equal, no emit (one A cancels one B)

        // Advance pointers (ModernGPU style)
        if (!pB) { a_idx++; }
        if (!pA) { b_idx++; }
    }

    return commit;
}

fn workgroup_reduce_sum(tid: u32, value: u32) -> u32 {
    shared_counts[tid] = value;
    workgroupBarrier();

    for (var stride = NT >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_counts[tid] += shared_counts[tid + stride];
        }
        workgroupBarrier();
    }

    return shared_counts[0];
}

// ============================================================================
// Main Kernel: Count Symmetric Difference By Key
// ============================================================================
@compute @workgroup_size(256)
fn count_availability(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block = get_workgroup_index(wg_id);

    if (block >= num_wg_total) {
        return;
    }

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
        let b_start = a_count2 + select(0u, 1u, extended);

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

    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    // Load only keys (values not needed for count phase)
    device_load_keys_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    let total_count = a_count2 + b_count2;
    let diag_start = i32(VT * tid) - i32(bit0);
    let diag = min(diag_start, i32(total_count));

    let bp = balanced_path_local_biased(a_count2, b_start, b_count2, diag);

    let a0tid = bp.x;
    let star = bp.y;
    let b0tid_true = i32(VT * tid) + i32(star) - i32(a0tid) - i32(bit0);
    let b_adjust = u32(max(0, -b0tid_true));
    let b0tid = u32(max(0, b0tid_true));

    var results: array<u32, 7>;
    var indices: array<u32, 7>;

    let commit = serial_set_sym_difference_by_key(
        a0tid,
        a_count2,
        b_start + b0tid,
        b_start + b_count2,
        star,
        b_adjust,
        extended,
        &results,
        &indices
    );

    let local_count = countOneBits(commit);
    let total = workgroup_reduce_sum(tid, local_count);

    if (tid == 0u) {
        counts[block] = total;
    }
}
