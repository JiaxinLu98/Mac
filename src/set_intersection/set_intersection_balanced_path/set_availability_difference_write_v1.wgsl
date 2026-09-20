// ============================================================================
// Set Availability - Difference Write Kernel (Strategy A / v1)
//
// ModernGPU-style DeviceComputeSetAvailability implementation
// Two-pass approach: Write phase - re-executes SerialSetOp and scatters results
//
// For multiset difference (A \ B): emit elements from A not fully covered by B
// - If A < B: emit A (A not in B), advance A
// - If B < A: advance B only (B element, skip)
// - If A == B: advance both (one B cancels one A, no emit)
//
// This naturally handles multiset semantics:
// - If A has 5 copies of x and B has 3 copies, result has 2 copies
//
// NOTE: This kernel MUST use identical logic to the count kernel!
// ============================================================================

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;  // Prefix sum of counts
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> a_length: u32;
@group(0) @binding(6) var<uniform> b_length: u32;
@group(0) @binding(7) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants (ModernGPU terminology)
// ============================================================================
const NT: u32 = 256u;           // Threads per workgroup
const VT: u32 = 7u;             // Values per thread
const NV: u32 = NT * VT;        // Total elements per workgroup = 1792

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// ============================================================================
// Shared Memory Layout
// ============================================================================
// Layout: [A elements] [B elements]
// Total size: NV + VT + 2 (maximum with extended frame)
var<workgroup> keys_shared: array<u32, 1801>;

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

// For workgroup exclusive scan
var<workgroup> shared_scan: array<u32, NT>;

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad2ToShared: Cooperative loading of A and B data into shared memory
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Load A elements: thread i loads indices i, i+NT, i+2*NT, ...
    var i = tid;
    while (i < a_load_count) {
        keys_shared[i] = a[a_global_offset + i];
        i += NT;
    }

    // Load B elements into shared memory starting at b_shared_start
    i = tid;
    while (i < b_load_count) {
        keys_shared[b_shared_start + i] = b[b_global_offset + i];
        i += NT;
    }
}

// ============================================================================
// Local Balanced Path with Biased Binary Search (ModernGPU Style)
//
// Complete implementation matching balanced_path_biased.wgsl but for shared memory.
// Includes MergePath search + balanced_adjust for correct duplicate handling.
//
// Shared memory layout:
//   A: keys_shared[0 .. a_count)
//   B: keys_shared[b_start .. b_start + b_count)
// ============================================================================

// Get biased search levels based on partition size (matches global version)
fn get_local_biased_levels(partition_size: u32) -> u32 {
    if (partition_size >= 512u) { return 4u; }
    if (partition_size >= 128u) { return 3u; }
    if (partition_size >= 32u)  { return 2u; }
    if (partition_size >= 16u)  { return 1u; }
    return 0u;
}

// Biased lower_bound for A in shared memory
// Searches [0, end_exclusive) for first element >= key
fn lower_bound_local_a(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;

    // Biased probing steps (probing near the end first)
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

    // Standard binary search to finish
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }

    return lo;
}

// Biased lower_bound for B in shared memory
// Searches [b_start, b_start + end_exclusive) for first element >= key
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

// Upper_bound for B in shared memory (non-biased, for balanced_adjust)
// Searches [range_begin, range_end) for first element > key
fn upper_bound_local_b(b_start: u32, range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keys_shared[b_start + mid] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

// Basic MergePath search in shared memory
// Returns the A index (p) where the merge path crosses the diagonal
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

// Complete Balanced Path with adjustment (matches global balanced_adjust)
// Returns (a_index, star) - ModernGPU style
//   - a_index: starting index in A
//   - star: star flag (0 or 1), used to adjust B start position and termination
fn balanced_path_local_biased(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    let diag_u = u32(max(0, diag));

    // Handle boundary cases
    if (diag_u == 0u) {
        return vec2<u32>(0u, 0u);  // (a_index=0, star=0)
    }
    if (diag_u >= a_count + b_count) {
        return vec2<u32>(a_count, 0u);  // (a_index=a_count, star=0)
    }

    // Step 1: Basic MergePath search
    let p = merge_path_local(a_count, b_start, b_count, diag_u);

    var a_index = p;
    var b_index = diag_u - p;
    var star: u32 = 0u;

    // Step 2: Balanced adjustment for duplicates
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

        // Calculate star bit: marks odd split for correct boundary handling
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        star = select(0u, 1u, round_up);

        a_index = a_start + a_advance;
        // Note: b_index is computed by caller using: diag - a_index + star
    }

    return vec2<u32>(a_index, star);
}

// ============================================================================
// Serial Set Difference (ModernGPU Style) - Multiset Version
//
// Emit A elements that are NOT fully covered by B (multiset difference)
// - pA (A < B): emit A, advance A (A not in B)
// - pB (B < A): advance B only (skip B element)
// - equal (!pA && !pB): advance both, no emit (one B cancels one A)
//
// This naturally handles multiset semantics:
// If A has 5 copies of x and B has 3 copies, 2 copies remain in difference.
//
// Parameters:
//   a_begin, a_end: A range in shared memory [0, a_count)
//   b_begin, b_end: B range in shared memory [b_start, b_start + b_count)
//   star: star parameter from BalancedPath (0 or 1)
//   b_adjust: adjustment for clamped b0tid (when b0tid was negative)
//   extended: if true, use relaxed termination (RangeCheck=false)
// ============================================================================
fn serial_set_difference(
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

    // ModernGPU: end = aBegin + bBegin + VT - star
    // When b0tid was clamped from negative, we need to subtract b_adjust to compensate
    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    // RangeCheck=true: end = min(end, aEnd + bEnd)
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

    for (var i: u32 = 0u; i < VT; i++) {
        // ModernGPU termination condition
        // RangeCheck only checks diagonal, boundary handling is done via pA/pB
        var test: bool;
        if (extended) {
            // RangeCheck=false: first MinIterations unconditional, then check diagonal
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else {
            // RangeCheck=true: only check diagonal (boundary handled below)
            test = (i32(a_idx) + i32(b_idx) < end_diag);
        }

        if (!test) {
            break;
        }

        let a_key = keys_shared[a_idx];
        let b_key = keys_shared[b_idx];

        // Always write results at position i (sparse storage)
        (*results)[i] = a_key;
        (*indices)[i] = a_idx;

        // ModernGPU boundary handling for RangeCheck=true
        var pA: bool = false;
        var pB: bool = false;
        if (!extended && a_idx >= a_end) {
            // A exhausted: only advance B, no output
            pB = true;
        } else if (!extended && b_idx >= b_end) {
            // B exhausted: output A and advance A (all remaining A are in difference)
            pA = true;
        } else {
            pA = a_key < b_key;  // A < B
            pB = b_key < a_key;  // B < A
        }

        // Advance pointers (ModernGPU style)
        if (!pB) { a_idx++; }  // a <= b, advance A
        if (!pA) { b_idx++; }  // b <= a, advance B

        // Difference: emit when pA is true (A < B or B exhausted)
        if (pA) {
            commit |= (1u << i);
        }
    }

    return commit;
}

// ============================================================================
// Workgroup exclusive scan (Hillis-Steele style)
// Returns the exclusive prefix sum for this thread
// ============================================================================
fn workgroup_exclusive_scan(tid: u32, value: u32) -> u32 {
    shared_scan[tid] = value;
    workgroupBarrier();

    // Up-sweep phase
    for (var offset: u32 = 1u; offset < NT; offset *= 2u) {
        var temp: u32 = 0u;
        if (tid >= offset) {
            temp = shared_scan[tid - offset];
        }
        workgroupBarrier();
        shared_scan[tid] += temp;
        workgroupBarrier();
    }

    // Convert to exclusive scan
    workgroupBarrier();

    if (tid == 0u) {
        return 0u;
    } else {
        return shared_scan[tid - 1u];
    }
}

// ============================================================================
// Main Kernel: Write Difference Results
// ============================================================================
@compute @workgroup_size(256)
fn write_availability(
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

    // ========================================================================
    // Step 2: Load data into shared memory
    // ========================================================================
    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    device_load_2_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    // ========================================================================
    // Step 3: Each thread finds its starting position using Local BalancedPath
    // ========================================================================
    let total_count = a_count2 + b_count2;

    // Calculate diagonal range for this thread (with bit0 adjustment)
    let diag_start = i32(VT * tid) - i32(bit0);
    let diag = min(diag_start, i32(total_count));

    // BalancedPath returns (a_index, star) - ModernGPU style
    let bp = balanced_path_local_biased(a_count2, b_start, b_count2, diag);

    // Thread starting positions (ModernGPU formula)
    let a0tid = bp.x;
    let star = bp.y;
    // ModernGPU: b0tid = VT * tid + bp.y - bp.x - bit0
    // b0tid can be negative when bit0=1 and tid=0, we clamp it but track the adjustment
    let b0tid_true = i32(VT * tid) + i32(star) - i32(a0tid) - i32(bit0);
    let b_adjust = u32(max(0, -b0tid_true));  // Amount that was clamped (0 if no clamping)
    let b0tid = u32(max(0, b0tid_true));

    // ========================================================================
    // Step 4: Serial set difference (re-execute to get results)
    // ========================================================================
    var results: array<u32, 7>;
    var indices: array<u32, 7>;

    // ModernGPU style: use star (end) for termination control
    // Pass b_adjust to compensate termination condition when b0tid was clamped
    let commit = serial_set_difference(
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

    // ========================================================================
    // Step 5: Workgroup exclusive scan to compute local offsets
    // ========================================================================
    let local_count = countOneBits(commit);
    let local_offset = workgroup_exclusive_scan(tid, local_count);

    // ========================================================================
    // Step 6: Scatter results to output
    // ========================================================================
    let global_offset = offsets[block] + local_offset;

    var write_pos = global_offset;
    for (var i: u32 = 0u; i < VT; i++) {
        if ((commit & (1u << i)) != 0u) {
            output[write_pos] = results[i];
            write_pos++;
        }
    }
}
