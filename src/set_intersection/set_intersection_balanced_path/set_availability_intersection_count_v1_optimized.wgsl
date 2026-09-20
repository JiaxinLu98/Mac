// ============================================================================
// Set Availability - Intersection Count Kernel (Strategy A / v1) - OPTIMIZED
//
// ModernGPU-style DeviceComputeSetAvailability implementation
// Two-pass approach: Count phase - executes SerialSetOp to count matches
//
// Optimizations applied (Step 2):
// - Opt 3: Branchless select() + pre-fetch in serial_set_intersection
// - Opt 2: Unrolled loads in device_load_2_to_shared (7 iterations)
//
// Flow:
// 1. Read partition boundaries from DPI
// 2. Load A and B data into shared memory
// 3. Each thread runs Local BalancedPath to find its starting position
// 4. Each thread runs SerialSetIntersection to process VT elements
// 5. Workgroup reduction to sum counts
// ============================================================================

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<uniform> a_length: u32;
@group(0) @binding(5) var<uniform> b_length: u32;
@group(0) @binding(6) var<uniform> num_wg_total: u32;
// Debug buffer: [block * NT * 7 + tid * 7 + offset]
// offset 0: tid, 1: a0tid, 2: b0tid, 3: star, 4: b_adjust, 5: diag, 6: commit
@group(0) @binding(7) var<storage, read_write> debug_out: array<u32>;

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

// For workgroup reduction
var<workgroup> shared_counts: array<u32, NT>;

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// DeviceLoad2ToShared: Cooperative loading of A and B data into shared memory
// Optimized: unrolled 7 iterations (NV/NT = 1792/256 = 7)
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Load A elements: unrolled 7 iterations
    if (tid < a_load_count) {
        keys_shared[tid] = a[a_global_offset + tid];
    }
    if (tid + 256u < a_load_count) {
        keys_shared[tid + 256u] = a[a_global_offset + tid + 256u];
    }
    if (tid + 512u < a_load_count) {
        keys_shared[tid + 512u] = a[a_global_offset + tid + 512u];
    }
    if (tid + 768u < a_load_count) {
        keys_shared[tid + 768u] = a[a_global_offset + tid + 768u];
    }
    if (tid + 1024u < a_load_count) {
        keys_shared[tid + 1024u] = a[a_global_offset + tid + 1024u];
    }
    if (tid + 1280u < a_load_count) {
        keys_shared[tid + 1280u] = a[a_global_offset + tid + 1280u];
    }
    if (tid + 1536u < a_load_count) {
        keys_shared[tid + 1536u] = a[a_global_offset + tid + 1536u];
    }

    // Load B elements: unrolled 7 iterations
    if (tid < b_load_count) {
        keys_shared[b_shared_start + tid] = b[b_global_offset + tid];
    }
    if (tid + 256u < b_load_count) {
        keys_shared[b_shared_start + tid + 256u] = b[b_global_offset + tid + 256u];
    }
    if (tid + 512u < b_load_count) {
        keys_shared[b_shared_start + tid + 512u] = b[b_global_offset + tid + 512u];
    }
    if (tid + 768u < b_load_count) {
        keys_shared[b_shared_start + tid + 768u] = b[b_global_offset + tid + 768u];
    }
    if (tid + 1024u < b_load_count) {
        keys_shared[b_shared_start + tid + 1024u] = b[b_global_offset + tid + 1024u];
    }
    if (tid + 1280u < b_load_count) {
        keys_shared[b_shared_start + tid + 1280u] = b[b_global_offset + tid + 1280u];
    }
    if (tid + 1536u < b_load_count) {
        keys_shared[b_shared_start + tid + 1536u] = b[b_global_offset + tid + 1536u];
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
// Serial Set Intersection (ModernGPU Style) - OPTIMIZED
//
// Branchless select() for pointer advancement + pre-fetch pattern.
// Processes elements serially, returning a commit bitmask.
// Uses diagonal-based termination condition.
// ============================================================================
fn serial_set_intersection(
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
    // because the "true" bBegin would have been (b_begin - b_adjust)
    var end_diag = i32(a_begin) + i32(b_begin) + i32(VT) - i32(star) - i32(b_adjust);
    // RangeCheck=true: end = min(end, aEnd + bEnd)
    if (!extended) {
        end_diag = min(end_diag, i32(a_end) + i32(b_end));
    }

    let min_iterations = VT / 2u;

    // Pre-fetch first keys
    var a_key = keys_shared[a_idx];
    var b_key = keys_shared[b_idx];

    for (var i: u32 = 0u; i < VT; i++) {
        // ModernGPU termination condition
        var test: bool;
        if (extended) {
            // RangeCheck=false: first MinIterations unconditional, then check diagonal
            test = (i < min_iterations) || (i32(a_idx) + i32(b_idx) < end_diag);
        } else {
            // RangeCheck=true: all three conditions must be true
            test = (i32(a_idx) + i32(b_idx) < end_diag) && (a_idx < a_end) && (b_idx < b_end);
        }

        if (!test) {
            break;
        }

        // Always write results at position i (sparse storage)
        (*results)[i] = a_key;
        (*indices)[i] = a_idx;

        let pA = a_key < b_key;
        let pB = b_key < a_key;

        // Match when pA == pB (both false means a == b)
        if (pA == pB) {
            commit |= (1u << i);
        }

        // Branchless pointer advancement using select()
        a_idx += select(0u, 1u, !pB);
        b_idx += select(0u, 1u, !pA);

        // Pre-fetch next keys
        a_key = select(a_key, keys_shared[a_idx], !pB);
        b_key = select(b_key, keys_shared[b_idx], !pA);
    }

    return commit;
}

// ============================================================================
// Workgroup reduction to sum all thread counts
// ============================================================================
fn workgroup_reduce_sum(tid: u32, value: u32) -> u32 {
    shared_counts[tid] = value;
    workgroupBarrier();

    // Tree reduction
    for (var stride = NT >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_counts[tid] += shared_counts[tid + stride];
        }
        workgroupBarrier();
    }

    return shared_counts[0];
}

// ============================================================================
// popcount for commit bitmask
// ============================================================================
fn countOneBits(n_in: u32) -> u32 {
    var n = n_in;
    n = n - ((n >> 1u) & 0x55555555u);
    n = (n & 0x33333333u) + ((n >> 2u) & 0x33333333u);
    return (((n + (n >> 4u)) & 0x0F0F0F0Fu) * 0x01010101u) >> 24u;
}

// ============================================================================
// Main Kernel: Count Intersections
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
    // let diag_end = min(diag_start + i32(VT), i32(total_count));

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
    // Step 4: Serial set intersection for this thread
    // ========================================================================
    var results: array<u32, 7>;
    var indices: array<u32, 7>;

    // ModernGPU style: use star (end) for termination control
    // Pass b_adjust to compensate termination condition when b0tid was clamped
    let commit = serial_set_intersection(
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
    // Step 5: Debug output
    // ========================================================================
    let debug_base = block * NT * 7u + tid * 7u;
    debug_out[debug_base + 0u] = tid;
    debug_out[debug_base + 1u] = a0tid;
    debug_out[debug_base + 2u] = b0tid;
    debug_out[debug_base + 3u] = star;
    debug_out[debug_base + 4u] = b_adjust;
    debug_out[debug_base + 5u] = u32(diag);
    debug_out[debug_base + 6u] = commit;

    // ========================================================================
    // Step 6: Count bits and reduce
    // ========================================================================
    let local_count = countOneBits(commit);
    let total = workgroup_reduce_sum(tid, local_count);

    if (tid == 0u) {
        counts[block] = total;
    }
}
