// ============================================================================
// count_intersections - Binary Search Version (MULTISET)
//
// For each unique value v in A's partition (first occurrence globally):
//   count += min(occurrences_in_A(v), occurrences_in_B(v))
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<uniform> a_length: u32;
@group(0) @binding(5) var<uniform> b_length: u32;
@group(0) @binding(6) var<uniform> num_wg_total: u32;

const WORKGROUP_SIZE: u32 = 256u;
const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// Shared memory for workgroup reduction
var<workgroup> shared_counts: array<u32, WORKGROUP_SIZE>;

// Convert 2D workgroup_id to 1D index
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// Lower bound in A: find first index where A[index] >= value
fn lower_bound_a(value: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = a_length;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (a[mid] < value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Upper bound in A: find first index where A[index] > value
fn upper_bound_a(value: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = a_length;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (a[mid] <= value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Lower bound in B: find first index where B[index] >= value
fn lower_bound_b(value: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = b_length;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b[mid] < value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Upper bound in B: find first index where B[index] > value
fn upper_bound_b(value: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = b_length;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b[mid] <= value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Count occurrences of value in entire A array
fn count_in_a(value: u32) -> u32 {
    return upper_bound_a(value) - lower_bound_a(value);
}

// Count occurrences of value in entire B array
fn count_in_b(value: u32) -> u32 {
    return upper_bound_b(value) - lower_bound_b(value);
}

// Workgroup reduction to sum all thread counts
fn workgroup_reduce_sum(local_id: u32, value: u32) -> u32 {
    shared_counts[local_id] = value;
    workgroupBarrier();

    // Tree reduction
    for (var stride = WORKGROUP_SIZE >> 1u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) {
            shared_counts[local_id] += shared_counts[local_id + stride];
        }
        workgroupBarrier();
    }

    return shared_counts[0];
}

@compute @workgroup_size(256)
fn count_intersections(@builtin(workgroup_id) wg_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>) {

    let num_wg = num_wg_total;
    let k = get_workgroup_index(wg_id);

    // Skip if workgroup index is out of range
    if (k >= num_wg) {
        return;
    }

    // Read partition boundaries from DPI
    let packed_a_start = dpi[k];
    let packed_a_end = dpi[k + 1u];

    let a_start = packed_a_start & INDEX_MASK;
    let a_end = packed_a_end & INDEX_MASK;

    // Calculate partition size
    let a_size = a_end - a_start;

    // Early exit if partition is empty
    if (a_size == 0u) {
        if (local_id.x == 0u) {
            counts[k] = 0u;
        }
        return;
    }

    // Each thread processes a subset of A's elements
    var local_count: u32 = 0u;
    let tid = local_id.x;

    // Iterate over A elements assigned to this thread
    var i = a_start + tid;
    while (i < a_end) {
        let val = a[i];

        // Only process if this is the FIRST occurrence of val in the ENTIRE A array
        // This ensures each unique value is processed exactly once
        var is_global_first = true;
        if (i > 0u) {
            is_global_first = (a[i - 1u] != val);
        }

        if (is_global_first) {
            // Count occurrences in both arrays
            let cnt_a = count_in_a(val);
            let cnt_b = count_in_b(val);

            // Add min(cnt_a, cnt_b) to the total (multiset intersection)
            local_count += min(cnt_a, cnt_b);
        }

        i += WORKGROUP_SIZE;
    }

    // Reduce counts across all threads in workgroup
    let total_count = workgroup_reduce_sum(local_id.x, local_count);

    // Thread 0 writes the final count
    if (local_id.x == 0u) {
        counts[k] = total_count;
    }
}
