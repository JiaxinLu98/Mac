// ============================================================================
// count_intersections_optimized - Binary Search with Partition Hints (MULTISET)
//
// Optimization: Use B partition boundaries from DPI as search hints
// Instead of searching [0, b_length), start from B partition and expand
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

// ============================================================================
// Standard binary search functions (for A array - no hints needed since we
// iterate A sequentially anyway)
// ============================================================================

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

fn count_in_a(value: u32) -> u32 {
    return upper_bound_a(value) - lower_bound_a(value);
}

// ============================================================================
// Optimized binary search with hints (for B array)
// Strategy: Start from hint, use galloping to find bounds, then binary search
// ============================================================================

// Galloping lower bound: find first index where B[index] >= value
// Start from hint and expand exponentially
fn lower_bound_b_gallop(value: u32, hint: u32) -> u32 {
    // Clamp hint to valid range
    var pos = min(hint, b_length);

    if (b_length == 0u) {
        return 0u;
    }

    // Check if hint is a good starting point
    if (pos < b_length && b[pos] >= value) {
        // Value might be at or before pos, search backward
        var step: u32 = 1u;
        var lo: u32 = 0u;

        // Gallop backward to find lower bound
        while (pos > 0u && b[pos - 1u] >= value) {
            let new_pos = pos - min(step, pos);
            if (new_pos == 0u || b[new_pos] < value) {
                lo = new_pos;
                break;
            }
            pos = new_pos;
            step *= 2u;
        }

        // Binary search in [lo, pos]
        var hi = pos;
        // Handle edge case where we reached the beginning
        if (pos > 0u && b[pos - 1u] >= value) {
            lo = 0u;
        }

        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            if (b[mid] < value) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        return lo;
    } else {
        // Value is after pos, search forward
        var step: u32 = 1u;
        var hi: u32 = b_length;

        // Gallop forward to find upper bound
        while (pos < b_length && b[pos] < value) {
            let new_pos = min(pos + step, b_length);
            if (new_pos >= b_length || b[new_pos] >= value) {
                hi = new_pos;
                break;
            }
            pos = new_pos;
            step *= 2u;
        }

        // Binary search in [pos, hi]
        var lo = pos;
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
}

// Galloping upper bound: find first index where B[index] > value
fn upper_bound_b_gallop(value: u32, hint: u32) -> u32 {
    var pos = min(hint, b_length);

    if (b_length == 0u) {
        return 0u;
    }

    // Check if hint is a good starting point
    if (pos < b_length && b[pos] > value) {
        // Value ends at or before pos, search backward
        var step: u32 = 1u;
        var lo: u32 = 0u;

        while (pos > 0u && b[pos - 1u] > value) {
            let new_pos = pos - min(step, pos);
            if (new_pos == 0u || b[new_pos] <= value) {
                lo = new_pos;
                break;
            }
            pos = new_pos;
            step *= 2u;
        }

        var hi = pos;
        if (pos > 0u && b[pos - 1u] > value) {
            lo = 0u;
        }

        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            if (b[mid] <= value) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        return lo;
    } else {
        // Value ends after pos, search forward
        var step: u32 = 1u;
        var hi: u32 = b_length;

        while (pos < b_length && b[pos] <= value) {
            let new_pos = min(pos + step, b_length);
            if (new_pos >= b_length || b[new_pos] > value) {
                hi = new_pos;
                break;
            }
            pos = new_pos;
            step *= 2u;
        }

        var lo = pos;
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
}

// Count in B with hint - uses partition boundaries as starting point
fn count_in_b_with_hint(value: u32, b_hint: u32) -> u32 {
    let lo = lower_bound_b_gallop(value, b_hint);
    let hi = upper_bound_b_gallop(value, lo);  // Use lo as hint for upper bound
    return hi - lo;
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

    // Read A partition boundaries from DPI
    let packed_a_start = dpi[k];
    let packed_a_end = dpi[k + 1u];

    let a_start = packed_a_start & INDEX_MASK;
    let a_end = packed_a_end & INDEX_MASK;

    // NEW: Read B partition boundaries from DPI
    let b_start = dpi[num_wg + 1u + k];
    let b_end = dpi[num_wg + 2u + k];

    // Use middle of B partition as hint for binary search
    let b_hint = (b_start + b_end) / 2u;

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
        var is_global_first = true;
        if (i > 0u) {
            is_global_first = (a[i - 1u] != val);
        }

        if (is_global_first) {
            // Count occurrences in A (standard search)
            let cnt_a = count_in_a(val);

            // Count occurrences in B (optimized with hint)
            let cnt_b = count_in_b_with_hint(val, b_hint);

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
