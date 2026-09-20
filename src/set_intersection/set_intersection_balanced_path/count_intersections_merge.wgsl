// ============================================================================
// count_intersections - Merge-Based Version (MULTISET)
//
// Uses dual-pointer merge traversal on A and B partitions.
// Only performs global binary search for MATCHING values.
//
// Key optimization: Instead of searching for every unique value in A,
// we only search when a value appears in BOTH A and B partitions.
// This dramatically reduces binary search calls when intersection rate is low.
//
// For each unique value v found in BOTH partitions (first occurrence globally):
//   count += min(occurrences_in_A(v), occurrences_in_B(v))
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<uniform> a_length: u32;
@group(0) @binding(5) var<uniform> b_length: u32;
@group(0) @binding(6) var<uniform> num_wg_total: u32;

const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

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

@compute @workgroup_size(256)
fn count_intersections(@builtin(workgroup_id) wg_id: vec3<u32>,
                       @builtin(local_invocation_id) local_id: vec3<u32>) {

    let num_wg = num_wg_total;
    let k = get_workgroup_index(wg_id);

    // Skip if workgroup index is out of range
    if (k >= num_wg) {
        return;
    }

    // Only thread 0 does the work (sequential merge traversal)
    if (local_id.x != 0u) {
        return;
    }

    // ========================================================================
    // Read BOTH A and B partition boundaries from DPI
    // DPI layout:
    //   dpi[0 .. num_wg]           : packed aIndex (MSB = star)
    //   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
    // ========================================================================
    let packed_a_start = dpi[k];
    let packed_a_end = dpi[k + 1u];
    let a_start = packed_a_start & INDEX_MASK;
    let a_end = packed_a_end & INDEX_MASK;

    // B partition boundaries (KEY: these were previously unused!)
    let b_start = dpi[num_wg + 1u + k];
    let b_end = dpi[num_wg + 2u + k];

    // Early exit if either partition is empty
    if (a_start >= a_end || b_start >= b_end) {
        counts[k] = 0u;
        return;
    }

    // ========================================================================
    // Merge-based traversal using dual pointers
    // Only perform global binary search when values MATCH in both partitions
    // ========================================================================
    var count: u32 = 0u;
    var i = a_start;  // pointer into A partition
    var j = b_start;  // pointer into B partition

    while (i < a_end && j < b_end) {
        let a_val = a[i];
        let b_val = b[j];

        if (a_val < b_val) {
            // A value not in B partition, skip all duplicates in A
            i += 1u;
            while (i < a_end && a[i] == a_val) {
                i += 1u;
            }
        } else if (a_val > b_val) {
            // B value not in A partition, skip all duplicates in B
            j += 1u;
            while (j < b_end && b[j] == b_val) {
                j += 1u;
            }
        } else {
            // MATCH FOUND: a_val == b_val
            // Only count if this is the FIRST occurrence of val in ENTIRE A array
            // This ensures each unique value is counted exactly once globally
            var is_global_first = true;
            if (i > 0u) {
                is_global_first = (a[i - 1u] != a_val);
            }

            if (is_global_first) {
                // Only now do we perform global binary search (4 searches total)
                let cnt_a = count_in_a(a_val);
                let cnt_b = count_in_b(a_val);
                count += min(cnt_a, cnt_b);
            }

            // Skip all duplicates in both partitions
            i += 1u;
            while (i < a_end && a[i] == a_val) {
                i += 1u;
            }
            j += 1u;
            while (j < b_end && b[j] == b_val) {
                j += 1u;
            }
        }
    }

    counts[k] = count;
}
