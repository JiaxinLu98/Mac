// ============================================================================
// write_intersections - Write intersection results to output buffer (MULTISET)
// Multi-threaded parallel version using workgroup prefix sum
//
// For each unique value v in A's partition (first occurrence globally):
//   Write min(occurrences_in_A(v), occurrences_in_B(v)) copies of v to output
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read> offsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> a_length: u32;
@group(0) @binding(6) var<uniform> b_length: u32;
@group(0) @binding(7) var<uniform> num_wg_total: u32;

const WORKGROUP_SIZE: u32 = 256u;
const STAR_MASK: u32 = 0x80000000u;
const INDEX_MASK: u32 = 0x7FFFFFFFu;
const MAX_DISPATCH_X: u32 = 65535u;

// Shared memory for workgroup prefix sum
var<workgroup> shared_counts: array<u32, 256>;

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

// Workgroup exclusive prefix sum (Blelloch scan)
fn workgroup_exclusive_scan(tid: u32, value: u32) -> u32 {
    shared_counts[tid] = value;
    workgroupBarrier();

    // Up-sweep (reduce) phase
    var offset: u32 = 1u;
    for (var d: u32 = WORKGROUP_SIZE >> 1u; d > 0u; d >>= 1u) {
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            shared_counts[bi] += shared_counts[ai];
        }
        offset <<= 1u;
        workgroupBarrier();
    }

    // Clear the last element
    if (tid == 0u) {
        shared_counts[WORKGROUP_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep phase
    for (var d: u32 = 1u; d < WORKGROUP_SIZE; d <<= 1u) {
        offset >>= 1u;
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let temp = shared_counts[ai];
            shared_counts[ai] = shared_counts[bi];
            shared_counts[bi] += temp;
        }
        workgroupBarrier();
    }

    return shared_counts[tid];
}

@compute @workgroup_size(256)
fn write_intersections(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let num_wg = num_wg_total;
    let k = get_workgroup_index(wg_id);
    let tid = local_id.x;

    // Skip if workgroup index is out of range
    if (k >= num_wg) {
        return;
    }

    // Read partition boundaries from DPI
    let packed_a_start = dpi[k];
    let packed_a_end = dpi[k + 1u];

    let a_start = packed_a_start & INDEX_MASK;
    let a_end = packed_a_end & INDEX_MASK;

    // Get output offset for this partition
    let out_offset = offsets[k];

    // Calculate partition size
    let partition_size = a_end - a_start;

    // Handle empty partition - all threads must participate in barriers
    if (partition_size == 0u) {
        // Still need to participate in the prefix sum barriers
        _ = workgroup_exclusive_scan(tid, 0u);
        return;
    }

    // Divide partition into blocks for each thread
    // Each thread handles a contiguous block to maintain sorted output order
    let block_size = (partition_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let my_start = a_start + tid * block_size;
    let my_end = min(my_start + block_size, a_end);

    // ========== Phase 1: Count how many values this thread will write ==========
    var my_count: u32 = 0u;

    if (my_start < a_end) {
        var i = my_start;
        while (i < my_end) {
            let val = a[i];

            // Only process if this is the FIRST occurrence of val in the ENTIRE A array
            var is_global_first = true;
            if (i > 0u) {
                is_global_first = (a[i - 1u] != val);
            }

            if (is_global_first) {
                // Count occurrences in both arrays
                let cnt_a = count_in_a(val);
                let cnt_b = count_in_b(val);
                my_count += min(cnt_a, cnt_b);
            }

            // Skip to next unique value
            i += 1u;
            while (i < my_end && a[i] == val) {
                i += 1u;
            }
        }
    }

    // ========== Phase 2: Workgroup exclusive prefix sum ==========
    let my_offset = workgroup_exclusive_scan(tid, my_count);

    // ========== Phase 3: Write values to output ==========
    if (my_start < a_end) {
        var write_pos = out_offset + my_offset;
        var i = my_start;

        while (i < my_end) {
            let val = a[i];

            // Only process if this is the FIRST occurrence of val in the ENTIRE A array
            var is_global_first = true;
            if (i > 0u) {
                is_global_first = (a[i - 1u] != val);
            }

            if (is_global_first) {
                // Count occurrences in both arrays
                let cnt_a = count_in_a(val);
                let cnt_b = count_in_b(val);

                // Write min(cnt_a, cnt_b) copies of val to output
                let write_count = min(cnt_a, cnt_b);
                for (var w: u32 = 0u; w < write_count; w++) {
                    output[write_pos] = val;
                    write_pos++;
                }
            }

            // Skip to next unique value
            i += 1u;
            while (i < my_end && a[i] == val) {
                i += 1u;
            }
        }
    }
}
