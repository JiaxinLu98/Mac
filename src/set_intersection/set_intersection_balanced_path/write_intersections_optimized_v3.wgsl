// ============================================================================
// write_intersections_optimized_v3 - Simplified optimization
//
// Key insight: We need count in ENTIRE array, not just partition!
// Optimization: Use partition hint to choose better search starting point
// - If value >= b[b_start], search [b_start, b_length] then [0, b_start] if needed
// - Simpler logic, fewer branches, less overhead
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

var<workgroup> shared_counts: array<u32, 256>;

fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// Standard binary search in A (no optimization needed - we process A linearly)
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

// V3: Simple hint-based lower bound in B
// If b_start > 0 and b[b_start-1] < value, start search from b_start
fn lower_bound_b_v3(value: u32, b_start: u32) -> u32 {
    var lo: u32;
    var hi: u32 = b_length;

    // Simple heuristic: if we're past the beginning and the previous element < value,
    // we can start from b_start (value is likely >= b[b_start])
    if (b_start > 0u && b_start < b_length && b[b_start - 1u] < value) {
        lo = b_start;
    } else {
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
}

// V3: Simple hint-based upper bound in B
fn upper_bound_b_v3(value: u32, lower_result: u32) -> u32 {
    // Start from lower_bound result (we know value exists at or after this point)
    var lo: u32 = lower_result;
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

fn count_in_a(value: u32) -> u32 {
    return upper_bound_a(value) - lower_bound_a(value);
}

fn count_in_b_v3(value: u32, b_start: u32) -> u32 {
    let lo = lower_bound_b_v3(value, b_start);
    let hi = upper_bound_b_v3(value, lo);
    return hi - lo;
}

fn workgroup_exclusive_scan(tid: u32, value: u32) -> u32 {
    shared_counts[tid] = value;
    workgroupBarrier();

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

    if (tid == 0u) {
        shared_counts[WORKGROUP_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

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

    if (k >= num_wg) {
        return;
    }

    let packed_a_start = dpi[k];
    let packed_a_end = dpi[k + 1u];
    let b_start = dpi[num_wg + 1u + k];  // Read B partition hint

    let a_start = packed_a_start & INDEX_MASK;
    let a_end = packed_a_end & INDEX_MASK;
    let out_offset = offsets[k];
    let partition_size = a_end - a_start;

    if (partition_size == 0u) {
        _ = workgroup_exclusive_scan(tid, 0u);
        return;
    }

    let block_size = (partition_size + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let my_start = a_start + tid * block_size;
    let my_end = min(my_start + block_size, a_end);

    // Phase 1: Count
    var my_count: u32 = 0u;

    if (my_start < a_end) {
        var i = my_start;
        while (i < my_end) {
            let val = a[i];

            var is_global_first = true;
            if (i > 0u) {
                is_global_first = (a[i - 1u] != val);
            }

            if (is_global_first) {
                let cnt_a = count_in_a(val);
                let cnt_b = count_in_b_v3(val, b_start);
                my_count += min(cnt_a, cnt_b);
            }

            i += 1u;
            while (i < my_end && a[i] == val) {
                i += 1u;
            }
        }
    }

    // Phase 2: Prefix sum
    let my_offset = workgroup_exclusive_scan(tid, my_count);

    // Phase 3: Write
    if (my_start < a_end) {
        var write_pos = out_offset + my_offset;
        var i = my_start;

        while (i < my_end) {
            let val = a[i];

            var is_global_first = true;
            if (i > 0u) {
                is_global_first = (a[i - 1u] != val);
            }

            if (is_global_first) {
                let cnt_a = count_in_a(val);
                let cnt_b = count_in_b_v3(val, b_start);
                let write_count = min(cnt_a, cnt_b);

                for (var w: u32 = 0u; w < write_count; w++) {
                    output[write_pos] = val;
                    write_pos++;
                }
            }

            i += 1u;
            while (i < my_end && a[i] == val) {
                i += 1u;
            }
        }
    }
}
