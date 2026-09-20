// Set Intersection using Balanced Path algorithm
// Based on ModernGPU's balanced path approach for handling duplicates correctly
//
// Balanced Path differs from Merge Path in how it handles duplicate keys:
// - Merge Path greedily consumes all duplicates from A, then from B
// - Balanced Path pairs elements by key-rank match for correct partitioning

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<storage, read_write> partitions: array<u32>; // [aIndex, starred flag in MSB]
@group(0) @binding(4) var<storage, read_write> wg_counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> wg_offsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> total_count: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

struct Params {
    a_len: u32,
    b_len: u32,
    num_partitions: u32,
    vt: u32,  // values per thread
}

// ============================================================================
// Constants
// ============================================================================
const WORKGROUP_SIZE: u32 = 128u;
const VT: u32 = 4u;  // Values per thread for serial processing
const NV: u32 = WORKGROUP_SIZE * VT;  // Elements per workgroup
const STAR_MASK: u32 = 0x80000000u;  // MSB for star flag
const MAX_DISPATCH_X: u32 = 65535u;  // WebGPU limit per dimension

// Convert 2D workgroup_id to 1D index (for large dispatches)
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// Biased Binary Search
// Optimized for cases where duplicates are near the search start point
// Uses weighted midpoints: 511/512, 127/128, 31/32, 15/16 before symmetric search
// ============================================================================

fn biased_binary_search_it_lower(data_ptr: u32, begin_ptr: ptr<function, u32>, end_ptr: ptr<function, u32>, key: u32, shift: u32) {
    let begin = *begin_ptr;
    let end = *end_ptr;
    let scale = (1u << shift) - 1u;
    let mid = (begin + scale * end) >> shift;

    let key2 = a[data_ptr + mid];
    // Lower bound: pred = key2 < key
    if (key2 < key) {
        *begin_ptr = mid + 1u;
    } else {
        *end_ptr = mid;
    }
}

fn biased_binary_search_it_lower_b(data_ptr: u32, begin_ptr: ptr<function, u32>, end_ptr: ptr<function, u32>, key: u32, shift: u32) {
    let begin = *begin_ptr;
    let end = *end_ptr;
    let scale = (1u << shift) - 1u;
    let mid = (begin + scale * end) >> shift;

    let key2 = b[data_ptr + mid];
    if (key2 < key) {
        *begin_ptr = mid + 1u;
    } else {
        *end_ptr = mid;
    }
}

fn biased_binary_search_it_upper_b(data_ptr: u32, begin_ptr: ptr<function, u32>, end_ptr: ptr<function, u32>, key: u32, shift: u32) {
    let begin = *begin_ptr;
    let end = *end_ptr;
    let scale = (1u << shift) - 1u;
    let mid = (begin + scale * end) >> shift;

    let key2 = b[data_ptr + mid];
    // Upper bound: pred = !(key < key2) = key >= key2
    if (key >= key2) {
        *begin_ptr = mid + 1u;
    } else {
        *end_ptr = mid;
    }
}

// Biased binary search for lower bound in A array
// levels: number of biased iterations before symmetric search
fn biased_binary_search_lower_a(data_start: u32, count: u32, key: u32, levels: u32) -> u32 {
    var begin: u32 = 0u;
    var end: u32 = count;

    if (levels >= 4u && begin < end) {
        biased_binary_search_it_lower(data_start, &begin, &end, key, 9u);
    }
    if (levels >= 3u && begin < end) {
        biased_binary_search_it_lower(data_start, &begin, &end, key, 7u);
    }
    if (levels >= 2u && begin < end) {
        biased_binary_search_it_lower(data_start, &begin, &end, key, 5u);
    }
    if (levels >= 1u && begin < end) {
        biased_binary_search_it_lower(data_start, &begin, &end, key, 4u);
    }

    // Symmetric binary search for remaining iterations
    while (begin < end) {
        let mid = (begin + end) >> 1u;
        if (a[data_start + mid] < key) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    return begin;
}

// Biased binary search for lower bound in B array
fn biased_binary_search_lower_b(data_start: u32, count: u32, key: u32, levels: u32) -> u32 {
    var begin: u32 = 0u;
    var end: u32 = count;

    if (levels >= 4u && begin < end) {
        biased_binary_search_it_lower_b(data_start, &begin, &end, key, 9u);
    }
    if (levels >= 3u && begin < end) {
        biased_binary_search_it_lower_b(data_start, &begin, &end, key, 7u);
    }
    if (levels >= 2u && begin < end) {
        biased_binary_search_it_lower_b(data_start, &begin, &end, key, 5u);
    }
    if (levels >= 1u && begin < end) {
        biased_binary_search_it_lower_b(data_start, &begin, &end, key, 4u);
    }

    while (begin < end) {
        let mid = (begin + end) >> 1u;
        if (b[data_start + mid] < key) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    return begin;
}

// Biased binary search for upper bound in B array
fn biased_binary_search_upper_b(data_start: u32, count: u32, key: u32, levels: u32) -> u32 {
    var begin: u32 = 0u;
    var end: u32 = count;

    if (levels >= 4u && begin < end) {
        biased_binary_search_it_upper_b(data_start, &begin, &end, key, 9u);
    }
    if (levels >= 3u && begin < end) {
        biased_binary_search_it_upper_b(data_start, &begin, &end, key, 7u);
    }
    if (levels >= 2u && begin < end) {
        biased_binary_search_it_upper_b(data_start, &begin, &end, key, 5u);
    }
    if (levels >= 1u && begin < end) {
        biased_binary_search_it_upper_b(data_start, &begin, &end, key, 4u);
    }

    while (begin < end) {
        let mid = (begin + end) >> 1u;
        if (key >= b[data_start + mid]) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    return begin;
}

// ============================================================================
// Standard Merge Path for finding initial intersection
// ============================================================================
fn merge_path(a_count: u32, b_count: u32, diag: u32) -> u32 {
    var begin = select(0u, diag - b_count, diag > b_count);
    var end = min(diag, a_count);

    while (begin < end) {
        let mid = (begin + end) >> 1u;
        let a_key = a[mid];
        let b_key = b[diag - 1u - mid];
        if (a_key <= b_key) {
            begin = mid + 1u;
        } else {
            end = mid;
        }
    }
    return begin;
}

// ============================================================================
// Balanced Path Algorithm
// Returns (aIndex, star) packed as: aIndex | (star ? STAR_MASK : 0)
// ============================================================================
fn balanced_path(a_count: u32, b_count: u32, diag: u32, levels: u32) -> u32 {
    // Step 1: Find Merge Path intersection with cross-diagonal
    let p = merge_path(a_count, b_count, diag);
    var a_index = p;
    let b_index = diag - p;

    var star = false;

    if (b_index < b_count) {
        // Get the key at B[bIndex]
        let x = b[b_index];

        // Step 2 & 3: Binary search to find where Balanced Path diverges from Merge Path
        // Find first occurrence of key x in both arrays
        let a_start = biased_binary_search_lower_a(0u, a_index, x, levels);
        let b_start = biased_binary_search_lower_b(0u, b_index, x, levels);

        // Step 4: Calculate run lengths
        let a_run = a_index - a_start;
        let b_run = b_index - b_start;
        let x_count = a_run + b_run;

        // Attempt to evenly distribute the run
        var b_advance = max(x_count >> 1u, x_count - a_run);

        // Find upper bound of key x in B to limit b_advance
        let b_end = min(b_count, b_start + b_advance + 1u);
        let b_run_end = biased_binary_search_upper_b(b_index, b_end - b_index, x, levels) + b_index;
        let actual_b_run = b_run_end - b_start;

        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;

        // Step 5: Check if we need to star this diagonal
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        a_index = a_start + a_advance;

        if (round_up) {
            star = true;
        }
    }

    return select(a_index, a_index | STAR_MASK, star);
}

// ============================================================================
// Workgroup shared memory
// ============================================================================
var<workgroup> shared_a: array<u32, NV + 1u>;
var<workgroup> shared_b: array<u32, NV + 1u>;
var<workgroup> wg_a_start: u32;
var<workgroup> wg_a_end: u32;
var<workgroup> wg_b_start: u32;
var<workgroup> wg_b_end: u32;
var<workgroup> wg_a_star: bool;
var<workgroup> wg_b_star: bool;
var<workgroup> local_count: atomic<u32>;
var<workgroup> wg_output_offset: u32;

// ============================================================================
// Phase 1: Compute Balanced Path partitions
// ============================================================================
@compute @workgroup_size(32)
fn compute_partitions(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Support 2D dispatch for large partition counts
    let partition_idx = global_id.x + global_id.y * MAX_DISPATCH_X * 32u;
    let num_partitions = params.num_partitions;

    if (partition_idx > num_partitions) {
        return;
    }

    let a_count = params.a_len;
    let b_count = params.b_len;
    let total = a_count + b_count;

    // Boundary cases
    if (partition_idx == 0u) {
        partitions[0] = 0u;  // aIndex = 0, no star
        return;
    }
    if (partition_idx == num_partitions) {
        partitions[num_partitions] = a_count;  // aIndex = a_count, no star
        return;
    }

    // Calculate diagonal for this partition
    let diag = (partition_idx * total) / num_partitions;

    // Use 4 levels of biased binary search for global partitioning
    let result = balanced_path(a_count, b_count, diag, 4u);
    partitions[partition_idx] = result;
}

// ============================================================================
// Phase 2: Count intersections using serial set intersection
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn count_intersections(@builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let wg_idx = get_workgroup_index(wg_id);  // Support 2D dispatch

    // Bounds check for 2D dispatch padding
    if (wg_idx >= params.num_partitions) {
        return;
    }

    // Load partition boundaries
    if (tid == 0u) {
        let p0 = partitions[wg_idx];
        let p1 = partitions[wg_idx + 1u];

        wg_a_start = p0 & ~STAR_MASK;
        wg_a_star = (p0 & STAR_MASK) != 0u;
        wg_a_end = p1 & ~STAR_MASK;
        wg_b_star = (p1 & STAR_MASK) != 0u;

        // Calculate B range from diagonal positions
        let total = params.a_len + params.b_len;
        let diag0 = (wg_idx * total) / params.num_partitions;
        let diag1 = ((wg_idx + 1u) * total) / params.num_partitions;

        wg_b_start = diag0 - wg_a_start;
        wg_b_end = diag1 - wg_a_end;

        // Handle star adjustments
        if (wg_a_star && wg_b_start > 0u) {
            wg_b_start -= 1u;  // Steal from left partition
        }
        if (wg_b_star && wg_b_end < params.b_len) {
            wg_b_end += 1u;  // Include the starred element
        }

        atomicStore(&local_count, 0u);
    }
    workgroupBarrier();

    let a_start = wg_a_start;
    let a_end = min(wg_a_end, params.a_len);
    let b_start = wg_b_start;
    let b_end = min(wg_b_end, params.b_len);

    let a_len = a_end - a_start;
    let b_len = b_end - b_start;

    // Load A elements into shared memory
    for (var i = tid; i < a_len; i += WORKGROUP_SIZE) {
        shared_a[i] = a[a_start + i];
    }
    // Add sentinel
    if (tid == 0u && a_len > 0u) {
        shared_a[a_len] = 0xFFFFFFFFu;
    }

    // Load B elements into shared memory
    for (var i = tid; i < b_len; i += WORKGROUP_SIZE) {
        shared_b[i] = b[b_start + i];
    }
    // Add sentinel
    if (tid == 0u && b_len > 0u) {
        shared_b[b_len] = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    // Serial set intersection within shared memory
    // Each thread processes VT elements using merge-style comparison
    let elements_per_thread = (a_len + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let thread_a_start = min(tid * elements_per_thread, a_len);
    let thread_a_end = min(thread_a_start + elements_per_thread, a_len);

    var thread_count = 0u;
    var b_ptr = 0u;

    // Binary search to find starting position in B for this thread's A range
    if (thread_a_start < a_len && b_len > 0u) {
        let first_a = shared_a[thread_a_start];
        var lo = 0u;
        var hi = b_len;
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            if (shared_b[mid] < first_a) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        b_ptr = lo;
    }

    // Process this thread's portion of A
    for (var i = thread_a_start; i < thread_a_end; i++) {
        let a_val = shared_a[i];

        // Advance b_ptr while B[b_ptr] < A[i]
        while (b_ptr < b_len && shared_b[b_ptr] < a_val) {
            b_ptr++;
        }

        // Check for match
        if (b_ptr < b_len && shared_b[b_ptr] == a_val) {
            // For duplicates: count based on rank matching
            // Find first occurrence of this value in A
            var a_first = i;
            while (a_first > 0u && shared_a[a_first - 1u] == a_val) {
                a_first--;
            }
            let a_rank = i - a_first;

            // Find first occurrence of this value in B
            var b_first = b_ptr;
            while (b_first > 0u && shared_b[b_first - 1u] == a_val) {
                b_first--;
            }

            // Count occurrences in B
            var b_last = b_ptr;
            while (b_last < b_len && shared_b[b_last] == a_val) {
                b_last++;
            }
            let b_count_val = b_last - b_first;

            // Only count if this A's rank < count in B
            if (a_rank < b_count_val) {
                thread_count++;
            }
        }
    }

    // Accumulate count
    if (thread_count > 0u) {
        atomicAdd(&local_count, thread_count);
    }

    workgroupBarrier();

    // Write workgroup count
    if (tid == 0u) {
        wg_counts[wg_idx] = atomicLoad(&local_count);
    }
}

// ============================================================================
// Phase 3: Write intersections to output
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn write_intersections(@builtin(local_invocation_id) local_id: vec3<u32>,
                       @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let wg_idx = get_workgroup_index(wg_id);  // Support 2D dispatch

    // Bounds check for 2D dispatch padding
    if (wg_idx >= params.num_partitions) {
        return;
    }

    // Load partition boundaries and output offset
    if (tid == 0u) {
        let p0 = partitions[wg_idx];
        let p1 = partitions[wg_idx + 1u];

        wg_a_start = p0 & ~STAR_MASK;
        wg_a_star = (p0 & STAR_MASK) != 0u;
        wg_a_end = p1 & ~STAR_MASK;
        wg_b_star = (p1 & STAR_MASK) != 0u;

        let total = params.a_len + params.b_len;
        let diag0 = (wg_idx * total) / params.num_partitions;
        let diag1 = ((wg_idx + 1u) * total) / params.num_partitions;

        wg_b_start = diag0 - wg_a_start;
        wg_b_end = diag1 - wg_a_end;

        if (wg_a_star && wg_b_start > 0u) {
            wg_b_start -= 1u;
        }
        if (wg_b_star && wg_b_end < params.b_len) {
            wg_b_end += 1u;
        }

        wg_output_offset = wg_offsets[wg_idx];
        atomicStore(&local_count, 0u);
    }
    workgroupBarrier();

    let a_start = wg_a_start;
    let a_end = min(wg_a_end, params.a_len);
    let b_start = wg_b_start;
    let b_end = min(wg_b_end, params.b_len);

    let a_len = a_end - a_start;
    let b_len = b_end - b_start;

    // Load elements into shared memory
    for (var i = tid; i < a_len; i += WORKGROUP_SIZE) {
        shared_a[i] = a[a_start + i];
    }
    if (tid == 0u && a_len > 0u) {
        shared_a[a_len] = 0xFFFFFFFFu;
    }

    for (var i = tid; i < b_len; i += WORKGROUP_SIZE) {
        shared_b[i] = b[b_start + i];
    }
    if (tid == 0u && b_len > 0u) {
        shared_b[b_len] = 0xFFFFFFFFu;
    }

    workgroupBarrier();

    // Serial set intersection - write matches
    let elements_per_thread = (a_len + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;
    let thread_a_start = min(tid * elements_per_thread, a_len);
    let thread_a_end = min(thread_a_start + elements_per_thread, a_len);

    var b_ptr = 0u;

    if (thread_a_start < a_len && b_len > 0u) {
        let first_a = shared_a[thread_a_start];
        var lo = 0u;
        var hi = b_len;
        while (lo < hi) {
            let mid = (lo + hi) >> 1u;
            if (shared_b[mid] < first_a) {
                lo = mid + 1u;
            } else {
                hi = mid;
            }
        }
        b_ptr = lo;
    }

    for (var i = thread_a_start; i < thread_a_end; i++) {
        let a_val = shared_a[i];

        while (b_ptr < b_len && shared_b[b_ptr] < a_val) {
            b_ptr++;
        }

        if (b_ptr < b_len && shared_b[b_ptr] == a_val) {
            var a_first = i;
            while (a_first > 0u && shared_a[a_first - 1u] == a_val) {
                a_first--;
            }
            let a_rank = i - a_first;

            var b_first = b_ptr;
            while (b_first > 0u && shared_b[b_first - 1u] == a_val) {
                b_first--;
            }

            var b_last = b_ptr;
            while (b_last < b_len && shared_b[b_last] == a_val) {
                b_last++;
            }
            let b_count_val = b_last - b_first;

            if (a_rank < b_count_val) {
                let write_idx = wg_output_offset + atomicAdd(&local_count, 1u);
                output[write_idx] = a_val;
            }
        }
    }
}
