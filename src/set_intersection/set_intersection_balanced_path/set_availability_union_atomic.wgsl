// ============================================================================
// Set Availability - Union Atomic Kernel (Strategy B)
//
// Single-pass approach using atomic operations for global offset allocation
// ============================================================================

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> global_counter: atomic<u32>;
@group(0) @binding(5) var<uniform> a_length: u32;
@group(0) @binding(6) var<uniform> b_length: u32;
@group(0) @binding(7) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants
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

var<workgroup> shared_scan: array<u32, NT>;
var<workgroup> wg_global_offset: u32;

// ============================================================================
// Helper Functions
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    var i = tid;
    while (i < a_load_count) {
        keys_shared[i] = a[a_global_offset + i];
        i += NT;
    }
    i = tid;
    while (i < b_load_count) {
        keys_shared[b_shared_start + i] = b[b_global_offset + i];
        i += NT;
    }
}

fn balanced_path_local_biased(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    let diag_u = u32(max(0, diag));

    if (diag_u == 0u) { return vec2<u32>(0u, 0u); }
    if (diag_u >= a_count + b_count) { return vec2<u32>(a_count, b_count); }

    var lo: u32 = select(0u, diag_u - b_count, diag_u > b_count);
    var hi: u32 = min(diag_u, a_count);

    if (lo < hi) {
        let scale = (1u << 5u) - 1u;
        let mid = (lo + scale * hi) >> 5u;
        let a_key = keys_shared[mid];
        let b_key = keys_shared[b_start + diag_u - 1u - mid];
        if (a_key <= b_key) { lo = mid + 1u; } else { hi = mid; }
    }

    if (lo < hi) {
        let scale = (1u << 4u) - 1u;
        let mid = (lo + scale * hi) >> 4u;
        let a_key = keys_shared[mid];
        let b_key = keys_shared[b_start + diag_u - 1u - mid];
        if (a_key <= b_key) { lo = mid + 1u; } else { hi = mid; }
    }

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let a_key = keys_shared[mid];
        let b_key = keys_shared[b_start + diag_u - 1u - mid];
        if (a_key <= b_key) { lo = mid + 1u; } else { hi = mid; }
    }

    return vec2<u32>(lo, diag_u - lo);
}

fn serial_set_union(
    a_begin: u32,
    a_end: u32,
    b_begin: u32,
    b_end: u32,
    results: ptr<function, array<u32, 7>>,
    indices: ptr<function, array<u32, 7>>
) -> u32 {
    var commit: u32 = 0u;
    var a_idx = a_begin;
    var b_idx = b_begin;

    for (var i: u32 = 0u; i < VT; i++) {
        let a_valid = a_idx < a_end;
        let b_valid = b_idx < b_end;

        if (!a_valid && !b_valid) { break; }

        var result_val: u32 = 0u;
        var result_idx: u32 = 0u;

        if (a_valid && b_valid) {
            let a_key = keys_shared[a_idx];
            let b_key = keys_shared[b_idx];

            if (a_key < b_key) {
                result_val = a_key;
                result_idx = a_idx;
                a_idx++;
            } else if (b_key < a_key) {
                result_val = b_key;
                result_idx = b_idx;
                b_idx++;
            } else {
                result_val = a_key;
                result_idx = a_idx;
                a_idx++;
                b_idx++;
            }
        } else if (a_valid) {
            result_val = keys_shared[a_idx];
            result_idx = a_idx;
            a_idx++;
        } else {
            result_val = keys_shared[b_idx];
            result_idx = b_idx;
            b_idx++;
        }

        (*results)[i] = result_val;
        (*indices)[i] = result_idx;
        commit |= (1u << i);
    }

    return commit;
}

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

    workgroupBarrier();
    if (tid == 0u) {
        shared_scan[tid] = 0u;
    } else {
        let temp = shared_scan[tid - 1u];
        workgroupBarrier();
        shared_scan[tid] = temp;
    }
    workgroupBarrier();

    return shared_scan[tid];
}

// ============================================================================
// Main Kernel
// ============================================================================
@compute @workgroup_size(256)
fn compute_availability_atomic(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let block = get_workgroup_index(wg_id);

    if (block >= num_wg_total) { return; }

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

    device_load_2_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    let total_count = a_count2 + b_count2;
    let diag = min(i32(VT * tid) - i32(bit0), i32(total_count));

    let bp = balanced_path_local_biased(a_count2, b_start, b_count2, diag);

    let a0tid = bp.x;
    let b0tid = bp.y;

    var results: array<u32, 7>;
    var indices: array<u32, 7>;

    let commit = serial_set_union(
        a0tid,
        a_count2,
        b_start + b0tid,
        b_start + b_count2,
        &results,
        &indices
    );

    let local_count = countOneBits(commit);
    var wg_total: u32 = 0u;
    let local_offset = workgroup_exclusive_scan_with_total(tid, local_count, &wg_total);

    if (tid == 0u) {
        if (wg_total > 0u) {
            wg_global_offset = atomicAdd(&global_counter, wg_total);
        } else {
            wg_global_offset = 0u;
        }
    }
    workgroupBarrier();

    if (local_count > 0u) {
        let global_offset = wg_global_offset + local_offset;

        var write_pos = global_offset;
        for (var i: u32 = 0u; i < VT; i++) {
            if ((commit & (1u << i)) != 0u) {
                output[write_pos] = results[i];
                write_pos++;
            }
        }
    }
}
