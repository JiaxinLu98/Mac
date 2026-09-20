// ============================================================================
// compute diagonals: MergePath + Balanced adjust + packed star
// dpi layout:
//   dpi[0 .. num_wg]           : packed aIndex (MSB = star)
//   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
// Boundary diagonals (0 and num_wg) are written by wg0/thread0.
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> dpi: array<u32>; // diagonal_path_intersections
@group(0) @binding(3) var<uniform> a_length: u32;
@group(0) @binding(4) var<uniform> b_length: u32;
@group(0) @binding(5) var<uniform> num_wg_uniform: u32;  // actual number of workgroups (partitions)

const STAR_MASK: u32 = 0x80000000u;
const WORKGROUP_SIZE: u32 = 32u;

fn pack_a(a_idx: u32, star: bool) -> u32 {
    return select(a_idx, a_idx | STAR_MASK, star);
}

fn cmp_lte(x: u32, y: u32) -> bool {
    return x <= y;
}

// Simple (non-biased) binary searches (correctness-first)
fn lower_bound_a(end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (a[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn lower_bound_b(end_exclusive: u32, key: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn upper_bound_b(range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (b[mid] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo; // one past last
}

// ---------------------------------------------------------------------
// Balanced adjust: takes MergePath intersection p at diagonal diag,
// returns (packed aIndex|star, bIndex)
// ---------------------------------------------------------------------
fn balanced_adjust(a_count: u32, b_count: u32, diag: u32, p_in: u32) -> vec2<u32> {
    // Safety: ensure p <= diag
    var p = p_in;
    if (p > diag) { p = diag; }

    var a_index = p;
    var b_index = diag - p;
    var star = false;

    if (b_index < b_count) {
        let x = b[b_index];

        // first occurrence up to (a_index, b_index)
        let a_start = lower_bound_a(a_index, x);
        let b_start = lower_bound_b(b_index, x);

        let a_run = a_index - a_start;
        let b_run = b_index - b_start;
        let x_count = a_run + b_run;

        // split duplicate run
        var b_advance = max(x_count >> 1u, x_count - a_run);

        // probe upper bound to clamp b_advance by actual run length in B
        var b_end_hint = min(b_count, b_start + b_advance + 1u);
        // ensure the probe range is not empty and starts at b_index
        b_end_hint = max(b_end_hint, min(b_count, b_index + 1u));

        let b_run_end = upper_bound_b(b_index, b_end_hint, x);
        let actual_b_run = b_run_end - b_start;

        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;

        // star tie-break for odd split
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);

        a_index = a_start + a_advance;
        b_index = diag - a_index;
        star = round_up;
    }

    return vec2<u32>(pack_a(a_index, star), b_index);
}

// ============================================================================
// Workgroup state for diagonal search (Merge Path + Balanced Path)
// ============================================================================

var<workgroup> x_top: i32;
var<workgroup> y_top: i32;
var<workgroup> x_bottom: i32;
var<workgroup> y_bottom: i32;
var<workgroup> oneorzero: array<u32, 32>;
var<workgroup> found: bool;

// Store the merge-path intersection (one per workgroup)
var<workgroup> found_x: u32;
var<workgroup> found_y: u32; 

@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_diagonals(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(num_workgroups) n_wgs: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) wg: vec3<u32>) {

    // Local thread index (0..31)
    let local_flat = local_id.x;
    let group_flat = wg.y * n_wgs.x + wg.x;
    let num_wg = num_wg_uniform;  // Use uniform instead of n_wgs.x * n_wgs.y to handle 2D dispatch correctly
    let k = group_flat;

    // ------------------------------------------------------------------------
    // Write boundary diagonals once (diag 0 and diag num_wg)
    // ------------------------------------------------------------------------
    if (local_flat == 0u && k == 0u) {
        // diagonal 0: (0, 0)
        dpi[0u] = pack_a(0u, false);
        dpi[num_wg + 1u] = 0u;

        // diagonal num_wg: (a_length, b_length)
        dpi[num_wg] = pack_a(a_length, false);
        dpi[num_wg + num_wg + 1u] = b_length;
    }

    // We do not compute diagonal 0 here (boundary already set).
    if (k == 0u) {
        return;
    }
    // Safety if dispatch count ever changes
    if (k >= num_wg) {
        return;
    }

    // ------------------------------------------------------------------------
    // Compute this workgroup's diagonal position (u32 diag)
    // diag = floor(k * (a_len+b_len) / num_wg)
    // Avoid overflow: k * total could exceed u32 max for large datasets
    // Use: diag = k * (total / num_wg) + (k * (total % num_wg)) / num_wg
    // ------------------------------------------------------------------------
    let total_u: u32 = a_length + b_length;
    let quotient: u32 = total_u / num_wg;
    let remainder: u32 = total_u % num_wg;
    let diag_u: u32 = k * quotient + (k * remainder) / num_wg;

    // For initializing the i32 search window
    let combined_index: i32 = i32(diag_u);
    let n_a: i32 = i32(a_length);
    let n_b: i32 = i32(b_length);

    // ------------------------------------------------------------------------
    // Init workgroup state (thread0) and barrier
    // ------------------------------------------------------------------------
    if (local_flat == 0u) {
        x_top = min(combined_index, n_a);
        y_top = max(0, combined_index - n_a);
        x_bottom = max(0, combined_index - n_b);
        y_bottom = min(combined_index, n_b);

        found = false;
        found_x = 0u;
        found_y = 0u;
    }
    workgroupBarrier();

    let thread_offset = i32(local_flat) - 16;

    // ------------------------------------------------------------------------
    // MergePath diagonal search (32 threads)
    // ------------------------------------------------------------------------
    while (!workgroupUniformLoad(&found)) {
        var current_x: i32 = 0;
        var current_y: i32 = 0;
        var r: u32 = 0u;

        current_x = x_top - ((x_top - x_bottom) >> 1) - thread_offset;
        current_y = y_top + ((y_bottom - y_top) >> 1) + thread_offset;

        if (current_x > n_a || current_y < 0) {
            r = 0u;
        } else if (current_y >= n_b || current_x < 1) {
            r = 1u;
        } else {
            // r = (a[current_x-1] <= b[current_y]) ? 1 : 0
            r = select(0u, 1u, cmp_lte(a[u32(current_x - 1)], b[u32(current_y)]));
        }

        oneorzero[local_flat] = r;
        workgroupBarrier();

        // Detect 0/1 boundary (unique due to monotonicity of sorted arrays)
        if (local_flat > 0u && oneorzero[local_flat] != oneorzero[local_flat - 1u]) {
            found_x = u32(current_x); // merge-path aIndex p
            found_y = u32(current_y); // merge-path bIndex (debug)
            found = true;
        }
        workgroupBarrier();

        // Shrink search window (single controlling thread)
        if (local_flat == 16u) {
            if (oneorzero[31] != 0u) {
                x_bottom = current_x;
                y_bottom = current_y;
            } else {
                x_top = current_x;
                y_top = current_y;
            }
        }
        workgroupBarrier();
    }

    // ------------------------------------------------------------------------
    // Balanced adjust once (thread0) + write dpi
    // ------------------------------------------------------------------------
    if (local_flat == 0u) {
        let out = balanced_adjust(a_length, b_length, diag_u, found_x);
        dpi[k] = out.x;                 // packed aIndex|star
        dpi[k + num_wg + 1u] = out.y;   // bIndex
    }
}
