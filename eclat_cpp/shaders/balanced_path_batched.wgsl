// ============================================================================
// Balanced Path with Biased Binary Search - BATCHED VERSION
//
// Same algorithm as balanced_path_biased.wgsl, but reads per-pair metadata
// from a wgInfo table instead of uniform buffers. This allows a single
// dispatch to compute DPI for all intersection pairs.
//
// wgInfo layout per workgroup (8 x u32, stride = WG_INFO_STRIDE):
//   [0] pairAStart   - offset into keysA for this pair
//   [1] pairALen     - length of A for this pair
//   [2] pairBStart   - offset into keysB for this pair
//   [3] pairBLen     - length of B for this pair
//   [4] localWgIdx   - workgroup index within this pair (0-based)
//   [5] pairNumWg    - total workgroups for this pair
//   [6] globalWgOffset - start of this pair's region in state buffer
//   [7] dpiOffset    - start of this pair's region in DPI buffer
// ============================================================================

enable subgroups;

@group(0) @binding(0) var<storage, read> keysA: array<u32>;
@group(0) @binding(1) var<storage, read> keysB: array<u32>;
@group(0) @binding(2) var<storage, read_write> dpi: array<u32>;
@group(0) @binding(3) var<storage, read> wgInfo: array<u32>;
@group(0) @binding(4) var<uniform> totalWg_uniform: u32;

const STAR_MASK: u32 = 0x80000000u;
const WORKGROUP_SIZE: u32 = 256u;
const WG_INFO_STRIDE: u32 = 8u;

const NT: u32 = 256u;
const VT: u32 = 12u;
const NV: u32 = NT * VT;

fn pack_a(a_idx: u32, star: bool) -> u32 {
    return select(a_idx, a_idx | STAR_MASK, star);
}

fn get_biased_search_levels(partition_size: u32) -> u32 {
    if (partition_size >= 512u) { return 4u; }
    if (partition_size >= 128u) { return 3u; }
    if (partition_size >= 32u)  { return 2u; }
    if (partition_size >= 16u)  { return 1u; }
    return 0u;
}

fn cmp_lte(x: u32, y: u32) -> bool {
    return x <= y;
}

// ============================================================================
// Binary search functions - read from keysA/keysB with offset
// ============================================================================
// These use module-scope variables set at kernel start
var<private> g_aStart: u32;
var<private> g_bStart: u32;

fn binary_search_step_a(
    lo_ptr: ptr<function, u32>,
    hi_ptr: ptr<function, u32>,
    key: u32,
    shift: u32
) {
    let lo = *lo_ptr;
    let hi = *hi_ptr;
    if (lo >= hi) { return; }
    let scale = (1u << shift) - 1u;
    let mid = (lo + scale * hi) >> shift;
    if (keysA[g_aStart + mid] < key) {
        *lo_ptr = mid + 1u;
    } else {
        *hi_ptr = mid;
    }
}

fn binary_search_step_b(
    lo_ptr: ptr<function, u32>,
    hi_ptr: ptr<function, u32>,
    key: u32,
    shift: u32
) {
    let lo = *lo_ptr;
    let hi = *hi_ptr;
    if (lo >= hi) { return; }
    let scale = (1u << shift) - 1u;
    let mid = (lo + scale * hi) >> shift;
    if (keysB[g_bStart + mid] < key) {
        *lo_ptr = mid + 1u;
    } else {
        *hi_ptr = mid;
    }
}

fn biased_lower_bound_a(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    if (levels >= 4u && lo < hi) { binary_search_step_a(&lo, &hi, key, 9u); }
    if (levels >= 3u && lo < hi) { binary_search_step_a(&lo, &hi, key, 7u); }
    if (levels >= 2u && lo < hi) { binary_search_step_a(&lo, &hi, key, 5u); }
    if (levels >= 1u && lo < hi) { binary_search_step_a(&lo, &hi, key, 4u); }
    while (lo < hi) {
        binary_search_step_a(&lo, &hi, key, 1u);
    }
    return lo;
}

fn biased_lower_bound_b(end_exclusive: u32, key: u32, levels: u32) -> u32 {
    var lo: u32 = 0u;
    var hi: u32 = end_exclusive;
    if (levels >= 4u && lo < hi) { binary_search_step_b(&lo, &hi, key, 9u); }
    if (levels >= 3u && lo < hi) { binary_search_step_b(&lo, &hi, key, 7u); }
    if (levels >= 2u && lo < hi) { binary_search_step_b(&lo, &hi, key, 5u); }
    if (levels >= 1u && lo < hi) { binary_search_step_b(&lo, &hi, key, 4u); }
    while (lo < hi) {
        binary_search_step_b(&lo, &hi, key, 1u);
    }
    return lo;
}

fn upper_bound_b(range_begin: u32, range_end: u32, key: u32) -> u32 {
    var lo: u32 = range_begin;
    var hi: u32 = range_end;
    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        if (keysB[g_bStart + mid] <= key) { lo = mid + 1u; } else { hi = mid; }
    }
    return lo;
}

fn balanced_adjust(a_count: u32, b_count: u32, diag: u32, p_in: u32, partition_size: u32) -> vec2<u32> {
    var p = p_in;
    if (p > diag) { p = diag; }
    var a_index = p;
    var b_index = diag - p;
    var star = false;

    if (b_index < b_count) {
        let x = keysB[g_bStart + b_index];
        let levels = get_biased_search_levels(partition_size);
        let a_start = biased_lower_bound_a(a_index, x, levels);
        let b_start = biased_lower_bound_b(b_index, x, levels);
        let a_run = a_index - a_start;
        let b_run = b_index - b_start;
        let x_count = a_run + b_run;
        var b_advance = max(x_count >> 1u, x_count - a_run);
        var b_end_hint = min(b_count, b_start + b_advance + 1u);
        b_end_hint = max(b_end_hint, min(b_count, b_index + 1u));
        let b_run_end = upper_bound_b(b_index, b_end_hint, x);
        let actual_b_run = b_run_end - b_start;
        b_advance = min(b_advance, actual_b_run);
        let a_advance = x_count - b_advance;
        let round_up = (a_advance == b_advance + 1u) && (b_advance < actual_b_run);
        a_index = a_start + a_advance;
        b_index = diag - a_index;
        star = round_up;
    }

    return vec2<u32>(pack_a(a_index, star), b_index);
}

fn ballot_find_first(ballot: vec4<u32>, sg_sz: u32) -> u32 {
    if (ballot.x != 0u) { return countTrailingZeros(ballot.x); }
    if (sg_sz > 32u && ballot.y != 0u) { return 32u + countTrailingZeros(ballot.y); }
    if (sg_sz > 64u && ballot.z != 0u) { return 64u + countTrailingZeros(ballot.z); }
    if (sg_sz > 96u && ballot.w != 0u) { return 96u + countTrailingZeros(ballot.w); }
    return sg_sz;
}

// ============================================================================
// Batched compute_diagonals
// ============================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn compute_diagonals_batched(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) n_wgs: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_lane: u32,
    @builtin(subgroup_size) sg_size: u32
) {
    let local_flat = local_id.x;
    let group_flat = wg.y * n_wgs.x + wg.x;
    let totalWg = totalWg_uniform;

    let subgroup_id = local_flat / sg_size;
    let subgroups_per_wg = WORKGROUP_SIZE / sg_size;
    let targetWg = group_flat * subgroups_per_wg + subgroup_id;

    // Read wgInfo for targetWg (all lanes read the same pair info)
    // Clamp to valid range for subgroup uniformity
    let safeWg = min(targetWg, max(totalWg, 1u) - 1u);
    let infoBase = safeWg * WG_INFO_STRIDE;
    let pairAStart = wgInfo[infoBase + 0u];
    let pairALen   = wgInfo[infoBase + 1u];
    let pairBStart = wgInfo[infoBase + 2u];
    let pairBLen   = wgInfo[infoBase + 3u];
    let localWgIdx = wgInfo[infoBase + 4u];
    let pairNumWg  = wgInfo[infoBase + 5u];
    let dpiOffset  = wgInfo[infoBase + 7u];

    // Set global offsets for binary search functions
    g_aStart = pairAStart;
    g_bStart = pairBStart;

    let valid_target = targetWg < totalWg;

    // Write boundary DPI entries for this pair (once per pair, by the first workgroup's lane 0)
    if (sg_lane == 0u && valid_target && localWgIdx == 0u) {
        // Start boundary: dpi[dpiOffset] = (0, 0)
        dpi[dpiOffset] = pack_a(0u, false);
        dpi[dpiOffset + pairNumWg + 1u] = 0u;
        // End boundary: dpi[dpiOffset + pairNumWg] = (aLen, bLen)
        dpi[dpiOffset + pairNumWg] = pack_a(pairALen, false);
        dpi[dpiOffset + pairNumWg + pairNumWg + 1u] = pairBLen;
    }

    // For the diagonal computation, we use localWgIdx within the pair
    // k=0 means the start boundary (already written above), so we compute for k = 1..pairNumWg-1
    let k = localWgIdx;
    let valid_k = valid_target && (k > 0u) && (k < pairNumWg);
    let effective_k = select(1u, k, valid_k);

    let diag_u: u32 = NV * effective_k;
    let combined_index: i32 = i32(diag_u);
    let n_a: i32 = i32(pairALen);
    let n_b: i32 = i32(pairBLen);

    var xt: i32 = min(combined_index, n_a);
    var yt: i32 = max(0, combined_index - n_a);
    var xb: i32 = max(0, combined_index - n_b);
    var yb: i32 = min(combined_index, n_b);

    let half_sg = i32(sg_size >> 1u);
    let thread_offset = i32(sg_lane) - half_sg;

    var fx: u32 = 0u;
    var fy: u32 = 0u;
    var done: bool = false;

    for (var iter: u32 = 0u; iter < 32u; iter++) {
        let current_x = xt - ((xt - xb) >> 1) - thread_offset;
        let current_y = yt + ((yb - yt) >> 1) + thread_offset;

        var r: u32 = 0u;
        if (current_x > n_a || current_y < 0) {
            r = 0u;
        } else if (current_y >= n_b || current_x < 1) {
            r = 1u;
        } else {
            r = select(0u, 1u, cmp_lte(keysA[g_aStart + u32(current_x - 1)], keysB[g_bStart + u32(current_y)]));
        }

        let prev_r = subgroupShuffleUp(r, 1u);
        let is_boundary = (sg_lane > 0u) && (r != prev_r);
        let ballot = subgroupBallot(is_boundary);
        let first_bit = ballot_find_first(ballot, sg_size);

        let safe_lane = min(first_bit, sg_size - 1u);
        let found_x = subgroupShuffle(u32(current_x), safe_lane);
        let found_y = subgroupShuffle(u32(current_y), safe_lane);

        let val_last = subgroupShuffle(r, sg_size - 1u);
        let cx_mid = i32(subgroupShuffle(u32(current_x), u32(half_sg)));
        let cy_mid = i32(subgroupShuffle(u32(current_y), u32(half_sg)));

        let found = first_bit < sg_size;
        fx = select(fx, found_x, found && !done);
        fy = select(fy, found_y, found && !done);
        done = done || found;

        let go_top = !done && (val_last == 0u);
        let go_bot = !done && (val_last != 0u);
        xt = select(xt, cx_mid, go_top);
        yt = select(yt, cy_mid, go_top);
        xb = select(xb, cx_mid, go_bot);
        yb = select(yb, cy_mid, go_bot);
    }

    if (sg_lane == 0u && valid_k) {
        let out = balanced_adjust(pairALen, pairBLen, diag_u, fx, NV);
        // Write to pair-local DPI slot: dpiOffset + k for A, dpiOffset + pairNumWg + 1 + k for B
        dpi[dpiOffset + k] = out.x;
        dpi[dpiOffset + k + pairNumWg + 1u] = out.y;
    }
}
