// ============================================================================
// TPC-H Q12 Group-By Aggregation Kernel
//
// Input: sorted (shipmode, is_high) pairs from Q12 join result.
//   shipmode: 0 = MAIL, 1 = SHIP (sorted, all MAILs first)
//   is_high: 1 = high priority (1-URGENT or 2-HIGH), 0 = low priority
//
// Output: 4 atomic counters:
//   result[0] = MAIL high count
//   result[1] = MAIL low count
//   result[2] = SHIP high count
//   result[3] = SHIP low count
//
// Algorithm: Workgroup-level tree reduction → atomicAdd to global counters.
// ============================================================================

@group(0) @binding(0) var<storage, read> shipmode: array<u32>;
@group(0) @binding(1) var<storage, read> is_high: array<u32>;

struct Params {
    total_count: u32,
}
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> result: array<atomic<u32>, 4>;

const WG_SIZE: u32 = 256u;

var<workgroup> sh_mail_high: array<u32, WG_SIZE>;
var<workgroup> sh_mail_low: array<u32, WG_SIZE>;
var<workgroup> sh_ship_high: array<u32, WG_SIZE>;
var<workgroup> sh_ship_low: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>
) {
    let tid = local_id.x;
    let total_threads = num_wgs.x * num_wgs.y * WG_SIZE;
    let global_idx = global_id.x + global_id.y * num_wgs.x * WG_SIZE;
    let total = params.total_count;

    // Each thread accumulates across its stride
    var mh: u32 = 0u;
    var ml: u32 = 0u;
    var sh: u32 = 0u;
    var sl: u32 = 0u;

    var idx = global_idx;
    while (idx < total) {
        let sm = shipmode[idx];
        let ih = is_high[idx];
        if (sm == 0u) {
            mh += ih;
            ml += (1u - ih);
        } else {
            sh += ih;
            sl += (1u - ih);
        }
        idx += total_threads;
    }

    // Store to shared memory
    sh_mail_high[tid] = mh;
    sh_mail_low[tid] = ml;
    sh_ship_high[tid] = sh;
    sh_ship_low[tid] = sl;
    workgroupBarrier();

    // Tree reduction
    for (var s: u32 = WG_SIZE / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            sh_mail_high[tid] += sh_mail_high[tid + s];
            sh_mail_low[tid] += sh_mail_low[tid + s];
            sh_ship_high[tid] += sh_ship_high[tid + s];
            sh_ship_low[tid] += sh_ship_low[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0: atomicAdd to global result
    if (tid == 0u) {
        atomicAdd(&result[0], sh_mail_high[0]);
        atomicAdd(&result[1], sh_mail_low[0]);
        atomicAdd(&result[2], sh_ship_high[0]);
        atomicAdd(&result[3], sh_ship_low[0]);
    }
}
