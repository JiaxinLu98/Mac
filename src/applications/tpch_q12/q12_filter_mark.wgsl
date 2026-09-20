// ============================================================================
// TPC-H Q12 Filter Mark Kernel
//
// Evaluates Q12 predicates on each lineitem row:
//   shipmode IN ('MAIL','SHIP')           → shipmode <= 1
//   commitdate < receiptdate
//   shipdate < commitdate
//   receiptdate >= '1994-01-01'           → receiptdate >= date_low
//   receiptdate < '1995-01-01'            → receiptdate < date_high
//
// Writes 1 (pass) or 0 (fail) to flags array.
// ============================================================================

@group(0) @binding(0) var<storage, read> shipdate: array<u32>;
@group(0) @binding(1) var<storage, read> commitdate: array<u32>;
@group(0) @binding(2) var<storage, read> receiptdate: array<u32>;
@group(0) @binding(3) var<storage, read> shipmode: array<u32>;
@group(0) @binding(4) var<storage, read_write> flags: array<u32>;
@group(0) @binding(5) var<storage, read_write> filter_count: array<atomic<u32>, 1>;

struct Params {
    len: u32,
    date_low: u32,
    date_high: u32,
}
@group(0) @binding(6) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * WORKGROUP_SIZE;
    if (idx >= params.len) { return; }

    let sm = shipmode[idx];
    let sd = shipdate[idx];
    let cd = commitdate[idx];
    let rd = receiptdate[idx];

    let keep = (sm <= 1u)
             & (cd < rd)
             & (sd < cd)
             & (rd >= params.date_low)
             & (rd < params.date_high);

    let val = select(0u, 1u, keep);
    flags[idx] = val;
    if (val == 1u) { atomicAdd(&filter_count[0], 1u); }
}
