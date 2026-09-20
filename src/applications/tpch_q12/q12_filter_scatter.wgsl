// ============================================================================
// TPC-H Q12 Filter Scatter Kernel
//
// Compacts passing rows using exclusive prefix sum offsets.
// Reads original flags (is_valid) and prefix sum result (scan_result).
// Outputs two arrays: compacted orderkeys and shipmodes.
// ============================================================================

@group(0) @binding(0) var<storage, read> in_orderkey: array<u32>;
@group(0) @binding(1) var<storage, read> in_shipmode: array<u32>;
@group(0) @binding(2) var<storage, read> scan_result: array<u32>;
@group(0) @binding(3) var<storage, read> is_valid: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_orderkey: array<u32>;
@group(0) @binding(5) var<storage, read_write> out_shipmode: array<u32>;

struct Params {
    len: u32,
}
@group(0) @binding(6) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * WORKGROUP_SIZE;
    if (idx >= params.len) { return; }

    if (is_valid[idx] == 1u) {
        let write_idx = scan_result[idx];
        out_orderkey[write_idx] = in_orderkey[idx];
        out_shipmode[write_idx] = in_shipmode[idx];
    }
}
