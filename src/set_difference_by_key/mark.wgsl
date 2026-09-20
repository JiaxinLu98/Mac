const THREADS_PER_DIMENSION: u32 = 16u;

struct Params {
    lenA: u32,
    lenB: u32,
};

@group(0) @binding(0) var<storage, read> setA: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> setB: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(THREADS_PER_DIMENSION, THREADS_PER_DIMENSION)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(num_workgroups) n_wgs: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {

    let wg_size_x = 16u;
    let wg_size_y = 16u;
    let local_flat = local_id.y * wg_size_x + local_id.x;

    let group_flat = wg.y * n_wgs.x + wg.x;
    let threads_per_group = wg_size_x * wg_size_y;

    let start = group_flat * threads_per_group + local_flat;

    let total_threads = n_wgs.x * n_wgs.y * threads_per_group;

    var idx = start;

    while (idx < params.lenA) {
        let keyA: u32 = setA[idx].x;

        var lowB: u32 = 0u;
        var highB: u32 = params.lenB;
        while (lowB < highB) {
            let mid: u32 = (lowB + highB) / 2u;
            let midKey: u32 = setB[mid].x;
            if (midKey < keyA) {
                lowB = mid + 1u;
            } else {
                highB = mid;
            }
        }
        let firstB: u32 = lowB;

        if (firstB == params.lenB || setB[firstB].x != keyA) {
            flags[idx] = 1u;
        } else {
            lowB  = firstB;
            highB = params.lenB;
            while (lowB < highB) {
                let mid: u32 = (lowB + highB) / 2u;
                let midKey: u32 = setB[mid].x;

                if (midKey <= keyA) {
                    lowB = mid + 1u;
                } else {
                    highB = mid;
                }
            }
            let lastBPlus1: u32 = lowB;
            let countB: u32 = lastBPlus1 - firstB;  // n

            var lowA: u32 = 0u;
            var highA: u32 = params.lenA;
            while (lowA < highA) {
                let mid: u32 = (lowA + highA) / 2u;
                let midKeyA: u32 = setA[mid].x;

                if (midKeyA < keyA) {
                    lowA = mid + 1u;
                } else {
                    highA = mid;
                }
            }
            let firstA: u32 = lowA;

            let local_rank: u32 = idx - firstA;

            flags[idx] = select(1u, 0u, local_rank < countB);
        }

        idx = idx + total_threads;
    }
}
