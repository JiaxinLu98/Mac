const THREADS_PER_DIMENSION: u32 = 16u;

struct Params {
    lenA: u32,
    lenB: u32, 
};

@group(0) @binding(0) var<storage, read> setA: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> setB: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(THREADS_PER_DIMENSION, THREADS_PER_DIMENSION)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups)       n_wgs:     vec3<u32>,
        @builtin(local_invocation_id)  local_id: vec3<u32>,
        @builtin(workgroup_id)         wg:       vec3<u32>) {

    // per-workgroup layout
    let wg_size_x: u32 = 16u;
    let wg_size_y: u32 = 16u;
    let local_flat: u32 = local_id.y * wg_size_x + local_id.x;

    // flatten (wg.x, wg.y) -> group_flat
    let group_flat: u32 = wg.y * n_wgs.x + wg.x;
    let threads_per_group: u32 = wg_size_x * wg_size_y;

    // global starting index for this thread
    let start: u32 = group_flat * threads_per_group + local_flat;

    // total #threads in the grid (for grid-stride loop)
    let total_threads: u32 = n_wgs.x * n_wgs.y * threads_per_group;

    var idx: u32 = start;

    while (idx < params.lenA) {
        // key from A (ignore value here)
        let keyA: u32 = setA[idx].x;

        // 1) lower_bound(keyA) in B: find first index in B where key >= keyA
        var lowB: u32 = 0u;
        var highB: u32 = params.lenB;
        while (lowB < highB) {
            let mid: u32 = (lowB + highB) / 2u;
            let midVal: u32 = setB[mid];

            if (midVal < keyA) {
                lowB = mid + 1u;
            } else {
                highB = mid;
            }
        }
        let firstB: u32 = lowB;

        // keyA does not exist in B -> not in intersection
        if (firstB == params.lenB || setB[firstB] != keyA) {
            flags[idx] = 0u;
        } else {
            // 2) upper_bound(keyA) in B: first index where key > keyA
            lowB = firstB;
            highB = params.lenB;
            while (lowB < highB) {
                let mid: u32 = (lowB + highB) / 2u;
                let midVal: u32 = setB[mid];

                if (midVal <= keyA) {
                    lowB = mid + 1u;
                } else {
                    highB = mid;
                }
            }
            let lastBPlus1: u32 = lowB;
            let countB: u32 = lastBPlus1 - firstB;

            // 3) lower_bound(keyA) in A (by key)
            var lowA: u32 = 0u;
            var highA: u32 = params.lenA;
            while (lowA < highA) {
                let mid: u32 = (lowA + highA) / 2u;
                let midVal: u32 = setA[mid].x;

                if (midVal < keyA) {
                    lowA = mid + 1u;
                } else {
                    highA = mid;
                }
            }
            let firstA: u32 = lowA;           // first index of this key in A
            let local_rank: u32 = idx - firstA;  // 0..(m-1) among equal keys

            // select(a, b, cond): cond==true -> b, else -> a
            flags[idx] = select(0u, 1u, local_rank < countB);
        }

        idx = idx + total_threads;
    }
}
