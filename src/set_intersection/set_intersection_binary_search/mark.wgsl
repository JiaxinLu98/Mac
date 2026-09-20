const THREADS_PER_DIMENSION: u32 = 16u;
const THREADS_PER_GROUP: u32 = THREADS_PER_DIMENSION * THREADS_PER_DIMENSION;

struct Params {
    lenA: u32,
    lenB: u32,
};

@group(0) @binding(0) var<storage, read> setA: array<u32>;
@group(0) @binding(1) var<storage, read> setB: array<u32>;
@group(0) @binding(2) var<storage, read_write> flags: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(THREADS_PER_DIMENSION, THREADS_PER_DIMENSION)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(num_workgroups) n_wgs: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {

    // Flatten local/global indices once to reduce ALU work inside the stride loop.
    let local_flat = local_id.y * THREADS_PER_DIMENSION + local_id.x;
    let group_flat = wg.y * n_wgs.x + wg.x;
    let start = group_flat * THREADS_PER_GROUP + local_flat;
    let total_threads = n_wgs.x * n_wgs.y * THREADS_PER_GROUP;

    let lenA = params.lenA;
    let lenB = params.lenB;

    var idx = start;

    loop {
        if (idx >= lenA) { break; }

        let value = setA[idx];

        // Lower bound of value in B.
        var lowB: u32 = 0u;
        var highB: u32 = lenB;
        while (lowB < highB) {
            let mid: u32 = (lowB + highB) / 2u;
            let midVal: u32 = setB[mid];
            if (midVal < value) {
                lowB = mid + 1u;
            } else {
                highB = mid;
            }
        }
        let firstB: u32 = lowB;

        // Value not found in B; early flag and continue.
        if (firstB == lenB || setB[firstB] != value) {
            flags[idx] = 0u;
            idx += total_threads;
            continue;
        }

        // Upper bound of value in B.
        lowB = firstB;
        highB = lenB;
        while (lowB < highB) {
            let mid: u32 = (lowB + highB) / 2u;
            let midVal: u32 = setB[mid];

            if (midVal <= value) {
                lowB = mid + 1u;
            } else {
                highB = mid;
            }
        }
        let countB: u32 = lowB - firstB;

        // Lower bound of value in A to compute local rank among duplicates.
        var lowA: u32 = 0u;
        var highA: u32 = lenA;
        while (lowA < highA) {
            let mid: u32 = (lowA + highA) / 2u;
            let midVal: u32 = setA[mid];

            if (midVal < value) {
                lowA = mid + 1u;
            } else {
                highA = mid;
            }
        }
        let local_rank: u32 = idx - lowA;
        flags[idx] = select(0u, 1u, local_rank < countB);

        idx += total_threads;
    }
}
