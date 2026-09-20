const THREADS_PER_DIMENSION: u32 = 16u;
const EMPTY_KEY: u32 = 0xffffffffu;

@group(0) @binding(0) var<storage, read_write> a_in_hash_table: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> a_in_hash_table_size: u32;
@group(0) @binding(2) var<storage, read_write> b_in: array<u32>;
@group(0) @binding(3) var<uniform> b_in_size: u32;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
@group(0) @binding(5) var<storage, read_write> result_size: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> a_in_used: array<atomic<u32>>;

fn get_position(key: u32, a_in_hash_table_size: u32) -> u32 {
    var mixed: u32 = key;
    mixed = mixed ^ (mixed >> 16);
    mixed = mixed * 0x85ebca6b;
    mixed = mixed ^ (mixed >> 13);
    mixed = mixed * 0xc2b2ae35;
    mixed = mixed ^ (mixed >> 16);
    return mixed & (a_in_hash_table_size - 1u);
}

@compute @workgroup_size(THREADS_PER_DIMENSION, THREADS_PER_DIMENSION)
fn get_join_result(@builtin(global_invocation_id) global_id: vec3<u32>, 
                   @builtin(num_workgroups) n_wgs: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) wg: vec3<u32>)  {

    // The local displacement of each thread in a workgroup
    let wg_size_x = 16u;
    let wg_size_y = 16u;
    let local_flat = local_id.y * wg_size_x + local_id.x;

    // Flatten the two-dimensional (group_id.x, group_id.y) into a flat group index
    let group_flat = wg.y * n_wgs.x + wg.x;
    let threads_per_group = wg_size_x * wg_size_y;

    // Calculate the starting index of the current thread globally
    let start = group_flat * threads_per_group + local_flat;

    // Calculate the total stride of the entire grid
    let total_threads = n_wgs.x * n_wgs.y * threads_per_group;

    let data_size = b_in_size;
    let stride = (data_size + total_threads - 1u) / total_threads;
    let begin = start * stride;
    if(begin >= data_size) { return; }
    let end = min(begin + stride, data_size);

    for(var i = begin; i < end; i = i + 1u) {
        let key = b_in[i];

        if(key == EMPTY_KEY) {
            continue;
        }

        var position = get_position(key, a_in_hash_table_size);
        var probe_count = 0u;

        loop {
            let current_key = atomicLoad(&a_in_hash_table[position]);
            
            if(key == current_key) {
                let claim = atomicCompareExchangeWeak(&a_in_used[position], 0u, 1u);

                if(claim.exchanged && claim.old_value == 0u) {
                    let result_index = atomicAdd(&result_size, 1u);
                    result[result_index] = key;
                    break;
                }
            }
            else if(current_key == EMPTY_KEY) {
                break;
            }
            position = (position + 1) & (a_in_hash_table_size - 1);
            probe_count = probe_count + 1u;

            if(probe_count >= a_in_hash_table_size) {
                break;
            }
        }
    }
}