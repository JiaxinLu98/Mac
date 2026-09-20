struct Params {
  count: u32,
  keyIndex: u32,
};

@group(0) @binding(0) var<storage, read>  pairs : array<u32>;
@group(0) @binding(1) var<storage, read_write> keys  : array<u32>;
@group(0) @binding(2) var<storage, read_write> vals  : array<u32>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(16, 16)
fn split_pairs(@builtin(global_invocation_id) global_id: vec3<u32>, 
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

    let data_size = params.count;
    let stride = (data_size + total_threads - 1u) / total_threads;
    let begin = start * stride;
    if(begin >= data_size) { return; }
    let end = min(begin + stride, data_size);

    for(var i = begin; i < end; i = i + 1u) {
      let x = pairs[2u * i];
      let y = pairs[2u * i + 1u];

      if (params.keyIndex == 0u) {
            keys[i] = x;
            vals[i] = y;
        } else {
            keys[i] = y;
            vals[i] = x;
        }
    }
}