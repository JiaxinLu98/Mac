const THREADS_PER_DIMENSION: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> data: array<vec2<u32>>;
@group(0) @binding(1) var<uniform> data_size: u32;
@group(0) @binding(2) var<storage, read_write> scan_result: array<u32>;
@group(0) @binding(3) var<storage, read_write> new_data: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> new_data_size: u32;
@group(0) @binding(5) var<storage, read> is_valid: array<u32>;

@compute @workgroup_size(THREADS_PER_DIMENSION, THREADS_PER_DIMENSION)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(num_workgroups) n_wgs: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) wg: vec3<u32>) {
    
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

    let stride = (data_size + total_threads - 1u) / total_threads;
    let begin = start * stride;
    if(begin >= data_size) { return; }
    let end = min(begin + stride, data_size);
    
    for (var i = begin; i < end; i = i + 1u) {
      if(is_valid[i] == 1u) {
          let write_index = scan_result[i];
          new_data[write_index] = data[i];
      }

      if (i == data_size - 1u) {
          new_data_size = scan_result[i] + is_valid[i];
      }
    }
}