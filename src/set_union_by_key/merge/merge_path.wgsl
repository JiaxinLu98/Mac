// compute diagonals

@group(0) @binding(0) var<storage, read> a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> c: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> dpi: array<u32>; // diagonal_path_intersections
@group(0) @binding(4) var<uniform> a_length: u32;
@group(0) @binding(5) var<uniform> b_length: u32;
@group(0) @binding(6) var<uniform> c_length: u32;

var<workgroup> x_top: i32;
var<workgroup> y_top: i32;
var<workgroup> x_bottom: i32;
var<workgroup> y_bottom: i32;
var<workgroup> oneorzero: array<u32, 32>;
var<workgroup> found: bool;
var<workgroup> local_count: u32;

fn cmp_gt(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x > b.x;
}

fn cmp_lte(a: vec2<u32>, b: vec2<u32>) -> bool {
    return a.x <= b.x;
}

@compute @workgroup_size(32)
fn compute_diagonals(@builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>,
                     @builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(num_workgroups) num_workgroups: vec3<u32>) {

    let n_a = i32(a_length);
    let n_b = i32(b_length);
    let total_length = n_a + n_b;

    // Initialize the search window on the diagonal from a single thread
    let combined_index = i32(wg_id.x) * (total_length / i32(num_workgroups.x));
    x_top = min(combined_index, n_a);
    y_top = max(0, combined_index - n_a);
    x_bottom = max(0, combined_index - n_b);
    y_bottom = min(combined_index, n_b);
    found = false;

    let thread_offset = i32(local_id.x) - 16;

    // Search the diagonal
    while !workgroupUniformLoad(&found) {

        // Update our coordinates within the 32-wide section of the diagonal
        let current_x = x_top - ((x_top - x_bottom) >> 1) - thread_offset;
        let current_y = y_top + ((y_bottom - y_top) >> 1) + thread_offset;

        // Are we a '1' or a '0' with respect to A[x] <= B[x]
        var r: u32 = 0u;
        if (current_x > n_a || current_y < 0) {
            r = 0u;
        } else if (current_y >= n_b || current_x < 1) {
            r = 1u;
        } else {
            r = select(0u, 1u, cmp_lte(a[u32(current_x - 1)], b[u32(current_y)]));
        }
        oneorzero[local_id.x] = r;
        workgroupBarrier();

        // If we find the meeting of the '1's and '0's, we found the
        // intersection of the path and diagonal
        if (local_id.x > 0u && oneorzero[local_id.x] != oneorzero[local_id.x - 1u]) {
            found = true; // Race condition, but matches CUDA's behavior
            dpi[wg_id.x] = u32(current_x);
            dpi[wg_id.x + num_workgroups.x + 1u] = u32(current_y);
        }
        workgroupBarrier();

        // Adjust the search window on the diagonal
        if (local_id.x == 16u) {
            if (oneorzero[31] != 0u) {
                x_bottom = current_x;
                y_bottom = current_y;
            } else {
                x_top = current_x;
                y_top = current_y;
            }
        }
        workgroupBarrier();
    }

    // Set the boundary diagonals (through 0,0 and n_a,n_b)
    if (local_id.x == 0u && wg_id.x == 0u) {
        dpi[0] = 0u;
        dpi[num_workgroups.x + 1u] = 0u;
        dpi[num_workgroups.x] = a_length;
        dpi[num_workgroups.x + num_workgroups.x + 1u] = b_length;
    }
}

// merge single path

// Define constants used in the kernel
const K = 1024u; // Merge window size. Must be 4 * workgroup_size_x
const POSITIVE_INFINITY = vec2<u32>(0xFFFFFFFFu, 0xFFFFFFFFu);
const NEGATIVE_INFINITY = vec2<u32>(0u, 0u);
const block_dim_x = 256u; // workgroup_size_x

// Shared (workgroup) memory for the local merge window
var<workgroup> elements: array<vec2<u32>, (K+2u) << 1 >; // K+2 elements for sentinels
const A_shared = 0;
const B_shared = K + 2u;


// Workgroup-shared variables for the bounds of this block's merge task
var<workgroup> x_block_top: u32;
var<workgroup> y_block_top: u32;
var<workgroup> x_block_stop: u32;
var<workgroup> y_block_stop: u32;

@compute @workgroup_size(256)
fn merge_single_path(@builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>,
                     @builtin(num_workgroups) num_workgroups: vec3<u32>) {

    let tid = local_id.x;
    
    // Step 1: Initialize this workgroup's merge boundaries from the diagonal intersections array.
    switch (tid) {
        case 0u: {
            x_block_top = dpi[wg_id.x];
            elements[A_shared + 0] = NEGATIVE_INFINITY;
            break;
        }
        case 32u: {
            x_block_stop = dpi[wg_id.x + 1u];
            elements[B_shared + 0] = NEGATIVE_INFINITY;
            break;
        }
        case 64u: {
            y_block_top = dpi[wg_id.x + num_workgroups.x + 1u];
            elements[A_shared + K + 1u] = POSITIVE_INFINITY;
            break;
        }
        case 96u: {
            y_block_stop = dpi[wg_id.x + num_workgroups.x + 2u];
            elements[B_shared + K + 1u] = POSITIVE_INFINITY;
            break;
        }
        default: {
            break;
        }
    }

    workgroupBarrier();

    // Step 2: Main loop. Continue merging K-element windows until this workgroup's section is complete.
    while (workgroupUniformLoad(&x_block_top) < workgroupUniformLoad(&x_block_stop) || workgroupUniformLoad(&y_block_top) < workgroupUniformLoad(&y_block_stop)) {

        // Step 2a: Load a window of data from global memory into shared memory.
        var sharedX = tid + 1u;  // +1: sentinel at index 0
        var tmp_idx_a = x_block_top + tid;
        var tmp_idx_b = y_block_top + tid;

        elements[A_shared + sharedX] = a[tmp_idx_a];
        elements[B_shared + sharedX] = b[tmp_idx_b];
        
        sharedX += block_dim_x; tmp_idx_a += block_dim_x; tmp_idx_b += block_dim_x;
        elements[A_shared + sharedX] = a[tmp_idx_a];
        elements[B_shared + sharedX] = b[tmp_idx_b];

        sharedX += block_dim_x; tmp_idx_a += block_dim_x; tmp_idx_b += block_dim_x;
        elements[A_shared + sharedX] = a[tmp_idx_a];
        elements[B_shared + sharedX] = b[tmp_idx_b];

        sharedX += block_dim_x; tmp_idx_a += block_dim_x; tmp_idx_b += block_dim_x;
        elements[A_shared + sharedX] = a[tmp_idx_a];
        elements[B_shared + sharedX] = b[tmp_idx_b];  
        
        let c_offset = x_block_top + y_block_top;
        workgroupBarrier();

        // Step 2b: Binary search for the merge path intersection on a local diagonal.
        var A_Left = 1u;
        var A_Right = tid * 4u + 1u;
        var Ax: u32 = 0u;
        var Bx: u32 = 0u;
        var found_path = false;

        while (A_Left <= A_Right && !found_path) {

            Ax = A_Left + ((A_Right - A_Left) >> 1u);
            Bx = tid * 4u + 2u - Ax;

            let val_A = elements[A_shared + Ax];
            let val_B_m1 = elements[B_shared + Bx - 1u];

            // if (A[Ax] > B[B-1])
            if (cmp_gt(val_A, val_B_m1)) {
                let val_A_m1 = elements[A_shared + Ax - 1u];
                let val_B = elements[B_shared + Bx];
                // if (A[Ax-1] <= B[B])
                if (cmp_lte(val_A_m1, val_B)) {
                    // Found it
                    found_path = true;
                } else { // partition is to the left
                    A_Right = Ax - 1u;
                }
            } else { // partition is to the right
                A_Left = Ax + 1u;
            }
        }

        // Step 2c: Merge 4 elements from the calculated path intersection.
        let thread_id_x4 = tid * 4u;
        let x_y_combined = c_offset + thread_id_x4;

        // Pre-calculate how many elements this thread should process
        let elements_to_process = min(4u, max(0u, c_length - x_y_combined));

        var Ai = elements[A_shared + Ax];
        var Bi = elements[B_shared + Bx];
        var Ci: vec2<u32>;
        var local_Ax = Ax;
        var local_Bx = Bx;

        // Process all 4 elements in a loop
        for(var elem_idx = 0u; elem_idx < 4u; elem_idx = elem_idx + 1u) {
            let output_idx = x_y_combined + elem_idx;

            // Branchless selection using select()
            let take_from_A = cmp_lte(Ai, Bi);
            let selected_value = select(Bi, Ai, take_from_A);

            // Branchless index increment
            local_Ax += select(0u, 1u, take_from_A);
            local_Bx += select(1u, 0u, take_from_A);

            // Update values for next iteration
            Ai = select(Ai, elements[A_shared + local_Ax], take_from_A);
            Bi = select(elements[B_shared + local_Bx], Bi, take_from_A);

            // Conditional write - only write if within bounds
            if(elem_idx < elements_to_process) {
                c[output_idx] = selected_value;
            }
        }

        // Step 2d: The last thread updates the block's progress for the next window.
        if (tid == block_dim_x - 1u) {
            x_block_top += (Ax - 1u);
            y_block_top += (Bx - 1u);
        }

        workgroupBarrier();
    }
}