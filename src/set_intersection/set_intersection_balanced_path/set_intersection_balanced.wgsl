// ============================================================================
// Set Intersection using Full Balanced Path (ModernGPU-style)
//
// Based on ModernGPU's DeviceComputeSetAvailability function.
//
// Flow:
// 1. Read partition boundaries from DPI (bp_global in ModernGPU)
// 2. Calculate A and B ranges, handle star bits
// 3. Load A and B data into shared memory (DeviceLoad2ToShared)
// 4. Each thread runs Local BalancedPath to find its starting position
// 5. Each thread runs SerialSetOp to process VT elements
// ============================================================================

// ============================================================================
// Bindings
// ============================================================================
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read> dpi: array<u32>;  // bp_global in ModernGPU
@group(0) @binding(3) var<storage, read_write> counts: array<u32>;
@group(0) @binding(4) var<uniform> a_length: u32;  // aCount
@group(0) @binding(5) var<uniform> b_length: u32;  // bCount
@group(0) @binding(6) var<uniform> num_wg_total: u32;

// ============================================================================
// Constants (matching ModernGPU terminology)
// ============================================================================
const NT: u32 = 256u;           // Threads per workgroup (WORKGROUP_SIZE)
const VT: u32 = 7u;             // Values per thread
const NV: u32 = NT * VT;        // Total elements per workgroup = 1792

const STAR_MASK: u32 = 0x80000000u;   // 0x80000000 - MSB for star flag
const INDEX_MASK: u32 = 0x7FFFFFFFu;  // 0x7fffffff - mask to extract index
const MAX_DISPATCH_X: u32 = 65535u;

// ============================================================================
// Shared Memory Layout (keys_shared in ModernGPU)
// ============================================================================
// Layout: [A elements (aCount2 + extended)] [B elements (bCount2 + extended)]
//         |<-------- bStart -------->|
//
// Total size needed: NV + VT + 1 (maximum)
// ModernGPU uses: DeviceLoad2ToShared<NT, VT, VT + 1>
// This loads (aCount2 + extended) A elements and (bCount2 + extended) B elements
// ============================================================================
var<workgroup> keys_shared: array<u32, NV + VT + 2u>;

// Workgroup-level variables
var<workgroup> wg_a0: u32;        // A partition start (global index)
var<workgroup> wg_a1: u32;        // A partition end (global index)
var<workgroup> wg_b0: u32;        // B partition start (global index)
var<workgroup> wg_b1: u32;        // B partition end (global index)
var<workgroup> wg_a_count: u32;   // aCount2 = a1 - a0
var<workgroup> wg_b_count: u32;   // bCount2 = b1 - b0
var<workgroup> wg_b_start: u32;   // bStart = aCount2 + extended (B's offset in shared memory)
var<workgroup> wg_extended: bool; // Whether we loaded extended frame
var<workgroup> wg_bit0: u32;      // Star bit from bp0 (0 or 1)

// For workgroup reduction
var<workgroup> shared_counts: array<u32, NT>;

// ============================================================================
// Helper: Convert 2D workgroup_id to 1D index
// ============================================================================
fn get_workgroup_index(wg_id: vec3<u32>) -> u32 {
    return wg_id.x + wg_id.y * MAX_DISPATCH_X;
}

// ============================================================================
// TODO: DeviceLoad2ToShared
// Cooperative loading of A and B data into shared memory
//
// ModernGPU signature:
//   DeviceLoad2ToShared<NT, VT, VT + 1>(
//       a_global + a0, aCount2 + extended,
//       b_global + b0, bCount2 + extended,
//       tid, keys_shared);
//
// This function loads:
//   - A[a0 .. a0 + aCount2 + extended) into keys_shared[0 .. aCount2 + extended)
//   - B[b0 .. b0 + bCount2 + extended) into keys_shared[bStart .. bStart + bCount2 + extended)
//
// Each thread loads multiple elements with coalesced access pattern.
// ============================================================================
fn device_load_2_to_shared(
    tid: u32,
    a_global_offset: u32,
    a_load_count: u32,
    b_global_offset: u32,
    b_load_count: u32,
    b_shared_start: u32
) {
    // Load A elements: thread i loads indices i, i+NT, i+2*NT, ...
    var i = tid;
    while (i < a_load_count) {
        keys_shared[i] = a[a_global_offset + i];
        i += NT;
    }

    // Load B elements into shared memory starting at b_shared_start
    i = tid;
    while (i < b_load_count) {
        keys_shared[b_shared_start + i] = b[b_global_offset + i];
        i += NT;
    }
}

// ============================================================================
// TODO: BalancedPath (Local version in shared memory)
// Find the intersection point of diagonal with merge path in shared memory
//
// ModernGPU signature:
//   int2 bp = BalancedPath<Duplicates, int>(
//       keys_shared, aCount2,
//       keys_shared + bStart, bCount2,
//       diag, 2, comp);
//
// Parameters:
//   - a_keys: keys_shared (A portion)
//   - a_count: aCount2
//   - b_keys: keys_shared + bStart (B portion)
//   - b_count: bCount2
//   - diag: the diagonal to search (VT * tid - bit0)
//   - levels: biased search levels (typically 2 for local search)
//
// Returns: (a_index, b_index) where a_index + b_index = diag
// ============================================================================
fn balanced_path_local(
    a_count: u32,
    b_start: u32,
    b_count: u32,
    diag: i32
) -> vec2<u32> {
    // TODO: Implement biased binary search in shared memory
    // For now, use simple merge path

    let diag_u = u32(max(0, diag));

    // Standard merge path search
    var lo: u32 = select(0u, diag_u - b_count, diag_u > b_count);
    var hi: u32 = min(diag_u, a_count);

    while (lo < hi) {
        let mid = (lo + hi) >> 1u;
        let a_key = keys_shared[mid];
        let b_idx = diag_u - 1u - mid;
        let b_key = keys_shared[b_start + b_idx];

        if (a_key <= b_key) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    let a_idx = lo;
    let b_idx = diag_u - lo;

    return vec2<u32>(a_idx, b_idx);
}

// ============================================================================
// TODO: SerialSetOp (Set Intersection)
// Each thread processes VT elements serially
//
// ModernGPU signature:
//   int commit = SerialSetOp<VT, Extended, Op>(
//       keys_shared, a0tid, aCount2,
//       bStart + b0tid, bStart + bCount2,
//       bp.y, results, indices, comp);
//
// Parameters:
//   - a0tid: thread's starting A index in shared memory
//   - aCount2: total A elements in partition
//   - b0tid: thread's starting B index in shared memory (relative to bStart)
//   - bp.y: for handling star bit adjustments
//
// For set intersection:
//   - Advances through A and B, counting matches
//   - Returns number of matched elements
// ============================================================================
fn serial_set_intersection(
    a_start_local: u32,
    a_end_local: u32,
    b_start_shared: u32,
    b_end_shared: u32
) -> u32 {
    // TODO: Implement proper serial set intersection with VT elements
    // For now, simple serial merge

    var a_idx = a_start_local;
    var b_idx = b_start_shared;
    var count: u32 = 0u;
    var processed: u32 = 0u;

    // Process up to VT elements
    while (a_idx < a_end_local && b_idx < b_end_shared && processed < VT) {
        let a_key = keys_shared[a_idx];
        let b_key = keys_shared[b_idx];

        if (a_key < b_key) {
            a_idx++;
        } else if (b_key < a_key) {
            b_idx++;
        } else {
            // Match found
            count++;
            a_idx++;
            b_idx++;
        }
        processed++;
    }

    return count;
}

// ============================================================================
// Workgroup reduction to sum all thread counts
// ============================================================================
fn workgroup_reduce_sum(tid: u32, value: u32) -> u32 {
    shared_counts[tid] = value;
    workgroupBarrier();

    // Tree reduction
    for (var stride = NT >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            shared_counts[tid] += shared_counts[tid + stride];
        }
        workgroupBarrier();
    }

    return shared_counts[0];
}

// ============================================================================
// Main Kernel: Count Intersections (DeviceComputeSetAvailability equivalent)
// ============================================================================
@compute @workgroup_size(256)
fn count_intersections_balanced(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;                    // Thread ID within workgroup
    let block = get_workgroup_index(wg_id);  // Workgroup index (block in ModernGPU)
    let num_wg = num_wg_total;

    // Skip if workgroup index is out of range
    if (block >= num_wg) {
        return;
    }

    // ========================================================================
    // Step 1: Read partition boundaries from DPI
    //
    // DPI layout (我们的实现已经存储了 B 的位置):
    //   dpi[0 .. num_wg]           : packed aIndex (MSB = star)
    //   dpi[num_wg+1 .. 2*num_wg+1]: bIndex
    //
    // 注意: ModernGPU 只存储 A 索引，B 需要从对角线计算 (b = gid - a)
    //       但我们的 balanced_path_biased.wgsl 已经存储了 B 索引
    // ========================================================================
    if (tid == 0u) {
        // Read A partition boundaries (packed with star bit)
        let bp0 = dpi[block];
        let bp1 = dpi[block + 1u];

        // ====================================================================
        // Step 2: Extract A indices (mask off star bit)
        // ====================================================================
        let a0 = bp0 & INDEX_MASK;
        let a1 = bp1 & INDEX_MASK;

        // ====================================================================
        // Step 3: Extract star bits FIRST (needed for B adjustment)
        // ModernGPU: bit0 = (0x80000000 & bp0) ? 1 : 0;
        // ====================================================================
        let bit0 = select(0u, 1u, (bp0 & STAR_MASK) != 0u);
        let bit1 = select(0u, 1u, (bp1 & STAR_MASK) != 0u);

        // ====================================================================
        // Step 4: Read B indices and apply star bit adjustment
        // IMPORTANT: DPI stores b = diag - a, but ModernGPU adds star bit:
        //   b0 = gid - a0 + bit0
        //   b1 = min(total, gid + NV) - a1 + bit1
        // We must apply the same adjustment!
        // ====================================================================
        let b0 = dpi[num_wg + 1u + block] + bit0;
        let b1 = dpi[num_wg + 1u + block + 1u] + bit1;

        // ====================================================================
        // Step 5: Calculate counts and extended frame
        // ModernGPU:
        //   int aCount2 = a1 - a0;
        //   int bCount2 = b1 - b0;
        //   extended = (a1 < aCount) && (b1 < bCount);
        //   int bStart = aCount2 + (int)extended;
        // ====================================================================
        let a_count2 = a1 - a0;
        let b_count2 = b1 - b0;
        let extended = (a1 < a_length) && (b1 < b_length);
        let b_start = a_count2 + select(0u, 1u, extended);

        // Store in workgroup shared variables
        wg_a0 = a0;
        wg_a1 = a1;
        wg_b0 = b0;
        wg_b1 = b1;
        wg_a_count = a_count2;
        wg_b_count = b_count2;
        wg_b_start = b_start;
        wg_extended = extended;
        wg_bit0 = bit0;
    }
    workgroupBarrier();

    // Read workgroup-level values
    let a0 = wg_a0;
    let a1 = wg_a1;
    let b0 = wg_b0;
    let b1 = wg_b1;
    let a_count2 = wg_a_count;
    let b_count2 = wg_b_count;
    let b_start = wg_b_start;
    let extended = wg_extended;
    let bit0 = wg_bit0;

    // ========================================================================
    // Step 6: Load data into shared memory (DeviceLoad2ToShared)
    // ModernGPU:
    //   DeviceLoad2ToShared<NT, VT, VT + 1>(
    //       a_global + a0, aCount2 + (int)extended,
    //       b_global + b0, bCount2 + (int)extended,
    //       tid, keys_shared);
    // ========================================================================
    let a_load_count = a_count2 + select(0u, 1u, extended);
    let b_load_count = b_count2 + select(0u, 1u, extended);

    device_load_2_to_shared(tid, a0, a_load_count, b0, b_load_count, b_start);
    workgroupBarrier();

    // ========================================================================
    // Step 7: Each thread finds its starting position using Local BalancedPath
    // ModernGPU:
    //   int count = aCount2 + bCount2;
    //   int diag = min(VT * tid - bit0, count);
    //   int2 bp = BalancedPath<Duplicates, int>(
    //       keys_shared, aCount2,
    //       keys_shared + bStart, bCount2,
    //       diag, 2, comp);
    //   int a0tid = bp.x;
    //   int b0tid = VT * tid + bp.y - bp.x - bit0;
    // ========================================================================
    let total_count = a_count2 + b_count2;
    let diag = min(i32(VT * tid) - i32(bit0), i32(total_count));

    // TODO: Use proper BalancedPath with biased search and duplicate handling
    let bp = balanced_path_local(a_count2, b_start, b_count2, diag);

    let a0tid = bp.x;                                    // Thread's A start in shared memory
    let b0tid = i32(VT * tid) + i32(bp.y) - i32(bp.x) - i32(bit0);  // Thread's B index adjustment

    // ========================================================================
    // Step 8: Serial set intersection for this thread
    // ModernGPU:
    //   int commit;
    //   if(extended)
    //       commit = SerialSetOp<VT, false, Op>(...);
    //   else
    //       commit = SerialSetOp<VT, true, Op>(...);
    // ========================================================================
    // Calculate this thread's range in shared memory
    let a_start_local = a0tid;
    let a_end_local = a_count2;  // Thread can access up to end of A partition
    let b_start_local = b_start + u32(max(0, b0tid));
    let b_end_local = b_start + b_count2;

    // TODO: Implement proper SerialSetOp with VT elements and extended frame handling
    let local_count = serial_set_intersection(a_start_local, a_end_local, b_start_local, b_end_local);

    // ========================================================================
    // Step 9: Workgroup reduction to sum counts
    // ========================================================================
    let total = workgroup_reduce_sum(tid, local_count);

    // Thread 0 writes final count
    if (tid == 0u) {
        counts[block] = total;
    }
}
