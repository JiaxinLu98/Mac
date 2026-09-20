// GPU preheat workload: unrelated integer work that keeps the GPU busy so it reaches
// its steady performance state before a benchmark's warmup runs. Each element goes
// through 16 steps of a linear congruential generator.
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&data)) {
        return;
    }
    var x = data[i];
    for (var k = 0u; k < 16u; k++) {
        x = x * 1664525u + 1013904223u;
    }
    data[i] = x;
}
