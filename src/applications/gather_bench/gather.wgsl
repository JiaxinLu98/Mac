// Gather/pack shader (rebuttal experiment).
// One workgroup per pair; the 256 threads stride-copy that pair's run
// [srcOff, srcOff+len) from a single consolidated source buffer into dst[dstOff].
// This is the WebGPU analogue of the CUDA pack_pairs_kernel — it replaces a
// per-pair copyBufferToBuffer loop with a single compute dispatch.

struct Meta { srcOff: u32, dstOff: u32, len: u32, pad: u32, };

@group(0) @binding(0) var<storage, read>       src    : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst    : array<u32>;
@group(0) @binding(2) var<storage, read>       infos   : array<Meta>;
@group(0) @binding(3) var<uniform>             params : vec4<u32>; // x = numPairs

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(num_workgroups) nwg : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
    let pair = wid.x + wid.y * nwg.x;
    if (pair >= params.x) { return; }
    let m = infos[pair];
    for (var i = lid.x; i < m.len; i = i + 256u) {
        dst[m.dstOff + i] = src[m.srcOff + i];
    }
}
