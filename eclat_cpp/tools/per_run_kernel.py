"""Pure GPU kernel time per run from an nsys sqlite export.

A run starts at the base-arena upload: a cudaMalloc directly followed by a synchronous
cudaMemcpy. Kernel time is the sum of CUPTI kernel durations inside each run. The last
`timed` runs are summarized in the same format as the benchmark logs.
Usage: python3 per_run_kernel.py <timed runs> <file.sqlite> [...]
"""
import math
import re
import sqlite3
import sys

timed = int(sys.argv[1])
for path in sys.argv[2:]:
    db = sqlite3.connect(path)
    # nsys records versioned API names such as cudaMalloc_v3020.
    names = {i: re.sub(r"_v\d+$", "", v) for i, v in db.execute("SELECT id, value FROM StringIds")}
    calls = [(s, names[n]) for s, n in db.execute("SELECT start, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start")]
    starts = [calls[i - 1][0] for i in range(1, len(calls)) if calls[i][1] == "cudaMemcpy" and calls[i - 1][1] == "cudaMalloc"]
    bounds = starts + [float("inf")]
    kernels = sorted(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"))

    per_run, counts, k = [], [], 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        total, n = 0, 0
        while k < len(kernels) and kernels[k][0] < hi:
            if kernels[k][0] >= lo:
                total += kernels[k][1] - kernels[k][0]
                n += 1
            k += 1
        per_run.append(total / 1e6)
        counts.append(n)

    runs = per_run[-timed:]
    mean = sum(runs) / len(runs)
    std = math.sqrt(sum((x - mean) ** 2 for x in runs) / (len(runs) - 1)) if len(runs) > 1 else 0.0
    print(f"{path}: {len(per_run)} runs found, kernels per run {counts[-1]}")
    print(f"  per-run GPU kernel ms: {[round(x, 1) for x in per_run]}")
    print(f"  GPU kernel (last {len(runs)} runs): {mean:.3f} ms (std {std:.3f}, min {min(runs):.3f}, max {max(runs):.3f})")
