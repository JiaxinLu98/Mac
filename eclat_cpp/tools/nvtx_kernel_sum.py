"""GPU kernel time per NVTX range from an nsys sqlite export.

A CUDA call made inside an NVTX range owns the kernels it launched (matched by
correlation id), so the kernel time of a range is the sum of the CUPTI durations
of those kernels, wherever they ran on the GPU timeline. The last `--runs` ranges
with the given name are summarized, and their per-range values are printed in a
[micro-result] JSON line.
Usage: python nvtx_kernel_sum.py <file.sqlite> <range name> [--runs 100]
                                 [--impl thrust] [--dataset 128e6]
"""
import argparse
import bisect
import json
import math
import sqlite3


def summarize(values):
    n = len(values)
    s = sorted(values)
    mean = sum(s) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in s) / (n - 1)) if n > 1 else 0.0
    median = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    p95 = s[math.ceil(0.95 * n) - 1]
    return mean, std, median, p95, s[0], s[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("range_name")
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--impl", default=None)
    parser.add_argument("--dataset", default=None)
    args = parser.parse_args()

    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))

    ranges = []
    for start, end, text, text_id in db.execute(
            "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
        name = text if text is not None else strings.get(text_id)
        if name == args.range_name:
            ranges.append((start, end))
    if not ranges:
        raise SystemExit(f"{args.sqlite}: no NVTX range named {args.range_name!r}")

    launches = sorted(db.execute("SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    launch_starts = [s for s, _ in launches]
    kernel_ns, kernel_count = {}, {}
    for corr, start, end in db.execute("SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kernel_ns[corr] = kernel_ns.get(corr, 0) + (end - start)
        kernel_count[corr] = kernel_count.get(corr, 0) + 1

    per_range, counts = [], []
    for lo, hi in ranges:
        total, n = 0, 0
        for i in range(bisect.bisect_left(launch_starts, lo), bisect.bisect_right(launch_starts, hi)):
            corr = launches[i][1]
            total += kernel_ns.get(corr, 0)
            n += kernel_count.get(corr, 0)
        per_range.append(total / 1e6)
        counts.append(n)

    runs = per_range[-args.runs:]
    mean, std, median, p95, lo, hi = summarize(runs)
    print(f"{args.sqlite}: {len(ranges)} ranges named {args.range_name!r}, "
          f"kernels per range in the last {len(runs)}: {min(counts[-len(runs):])}-{max(counts[-len(runs):])}")
    print(f"  GPU kernel: mean {mean:.4f} ms (std {std:.4f}, median {median:.4f}, p95 {p95:.4f}, "
          f"min {lo:.4f}, max {hi:.4f}, n={len(runs)})")
    record = {"impl": args.impl or args.range_name, "dataset": args.dataset, "source": "nsys",
              "range": args.range_name, "runs": len(runs), "kernel_ms": [round(x, 5) for x in runs]}
    print(f"[micro-result] {json.dumps(record)}")


if __name__ == "__main__":
    main()
