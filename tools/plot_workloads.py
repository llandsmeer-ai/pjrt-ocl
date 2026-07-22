#!/usr/bin/env python3
"""Plot the application-workload suite scoreboard as a gap-per-workload bar chart.

Reads the markdown table written by ``tools/bench_suite/run_suite.py``
(``docs/workload-coverage.md`` by default) and renders a horizontal bar chart of
``gap = ours_ms / cuda_ms`` per workload — log x-axis, a dashed line at 1x (CUDA
parity), bars coloured by category (AI / SCI / PHYS). No benchmarking here; this
is a pure re-render of the suite's results so the figure regenerates cheaply.

    . ./env.sh && python tools/plot_workloads.py \
        --md docs/workload-coverage.md --out docs/bench_workloads.png
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_table(md_path):
    rows = []
    with open(md_path) as f:
        for line in f:
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 6 or cells[0] in ("workload", "") or set(cells[0]) <= {"-", ":"}:
                continue
            name, cat, status, ours, cuda, gap = cells[:6]
            m = re.match(r"([0-9.]+)x", gap)
            if status != "PASS" or not m:
                continue
            rows.append((name, cat, float(ours), float(cuda), float(m.group(1))))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="docs/workload-coverage.md")
    ap.add_argument("--out", default="docs/bench_workloads.png")
    args = ap.parse_args()

    rows = parse_table(args.md)
    rows.sort(key=lambda r: r[4])  # by gap, fastest-relative first

    colors = {"AI": "#d1495b", "SCI": "#2a9d8f", "PHYS": "#e9c46a"}
    names = [r[0] for r in rows]
    gaps = [r[4] for r in rows]
    bar_colors = [colors.get(r[1], "#888") for r in rows]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.2))
    y = range(len(rows))
    ax.barh(list(y), gaps, color=bar_colors, edgecolor="none")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("gap = ours / native CUDA   (log; lower is better, dashed = CUDA parity)")
    ax.set_title("Application-workload suite: pjrt-ocl (OpenCL) vs native CUDA, RTX PRO 6000")

    for yi, (name, cat, ours, cuda, gap) in zip(y, rows):
        ax.text(gap * 1.05, yi, f"{gap:.2f}x  ({ours:.2f} ms)", va="center", fontsize=7.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, colors.keys(), title="category", loc="upper right", fontsize=8)
    ax.set_xlim(right=max(gaps) * 2.2)
    ax.grid(axis="x", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}  ({len(rows)} workloads)")


if __name__ == "__main__":
    main()
