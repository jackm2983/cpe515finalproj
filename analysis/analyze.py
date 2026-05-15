#!/usr/bin/env python3
"""
analyze cfu fir cycle sweeps.

usage:
    python3 analyze.py log.txt [log2.txt ...]

reads sim console logs, extracts CSV rows (lines starting with "CSV,"),
parses them, and produces:
    - summary_taps.csv         cycles/output by variant x taps at N=64
    - summary_n.csv            cycles/output by variant x N at taps=4
    - plot_taps.png            line plot, per_output vs taps
    - plot_n.png               line plot, per_output vs N
    - plot_speedup.png         bar chart, speedup vs scalar by taps
    - results_table.md         markdown table for the paper

supports both old (per_output) and new (per_out_x100) CSV schemas.
"""

import csv
import io
import os
import re
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


# expected header field names (both schemas)
NEW_HEADER = ["variant", "taps", "N", "iters", "cycles", "overhead",
              "net", "per_out_x100", "checksum"]
OLD_HEADER = ["variant", "taps", "N", "iters", "cycles", "overhead",
              "net", "per_output", "checksum"]


def parse_csv_lines(paths):
    """extract all CSV, rows from one or more sim log files.
    returns a list of dicts, one per row, with per_output as float."""
    rows = []
    schema = None  # "new" or "old"
    for path in paths:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line.startswith("CSV,"):
                    continue
                body = line[len("CSV,"):]
                # detect header rows. they start with "variant,"
                if body.startswith("variant,"):
                    if "per_out_x100" in body:
                        schema = "new"
                    elif "per_output" in body:
                        schema = "old"
                    continue
                fields = list(csv.reader(io.StringIO(body)))[0]
                if len(fields) < 9:
                    continue
                try:
                    row = {
                        "variant": fields[0],
                        "taps":    int(fields[1]),
                        "N":       int(fields[2]),
                        "iters":   int(fields[3]),
                        "cycles":  int(fields[4]),
                        "overhead": int(fields[5]),
                        "net":     int(fields[6]),
                        "checksum": int(fields[8]),
                    }
                    # compute per_output as float regardless of source schema
                    raw = int(fields[7])
                    if schema == "new":
                        row["per_output"] = raw / 100.0
                    else:
                        # legacy schema: integer-floored cycles/output.
                        # recompute from net for better precision.
                        outs = (row["N"] - row["taps"] + 1) * row["iters"]
                        row["per_output"] = row["net"] / outs if outs else 0.0
                except (ValueError, IndexError):
                    continue
                rows.append(row)
    return rows, schema


def index_by(rows, *keys):
    """build a nested dict keyed by the given fields."""
    out = {}
    for r in rows:
        d = out
        for k in keys[:-1]:
            d = d.setdefault(r[k], {})
        d.setdefault(r[keys[-1]], []).append(r)
    return out


def best_row(rows):
    """pick the row with the lowest per_output (for dedup if multiple runs)."""
    return min(rows, key=lambda r: r["per_output"])


def write_taps_summary(rows, out_path, N_filter=64):
    """cycles/output by variant x taps, at fixed N."""
    by_var_taps = defaultdict(dict)
    for r in rows:
        if r["N"] != N_filter:
            continue
        by_var_taps[r["variant"]][r["taps"]] = r["per_output"]

    if not by_var_taps:
        print(f"no rows found at N={N_filter}")
        return None

    taps_sorted = sorted({t for d in by_var_taps.values() for t in d})
    variants_order = ["scalar", "mac16", "unroll2", "acc_reg", "circ",
                      "swin", "loaded"]
    variants_present = [v for v in variants_order if v in by_var_taps]

    with open(out_path, "w") as f:
        f.write("variant," + ",".join(f"taps={t}" for t in taps_sorted) + "\n")
        for v in variants_present:
            cells = []
            for t in taps_sorted:
                val = by_var_taps[v].get(t)
                cells.append(f"{val:.2f}" if val is not None else "-")
            f.write(v + "," + ",".join(cells) + "\n")
    print(f"wrote {out_path}")
    return by_var_taps, taps_sorted, variants_present


def write_n_summary(rows, out_path, taps_filter=4):
    """cycles/output by variant x N, at fixed taps."""
    by_var_N = defaultdict(dict)
    for r in rows:
        if r["taps"] != taps_filter:
            continue
        by_var_N[r["variant"]][r["N"]] = r["per_output"]

    if not by_var_N:
        print(f"no rows found at taps={taps_filter}")
        return None

    N_sorted = sorted({n for d in by_var_N.values() for n in d})
    variants_order = ["scalar", "mac16", "unroll2", "acc_reg", "circ",
                      "swin", "loaded"]
    variants_present = [v for v in variants_order if v in by_var_N]

    with open(out_path, "w") as f:
        f.write("variant," + ",".join(f"N={n}" for n in N_sorted) + "\n")
        for v in variants_present:
            cells = []
            for n in N_sorted:
                val = by_var_N[v].get(n)
                cells.append(f"{val:.2f}" if val is not None else "-")
            f.write(v + "," + ",".join(cells) + "\n")
    print(f"wrote {out_path}")
    return by_var_N, N_sorted, variants_present


def plot_taps(by_var_taps, taps_sorted, variants_present, out_path,
              N_label=64):
    if not HAVE_PLT:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"scalar": "o", "mac16": "s", "unroll2": "^",
               "acc_reg": "D", "circ": "v", "swin": "*", "loaded": "P"}
    for v in variants_present:
        xs, ys = [], []
        for t in taps_sorted:
            if t in by_var_taps[v]:
                xs.append(t)
                ys.append(by_var_taps[v][t])
        ax.plot(xs, ys, marker=markers.get(v, "."), label=v, linewidth=1.5)
    ax.set_xlabel("filter taps")
    ax.set_ylabel("cycles per output sample")
    ax.set_title(f"FIR cycles/output vs filter length (N={N_label})")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_n(by_var_N, N_sorted, variants_present, out_path, taps_label=4):
    if not HAVE_PLT:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"scalar": "o", "mac16": "s", "unroll2": "^",
               "acc_reg": "D", "circ": "v", "swin": "*", "loaded": "P"}
    for v in variants_present:
        xs, ys = [], []
        for n in N_sorted:
            if n in by_var_N[v]:
                xs.append(n)
                ys.append(by_var_N[v][n])
        ax.plot(xs, ys, marker=markers.get(v, "."), label=v, linewidth=1.5)
    ax.set_xlabel("sample buffer size N")
    ax.set_ylabel("cycles per output sample")
    ax.set_title(f"FIR cycles/output vs N (taps={taps_label})")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_speedup(by_var_taps, taps_sorted, variants_present, out_path,
                 N_label=64):
    """grouped bar chart: speedup vs scalar at each tap count."""
    if not HAVE_PLT:
        return
    if "scalar" not in by_var_taps:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    non_scalar = [v for v in variants_present if v != "scalar"]
    n_var = len(non_scalar)
    n_taps = len(taps_sorted)
    bar_w = 0.8 / n_var

    for i, v in enumerate(non_scalar):
        xs, ys = [], []
        for j, t in enumerate(taps_sorted):
            scalar_v = by_var_taps["scalar"].get(t)
            var_v = by_var_taps[v].get(t)
            if scalar_v is None or var_v is None or var_v == 0:
                continue
            xs.append(j + i * bar_w - 0.4 + bar_w / 2)
            ys.append(scalar_v / var_v)
        if xs:
            ax.bar(xs, ys, width=bar_w, label=v)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(range(n_taps))
    ax.set_xticklabels([f"taps={t}" for t in taps_sorted])
    ax.set_ylabel("speedup vs scalar")
    ax.set_title(f"Speedup over scalar baseline (N={N_label})")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def write_markdown_table(by_var_taps, taps_sorted, variants_present,
                         out_path, N_label=64):
    """produce a markdown table for the paper."""
    lines = []
    lines.append(f"## FIR cycles/output by variant and tap count (N={N_label})")
    lines.append("")
    header = "| variant | " + " | ".join(f"taps={t}" for t in taps_sorted) + " | trend |"
    sep    = "| --- | " + " | ".join("---" for _ in taps_sorted) + " | --- |"
    lines.append(header)
    lines.append(sep)
    for v in variants_present:
        cells = []
        for t in taps_sorted:
            val = by_var_taps[v].get(t)
            cells.append(f"{val:.2f}" if val is not None else "-")
        # crude trend: ratio of largest/smallest taps
        vals = [by_var_taps[v].get(t) for t in taps_sorted
                if by_var_taps[v].get(t) is not None]
        trend = ""
        if len(vals) >= 2 and vals[0] > 0:
            ratio = vals[-1] / vals[0]
            trend = f"{ratio:.1f}x growth"
        lines.append(f"| {v} | " + " | ".join(cells) + f" | {trend} |")
    lines.append("")

    if "scalar" in by_var_taps:
        lines.append(f"## Speedup vs scalar (N={N_label})")
        lines.append("")
        header = "| variant | " + " | ".join(f"taps={t}" for t in taps_sorted) + " |"
        sep    = "| --- | " + " | ".join("---" for _ in taps_sorted) + " |"
        lines.append(header)
        lines.append(sep)
        for v in variants_present:
            if v == "scalar":
                continue
            cells = []
            for t in taps_sorted:
                sv = by_var_taps["scalar"].get(t)
                vv = by_var_taps[v].get(t)
                if sv is None or vv is None or vv == 0:
                    cells.append("-")
                else:
                    cells.append(f"{sv/vv:.2f}x")
            lines.append(f"| {v} | " + " | ".join(cells) + " |")
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def print_console_summary(rows):
    if not rows:
        print("no rows parsed")
        return
    print(f"\nparsed {len(rows)} rows")
    print("variants seen:", sorted({r['variant'] for r in rows}))
    print("taps seen:    ", sorted({r['taps'] for r in rows}))
    print("N seen:       ", sorted({r['N'] for r in rows}))

    # check checksum consistency per (taps, N): all variants should agree
    by_tn = defaultdict(dict)
    for r in rows:
        by_tn[(r["taps"], r["N"])][r["variant"]] = r["checksum"]
    bad = []
    for (t, n), vs in by_tn.items():
        cks = set(vs.values())
        # variants returning 0 (swin/loaded skipped) shouldn't count
        cks.discard(0)
        if len(cks) > 1:
            bad.append((t, n, vs))
    if bad:
        print("\nWARNING: checksum mismatches detected:")
        for t, n, vs in bad:
            print(f"  taps={t} N={n}: {vs}")
    else:
        print("\nall checksums agree per (taps, N) where applicable")


def main():
    if len(sys.argv) < 2:
        print("usage: analyze.py <sim_log.txt> [more_logs...]")
        sys.exit(1)

    paths = sys.argv[1:]
    for p in paths:
        if not os.path.exists(p):
            print(f"missing: {p}")
            sys.exit(1)

    rows, schema = parse_csv_lines(paths)
    print(f"detected schema: {schema}")
    print_console_summary(rows)

    if not rows:
        sys.exit(1)

    # dedup: if same (variant, taps, N) appears multiple times across logs,
    # keep the row with the lowest per_output (best measurement)
    dedup = {}
    for r in rows:
        key = (r["variant"], r["taps"], r["N"])
        if key not in dedup or r["per_output"] < dedup[key]["per_output"]:
            dedup[key] = r
    rows = list(dedup.values())

    # taps sweep: pick the largest N that has the most variants populated
    N_counts = defaultdict(int)
    for r in rows:
        N_counts[r["N"]] += 1
    N_for_taps_sweep = max(N_counts, key=lambda n: N_counts[n])

    res_t = write_taps_summary(rows, "summary_taps.csv", N_filter=N_for_taps_sweep)
    if res_t:
        by_var_taps, taps_sorted, variants_present = res_t
        plot_taps(by_var_taps, taps_sorted, variants_present,
                  "plot_taps.png", N_label=N_for_taps_sweep)
        plot_speedup(by_var_taps, taps_sorted, variants_present,
                     "plot_speedup.png", N_label=N_for_taps_sweep)
        write_markdown_table(by_var_taps, taps_sorted, variants_present,
                             "results_table.md", N_label=N_for_taps_sweep)

    # N sweep at taps=4 (most variants compatible there)
    res_n = write_n_summary(rows, "summary_n.csv", taps_filter=4)
    if res_n:
        by_var_N, N_sorted, variants_present_n = res_n
        plot_n(by_var_N, N_sorted, variants_present_n,
               "plot_n.png", taps_label=4)

    print("\ndone.")


if __name__ == "__main__":
    main()