#!/usr/bin/env python3
"""Turn results/eval.csv into the Markdown comparison table used in the README.

For each label (base_zero_shot / finetuned / scratch ...) and each terrain level it reports
mean ± std over seeds of survival, linear velocity error and distance, plus the hardest level
with survival >= ``--threshold`` (default 0.90) on every seed.

Usage: python scripts/make_table.py results/eval.csv [--threshold 0.9] > results/table.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def load(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("seed", "level", "num_envs"):
            r[k] = int(r[k])
        for k in ("survival", "vel_err_lin", "vel_err_yaw", "distance"):
            r[k] = float(r[k])
    return rows


def fmt(xs: list[float], digits: int = 2) -> str:
    return f"{mean(xs):.{digits}f} ± {pstdev(xs):.{digits}f}"


def max_level(per_level: dict[int, list[float]], threshold: float) -> int | None:
    """Hardest level such that every level up to it has survival >= threshold on all seeds."""
    best = None
    for lvl in sorted(per_level):
        if min(per_level[lvl]) < threshold:
            break
        best = lvl
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--subterrain", default="all")
    args = ap.parse_args()

    rows = [r for r in load(args.csv) if r["subterrain"] == args.subterrain]
    labels = list(dict.fromkeys(r["label"] for r in rows))
    levels = sorted({r["level"] for r in rows})
    data: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        data[(r["label"], r["level"])].append(r)

    seeds = sorted({r["seed"] for r in rows})
    print(f"Seeds: {seeds}. Survival = no fall in 20 s at vx = 0.8 m/s. mean ± std over seeds.\n")

    print(
        "| Level | "
        + " | ".join(f"{lb}: survival / vel err [m/s] / dist [m]" for lb in labels)
        + " |"
    )
    print("|---|" + "---|" * len(labels))
    for lvl in levels:
        cells = []
        for lb in labels:
            g = data.get((lb, lvl))
            cells.append(
                "n/a"
                if not g
                else f"{fmt([r['survival'] for r in g])} / {fmt([r['vel_err_lin'] for r in g], 3)} / {fmt([r['distance'] for r in g], 1)}"
            )
        print(f"| {lvl} | " + " | ".join(cells) + " |")

    print(
        f"\nHardest level with survival >= {args.threshold:.2f} on every seed (all levels below it also pass):\n"
    )
    print("| Policy | Max level |")
    print("|---|---|")
    for lb in labels:
        per = {
            lvl: [r["survival"] for r in data[(lb, lvl)]] for lvl in levels if data.get((lb, lvl))
        }
        ml = max_level(per, args.threshold)
        print(f"| {lb} | {'none' if ml is None else ml} |")


if __name__ == "__main__":
    main()
