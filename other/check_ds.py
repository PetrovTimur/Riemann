#!/usr/bin/env python3
"""
Quick dataset checker:
- Loads a CSV file
- Takes the last two columns (x, y)
- Prints basic statistics
- Plots a scatter of (x, y)

Usage:
  python other/check_ds.py [--file PATH] [--limit N] [--no-show]

Defaults:
  --file defaults to datasets/riemann_invariants_only_train.csv
  --limit optionally limits rows for faster preview
  --no-show skips interactive plot (still saves PNG next to CSV)
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot last two columns of a CSV as scatter and print statistics")
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join("datasets", "riemann_invariants_only_train.csv"),
        help="Path to CSV file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to load",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not show the interactive plot (still saves PNG)",
    )
    return parser.parse_args()


def load_last_two_columns(csv_path: str, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str], Optional[np.ndarray]]:
    if not os.path.exists(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Read with pandas for flexible handling; fallback to numpy if needed
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV with pandas: {e}", file=sys.stderr)
        sys.exit(2)

    if df.shape[1] < 2:
        print("Error: CSV must have at least two columns", file=sys.stderr)
        sys.exit(3)

    # Extract category from meta before dropping it
    categories = None
    if "meta" in df.columns:
        def _parse_category(val):
            try:
                return json.loads(val).get("category")
            except Exception:
                return None
        parsed = df["meta"].map(_parse_category)
        if parsed.notna().any():
            categories = parsed.to_numpy()

    df = df.drop(columns=["meta"], errors="ignore")

    # Optionally limit rows
    if limit is not None and limit > 0:
        df = df.head(limit)
        if categories is not None:
            categories = categories[:limit]

    print(df.max())

    # Get last two columns
    col_names = list(df.columns)
    x_col, y_col = col_names[-2], col_names[-1]

    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()

    # Ensure numeric
    try:
        x = x.astype(float)
        y = y.astype(float)
    except Exception as e:
        print(f"Error converting last two columns to float: {e}", file=sys.stderr)
        sys.exit(4)

    return x, y, (x_col, y_col), categories


def print_stats(x: np.ndarray, y: np.ndarray, names: Tuple[str, str]) -> None:
    x_name, y_name = names

    def stats(arr: np.ndarray):
        total = arr.size
        non_nan = int(np.count_nonzero(~np.isnan(arr)))
        nan = int(total - non_nan)
        return {
            "count": non_nan,
            "nan": nan,
            "mean": float(np.nanmean(arr)) if non_nan > 0 else float("nan"),
            "std": float(np.nanstd(arr)) if non_nan > 0 else float("nan"),
            "min": float(np.nanmin(arr)) if non_nan > 0 else float("nan"),
            "max": float(np.nanmax(arr)) if non_nan > 0 else float("nan"),
        }

    x_stats = stats(x)
    y_stats = stats(y)

    print("Columns:")
    print(f"  X: {x_name}")
    print(f"  Y: {y_name}")
    print("Statistics:")
    print(
        f"  {x_name}: count={x_stats['count']} nan={x_stats['nan']} "
        f"mean={x_stats['mean']:.6g} std={x_stats['std']:.6g} "
        f"min={x_stats['min']:.6g} max={x_stats['max']:.6g}"
    )
    print(
        f"  {y_name}: count={y_stats['count']} nan={y_stats['nan']} "
        f"mean={y_stats['mean']:.6g} std={y_stats['std']:.6g} "
        f"min={y_stats['min']:.6g} max={y_stats['max']:.6g}"
    )


def plot_scatter(x: np.ndarray, y: np.ndarray, names: Tuple[str, str], csv_path: str,
                 show: bool = True, categories: Optional[np.ndarray] = None) -> str:
    x_name, y_name = names

    plt.figure(figsize=(8, 6))
    if categories is not None:
        unique_cats = sorted(set(c for c in categories if c is not None))
        cmap = plt.get_cmap("tab10", len(unique_cats))
        cat_to_idx = {c: i for i, c in enumerate(unique_cats)}
        colors = [cmap(cat_to_idx[c]) if c is not None else "grey" for c in categories]
        sc = plt.scatter(x, y, s=6, alpha=0.7, edgecolors="none", c=colors)
        handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(cat_to_idx[c]),
                              markersize=6, label=str(c)) for c in unique_cats]
        plt.legend(handles=handles, title="category", fontsize=7, title_fontsize=8)
    else:
        plt.scatter(x, y, s=6, alpha=0.7, edgecolors="none")
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.plot([-100, 100], [-100, 100], color="black")
    plt.plot([100, -100], [-100, 100], color="red")
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"Scatter of {x_name} vs {y_name}\n{os.path.basename(csv_path)}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.gca().set_aspect("equal", adjustable="box")

    # Save next to CSV (same directory)
    out_dir = os.path.dirname(csv_path) or "."
    out_name = os.path.splitext(os.path.basename(csv_path))[0] + "_last2_scatter.png"
    out_path = os.path.join(out_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)

    if show:
        plt.show()
    else:
        # Close to free memory in non-interactive mode
        plt.close()

    print(f"Saved scatter plot to: {out_path}")
    return out_path


def main():
    args = parse_args()
    x, y, names, categories = load_last_two_columns(args.file, limit=args.limit)
    print_stats(x, y, names)
    plot_scatter(x, y, names, csv_path=args.file, show=not args.no_show, categories=categories)


if __name__ == "__main__":
    main()
