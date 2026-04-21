#!/usr/bin/env python3
"""Quick dataset checker (with loss coloring):

- Loads a CSV file
- Uses the last two columns *before* the loss column as (x, y)
- Colors points by the loss column
- Prints basic statistics
- Saves a scatter plot next to the CSV

Typical usage with eval output table produced by `training/run_eval.py`:
  python other/check_ds_loss.py --file outputs/.../try.csv

Args:
  --file PATH       path to CSV
  --loss-col NAME   loss column name (default: loss)
  --limit N         optionally limit number of rows
  --no-show         do not open an interactive window (still saves png)
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter plot of last two columns (before loss col) colored by loss"
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to CSV file (must contain loss column)",
    )
    parser.add_argument(
        "--loss-col",
        type=str,
        default="loss",
        help="Name of loss column to color by",
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


def _coerce_float(a: np.ndarray, name: str) -> np.ndarray:
    try:
        return a.astype(float)
    except Exception as e:
        raise ValueError(f"Error converting '{name}' to float: {e}")


def load_xy_loss(
    csv_path: str, loss_col: str = "loss", limit: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[str, str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Keep behavior consistent with other/check_ds.py
    if "meta" in df.columns:
        df = df.drop(columns=["meta"])

    if loss_col not in df.columns:
        raise ValueError(
            f"CSV does not contain loss column '{loss_col}'. Available columns: {list(df.columns)}"
        )

    if limit is not None and limit > 0:
        df = df.head(limit)

    cols = list(df.columns)
    loss_idx = cols.index(loss_col)
    if loss_idx < 2:
        raise ValueError(
            f"Loss column '{loss_col}' is too early in the table to take two columns before it."
        )

    x_col = cols[loss_idx - 2]
    y_col = cols[loss_idx - 1]

    x = _coerce_float(df[x_col].to_numpy(), x_col)
    y = _coerce_float(df[y_col].to_numpy(), y_col)
    loss = _coerce_float(df[loss_col].to_numpy(), loss_col)

    return x, y, loss, (x_col, y_col)


def print_stats(x: np.ndarray, y: np.ndarray, loss: np.ndarray, names: Tuple[str, str], loss_col: str) -> None:
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

    xs = stats(x)
    ys = stats(y)
    ls = stats(loss)

    print("Columns:")
    print(f"  X: {x_name}")
    print(f"  Y: {y_name}")
    print(f"  Color: {loss_col}")
    print("Statistics:")
    print(
        f"  {x_name}: count={xs['count']} nan={xs['nan']} mean={xs['mean']:.6g} std={xs['std']:.6g} min={xs['min']:.6g} max={xs['max']:.6g}"
    )
    print(
        f"  {y_name}: count={ys['count']} nan={ys['nan']} mean={ys['mean']:.6g} std={ys['std']:.6g} min={ys['min']:.6g} max={ys['max']:.6g}"
    )
    print(
        f"  {loss_col}: count={ls['count']} nan={ls['nan']} mean={ls['mean']:.6g} std={ls['std']:.6g} min={ls['min']:.6g} max={ls['max']:.6g}"
    )


def plot_scatter_loss(
    x: np.ndarray,
    y: np.ndarray,
    loss: np.ndarray,
    names: Tuple[str, str],
    csv_path: str,
    loss_col: str,
    show: bool = True,
) -> str:
    x_name, y_name = names

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        x,
        y,
        c=loss,
        s=6,
        alpha=0.8,
        edgecolors="none",
        cmap="viridis",
    )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(
        f"Scatter of {x_name} vs {y_name} colored by {loss_col}\n{os.path.basename(csv_path)}"
    )
    plt.grid(True, linestyle="--", alpha=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label(loss_col)

    out_dir = os.path.dirname(csv_path) or "."
    out_name = os.path.splitext(os.path.basename(csv_path))[0] + "_last2_colored_by_loss.png"
    out_path = os.path.join(out_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved loss-colored scatter plot to: {out_path}")
    return out_path


def main() -> int:
    args = parse_args()

    try:
        x, y, loss, names = load_xy_loss(
            args.file, loss_col=args.loss_col, limit=args.limit
        )
        print_stats(x, y, loss, names, args.loss_col)
        plot_scatter_loss(
            x,
            y,
            loss,
            names,
            csv_path=args.file,
            loss_col=args.loss_col,
            show=not args.no_show,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

