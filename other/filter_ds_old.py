#!/usr/bin/env python3
"""
Filter CSV rows by absolute value threshold across all columns.

- Loads a CSV file
- Drops rows where ANY column has abs(value) > threshold
- Prints how many rows were filtered and how many remain
- Saves the filtered CSV

Usage:
  python other/filter_ds.py --in datasets/riemann_invariants_only_train.csv \
                            [--out PATH] [--threshold 100] [--limit N] [--no-save]

Defaults:
  --threshold: 100
  --out: same directory as input, with suffix _filtered.csv
  --limit: optionally limit rows to process (preview mode)
  --no-save: skip writing the filtered CSV
"""

import argparse
import os
import sys
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rows where any column exceeds abs(threshold)")
    parser.add_argument(
        "--in",
        dest="in_file",
        type=str,
        default=os.path.join("datasets", "riemann_invariants_only_train.csv"),
        help="Path to input CSV",
    )
    parser.add_argument(
        "--out",
        dest="out_file",
        type=str,
        default=None,
        help="Path to output CSV (defaults to <input>_filtered.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        help="Absolute value threshold; rows with any |value| > threshold are removed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows to process (preview)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the filtered CSV",
    )
    return parser.parse_args()


def load_csv(path: str, limit: Optional[int] = None) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read CSV: {e}", file=sys.stderr)
        sys.exit(2)
    if limit is not None and limit > 0:
        df = df.head(limit)
    return df


def filter_by_abs_threshold(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    # Try to operate on numeric columns only, preserving non-numeric columns
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    # Build a mask: True for rows to keep (all abs <= threshold or NaN treated as not exceeding)
    # NaNs are considered safe (won't trigger removal) but we can decide to drop rows with NaN if desired later
    exceeds = (numeric_df.abs() > threshold)
    to_remove_mask = exceeds.any(axis=1)
    filtered_df = df.loc[~to_remove_mask].copy()
    return filtered_df


def default_out_path(in_path: str) -> str:
    base_dir = os.path.dirname(in_path)
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(base_dir, f"{base_name}_filtered.csv")


def main():
    args = parse_args()
    df = load_csv(args.in_file, limit=args.limit)
    total_rows = len(df)

    filtered_df = filter_by_abs_threshold(df, threshold=args.threshold)
    kept_rows = len(filtered_df)
    removed_rows = total_rows - kept_rows

    print(f"Input: {args.in_file}")
    print(f"Threshold: {args.threshold}")
    if args.limit:
        print(f"Preview mode: processing first {args.limit} rows of the file")
    print(f"Rows total: {total_rows}")
    print(f"Rows removed (any |value| > {args.threshold}): {removed_rows}")
    print(f"Rows kept: {kept_rows}")

    if not args.no_save:
        out_path = args.out_file or default_out_path(args.in_file)
        try:
            filtered_df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Failed to write filtered CSV: {e}", file=sys.stderr)
            sys.exit(3)
        print(f"Saved filtered CSV to: {out_path}")
    else:
        print("Skipping save (--no-save)")


if __name__ == "__main__":
    main()
