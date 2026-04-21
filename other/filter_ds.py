#!/usr/bin/env python3
"""
Filter CSV rows by absolute value threshold across all columns.

- Loads a CSV file
- Drops rows where ANY column has abs(value) > threshold
- Prints how many rows were filtered and how many remain
- Saves the filtered CSV

Extra optional filter:
- Drop rows where | |col_b| - |col_a| | > diff-threshold (defaults to last *numeric* two columns)

Audit/inspection:
- Optionally print and/or save the rows that were filtered out, with reasons.

Usage:
  python other/filter_ds.py --in datasets/riemann_invariants_only_train.csv \
                            [--out PATH] [--threshold 100] [--limit N] [--no-save]

Diff filter usage:
  python other/filter_ds.py --in input.csv --diff-threshold 0.5

Defaults:
  --threshold: 100
  --out: same directory as input, with suffix _filtered.csv
  --limit: optionally limit rows to process (preview mode)
  --no-save: skip writing the filtered CSV
"""

import argparse
import os
import sys
from typing import Iterable, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter rows where any column exceeds abs(threshold)")
    parser.add_argument(
        "--in",
        dest="in_file",
        type=str,
        default=os.path.join("datasets", "train.csv"),
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
        "--diff-threshold",
        type=float,
        default=None,
        help=(
            "Optional: remove rows where abs(abs(col_b) - abs(col_a)) exceeds this threshold. "
            "By default picks the last two *numeric* columns (skipping meta/non-numeric columns)."
        ),
    )
    parser.add_argument(
        "--diff-cols",
        type=str,
        default=None,
        help=(
            "Optional: specify two column names for the diff filter as 'col_a,col_b'. "
            "If not set, the last two numeric columns are used."
        ),
    )
    parser.add_argument(
        "--meta-col",
        type=str,
        default="meta",
        help=(
            "Name of meta column (non-numeric) to exclude from automatic numeric-column selection. "
            "Set to empty string to disable name-based exclusion."
        ),
    )
    parser.add_argument(
        "--show-removed",
        type=int,
        default=0,
        help=(
            "Print the first N removed rows (with reasons) to stdout. "
            "0 disables printing."
        ),
    )
    parser.add_argument(
        "--save-removed",
        type=str,
        default=None,
        help=(
            "Optional path to save removed rows (including reason columns). "
            "Defaults to <input>_removed.csv if set to 'auto'."
        ),
    )
    parser.add_argument(
        "--save-reasons",
        type=str,
        default=None,
        help=(
            "Optional path to save only the reasons mask table (index + reason columns). "
            "Defaults to <input>_reasons.csv if set to 'auto'."
        ),
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


def _normalize_exclude_cols(exclude_cols: Optional[Iterable[str]]) -> set[str]:
    if not exclude_cols:
        return set()
    return {c for c in exclude_cols if c}


def get_numeric_columns(df: pd.DataFrame, exclude_cols: Optional[Iterable[str]] = None) -> list[str]:
    """Return columns that are numeric *or* can be parsed as numeric with minimal coercion.

    We exclude explicit columns (e.g. meta) and keep the original column order.
    """

    exclude = _normalize_exclude_cols(exclude_cols)

    # Identify columns that are already numeric.
    numeric = set(df.select_dtypes(include="number").columns)

    # For object/string columns, include them if they parse to mostly-numeric values.
    # Heuristic: at least one non-NaN numeric after coercion.
    for col in df.columns:
        if col in exclude or col in numeric:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            numeric.add(col)

    return [c for c in df.columns if c in numeric and c not in exclude]


def mask_abs_threshold_exceeded(
    df: pd.DataFrame,
    threshold: float,
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.Series:
    """Boolean mask: True where row should be removed due to any |value| > threshold."""

    numeric_cols = get_numeric_columns(df, exclude_cols=exclude_cols)
    if not numeric_cols:
        # No numeric columns -> nothing to remove by numeric threshold.
        return pd.Series(False, index=df.index)

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    exceeds = numeric_df.abs() > threshold
    return exceeds.any(axis=1)


def _parse_diff_cols_arg(diff_cols: Optional[str]) -> Optional[tuple[str, str]]:
    if diff_cols is None:
        return None
    parts = [p.strip() for p in diff_cols.split(",") if p.strip()]
    if len(parts) != 2:
        print("Error: --diff-cols must be two column names separated by a comma, e.g. --diff-cols a,b", file=sys.stderr)
        sys.exit(2)
    return parts[0], parts[1]


def _resolve_diff_columns(
    df: pd.DataFrame,
    diff_cols: Optional[tuple[str, str]],
    exclude_cols: Optional[Iterable[str]] = None,
) -> tuple[str, str]:
    if diff_cols is not None:
        col_a, col_b = diff_cols
        if col_a not in df.columns or col_b not in df.columns:
            missing = [c for c in (col_a, col_b) if c not in df.columns]
            print(f"Error: diff columns not found in CSV: {missing}", file=sys.stderr)
            sys.exit(2)
        return col_a, col_b

    numeric_cols = get_numeric_columns(df, exclude_cols=exclude_cols)
    if len(numeric_cols) < 2:
        print(
            "Error: diff filter requires at least 2 numeric columns in the CSV (after excluding meta/non-numeric)",
            file=sys.stderr,
        )
        sys.exit(2)

    return numeric_cols[-2], numeric_cols[-1]


def mask_abs_diff_exceeded(
    df: pd.DataFrame,
    diff_threshold: float,
    diff_cols: Optional[tuple[str, str]] = None,
    exclude_cols: Optional[Iterable[str]] = None,
) -> pd.Series:
    """Boolean mask: True where row should be removed due to abs(abs(b) - abs(a)) > diff_threshold."""

    if diff_threshold < 0:
        print("Error: --diff-threshold must be >= 0", file=sys.stderr)
        sys.exit(2)

    col_a, col_b = _resolve_diff_columns(df, diff_cols=diff_cols, exclude_cols=exclude_cols)

    s1 = pd.to_numeric(df[col_a], errors="coerce")
    s2 = pd.to_numeric(df[col_b], errors="coerce")

    # NaNs won't trigger removal (treated as safe)
    diff = (s1.abs() - s2.abs()).abs()
    return diff > diff_threshold


def default_out_path(in_path: str) -> str:
    base_dir = os.path.dirname(in_path)
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(base_dir, f"{base_name}_filtered.csv")


def default_removed_path(in_path: str) -> str:
    base_dir = os.path.dirname(in_path)
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(base_dir, f"{base_name}_removed.csv")


def default_reasons_path(in_path: str) -> str:
    base_dir = os.path.dirname(in_path)
    base_name = os.path.splitext(os.path.basename(in_path))[0]
    return os.path.join(base_dir, f"{base_name}_reasons.csv")


def build_filter_audit(
    df: pd.DataFrame,
    threshold: float,
    diff_threshold: Optional[float],
    diff_cols: Optional[tuple[str, str]],
    meta_col: Optional[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (kept_df, removed_df_with_reasons, reasons_df).

    reasons_df columns:
      - rm_abs_threshold
      - rm_abs_diff (only if diff_threshold provided)
      - rm_any

    removed_df_with_reasons = original columns + reason columns appended.
    """

    exclude_cols: list[str] = []
    if meta_col:
        exclude_cols.append(meta_col)

    # rm_abs = mask_abs_threshold_exceeded(df, threshold=threshold, exclude_cols=exclude_cols)

    reasons = {
    }

    # if diff_threshold is not None:
    rm_diff = mask_abs_diff_exceeded(
        df,
        diff_threshold=30,
        diff_cols=diff_cols,
        exclude_cols=exclude_cols,
    )
    reasons["rm_abs_diff"] = rm_diff

    reasons_df = pd.DataFrame(reasons, index=df.index)
    reasons_df["rm_any"] = reasons_df.any(axis=1)

    kept_df = df.loc[~reasons_df["rm_any"]].copy()

    removed_df = df.loc[reasons_df["rm_any"]].copy()
    removed_with_reasons = pd.concat([removed_df, reasons_df.loc[removed_df.index]], axis=1)

    return kept_df, removed_with_reasons, reasons_df


def main():
    args = parse_args()
    df = load_csv(args.in_file, limit=args.limit)

    meta_col = (args.meta_col or "").strip() or None

    # Build audit masks first, so we can report and optionally save removed rows.
    diff_cols = _parse_diff_cols_arg(args.diff_cols)
    kept_df, removed_with_reasons, reasons_df = build_filter_audit(
        df,
        threshold=args.threshold,
        diff_threshold=args.diff_threshold,
        diff_cols=diff_cols,
        meta_col=meta_col,
    )

    total_rows = len(df)
    kept_rows = len(kept_df)
    removed_rows = total_rows - kept_rows

    print(f"Input: {args.in_file}")
    print(f"Threshold: {args.threshold}")
    if args.diff_threshold is not None:
        if args.diff_cols:
            print(f"Diff-threshold: {args.diff_threshold} (cols: {args.diff_cols})")
        else:
            # Resolve to show which columns were used
            exclude = [meta_col] if meta_col else []
            col_a, col_b = _resolve_diff_columns(df, diff_cols=None, exclude_cols=exclude)
            print(f"Diff-threshold: {args.diff_threshold} (cols: {col_a},{col_b})")
    if meta_col:
        print(f"Meta column excluded from auto numeric selection: {meta_col}")
    if args.limit:
        print(f"Preview mode: processing first {args.limit} rows of the file")

    print(f"Rows total: {total_rows}")
    print(f"Rows removed (combined filters): {removed_rows}")
    print(f"Rows kept: {kept_rows}")

    # Per-reason counts
    per_reason = reasons_df.drop(columns=["rm_any"]).sum().to_dict()
    if per_reason:
        print("Removed-by-reason counts (note: rows can match multiple reasons):")
        for k, v in per_reason.items():
            print(f"  {k}: {int(v)}")

    if args.show_removed and args.show_removed > 0:
        to_show = removed_with_reasons.head(args.show_removed)
        # Avoid pandas truncation surprises.
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print("\nFirst removed rows (with reasons):")
            print(to_show)

    if args.save_removed:
        out_removed = args.save_removed
        if out_removed == "auto":
            out_removed = default_removed_path(args.in_file)
        try:
            removed_with_reasons.to_csv(out_removed, index=False)
        except Exception as e:
            print(f"Failed to write removed-rows CSV: {e}", file=sys.stderr)
            sys.exit(3)
        print(f"Saved removed rows (with reasons) to: {out_removed}")

    if args.save_reasons:
        out_reasons = args.save_reasons
        if out_reasons == "auto":
            out_reasons = default_reasons_path(args.in_file)
        try:
            # Keep original index to help trace back; write it as a column.
            reasons_df.reset_index(names="row_index").to_csv(out_reasons, index=False)
        except Exception as e:
            print(f"Failed to write reasons CSV: {e}", file=sys.stderr)
            sys.exit(3)
        print(f"Saved reasons table to: {out_reasons}")

    if not args.no_save:
        out_path = args.out_file or default_out_path(args.in_file)
        try:
            kept_df.to_csv(out_path, index=False)
        except Exception as e:
            print(f"Failed to write filtered CSV: {e}", file=sys.stderr)
            sys.exit(3)
        print(f"Saved filtered CSV to: {out_path}")
    else:
        print("Skipping save (--no-save)")


if __name__ == "__main__":
    main()
