import argparse
import os
import sys

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split a CSV into two parts by percentage.")
    p.add_argument("--in", dest="in_file", required=True, help="Input CSV path")
    p.add_argument("--pct", type=float, default=80.0, help="Percent to put in part 1 (0-100)")
    p.add_argument("--out1", type=str, default=None, help="Output path for part 1")
    p.add_argument("--out2", type=str, default=None, help="Output path for part 2")
    p.add_argument("--shuffle", action="store_true", help="Shuffle rows before splitting")
    p.add_argument("--seed", type=int, default=42, help="Random seed (used with --shuffle)")
    return p.parse_args()


def default_out_paths(in_path: str) -> tuple[str, str]:
    base_dir = os.path.dirname(in_path) or "."
    base = os.path.splitext(os.path.basename(in_path))[0]
    return (
        os.path.join(base_dir, f"{base}_part1.csv"),
        os.path.join(base_dir, f"{base}_part2.csv"),
    )


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.pct <= 100.0):
        print("Error: --pct must be between 0 and 100", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.in_file):
        print(f"Error: file not found: {args.in_file}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.in_file)

    if args.shuffle:
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    split_idx = int(round(len(df) * (args.pct / 100.0)))
    df1 = df.iloc[:split_idx].copy()
    df2 = df.iloc[split_idx:].copy()

    out1, out2 = default_out_paths(args.in_file)
    out1 = args.out1 or out1
    out2 = args.out2 or out2

    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Part 1 rows ({args.pct}%): {len(df1)} -> {out1}")
    print(f"Part 2 rows ({100.0 - args.pct}%): {len(df2)} -> {out2}")


if __name__ == "__main__":
    main()
