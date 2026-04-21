import argparse
import ast
import math

import pandas as pd

G = 9.81


def should_keep(meta_str: str) -> bool:
    meta = ast.literal_eval(meta_str)
    hL = meta["hL"]
    huL = meta["huL"]
    hR = meta["hR"]
    huR = meta["huR"]

    uL = huL / hL if hL > 0 else 0.0
    uR = huR / hR if hR > 0 else 0.0
    cL = math.sqrt(G * hL) if hL > 0 else 0.0
    cR = math.sqrt(G * hR) if hR > 0 else 0.0

    return (uR - uL) <= 2 * (cL + cR)


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows based on dry-bed condition in meta column.")
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path (default: <input>_filtered.csv)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    mask = df["meta"].apply(should_keep)
    filtered = df[mask]

    out_path = args.output or args.csv_path.replace(".csv", "_filtered.csv")
    filtered.to_csv(out_path, index=False)

    print(f"Original rows:  {len(df)}")
    print(f"Filtered rows:  {len(filtered)}")
    print(f"Removed rows:   {len(df) - len(filtered)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()

