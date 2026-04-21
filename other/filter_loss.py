import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter CSV by loss and drop last 3 columns")
    parser.add_argument("--input", "-i", type=str, default="outputs/2026-04-20/14-03-18/try.csv", help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, default="outputs/2026-04-20/14-03-18/try_filtered.csv", help="Path to output CSV file")
    parser.add_argument("--threshold", "-t", type=float, default=0.1, help="Maximum allowed loss threshold")

    args = parser.parse_args()

    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)

    original_size = len(df)

    # Filter rows where loss is less than or equal to the threshold
    # Note: 'loss' is typically the 3rd to last column, we use the column name directly to be safe
    df_filtered = df[df['loss'] <= args.threshold]

    # Drop the last 3 columns
    df_filtered = df_filtered.iloc[:, :-3]

    filtered_size = len(df_filtered)
    print(f"Filtered out {original_size - filtered_size} rows where loss > {args.threshold}.")
    print(f"Remaining rows: {filtered_size}")

    print(f"Saving to {args.output}...")
    df_filtered.to_csv(args.output, index=False)
    print("Done!")

if __name__ == "__main__":
    main()

