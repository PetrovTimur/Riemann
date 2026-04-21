#!/usr/bin/env python3
"""
Utility to check for dry bed formation in pandas DataFrame
"""
import numpy as np
import pandas as pd
import json


def check_dry_bed_df(df, meta_col='meta', g=9.8066):
    """
    Check for vacuum/dry bed formation in DataFrame where meta column contains dicts.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with meta column containing dicts like {'hl': ..., 'hul': ..., 'hr': ..., 'hur': ...}
    meta_col : str
        Name of column containing the dict
    g : float
        Gravitational acceleration

    Returns:
    --------
    pd.Series : Boolean series indicating vacuum formation (True = vacuum)
    """
    hl = df[meta_col].apply(lambda x: x['hL'])
    hul = df[meta_col].apply(lambda x: x['huL'])
    hr = df[meta_col].apply(lambda x: x['hR'])
    hur = df[meta_col].apply(lambda x: x['huR'])
    u_diff = hur / hr - hul / hl  # u_R - u_L
    c_sum = np.sqrt(g * hl) + np.sqrt(g * hr)  # c_L + c_R
    return u_diff > 2 * c_sum


def add_dry_bed_column(df, meta_col='meta', g=9.8066, col_name='is_vacuum'):
    """Add dry bed detection column to DataFrame (in-place)"""
    df[col_name] = check_dry_bed_df(df, meta_col, g)
    return df


def filter_valid_problems(df, meta_col='meta', g=9.8066, drop_vacuum=True):
    """Filter DataFrame to remove vacuum cases"""
    is_vacuum = check_dry_bed_df(df, meta_col, g)
    if drop_vacuum:
        return df[~is_vacuum].copy()
    else:
        return df[is_vacuum].copy()


# Example usage
if __name__ == "__main__":
    # Create sample DataFrame with meta column containing dicts

    df = pd.read_csv("datasets/check_removed_with_meta.csv")
    df["meta"] = df["meta"].apply(json.loads)


    print("Original DataFrame:")
    print(df)
    print("\n" + "="*70 + "\n")

    # Add vacuum detection column
    df = add_dry_bed_column(df)

    print("With vacuum detection:")
    print(df[['is_vacuum']])
    print("\n" + "="*70 + "\n")

    # Show statistics
    n_vacuum = df['is_vacuum'].sum()
    print(f"Statistics:")
    print(f"  Total problems: {len(df)}")
    print(f"  Vacuum cases: {n_vacuum}")
    print(f"  Valid cases: {len(df) - n_vacuum}")
    print("\n" + "="*70 + "\n")

    # Filter out vacuum cases
    df_valid = filter_valid_problems(df, drop_vacuum=True)
    print("Valid problems (no vacuum):")
    # print(df_valid[['id', 'description']])
    print("\n" + "="*70 + "\n")

    # Show only vacuum cases
    df_vacuum = filter_valid_problems(df, drop_vacuum=False)
    print("Vacuum problems:")
    # print(df_vacuum[['id', 'description']])

    # Detailed analysis
    print("\n" + "="*70)
    print("Detailed Analysis:")
    print("="*70)
    print(df.iterrows())
    for idx, row in df.iterrows():
        m = row['meta']
        uL = m['huL'] / m['hL']
        uR = m['huR'] / m['hR']
        cL = np.sqrt(9.8066 * m['hL'])
        cR = np.sqrt(9.8066 * m['hR'])
        u_diff = uR - uL
        c_threshold = 2 * (cL + cR)

        status = "🌵 VACUUM" if row['is_vacuum'] else "💧 Normal"
        print(f"\n({status}):")
        print(f"  Left:  h={m['hL']:.4f}, u={uL:.2f}")
        print(f"  Right: h={m['hR']:.4f}, u={uR:.2f}")
        print(f"  Criterion: u_R - u_L = {u_diff:.2f} vs 2(c_L+c_R) = {c_threshold:.2f}")



