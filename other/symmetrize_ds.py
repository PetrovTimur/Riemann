import pandas as pd

# Column names in the CSV (as provided by the user)
COL_I1_NODE_L = "I1_node_n[j-1]"
COL_I1_NODE_C = "I1_node_n[j]"
COL_I1_NODE_R = "I1_node_n[j+1]"
COL_I2_NODE_L = "I2_node_n[j-1]"
COL_I2_NODE_C = "I2_node_n[j]"
COL_I2_NODE_R = "I2_node_n[j+1]"
COL_I1_CELL_PH_L = "I1_cell_n_plus_half[j-1]"
COL_I1_CELL_PH_C = "I1_cell_n_plus_half[j]"
COL_I2_CELL_PH_L = "I2_cell_n_plus_half[j-1]"
COL_I2_CELL_PH_C = "I2_cell_n_plus_half[j]"
COL_I1_CELL_L = "I1_cell_n[j-1]"
COL_I1_CELL_C = "I1_cell_n[j]"
COL_I2_CELL_L = "I2_cell_n[j-1]"
COL_I2_CELL_C = "I2_cell_n[j]"
COL_I1_NODE_NP1 = "I1_node_n_plus_1[j]"
COL_I2_NODE_NP1 = "I2_node_n_plus_1[j]"


def symmetrize_csv(
    input_path: str,
    output_path: str,
) -> None:
    """Load a CSV, filter rows using the symmetry condition from BaseModule._symmetrize, and save.

    The condition mirrors the (commented-out) logic in `BaseModule._symmetrize`:

        feats = data["feats"]  # [bs, F = 14]
        I1_L = feats[:, -4]
        I1_R = feats[:, -3]
        I2_L = feats[:, -2]
        I2_R = feats[:, -1]

        u_L = (I1_L + I2_L) / 2
        u_R = (I1_R + I2_R) / 2

        c_L = (I1_L - I2_L) / 4
        c_R = (I1_R - I2_R) / 4

        cond = ((u_L < -c_L) & (u_R < -c_R)) |
                ((u_L < -c_L) & (u_R < -c_R)) |
                ((u_L < -c_L) & (u_R < -c_R))

    Since the three OR parts are identical, this reduces to a single

        cond = (u_L < -c_L) & (u_R < -c_R)

    Here we apply this condition using the CSV columns for I1/I2 on left/right.
    We assume that in the CSV the last four features correspond to
    I1_cell_n[j-1], I1_cell_n[j], I2_cell_n[j-1], I2_cell_n[j],
    in the same order as in the training `feats` tensor.
    """

    # Load CSV
    df = pd.read_csv(input_path)

    # Map CSV columns to the invariants used in _symmetrize.
    # We interpret left/right as j-1/j; adjust this if your convention differs.
    I1_L = df[COL_I1_CELL_L]
    I1_R = df[COL_I1_CELL_C]
    I2_L = df[COL_I2_CELL_L]
    I2_R = df[COL_I2_CELL_C]

    # Compute u and c on left/right, same as in BaseModule._symmetrize
    u_L = (I1_L + I2_L) / 2.0
    u_R = (I1_R + I2_R) / 2.0

    c_L = (I1_L - I2_L) / 4.0
    c_R = (I1_R - I2_R) / 4.0

    # Symmetry condition using elementwise logical operators
    cond = ((u_R < -c_R) & (u_L < c_L)) | ((u_L < -c_L) & (u_R < c_R))

    # Filter rows
    df_filtered = df[cond].copy()

    # Print sizes of original and filtered tables
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(df_filtered)}")

    # Save filtered CSV
    df_filtered.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage; adjust filenames as needed.
    input_csv = "datasets/riemann_invariants/riemann_invariants_only_val.csv"          # path to your original CSV
    output_csv = "datasets/riemann_invariants_sym/riemann_invariants_only_val.csv"     # path for the symmetrized/filtered CSV

    symmetrize_csv(input_csv, output_csv)
