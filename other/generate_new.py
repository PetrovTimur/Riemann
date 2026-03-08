import os

import torch
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from training.simulation import Simulation
from training.solvers import CabaretSolverPlus, RiemannSolver, riemann_solver_newton

# Constants
G = 9.806


def load_and_preprocess_data(file_path: str, exclude_category: int = 8) -> pd.DataFrame:
    """
    Load Riemann problem data from CSV and compute derived quantities.

    Args:
        file_path: Path to CSV file with Riemann data
        exclude_category: Category to exclude from dataset

    Returns:
        DataFrame with original data plus computed velocities and wave speeds
    """
    df = pd.read_csv(file_path)
    df = df[df['category'] != exclude_category]

    # Compute velocities and wave speeds
    df['uL'] = df['huL'] / df['hL']
    df['uR'] = df['huR'] / df['hR']
    df['cL'] = (df['hL'] * G) ** 0.5
    df['cR'] = (df['hR'] * G) ** 0.5

    return df


def plot_condition_heatmap(df: pd.DataFrame, show: bool = True):
    """
    Create and display a heatmap showing the distribution of data across
    different characteristic regions (supersonic/subsonic flows).

    Args:
        df: DataFrame with velocity (u) and wave speed (c) columns
        show: Whether to display the plot
    """
    # Define conditions based on characteristic directions
    row_conditions = [
        df['uL'] < -df['cL'],
        (df['uL'] >= -df['cL']) & (df['uL'] <= df['cL']),
        df['uL'] > df['cL']
    ]

    col_conditions = [
        df['uR'] < -df['cR'],
        (df['uR'] >= -df['cR']) & (df['uR'] <= df['cR']),
        df['uR'] > df['cR']
    ]

    # Count samples in each region
    heatmap_data = np.zeros((3, 3))
    for i, row_cond in enumerate(row_conditions):
        for j, col_cond in enumerate(col_conditions):
            heatmap_data[i, j] = df[row_cond & col_cond].shape[0]

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu",
                xticklabels=['uR < -cR', '-cR < uR < cR', 'uR > cR'],
                yticklabels=['uL < -cL', '-cL < uL < cL', 'uL > cL'])
    plt.title("Distribution of Initial Conditions")
    plt.xlabel("Right State (uR, cR)")
    plt.ylabel("Left State (uL, cL)")

    if show:
        plt.show()


def prepare_datasets(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and validation sets and convert to PyTorch tensors.

    Args:
        df: DataFrame with Riemann problem data
        test_size: Fraction of data to use for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, train_X_tensor) as TensorDatasets
    """
    # Define features and targets
    X = df[['hL', 'huL', 'hR', 'huR']]
    y = df[['h_star', 'u_star']]

    # Stratified split if category column exists
    stratify = df['category'] if 'category' in df.columns else None
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"Training input shape: {train_X.shape}")
    print(f"Validation input shape: {val_X.shape}")

    # Convert to PyTorch tensors
    train_X_tensor = torch.tensor(train_X.values, dtype=torch.float32)
    val_X_tensor = torch.tensor(val_X.values, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y.values, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y.values, dtype=torch.float32)

    # Create datasets
    train_dataset = TensorDataset(train_X_tensor, train_y_tensor)
    val_dataset = TensorDataset(val_X_tensor, val_y_tensor)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_dataset, val_dataset, train_X_tensor


def create_output_dataframe() -> pd.DataFrame:
    """
    Create an empty DataFrame with columns for Riemann invariants at various
    spatial and temporal locations.

    Returns:
        Empty DataFrame with appropriate column names
    """
    columns = [
        'I1_node_n[j-1]', 'I1_node_n[j]', 'I1_node_n[j+1]',
        'I2_node_n[j-1]', 'I2_node_n[j]', 'I2_node_n[j+1]',
        'I1_cell_n_plus_half[j-1]', 'I1_cell_n_plus_half[j]',
        'I2_cell_n_plus_half[j-1]', 'I2_cell_n_plus_half[j]',
        'I1_cell_n[j-1]', 'I1_cell_n[j]',
        'I2_cell_n[j-1]', 'I2_cell_n[j]',
        'I1_node_true', 'I2_node_true',
        'meta'
    ]
    df = pd.DataFrame(columns=columns)
    # Make sure pandas treats this column as generic Python objects (strings/dicts)
    # so we can safely store JSON without dtype warnings.
    df['meta'] = df['meta'].astype('object')
    return df



def run_simulation(hL: float, huL: float, hR: float, huR: float,
                   L: float = 1.0, nx: int = 20, t_end: float = 5e-3):
    """
    Run a CABARET simulation for given Riemann initial conditions.

    Args:
        hL: Left water depth
        huL: Left momentum (h*u)
        hR: Right water depth
        huR: Right momentum (h*u)
        L: Domain length
        nx: Number of spatial cells
        t_end: End time for simulation

    Returns:
        Tuple of (simulation object, (h_true, u_true)) where true values
        are from exact Riemann solver
    """
    config = {
        'L': L,
        'nx': nx,
        'h_l': hL,
        'h_r': hR,
        'u_l': huL / hL,
        'u_r': huR / hR,
        'solver': CabaretSolverPlus(model=None),
        't_end': t_end,
        't_start': 0,
    }

    t_rand = 5e-3
    config["t_start"] = t_rand
    config["t_end"] = t_rand + 1e-12


    sim = Simulation(config)
    sim.run()

    # print(sim.t)

    # Compute exact solution for comparison
    true_solver = RiemannSolver()
    h, u = true_solver.solve(
        sim.x, sim.t, config['h_l'], config['u_l'], config['h_r'], config['u_r']
    )['vals']

    return sim, (h, u)


def extract_riemann_invariants(solver, target, output_df: pd.DataFrame, use_local_riemann: bool = False):
    """
    Extract Riemann invariants from solver state at various spatial/temporal positions
    and append to output DataFrame.

    This extracts invariants I1 = u + 2*c and I2 = u - 2*c at:
    - Nodes at time n
    - Cells at time n
    - Cells at time n+1/2
    - True values at cells from exact solver OR local Riemann solution

    Only extracts points where the flow is "interesting" (supersonic: |u| > c).

    Args:
        solver: CABARET solver object with solution data
        target: Tuple of (h_true, u_true) from exact solver
        output_df: DataFrame to append extracted data to
        use_local_riemann: If True, use local Riemann solution from n+1/2 cell values
                          instead of exact solution for true targets

    Returns:
        Number of interesting points extracted
    """
    g = solver.g

    # Compute Riemann invariants for ALL nodes at time n
    node_count = solver.h_node_n.shape[0]
    h_node_n = solver.h_node_n
    hu_node_n = solver.hu_node_n

    u_node_n = hu_node_n / (h_node_n + 1e-12)
    c_node_n = np.sqrt(g * np.maximum(0.0, h_node_n))
    I1_node_n = u_node_n + 2 * c_node_n
    I2_node_n = u_node_n - 2 * c_node_n

    # Compute Riemann invariants for ALL cells at time n
    cell_count = solver.h_cell_n.shape[0]
    h_cell_n = solver.h_cell_n
    hu_cell_n = solver.hu_cell_n

    u_cell_n = hu_cell_n / (h_cell_n + 1e-12)
    c_cell_n = np.sqrt(g * np.maximum(0.0, h_cell_n))
    I1_cell_n = u_cell_n + 2 * c_cell_n
    I2_cell_n = u_cell_n - 2 * c_cell_n

    # Compute Riemann invariants for ALL cells at time n+1/2
    h_cell_n_plus_half = solver.h_cell_n_plus_half
    hu_cell_n_plus_half = solver.hu_cell_n_plus_half

    u_cell_n_plus_half = hu_cell_n_plus_half / (h_cell_n_plus_half + 1e-12)
    c_cell_n_plus_half = np.sqrt(g * np.maximum(0.0, h_cell_n_plus_half))
    I1_cell_n_plus_half = u_cell_n_plus_half + 2 * c_cell_n_plus_half
    I2_cell_n_plus_half = u_cell_n_plus_half - 2 * c_cell_n_plus_half

    # Create interesting mask: supersonic flow (|u| > c)
    interesting_mask = np.abs(u_cell_n_plus_half) > c_cell_n_plus_half

    # Compute true solution Riemann invariants
    if use_local_riemann:
        # Use local Riemann solution from cell n+1/2 values
        # Only compute for interesting cells
        I1_cell_true = np.zeros(cell_count)
        I2_cell_true = np.zeros(cell_count)

        # Get indices of interesting cells
        interesting_indices = np.where(interesting_mask)[0]

        for cell_idx in interesting_indices:
            # Skip boundary cells
            if cell_idx == 0 or cell_idx >= cell_count - 1:
                continue

            # Get left and right states at cell interfaces from n+1/2
            hL = h_cell_n_plus_half[cell_idx]
            huL = hu_cell_n_plus_half[cell_idx]
            hR = h_cell_n_plus_half[cell_idx + 1]
            huR = hu_cell_n_plus_half[cell_idx + 1]

            # Solve local Riemann problem
            solution = riemann_solver_newton(hL, huL, hR, huR, g=g)
            h_star, u_star = solution['star']

            # Compute Riemann invariants from solution
            c_star = np.sqrt(g * np.maximum(0.0, h_star))
            I1_cell_true[cell_idx] = u_star + 2 * c_star
            I2_cell_true[cell_idx] = u_star - 2 * c_star
    else:
        # Use exact solution from global Riemann solver
        h, u = target
        h_cell_true = h[1::2]  # Extract cell values (odd indices)
        u_cell_true = u[1::2]

        c_cell_true = np.sqrt(g * np.maximum(0.0, h_cell_true))
        I1_cell_true = u_cell_true + 2 * c_cell_true
        I2_cell_true = u_cell_true - 2 * c_cell_true

    # Extract features for interesting cells
    # For each interesting cell i, we need:
    # - Nodes: i-1, i, i+1, i+2 (cell i is between nodes i and i+1)
    # - Cells: i-1, i
    count = 0
    for cell_idx in range(1, cell_count - 1):
        if not interesting_mask[cell_idx]:
            continue

        # Node indices: cell i is between node i and node i+1
        # We need nodes [i, i+1, i+2] for the stencil
        node_idx = cell_idx

        # Check bounds for nodes (need i-1, i, i+1 relative to node_idx+1)
        if node_idx < 1 or node_idx + 2 >= node_count:
            continue

        # For cell i, we need the stencil around node i+1 (the right node of cell i)
        # Nodes: i, i+1, i+2 (which are j-1, j, j+1 where j = i+1)
        inv_list = [
            I1_node_n[node_idx], I1_node_n[node_idx + 1], I1_node_n[node_idx + 2],
            I2_node_n[node_idx], I2_node_n[node_idx + 1], I2_node_n[node_idx + 2],
            I1_cell_n_plus_half[cell_idx - 1], I1_cell_n_plus_half[cell_idx],
            I2_cell_n_plus_half[cell_idx - 1], I2_cell_n_plus_half[cell_idx],
            I1_cell_n[cell_idx - 1], I1_cell_n[cell_idx],
            I2_cell_n[cell_idx - 1], I2_cell_n[cell_idx],
            I1_cell_true[cell_idx], I2_cell_true[cell_idx],
            None,
        ]

        output_df.loc[len(output_df)] = inv_list
        count += 1

    return count




def generate_training_data(input_tensor: torch.Tensor, output_df: pd.DataFrame, use_local_riemann: bool = False):
    """
    Generate training data by running simulations for all input samples.

    Args:
        input_tensor: Tensor of shape (N, 4) with [hL, huL, hR, huR] for each sample
        output_df: DataFrame to store extracted Riemann invariants
        use_local_riemann: If True, use local Riemann solution from n+1/2 cell values
    """
    pbar = tqdm(input_tensor, desc="Generating training data", unit="sample")
    for episode in pbar:
        hL, huL, hR, huR = map(lambda x: x.item(), episode)

        sim, target = run_simulation(hL, huL, hR, huR)

        # Track how many rows existed before extraction so we can attach meta
        start_len = len(output_df)
        extract_riemann_invariants(sim.solver, target, output_df, use_local_riemann=use_local_riemann)

        # Attach meta to every row that was added for this episode.
        # Use .at/.loc (not chained indexing) so the assignment is guaranteed to persist.
        if len(output_df) > start_len:

            meta = {
                'hL': float(hL),
                'huL': float(huL),
                'hR': float(hR),
                'huR': float(huR),
            }
            # Storing dicts in a CSV column is awkward; a JSON string round-trips cleanly.
            import json
            meta_json = json.dumps(meta, separators=(",", ":"))

            for idx in range(start_len, len(output_df)):
                output_df.at[idx, 'meta'] = meta_json

        # Update progress bar with current dataframe size
        pbar.set_postfix({"df_len": len(output_df)})


def split_and_save_dataset(df: pd.DataFrame, output_dir: str,
                           test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train/validation and save to CSV files.

    Args:
        df: DataFrame to split
        output_dir: Directory to save train.csv and val.csv
        test_size: Fraction of data for validation
        random_state: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")

    # Check if stratify column exists
    stratify_col = df['category'] if 'category' in df.columns else None

    # Split dataset
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=stratify_col
    )

    # Reset indices and save
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print(f"Saved training data: {len(train_df)} samples to {train_path}")
    print(f"Saved validation data: {len(val_df)} samples to {val_path}")

    return train_df, val_df


def main():
    """
    Main execution function to generate Riemann invariant training data.
    """
    # Configuration
    input_file = 'datasets/old/riemann_training_data_balanced2.csv'
    output_file = 'datasets/check.csv'
    output_dir = 'datasets/new_dataset'
    use_local_riemann = True  # Set to True to use local Riemann solution from n+1/2 cell values

    # Load and preprocess data
    print("Loading data...")
    df = load_and_preprocess_data(input_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle the dataframe
    df = df[:1000]  # Take only first 100 samples for testing

    # Visualize distribution (optional)
    plot_condition_heatmap(df, show=False)

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset, val_dataset, train_X_tensor = prepare_datasets(df)

    # Create output dataframe for Riemann invariants
    new_df = create_output_dataframe()

    # Generate training data from simulations
    generate_training_data(train_X_tensor, new_df, use_local_riemann=use_local_riemann)

    print(f"\nGenerated {len(new_df)} samples")

    # Save complete dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    new_df.to_csv(output_file, index=False)
    print(f"Saved complete dataset to {output_file}")

    # Split and save train/val
    print("\nSplitting into train/validation sets...")
    train_df, val_df = split_and_save_dataset(new_df, output_dir)

    print("\nData generation complete!")
    return new_df


if __name__ == "__main__":
    result_df = main()

