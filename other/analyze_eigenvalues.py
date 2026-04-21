#!/usr/bin/env python3
"""
Analyze eigenvalues to understand when sonic points should occur
"""
import numpy as np
import sys
sys.path.insert(0, '/')

from training.solvers import CabaretSolverImproved

def analyze_eigenvalues():
    """Analyze eigenvalues during dam break simulation"""
    print("Analyzing Eigenvalues for Sonic Point Detection")
    print("="*70)

    # Grid parameters
    N_cells = 20
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1

    L = 4.0
    dx = L / N_cells
    dt = 0.01  # Larger time step to create stronger gradients

    # Initial condition: sharp dam break
    x = np.linspace(0, L, N_total)
    h = np.ones(N_total)
    u = np.zeros(N_total)

    # Very sharp discontinuity
    h[x < L/2] = 4.0
    h[x >= L/2] = 0.3

    hu = h * u

    solver = CabaretSolverImproved(model=None, g=9.81)

    print(f"Grid: {N_total} points ({N_cells} cells), dx={dx:.4f}, dt={dt:.6f}")
    print(f"CFL condition: dt/dx = {dt/dx:.4f}")
    print(f"Dam break: h_left={h[0]:.1f}, h_right={h[-1]:.1f}")
    print()

    # Run one step and inspect
    h_before = h.copy()
    h, hu = solver.step(h, hu, dx, dt)

    # Access solver internals after step 1
    eps = 1e-12
    u_cell_n_plus_half = solver.hu_cell_n_plus_half / (solver.h_cell_n_plus_half + eps)
    c_cell_n_plus_half = np.sqrt(solver.g * np.maximum(0.0, solver.h_cell_n_plus_half))

    lambda1 = u_cell_n_plus_half + c_cell_n_plus_half
    lambda2 = u_cell_n_plus_half - c_cell_n_plus_half

    print("After Step 1 (n+1/2 time level at cell centers):")
    print("-" * 70)
    print("Cell |      x      |     h      |     u      | lambda1 | lambda2 | Sonic?")
    print("-" * 70)

    x_cells = 0.5 * (x[::2][:-1] + x[::2][1:])

    for i in range(N_cells):
        sonic_I1 = ""
        sonic_I2 = ""

        if i > 0:
            if lambda1[i-1] * lambda1[i] <= 0:
                sonic_I1 = "I1"
            if lambda2[i-1] * lambda2[i] <= 0:
                sonic_I2 = "I2"

        sonic_str = f"{sonic_I1} {sonic_I2}".strip() or "-"

        print(f"{i:4d} | {x_cells[i]:10.4f} | {solver.h_cell_n_plus_half[i]:10.4f} | "
              f"{u_cell_n_plus_half[i]:10.4f} | {lambda1[i]:7.3f} | {lambda2[i]:7.3f} | {sonic_str}")

    print()
    print("Looking for sign changes in eigenvalues between adjacent cells...")

    sonic_transitions = []
    for i in range(1, N_cells):
        if lambda1[i-1] * lambda1[i] <= 0:
            sonic_transitions.append((i, "lambda1", lambda1[i-1], lambda1[i]))
        if lambda2[i-1] * lambda2[i] <= 0:
            sonic_transitions.append((i, "lambda2", lambda2[i-1], lambda2[i]))

    if sonic_transitions:
        print(f"\nFound {len(sonic_transitions)} sonic transitions:")
        for cell_idx, eig_type, val_before, val_after in sonic_transitions:
            print(f"  Between cells {cell_idx-1} and {cell_idx}: {eig_type} changes from "
                  f"{val_before:.4f} to {val_after:.4f}")
    else:
        print("\nNo sonic transitions detected in eigenvalues.")
        print("\nPossible reasons:")
        print("  1. Time step too small (shock hasn't formed yet)")
        print("  2. Grid too coarse (transition happens within a cell)")
        print("  3. Initial condition not creating transonic flow")

        # Check Froude numbers
        Fr = np.abs(u_cell_n_plus_half) / c_cell_n_plus_half
        print(f"\n  Max Froude number: {Fr.max():.4f}")
        print(f"  Min Froude number: {Fr.min():.4f}")

        if Fr.max() < 1.0:
            print("  → Flow is everywhere subsonic (Fr < 1)")
        elif Fr.min() > 1.0:
            print("  → Flow is everywhere supersonic (Fr > 1)")

    print("\n" + "="*70)


if __name__ == "__main__":
    analyze_eigenvalues()

