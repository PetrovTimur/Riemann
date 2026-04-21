#!/usr/bin/env python3
"""
Detailed test to verify sonic point detection and implicit solver activation
"""
import numpy as np
import sys
sys.path.insert(0, '/')

from training.solvers import CabaretSolverImproved

def test_sonic_detection_detailed():
    """Test with detailed output showing when sonic points are detected"""
    print("Testing Sonic Point Detection in CabaretSolverImproved")
    print("="*70)

    # Grid parameters
    N_cells = 20
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1

    L = 4.0
    dx = L / N_cells
    dt = 0.005

    # Initial condition: dam break (creates strong sonic transition)
    x = np.linspace(0, L, N_total)
    h = np.ones(N_total)
    u = np.zeros(N_total)

    # Strong discontinuity
    h[x < L/2] = 3.0
    h[x >= L/2] = 0.5

    hu = h * u

    # Create custom solver with instrumentation
    class InstrumentedSolver(CabaretSolverImproved):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.sonic_I1_count = 0
            self.sonic_I2_count = 0
            self.sonic_both_count = 0
            self.standard_count = 0

        def _step2(self):
            """Override to add instrumentation"""
            eps = 1e-12

            # Compute intermediate values
            u_cell_n_plus_half = self.hu_cell_n_plus_half / (self.h_cell_n_plus_half + eps)
            c_cell_n_plus_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

            lambda1_cell_n_plus_half = u_cell_n_plus_half + c_cell_n_plus_half
            lambda2_cell_n_plus_half = u_cell_n_plus_half - c_cell_n_plus_half

            # Reset counters
            self.sonic_I1_count = 0
            self.sonic_I2_count = 0
            self.sonic_both_count = 0
            self.standard_count = 0

            # Detect sonic points
            for j in range(1, self.N_nodes - 1):
                sonic_I1 = (lambda1_cell_n_plus_half[j - 1] * lambda1_cell_n_plus_half[j] <= 0)
                sonic_I2 = (lambda2_cell_n_plus_half[j - 1] * lambda2_cell_n_plus_half[j] <= 0)

                if sonic_I1 and sonic_I2:
                    self.sonic_both_count += 1
                elif sonic_I1:
                    self.sonic_I1_count += 1
                elif sonic_I2:
                    self.sonic_I2_count += 1
                else:
                    self.standard_count += 1

            # Call parent implementation
            super()._step2()

    solver = InstrumentedSolver(model=None, g=9.81, newton_tol=1e-8, newton_max_iter=100)

    print(f"Grid: {N_total} points, dx={dx:.4f}, dt={dt:.6f}")
    print(f"Dam break: h_left={h[0]:.1f}, h_right={h[-1]:.1f}")
    print()

    # Time stepping with detailed output
    n_steps = 20

    for step in range(n_steps):
        h, hu = solver.step(h, hu, dx, dt)

        u_current = hu / (h + 1e-12)
        c_current = np.sqrt(9.81 * np.maximum(0, h))

        # Compute Froude number
        Fr = np.abs(u_current) / (c_current + 1e-12)

        print(f"Step {step:3d}:")
        print(f"  Sonic points - I1 only: {solver.sonic_I1_count:2d}, "
              f"I2 only: {solver.sonic_I2_count:2d}, "
              f"Both: {solver.sonic_both_count:2d}, "
              f"Standard: {solver.standard_count:2d}")
        print(f"  h: [{h.min():.4f}, {h.max():.4f}], "
              f"u: [{u_current.min():.4f}, {u_current.max():.4f}], "
              f"Fr_max: {Fr.max():.4f}")

        if step < 5 or step % 5 == 0:
            # Show where sonic points occur
            if solver.sonic_I1_count + solver.sonic_I2_count + solver.sonic_both_count > 0:
                print(f"  → Implicit solver activated!")
        print()

    print("="*70)
    print("✓ Sonic point detection test completed successfully!")


if __name__ == "__main__":
    test_sonic_detection_detailed()

