#!/usr/bin/env python3
"""
Test script for CabaretSolverImproved with implicit sonic point processing
"""
import numpy as np
import sys
sys.path.insert(0, '/')

from training.solvers import CabaretSolverImproved, CabaretSolverPlus

def test_dam_break():
    """Test the improved solver on a classic dam break problem (generates sonic point)"""
    print("Testing CabaretSolverImproved on dam break problem...")
    
    # Grid parameters
    N_cells = 50
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    L = 10.0  # Domain length
    dx = L / N_cells
    
    # Time parameters
    dt = 0.001
    T_final = 0.1
    
    # Initial condition: dam break
    x = np.linspace(0, L, N_total)
    h = np.ones(N_total)
    u = np.zeros(N_total)
    
    # Left side: high water level, Right side: low water level
    h[x < L/2] = 2.0
    h[x >= L/2] = 1.0
    
    hu = h * u
    
    # Create solver
    solver = CabaretSolverImproved(model=None, g=9.81, newton_tol=1e-8, newton_max_iter=100)
    
    print(f"Grid: {N_total} points, dx={dx:.4f}, dt={dt:.6f}")
    print(f"Initial condition: Dam break at x={L/2}")
    print(f"  Left:  h={h[0]:.2f}, u={u[0]:.2f}")
    print(f"  Right: h={h[-1]:.2f}, u={u[-1]:.2f}")
    
    # Time stepping
    t = 0
    n_steps = int(T_final / dt)
    
    for step in range(n_steps):
        h, hu = solver.step(h, hu, dx, dt)
        t += dt
        
        if step % 10 == 0 or step == n_steps - 1:
            u_current = hu / (h + 1e-12)
            print(f"Step {step:4d}, t={t:.4f}: h_min={h.min():.4f}, h_max={h.max():.4f}, "
                  f"u_min={u_current.min():.4f}, u_max={u_current.max():.4f}")
    
    print("\n✓ Test completed successfully!")
    return h, hu, x


def test_subsonic_flow():
    """Test the solver on a subsonic flow (no sonic points expected)"""
    print("\nTesting CabaretSolverImproved on subsonic flow...")
    
    # Grid parameters
    N_cells = 30
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    L = 10.0
    dx = L / N_cells
    dt = 0.001
    
    # Initial condition: smooth perturbation
    x = np.linspace(0, L, N_total)
    h = 1.0 + 0.1 * np.sin(2 * np.pi * x / L)
    u = 0.1 * np.ones(N_total)
    hu = h * u
    
    # Create solver
    solver = CabaretSolverImproved(model=None, g=9.81)
    
    print(f"Grid: {N_total} points, dx={dx:.4f}, dt={dt:.6f}")
    print(f"Initial: Smooth wave with small velocity")
    
    # Run a few steps
    for step in range(20):
        h, hu = solver.step(h, hu, dx, dt)
        
        if step % 5 == 0:
            u_current = hu / (h + 1e-12)
            print(f"Step {step:4d}: h_min={h.min():.4f}, h_max={h.max():.4f}")
    
    print("✓ Subsonic test completed successfully!")
    return h, hu, x


def compare_with_standard():
    """Compare CabaretSolverImproved with CabaretSolverPlus"""
    print("\nComparing CabaretSolverImproved with CabaretSolverPlus...")
    
    # Grid parameters
    N_cells = 40
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    L = 10.0
    dx = L / N_cells
    dt = 0.001
    
    # Initial condition: dam break
    x = np.linspace(0, L, N_total)
    h0 = np.ones(N_total)
    h0[x < L/2] = 2.0
    u0 = np.zeros(N_total)
    hu0 = h0 * u0
    
    # Standard solver
    solver_std = CabaretSolverPlus(model=None, g=9.81)
    h_std, hu_std = h0.copy(), hu0.copy()
    
    # Improved solver
    solver_imp = CabaretSolverImproved(model=None, g=9.81)
    h_imp, hu_imp = h0.copy(), hu0.copy()
    
    # Run both for 50 steps
    n_steps = 50
    for step in range(n_steps):
        h_std, hu_std = solver_std.step(h_std, hu_std, dx, dt)
        h_imp, hu_imp = solver_imp.step(h_imp, hu_imp, dx, dt)
    
    # Compare results
    h_diff = np.abs(h_std - h_imp)
    hu_diff = np.abs(hu_std - hu_imp)
    
    print(f"After {n_steps} steps:")
    print(f"  Max h difference:  {h_diff.max():.6e}")
    print(f"  Max hu difference: {hu_diff.max():.6e}")
    print(f"  Mean h difference:  {h_diff.mean():.6e}")
    print(f"  Mean hu difference: {hu_diff.mean():.6e}")
    
    print("✓ Comparison completed!")


if __name__ == "__main__":
    print("="*70)
    print("Testing CabaretSolverImproved Implementation")
    print("="*70)
    
    try:
        # Test 1: Dam break (sonic point)
        test_dam_break()
        
        # Test 2: Subsonic flow
        test_subsonic_flow()
        
        # Test 3: Comparison with standard solver
        compare_with_standard()
        
        print("\n" + "="*70)
        print("All tests passed! ✓")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

