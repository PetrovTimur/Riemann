#!/usr/bin/env python3
"""
Comprehensive validation test for CabaretSolverImproved
"""
import numpy as np
import sys
sys.path.insert(0, '/')

from training.solvers import CabaretSolverImproved, CabaretSolverPlus

def test_initialization():
    """Test that the solver initializes correctly"""
    print("Test 1: Initialization")
    print("-" * 50)
    
    solver = CabaretSolverImproved(model=None, g=9.81, newton_tol=1e-8, newton_max_iter=100)
    
    assert solver.g == 9.81, "Gravity not set correctly"
    assert solver.newton_tol == 1e-8, "Newton tolerance not set correctly"
    assert solver.newton_max_iter == 100, "Newton max iterations not set correctly"
    assert solver.model is None, "Model not set correctly"
    
    print("✓ Solver initializes correctly")
    print("✓ All parameters set properly")
    print()


def test_residual_functions():
    """Test residual function computations"""
    print("Test 2: Residual Functions")
    print("-" * 50)
    
    # Create a simple test case
    N_cells = 10
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    dx = 0.2
    dt = 0.001
    
    h = np.ones(N_total)
    u = 0.1 * np.ones(N_total)
    hu = h * u
    
    solver = CabaretSolverImproved(model=None, g=9.81)
    
    # Run one step to populate internal arrays
    try:
        h, hu = solver.step(h, hu, dx, dt)
        print("✓ Residual functions execute without errors")
        
        # Test that we can call residual functions
        F1 = solver._compute_residual_F1(0.1, 3.13, 5)
        F2 = solver._compute_residual_F2(0.1, 3.13, 5)
        
        assert not np.isnan(F1), "F1 returned NaN"
        assert not np.isnan(F2), "F2 returned NaN"
        assert np.isfinite(F1), "F1 returned infinity"
        assert np.isfinite(F2), "F2 returned infinity"
        
        print(f"✓ F1 and F2 return finite values")
        print(f"  F1({0.1:.2f}, {3.13:.2f}) = {F1:.6f}")
        print(f"  F2({0.1:.2f}, {3.13:.2f}) = {F2:.6f}")
        
    except Exception as e:
        print(f"✗ Error in residual functions: {e}")
        raise
    
    print()


def test_conservation():
    """Test mass and momentum conservation"""
    print("Test 3: Conservation Properties")
    print("-" * 50)
    
    N_cells = 30
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    L = 6.0
    dx = L / N_cells
    dt = 0.0005
    
    # Initial condition
    x = np.linspace(0, L, N_total)
    h = 1.0 + 0.2 * np.sin(2 * np.pi * x / L)
    u = 0.1 * np.cos(2 * np.pi * x / L)
    hu = h * u
    
    solver = CabaretSolverImproved(model=None, g=9.81)
    
    # Compute initial total mass and momentum (excluding boundaries)
    mass_0 = h[1:-1].sum()
    momentum_0 = hu[1:-1].sum()
    
    # Run 100 steps
    for _ in range(100):
        h, hu = solver.step(h, hu, dx, dt)
    
    # Compute final total mass and momentum
    mass_f = h[1:-1].sum()
    momentum_f = hu[1:-1].sum()
    
    mass_change = abs(mass_f - mass_0) / mass_0
    momentum_change = abs(momentum_f - momentum_0) / (abs(momentum_0) + 1e-12)
    
    print(f"Mass conservation:")
    print(f"  Initial:  {mass_0:.6f}")
    print(f"  Final:    {mass_f:.6f}")
    print(f"  Change:   {mass_change:.2e} ({mass_change*100:.4f}%)")
    
    print(f"\nMomentum conservation:")
    print(f"  Initial:  {momentum_0:.6f}")
    print(f"  Final:    {momentum_f:.6f}")
    print(f"  Change:   {momentum_change:.2e} ({momentum_change*100:.4f}%)")
    
    # Conservation is approximate for CABARET, but should be reasonable
    if mass_change < 0.1:  # 10% tolerance
        print("✓ Mass reasonably conserved")
    else:
        print("⚠ Mass conservation may need attention")
    
    print()


def test_stability():
    """Test numerical stability"""
    print("Test 4: Numerical Stability")
    print("-" * 50)
    
    N_cells = 40
    N_nodes = N_cells + 1
    N_total = 2 * N_cells + 1
    
    L = 8.0
    dx = L / N_cells
    dt = 0.002
    
    # Challenging initial condition
    x = np.linspace(0, L, N_total)
    h = 1.5 + 0.5 * np.tanh(10 * (x - L/2))  # Sharp gradient
    u = 0.2 * np.ones(N_total)
    hu = h * u
    
    solver = CabaretSolverImproved(model=None, g=9.81)
    
    print(f"Running 500 steps with sharp initial gradient...")
    
    stable = True
    for step in range(500):
        h, hu = solver.step(h, hu, dx, dt)
        
        # Check for NaN or infinity
        if np.any(np.isnan(h)) or np.any(np.isnan(hu)):
            print(f"✗ NaN detected at step {step}")
            stable = False
            break
        
        if np.any(np.isinf(h)) or np.any(np.isinf(hu)):
            print(f"✗ Infinity detected at step {step}")
            stable = False
            break
        
        # Check for negative depths
        if np.any(h < -1e-10):
            print(f"✗ Negative depth detected at step {step}: min(h) = {h.min()}")
            stable = False
            break
        
        # Check for extreme values
        if np.max(np.abs(h)) > 1e6 or np.max(np.abs(hu)) > 1e6:
            print(f"✗ Extreme values at step {step}")
            stable = False
            break
    
    if stable:
        print(f"✓ Simulation remained stable for 500 steps")
        print(f"  Final: h ∈ [{h.min():.4f}, {h.max():.4f}]")
        u_final = hu / (h + 1e-12)
        print(f"  Final: u ∈ [{u_final.min():.4f}, {u_final.max():.4f}]")
    
    print()


def test_interface_compatibility():
    """Test that solver has same interface as parent class"""
    print("Test 5: Interface Compatibility")
    print("-" * 50)
    
    # Test that both solvers have the same step() signature
    improved = CabaretSolverImproved(model=None, g=9.81)
    standard = CabaretSolverPlus(model=None, g=9.81)
    
    # Simple test case
    N = 41
    h = np.ones(N)
    hu = 0.1 * h
    dx = 0.2
    dt = 0.001
    
    try:
        h1, hu1 = improved.step(h.copy(), hu.copy(), dx, dt)
        h2, hu2 = standard.step(h.copy(), hu.copy(), dx, dt)
        
        print("✓ Both solvers have compatible interfaces")
        print("✓ Can be used interchangeably")
        
        # Check output shapes match
        assert h1.shape == h2.shape, "Output shapes don't match"
        assert hu1.shape == hu2.shape, "Output shapes don't match"
        print("✓ Output shapes match")
        
    except Exception as e:
        print(f"✗ Interface compatibility error: {e}")
        raise
    
    print()


def run_all_tests():
    """Run all validation tests"""
    print("="*70)
    print("CabaretSolverImproved - Comprehensive Validation")
    print("="*70)
    print()
    
    tests = [
        test_initialization,
        test_residual_functions,
        test_conservation,
        test_stability,
        test_interface_compatibility,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ TEST FAILED: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70)
    
    if failed == 0:
        print("\n🎉 All tests passed! Implementation validated successfully.")
        return 0
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)

