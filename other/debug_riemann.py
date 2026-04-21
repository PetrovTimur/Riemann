#!/usr/bin/env python3
"""
Debug the Riemann solver issue
"""
import numpy as np
import sys
sys.path.insert(0, '/')

from training.solvers import riemann_solver_newton, riemann_solver_scipy

# Problematic case
h_l = 0.10531860589981079
u_l = -9.377503395080566 / 0.10531860589981079
h_r = 1.1079761981964111
u_r = -8.32735824584961 / 1.1079761981964111

hu_l = h_l * u_l
hu_r = h_r * u_r

print("=" * 70)
print("Debugging Riemann Solver Issue")
print("=" * 70)
print(f"\nInput state:")
print(f"  Left:  h_l = {h_l:.6f}, u_l = {u_l:.6f}, hu_l = {hu_l:.6f}")
print(f"  Right: h_r = {h_r:.6f}, u_r = {u_r:.6f}, hu_r = {hu_r:.6f}")

# Compute wave speeds
g = 9.8066
c_l = np.sqrt(g * h_l)
c_r = np.sqrt(g * h_r)
print(f"\nWave speeds:")
print(f"  c_l = {c_l:.6f}")
print(f"  c_r = {c_r:.6f}")

# Check Froude numbers
Fr_l = abs(u_l) / c_l
Fr_r = abs(u_r) / c_r
print(f"\nFroude numbers:")
print(f"  Fr_l = {Fr_l:.6f} ({'supersonic' if Fr_l > 1 else 'subsonic'})")
print(f"  Fr_r = {Fr_r:.6f} ({'supersonic' if Fr_r > 1 else 'subsonic'})")

# Check if this is a dry state problem
if h_l < 1e-6 or h_r < 1e-6:
    print("\n⚠ WARNING: Near-dry state detected!")

# Try manual Newton solver
print("\n" + "=" * 70)
print("Testing Newton Solver (manual)")
print("=" * 70)
try:
    result_newton = riemann_solver_newton(h_l, hu_l, h_r, hu_r, g=g)
    print(f"\nResult:")
    print(f"  h_star = {result_newton['star'][0]:.6f}")
    print(f"  u_star = {result_newton['star'][1]:.6f}")
    print(f"  Iterations: {result_newton['data'][0]}")
    print(f"  Flux: {result_newton['flux']}")
    
    # Check if result is physical
    if result_newton['star'][0] < 0:
        print("  ❌ UNPHYSICAL: Negative depth!")
    elif result_newton['star'][0] > 100:
        print("  ⚠ WARNING: Very large depth (> 100m)")
    if abs(result_newton['star'][1]) > 100:
        print("  ⚠ WARNING: Very large velocity (> 100 m/s)")
        
except Exception as e:
    print(f"❌ Newton solver failed: {e}")
    import traceback
    traceback.print_exc()

# Try scipy solver
print("\n" + "=" * 70)
print("Testing Scipy Solver")
print("=" * 70)
try:
    result_scipy = riemann_solver_scipy(h_l, hu_l, h_r, hu_r, g=g)
    print(f"\nResult:")
    print(f"  h_star = {result_scipy['star'][0]:.6f}")
    print(f"  u_star = {result_scipy['star'][1]:.6f}")
    print(f"  Flux: {result_scipy['flux']}")
    
    # Check if result is physical
    if result_scipy['star'][0] < 0:
        print("  ❌ UNPHYSICAL: Negative depth!")
    elif result_scipy['star'][0] > 100:
        print("  ⚠ WARNING: Very large depth (> 100m)")
    if abs(result_scipy['star'][1]) > 100:
        print("  ⚠ WARNING: Very large velocity (> 100 m/s)")
        
except Exception as e:
    print(f"❌ Scipy solver failed: {e}")
    import traceback
    traceback.print_exc()

# Manual calculation to verify
print("\n" + "=" * 70)
print("Manual Verification")
print("=" * 70)

# Initial guess for c_star
c_initial = 0.25 * (u_l - u_r) + 0.5 * (c_l + c_r)
print(f"\nInitial guess for c_star: {c_initial:.6f}")

# Let's trace through the Newton iteration manually
h_k = np.array([h_l, h_r])
u_k = np.array([u_l, u_r])
c_k = np.array([c_l, c_r])

c = c_initial
print(f"\nNewton iteration trace:")
for iteration in range(10):
    s_k = c / c_k
    phi_k = np.where(s_k >= 1, 
                     ((c - c_k) * (s_k + 1) * np.sqrt(1 + s_k ** -2) / np.sqrt(2)), 
                     2 * (c - c_k))
    dphi_k = np.where(s_k > 1, 
                      ((2 * s_k ** 2 + 1 + s_k ** -2) / (np.sqrt(2) * s_k * np.sqrt(1 + s_k ** -2))), 
                      2)
    
    residual = phi_k.sum() - (u_k[0] - u_k[1])
    gradient = dphi_k.sum()
    
    print(f"  iter {iteration}: c={c:.6f}, residual={residual:.6e}, gradient={gradient:.6f}")
    
    if abs(residual) < 1e-6:
        print(f"  Converged!")
        break
    
    delta = residual / gradient
    c_old = c
    c = c - delta
    
    if abs(c - c_old) < 1e-10:
        print(f"  Converged (delta too small)")
        break

print(f"\nFinal c_star: {c:.6f}")
h_star_manual = c ** 2 / g
u_star_manual = 0.5 * (u_k.sum() + phi_k[1] - phi_k[0])
print(f"Final h_star: {h_star_manual:.6f}")
print(f"Final u_star: {u_star_manual:.6f}")

# Check the residual equation
print(f"\nResidual check:")
print(f"  phi_k[0] = {phi_k[0]:.6f}")
print(f"  phi_k[1] = {phi_k[1]:.6f}")
print(f"  phi_k.sum() = {phi_k.sum():.6f}")
print(f"  u_l - u_r = {u_l - u_r:.6f}")
print(f"  Difference: {abs(phi_k.sum() - (u_l - u_r)):.6e}")

print("\n" + "=" * 70)
print("Diagnosis")
print("=" * 70)

# The issue might be extreme velocities causing numerical issues
if abs(u_l) > 10 * c_l or abs(u_r) > 10 * c_r:
    print("⚠ ISSUE: Extreme supersonic flow detected!")
    print("  The left state is highly supersonic (Fr_l ≈ {:.1f})".format(Fr_l))
    print("  This may cause the Riemann solver to fail or give unphysical results.")
    print("\nPossible causes:")
    print("  1. Input data error (velocity too high)")
    print("  2. Numerical instability in upstream solver")
    print("  3. Vacuum/near-vacuum state forming")

# Check for vacuum condition
if u_r - u_l > 2 * (c_l + c_r):
    print("\n⚠ VACUUM FORMATION:")
    print(f"  u_r - u_l = {u_r - u_l:.6f}")
    print(f"  2(c_l + c_r) = {2 * (c_l + c_r):.6f}")
    print("  A vacuum (dry bed) may be forming between the two states!")
    print("  Standard Riemann solver doesn't handle vacuum states.")

