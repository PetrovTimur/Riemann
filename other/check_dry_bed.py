#!/usr/bin/env python3
"""
Utility to check for dry bed (vacuum) formation in Riemann problems
"""
import numpy as np

def check_dry_bed(problems, g=9.8066):
    """Check for vacuum/dry bed formation in array of Riemann problems"""
    arr = np.array([[p['hl'], p['hul'], p['hr'], p['hur']] for p in problems])
    u_diff = arr[:, 3] / arr[:, 2] - arr[:, 1] / arr[:, 0]  # u_R - u_L
    c_sum = np.sqrt(g * arr[:, 0]) + np.sqrt(g * arr[:, 2])  # c_L + c_R
    return u_diff > 2 * c_sum  # Boolean array: True = vacuum forming


# Example usage and test
if __name__ == "__main__":
    # Test cases
    problems = [
        # Normal dam break - no vacuum
        {'hl': 2.0, 'hul': 0.0, 'hr': 1.0, 'hur': 0.0},
        
        # The problematic case from earlier - vacuum forming
        {'hl': 0.10531860589981079, 'hul': -9.377503395080566, 
         'hr': 1.1079761981964111, 'hur': -8.32735824584961},
        
        # Diverging flow - vacuum
        {'hl': 1.0, 'hul': -10.0, 'hr': 1.0, 'hur': 20.0},
        
        # Converging flow - no vacuum
        {'hl': 1.0, 'hul': 10.0, 'hr': 1.0, 'hur': -10.0},
    ]
    
    vacuum_mask = check_dry_bed(problems)
    
    print("Dry Bed Detection Results:")
    print("-" * 60)
    for i, (problem, has_vacuum) in enumerate(zip(problems, vacuum_mask)):
        uL = problem['hul'] / problem['hl']
        uR = problem['hur'] / problem['hr']
        cL = np.sqrt(9.8066 * problem['hl'])
        cR = np.sqrt(9.8066 * problem['hr'])
        
        status = "🌵 VACUUM" if has_vacuum else "💧 Normal"
        print(f"Case {i+1}: {status}")
        print(f"  h_L={problem['hl']:.4f}, u_L={uL:.2f}, h_R={problem['hr']:.4f}, u_R={uR:.2f}")
        print(f"  u_R - u_L = {uR - uL:.4f}, 2(c_L + c_R) = {2*(cL + cR):.4f}")
        print()
    
    print(f"Summary: {vacuum_mask.sum()}/{len(problems)} cases have vacuum formation")

