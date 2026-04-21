import numpy as np
import sys

from training.simulation import Simulation
from training.solvers import CabaretSolverImproved

# ---------------------------------------------------------------
# Time-reversibility test (Appendix A of the paper)
#
# 1. Start with state (H^0, u^0)
# 2. Run one step forward with dt  ->  get (H^1, u^1)
# 3. Negate velocities: hu^1 -> -hu^1
# 4. Run one step forward with same dt  ->  get (H^2, u^2)
# 5. Check: H^2 == H^0  and  u^2 == -u^0
#
# Monotonization MUST be disabled for this to hold exactly.
# We use a wet test case (no dry beds) to avoid dry_eps effects.
# ---------------------------------------------------------------

out = open('test_output.txt', 'w')

# Use a transonic Riemann problem (sonic point present) so the
# implicit algorithm is actually exercised.
# Dam break: H_L=10, H_R=1, u=0  (paper eq 10)
config = {
    'L': 50, 'nx': 100,
    'h_l': 10.0, 'h_r': 1.0, 'u_l': 0.0, 'u_r': 0.0,
    't_end': 0.01,  # dummy, we do manual stepping
    't_start': 0,
    'init_type': 'A',
}

g = 9.81
solver = CabaretSolverImproved(g=g, monotonize=False, dry_eps=1e-15)
config['solver'] = solver
sim = Simulation(config)

h0 = sim.h.copy()
hu0 = sim.hu.copy()
dx = sim.dx

# Compute a stable dt (CFL = 0.3)
u_vals = np.where(h0 > 1e-12, hu0 / np.maximum(h0, 1e-12), 0.0)
c_vals = np.sqrt(g * np.maximum(0.0, h0))
max_speed = np.max(np.abs(np.concatenate([u_vals + c_vals, u_vals - c_vals])))
dt = 0.3 * dx / max_speed

out.write("dt = %.10f\n" % dt)
out.write("dx = %.10f\n\n" % dx)

# --- Step 1: Forward step ---
h1, hu1 = solver.step(h0, hu0, dx, dt)

# --- Step 2: Negate velocities ---
hu1_neg = -hu1.copy()

# --- Step 3: Backward step with same dt ---
h2, hu2 = solver.step(h1.copy(), hu1_neg, dx, dt)

# --- Step 4: Check ---
# Expected: h2 == h0, hu2 == -hu0
h_err = np.max(np.abs(h2 - h0))
hu_err = np.max(np.abs(hu2 - (-hu0)))

# Also relative errors
h_rel = h_err / (np.max(np.abs(h0)) + 1e-15)
hu_rel = hu_err / (np.max(np.abs(hu0)) + 1e-15)

out.write("=== Time-Reversibility Test (monotonize=False) ===\n")
out.write("max |H^2 - H^0|       = %.4e\n" % h_err)
out.write("max |hu^2 - (-hu^0)|  = %.4e\n" % hu_err)
out.write("relative H error      = %.4e\n" % h_rel)
out.write("relative hu error     = %.4e\n\n" % hu_rel)

# Print a few values around the sonic point for inspection
N = len(h0)
mid = N // 2
window = 6

out.write("  idx        x          H^0          H^2        H^2-H^0\n")
for i in range(mid - window, mid + window + 1):
    out.write("  %3d  %8.3f  %12.8f  %12.8f  %+.4e\n" % (
        i, sim.x[i], h0[i], h2[i], h2[i] - h0[i]))

out.write("\n")
out.write("  idx        x         hu^0        -hu^2      err(hu)\n")
for i in range(mid - window, mid + window + 1):
    out.write("  %3d  %8.3f  %12.8f  %12.8f  %+.4e\n" % (
        i, sim.x[i], hu0[i], -hu2[i], hu2[i] - (-hu0[i])))

# --- Also test WITH monotonization to show the contrast ---
solver_mono = CabaretSolverImproved(g=g, monotonize=True, dry_eps=1e-15)
h1m, hu1m = solver_mono.step(h0.copy(), hu0.copy(), dx, dt)
h2m, hu2m = solver_mono.step(h1m.copy(), -hu1m.copy(), dx, dt)

h_err_m = np.max(np.abs(h2m - h0))
hu_err_m = np.max(np.abs(hu2m - (-hu0)))

out.write("\n=== Time-Reversibility Test (monotonize=True) ===\n")
out.write("max |H^2 - H^0|       = %.4e\n" % h_err_m)
out.write("max |hu^2 - (-hu^0)|  = %.4e\n" % hu_err_m)
out.write("(Expected: larger errors because monotonization breaks reversibility)\n")

out.close()
print("Done")
