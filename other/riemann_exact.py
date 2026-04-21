"""
Exact Riemann solver for the 1D shallow water equations.

Strictly follows:
    "Exact solution of the Riemann problem for the shallow water equations"
    by Tom Kent, MATHS-CDT, University of Leeds (2013/14).

Solves the system (eq. 1–2):
    ∂h/∂t  + ∂(hu)/∂x = 0
    ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = 0

with piecewise constant initial data (eq. 3):
    (h, u) = (h_L, u_L)  if x < 0
    (h, u) = (h_R, u_R)  if x > 0

The solution consists of:
    - A left wave  (shock or rarefaction)
    - A star region (h*, u*)
    - A right wave (shock or rarefaction)
separated by a contact discontinuity at speed u*.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum


class WaveType(Enum):
    SHOCK = "shock"
    RAREFACTION = "rarefaction"


# ============================================================
# Section 2.1 — Shock waves (Rankine-Hugoniot jump conditions)
# ============================================================

def _shock_function(h_star: float, h_K: float, g: float) -> float:
    """
    Shock branch of f_K (eq. 18):
        f_K = (h* - h_K) * sqrt( g/2 * (1/h* + 1/h_K) )

    Valid when h* >= h_K.
    """
    return (h_star - h_K) * np.sqrt(0.5 * g * (1.0 / h_star + 1.0 / h_K))


def _shock_function_deriv(h_star: float, h_K: float, g: float) -> float:
    """
    Derivative of shock branch d(f_K)/d(h*).

    From differentiating eq. 18:
        Q = g/2 * (1/h* + 1/h_K)
        f_K = (h* - h_K) * sqrt(Q)
        df_K/dh* = sqrt(Q) + (h* - h_K) * dQ/dh* / (2*sqrt(Q))
    where dQ/dh* = -g / (2 h*²).
    """
    Q = 0.5 * g * (1.0 / h_star + 1.0 / h_K)
    sqrt_Q = np.sqrt(Q)
    dQ = -0.5 * g / (h_star ** 2)
    return sqrt_Q + (h_star - h_K) * dQ / (2.0 * sqrt_Q)


# ============================================================
# Section 2.2 — Rarefaction waves (Riemann invariants)
# ============================================================

def _rarefaction_function(h_star: float, h_K: float, g: float) -> float:
    """
    Rarefaction branch of f_K (eq. 28):
        f_K = 2 * ( sqrt(g*h*) - sqrt(g*h_K) )

    Valid when h* < h_K.
    """
    return 2.0 * (np.sqrt(g * h_star) - np.sqrt(g * h_K))


def _rarefaction_function_deriv(h_star: float, h_K: float, g: float) -> float:
    """
    Derivative of rarefaction branch d(f_K)/d(h*):
        df_K/dh* = sqrt(g / h*)
    """
    return np.sqrt(g / h_star)


# ============================================================
# Section 2.3 — Combined wave function f_K
# ============================================================

def _f_K(h_star: float, h_K: float, g: float) -> float:
    """
    Wave function f_K(h*, h_K) — eq. 30:
        f_K = shock_function    if h* >= h_K  (shock)
        f_K = rarefaction_func  if h* <  h_K  (rarefaction)
    """
    if h_star >= h_K:
        return _shock_function(h_star, h_K, g)
    else:
        return _rarefaction_function(h_star, h_K, g)


def _df_K(h_star: float, h_K: float, g: float) -> float:
    """Derivative d(f_K)/d(h*)."""
    if h_star >= h_K:
        return _shock_function_deriv(h_star, h_K, g)
    else:
        return _rarefaction_function_deriv(h_star, h_K, g)


# ============================================================
# Section 2.4 — Finding h* via Newton–Raphson iteration
# ============================================================

def _phi(h_star: float, h_L: float, h_R: float, u_L: float, u_R: float,
         g: float) -> float:
    """
    Residual function φ(h*) — eq. 34:
        φ(h*) = f_L(h*, h_L) + f_R(h*, h_R) + (u_R - u_L) = 0
    """
    return _f_K(h_star, h_L, g) + _f_K(h_star, h_R, g) + (u_R - u_L)


def _dphi(h_star: float, h_L: float, h_R: float, g: float) -> float:
    """Derivative φ'(h*) = f_L'(h*) + f_R'(h*)."""
    return _df_K(h_star, h_L, g) + _df_K(h_star, h_R, g)


def _initial_guess(h_L: float, h_R: float, u_L: float, u_R: float,
                   g: float) -> float:
    """
    Two-rarefaction approximation for initial guess h*₀ — eq. 35:
        h*₀ = (1/g) * [ ½(c_L + c_R) + ¼(u_L - u_R) ]²

    where c_K = sqrt(g * h_K).  This is exact if both waves turn out to
    be rarefactions and provides a good starting point otherwise.
    """
    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)
    h0 = (1.0 / g) * (0.5 * (c_L + c_R) + 0.25 * (u_L - u_R)) ** 2
    return max(h0, 1e-12)  # guard against zero


def _find_h_star(h_L: float, h_R: float, u_L: float, u_R: float,
                 g: float, tol: float, max_iter: int) -> tuple[float, int]:
    """
    Newton–Raphson iteration to solve φ(h*) = 0 (Section 2.4).

    Returns (h_star, iterations).
    """
    h_star = _initial_guess(h_L, h_R, u_L, u_R, g)

    for k in range(1, max_iter + 1):
        phi_val = _phi(h_star, h_L, h_R, u_L, u_R, g)
        dphi_val = _dphi(h_star, h_L, h_R, g)

        if abs(dphi_val) < 1e-30:
            break

        dh = phi_val / dphi_val
        h_new = h_star - dh

        # Positivity guard
        if h_new <= 0:
            h_new = h_star * 0.5

        # Convergence check (relative change)
        if abs(dh) / (0.5 * (h_star + h_new) + 1e-30) < tol:
            return h_new, k

        h_star = h_new

    raise RuntimeError(
        f"Newton iteration did not converge after {max_iter} iterations. "
        f"h_L={h_L}, u_L={u_L}, h_R={h_R}, u_R={u_R}, last h*={h_star}"
    )


# ============================================================
# Solution data structure & sampling (Section 3)
# ============================================================

@dataclass
class RiemannSolution:
    """
    Complete solution to the Riemann problem.

    Stores the star-region state (h*, u*), the original left/right states,
    the wave types, and all wave speeds needed to reconstruct the full
    self-similar solution u(x/t).
    """
    # Star region (Section 2.3–2.4)
    h_star: float
    u_star: float

    # Input states
    h_L: float
    u_L: float
    h_R: float
    u_R: float

    # Wave types
    left_wave: WaveType
    right_wave: WaveType

    # Left wave speeds
    S_L: float | None = None         # left shock speed  (eq. 14)
    S_L_head: float | None = None    # left rarefaction head  = u_L - c_L
    S_L_tail: float | None = None    # left rarefaction tail  = u* - c*

    # Right wave speeds
    S_R: float | None = None         # right shock speed (eq. 16)
    S_R_head: float | None = None    # right rarefaction head = u_R + c_R
    S_R_tail: float | None = None    # right rarefaction tail = u* + c*

    # Solver metadata
    iterations: int = 0
    g: float = 9.81

    # ----------------------------------------------------------
    # Sampling (Section 3, Fig. 3–5)
    # ----------------------------------------------------------

    def sample(self, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the exact solution at positions *x* and time *t* > 0.

        Uses the similarity variable ξ = x/t to select the correct region
        of the wave pattern:

            left state → [left wave] → star region → [right wave] → right state

        Parameters
        ----------
        x : array_like
            Spatial positions.
        t : float
            Time (must be > 0).

        Returns
        -------
        h, u : ndarray, ndarray
            Water depth and velocity at each point.
        """
        if t <= 0:
            raise ValueError("Time t must be positive.")

        x = np.asarray(x, dtype=float)
        xi = x / t                       # similarity variable

        h = np.empty_like(x)
        u = np.empty_like(x)

        for i in range(xi.size):
            h[i], u[i] = self._sample_point(xi[i])

        return h, u

    def _sample_point(self, xi: float) -> tuple[float, float]:
        """
        Sample at a single similarity variable ξ = x/t.

        Logic follows Section 3 / Fig. 3–5 of the paper.
        """
        g = self.g

        # ---- Left wave ----
        if self.left_wave == WaveType.SHOCK:
            # Shock: single discontinuity at speed S_L (eq. 14)
            if xi < self.S_L:
                return self.h_L, self.u_L
        else:
            # Rarefaction fan: head at S_L_head, tail at S_L_tail
            if xi < self.S_L_head:
                return self.h_L, self.u_L
            elif xi < self.S_L_tail:
                # Inside the left rarefaction fan (eq. 39–40)
                c_L = np.sqrt(g * self.h_L)
                h_fan = (1.0 / (9.0 * g)) * (self.u_L + 2.0 * c_L - xi) ** 2
                u_fan = (1.0 / 3.0) * (self.u_L + 2.0 * c_L + 2.0 * xi)
                return h_fan, u_fan

        # ---- Right wave ----
        if self.right_wave == WaveType.SHOCK:
            if xi >= self.S_R:
                return self.h_R, self.u_R
        else:
            if xi >= self.S_R_head:
                return self.h_R, self.u_R
            elif xi >= self.S_R_tail:
                # Inside the right rarefaction fan (eq. 39–40, mirrored)
                c_R = np.sqrt(g * self.h_R)
                h_fan = (1.0 / (9.0 * g)) * (-self.u_R + 2.0 * c_R + xi) ** 2
                u_fan = (1.0 / 3.0) * (self.u_R - 2.0 * c_R + 2.0 * xi)
                return h_fan, u_fan

        # ---- Star region ----
        return self.h_star, self.u_star

    def __repr__(self) -> str:
        lw = self.left_wave.value
        rw = self.right_wave.value
        return (f"RiemannSolution({lw}–{rw}, "
                f"h*={self.h_star:.6f}, u*={self.u_star:.6f}, "
                f"iters={self.iterations})")


# ============================================================
# Main solver interface
# ============================================================

def solve(h_L: float, u_L: float, h_R: float, u_R: float,
          g: float = 9.81, tol: float = 1e-10,
          max_iter: int = 100) -> RiemannSolution:
    """
    Solve the exact Riemann problem for the 1D shallow water equations.

    Implements the full algorithm from the Kent paper:
        1. Initial guess h*₀ via two-rarefaction approx (eq. 35).
        2. Newton–Raphson iteration on φ(h*) = 0 (eq. 34).
        3. Compute u* from eq. 36.
        4. Classify left/right waves as shock or rarefaction.
        5. Compute all wave speeds.

    Parameters
    ----------
    h_L, u_L : float
        Left state (water depth and velocity), h_L > 0.
    h_R, u_R : float
        Right state (water depth and velocity), h_R > 0.
    g : float
        Gravitational acceleration (default 9.81 m/s²).
    tol : float
        Relative convergence tolerance for Newton iteration.
    max_iter : int
        Maximum Newton iterations.

    Returns
    -------
    RiemannSolution
        Solution object; call ``sol.sample(x, t)`` to evaluate.
    """
    if h_L <= 0 or h_R <= 0:
        raise ValueError(
            f"Water depths must be positive: h_L={h_L}, h_R={h_R}. "
            f"Dry-bed cases are outside the scope of the paper."
        )

    c_L = np.sqrt(g * h_L)
    c_R = np.sqrt(g * h_R)

    # Vacuum check: if the states separate faster than the fastest
    # rarefaction waves can fill in, no wet-bed solution h* > 0 exists.
    #   u_R - u_L >= 2(c_L + c_R)  →  vacuum forms in the star region
    # This is the necessary condition for the two-rarefaction initial
    # guess (eq. 35) to be positive, and for φ(h*) = 0 to have a root.
    if u_R - u_L >= 2.0 * (c_L + c_R):
        raise ValueError(
            f"Vacuum / dry-bed formation detected: "
            f"u_R - u_L = {u_R - u_L:.4f} >= 2(c_L + c_R) = {2*(c_L + c_R):.4f}. "
            f"No wet-bed solution exists for these inputs. "
            f"The paper only covers the wet-bed Riemann problem."
        )

    # --- Step 1–2: find h* (Section 2.4) ---
    h_star, iters = _find_h_star(h_L, h_R, u_L, u_R, g, tol, max_iter)

    # --- Step 3: compute u* (eq. 36) ---
    #   u* = ½(u_L + u_R) + ½(f_R - f_L)
    f_L = _f_K(h_star, h_L, g)
    f_R = _f_K(h_star, h_R, g)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R - f_L)

    # --- Step 4: classify waves (Section 2.3) ---
    #   h* >= h_K  →  shock  ;  h* < h_K  →  rarefaction
    left_wave = WaveType.SHOCK if h_star >= h_L else WaveType.RAREFACTION
    right_wave = WaveType.SHOCK if h_star >= h_R else WaveType.RAREFACTION

    c_star = np.sqrt(g * h_star)

    # --- Step 5: wave speeds ---
    sol = RiemannSolution(
        h_star=h_star, u_star=u_star,
        h_L=h_L, u_L=u_L, h_R=h_R, u_R=u_R,
        left_wave=left_wave, right_wave=right_wave,
        iterations=iters, g=g,
    )

    # Left wave
    if left_wave == WaveType.SHOCK:
        # Shock speed from Rankine-Hugoniot (eq. 14):
        #   S_L = u_L - q_L,  q_L = sqrt( g*h*(h*+h_L) / (2*h_L) )
        q_L = np.sqrt(0.5 * g * h_star * (h_star + h_L)) / h_L
        sol.S_L = u_L - q_L
    else:
        # Rarefaction (Section 2.2):
        #   head speed = u_L - c_L   (leading edge, into undisturbed state)
        #   tail speed = u* - c*     (trailing edge, at star region)
        sol.S_L_head = u_L - c_L
        sol.S_L_tail = u_star - c_star

    # Right wave
    if right_wave == WaveType.SHOCK:
        # Shock speed (eq. 16):
        #   S_R = u_R + q_R,  q_R = sqrt( g*h*(h*+h_R) / (2*h_R) )
        q_R = np.sqrt(0.5 * g * h_star * (h_star + h_R)) / h_R
        sol.S_R = u_R + q_R
    else:
        # Rarefaction:
        #   tail speed = u* + c*
        #   head speed = u_R + c_R
        sol.S_R_tail = u_star + c_star
        sol.S_R_head = u_R + c_R

    return sol


def solve_profile(h_L: float, u_L: float, h_R: float, u_R: float,
                  x_min: float = -10.0, x_max: float = 10.0,
                  nx: int = 1000, t: float = 1.0,
                  g: float = 9.81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: solve and return arrays (x, h, u) on a uniform grid.
    """
    sol = solve(h_L, u_L, h_R, u_R, g=g)
    x = np.linspace(x_min, x_max, nx)
    h, u = sol.sample(x, t)
    return x, h, u


# ============================================================
# Plotting
# ============================================================

def plot_solution(sol: RiemannSolution, t: float = 1.0,
                  x_min: float = -10.0, x_max: float = 10.0,
                  nx: int = 1000, title: str = None):
    """Plot h, u, hu profiles at time t."""
    import matplotlib.pyplot as plt

    x = np.linspace(x_min, x_max, nx)
    h, u = sol.sample(x, t)
    hu = h * u

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x, h, 'b-', lw=1.5)
    axes[0].set(xlabel='x', ylabel='h', title='Water depth h')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, u, 'r-', lw=1.5)
    axes[1].set(xlabel='x', ylabel='u', title='Velocity u')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, hu, 'g-', lw=1.5)
    axes[2].set(xlabel='x', ylabel='hu', title='Momentum hu')
    axes[2].grid(True, alpha=0.3)

    if title is None:
        lw = sol.left_wave.value
        rw = sol.right_wave.value
        title = (f"t={t}:  {lw} – {rw}\n"
                 f"(h_L={sol.h_L}, u_L={sol.u_L}, "
                 f"h_R={sol.h_R}, u_R={sol.u_R})")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# Test cases from the paper (Section 3)
# ============================================================

if __name__ == "__main__":
    g = 9.81

    print("=" * 65)
    print("Exact Riemann Solver — 1D Shallow Water Equations")
    print("Following: Tom Kent, 'Exact solution of the Riemann problem")
    print("           for the shallow water equations' (2013/14)")
    print("=" * 65)

    # ----------------------------------------------------------
    # Test 1: Dam break  (Section 1.2, Fig. 1–2)
    #   h_L=2, h_R=1, u_L=u_R=0
    #   Expected: left rarefaction, right shock
    # ----------------------------------------------------------
    print("\n[Test 1] Dam break: h_L=2, h_R=1, u=0")
    sol1 = solve(2.0, 0.0, 1.0, 0.0, g=g)
    print(f"  {sol1}")
    print(f"  h* = {sol1.h_star:.8f},  u* = {sol1.u_star:.8f}")

    # ----------------------------------------------------------
    # Test 2: Two shocks  (Section 3.1, eq. 41)
    #   h_L=h_R=1, u_L=2, u_R=0
    #   Expected: left shock, right shock, u*=1 by symmetry
    # ----------------------------------------------------------
    print("\n[Test 2] Two shocks: h_L=h_R=1, u_L=2, u_R=0")
    sol2 = solve(1.0, 2.0, 1.0, 0.0, g=g)
    print(f"  {sol2}")
    print(f"  h* = {sol2.h_star:.8f},  u* = {sol2.u_star:.8f}")

    # ----------------------------------------------------------
    # Test 3: Two rarefactions  (Section 3.2, eq. 42)
    #   h_L=h_R=1, u_L=0, u_R=2
    #   Expected: left rarefaction, right rarefaction, u*=1
    # ----------------------------------------------------------
    print("\n[Test 3] Two rarefactions: h_L=h_R=1, u_L=0, u_R=2")
    sol3 = solve(1.0, 0.0, 1.0, 2.0, g=g)
    print(f"  {sol3}")
    print(f"  h* = {sol3.h_star:.8f},  u* = {sol3.u_star:.8f}")

    # ----------------------------------------------------------
    # Test 4: Mirror of dam break
    #   h_L=1, h_R=2, u=0  →  left shock, right rarefaction
    # ----------------------------------------------------------
    print("\n[Test 4] Mirror dam break: h_L=1, h_R=2, u=0")
    sol4 = solve(1.0, 0.0, 2.0, 0.0, g=g)
    print(f"  {sol4}")
    print(f"  h* = {sol4.h_star:.8f},  u* = {sol4.u_star:.8f}")

    # ----------------------------------------------------------
    # Test 5: Vacuum detection
    #   u_R - u_L = 84.7 >= 2*(c_L + c_R) ≈ 8.55  →  dry bed
    #   The solver should reject this cleanly.
    # ----------------------------------------------------------
    print("\n[Test 5] Vacuum case: h_L=0.1, u_L=-93, h_R=1.1, u_R=-8.3")
    try:
        sol5 = solve(0.1, -93., 1.1, -8.3, g=g)
        print(f"  ERROR: should have raised ValueError!")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    # ----------------------------------------------------------
    # Test 6: Large but valid case (no vacuum)
    #   u_R - u_L = 4.0 < 2*(c_L + c_R) ≈ 8.55  →  still wet
    # ----------------------------------------------------------
    print("\n[Test 6] Large valid: h_L=0.1, u_L=-6, h_R=1.1, u_R=-2")
    sol6 = solve(0.1, -6.0, 1.1, -2.0, g=g)
    print(f"  {sol6}")
    print(f"  h* = {sol6.h_star:.8f},  u* = {sol6.u_star:.8f}")

    # ----------------------------------------------------------
    # Verification
    # ----------------------------------------------------------
    print("\n--- Verification ---")
    print(f"  Test 2  u* ≈ 1.0 :  {abs(sol2.u_star - 1.0) < 1e-8}")
    print(f"  Test 3  u* ≈ 1.0 :  {abs(sol3.u_star - 1.0) < 1e-8}")
    # Tests 1 & 4 should give same h* (mirror symmetry)
    print(f"  Tests 1&4 same h* :  {abs(sol1.h_star - sol4.h_star) < 1e-8}")

    # ----------------------------------------------------------
    # Sample dam-break profile
    # ----------------------------------------------------------
    print("\n--- Dam-break profile at t = 1.0 ---")
    x, h, u = solve_profile(2.0, 0.0, 1.0, 0.0,
                             x_min=-10, x_max=10, nx=20, t=1.0, g=g)
    print(f"  {'x':>8s}  {'h':>10s}  {'u':>10s}")
    for xi, hi, ui in zip(x, h, u):
        print(f"  {xi:8.3f}  {hi:10.6f}  {ui:10.6f}")

    # ----------------------------------------------------------
    # Plots
    # ----------------------------------------------------------
    try:
        for sol in [sol1, sol2, sol3, sol4, sol6]:
            plot_solution(sol, t=1.0, x_min=-10, x_max=10)
    except Exception as e:
        print(f"\n(plotting skipped: {e})")
