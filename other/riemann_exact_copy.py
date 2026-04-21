"""
Exact Riemann solver for the 1D shallow water equations.

Based on: "Exact solution of the Riemann problem for the shallow water equations"
          by Tom Kent (2013/14).

Solves the system:
    ∂h/∂t + ∂(hu)/∂x = 0
    ∂(hu)/∂t + ∂(hu² + ½gh²)/∂x = 0

with piecewise constant initial data:
    (h, u) = (h_l, u_l) for x < 0
    (h, u) = (h_r, u_r) for x >= 0
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class WaveType(Enum):
    SHOCK = "shock"
    RAREFACTION = "rarefaction"


@dataclass
class RiemannSolution:
    """Complete solution to the Riemann problem."""
    # Star region values
    h_star: float
    u_star: float

    # Input states
    h_l: float
    u_l: float
    h_r: float
    u_r: float

    # Wave types
    left_wave: WaveType
    right_wave: WaveType

    # Wave speeds
    # Shock: single speed; Rarefaction: head and tail speeds
    S_l: Optional[float] = None        # left shock speed
    S_l_head: Optional[float] = None   # left rarefaction head speed
    S_l_tail: Optional[float] = None   # left rarefaction tail speed
    S_r: Optional[float] = None        # right shock speed
    S_r_head: Optional[float] = None   # right rarefaction head speed
    S_r_tail: Optional[float] = None   # right rarefaction tail speed

    # Dry bed flag
    is_dry_bed: bool = False

    # Iteration count (for Newton solver)
    iterations: int = 0

    # Gravity
    g: float = 9.81

    def sample(self, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample the solution at positions x and time t.

        Parameters
        ----------
        x : array_like
            Spatial positions to evaluate the solution.
        t : float
            Time at which to evaluate the solution (must be > 0).

        Returns
        -------
        h : ndarray
            Water depth at each position.
        u : ndarray
            Velocity at each position.
        """
        if t <= 0:
            raise ValueError("Time t must be positive.")

        x = np.asarray(x, dtype=float)
        S = x / t  # similarity variable

        h = np.empty_like(x)
        u = np.empty_like(x)

        if self.is_dry_bed:
            return self._sample_dry_bed(S, h, u)

        for i, s in enumerate(S):
            h[i], u[i] = self._sample_point(s)

        return h, u

    def _sample_point(self, s: float) -> tuple[float, float]:
        """Sample solution at a single point with similarity variable s = x/t."""
        g = self.g

        # --- Left wave ---
        if self.left_wave == WaveType.SHOCK:
            # Left of shock: left state; right of shock: star state
            if s < self.S_l:
                return self.h_l, self.u_l
            # Between left shock and contact (or right wave)
        else:
            # Left rarefaction
            if s < self.S_l_head:
                return self.h_l, self.u_l
            elif s < self.S_l_tail:
                # Inside left rarefaction fan (eq. 39, 40)
                h_fan = (1.0 / (9.0 * g)) * (self.u_l + 2.0 * np.sqrt(g * self.h_l) - s) ** 2
                u_fan = self.u_l + (2.0 / 3.0) * (s - self.u_l + np.sqrt(g * self.h_l))
                return h_fan, u_fan

        # --- Right wave ---
        if self.right_wave == WaveType.SHOCK:
            if s >= self.S_r:
                return self.h_r, self.u_r
        else:
            # Right rarefaction
            if s >= self.S_r_head:
                return self.h_r, self.u_r
            elif s >= self.S_r_tail:
                # Inside right rarefaction fan (eq. 39, 40)
                h_fan = (1.0 / (9.0 * g)) * (s - self.u_r + 2.0 * np.sqrt(g * self.h_r)) ** 2
                u_fan = self.u_r + (2.0 / 3.0) * (s - self.u_r - np.sqrt(g * self.h_r))
                return h_fan, u_fan

        # --- Star region ---
        return self.h_star, self.u_star

    def _sample_dry_bed(self, S: np.ndarray, h: np.ndarray, u: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the solution for dry-bed (vacuum) cases."""
        g = self.g

        # Both sides dry
        if self.h_l <= 0 and self.h_r <= 0:
            h[:] = 0.0
            u[:] = 0.0
            return h, u

        # Left dry, right wet
        if self.h_l <= 0:
            c_r = np.sqrt(g * self.h_r)
            for i, s in enumerate(S):
                if s >= self.u_r + c_r:
                    h[i], u[i] = self.h_r, self.u_r
                elif s >= self.u_r - 2.0 * c_r:
                    # Inside right rarefaction fan connecting vacuum to right state
                    h[i] = (1.0 / (9.0 * g)) * (s - self.u_r + 2.0 * c_r) ** 2
                    u[i] = self.u_r + (2.0 / 3.0) * (s - self.u_r - c_r)
                else:
                    h[i], u[i] = 0.0, 0.0
            return h, u

        # Right dry, left wet
        if self.h_r <= 0:
            c_l = np.sqrt(g * self.h_l)
            for i, s in enumerate(S):
                if s <= self.u_l - c_l:
                    h[i], u[i] = self.h_l, self.u_l
                elif s <= self.u_l + 2.0 * c_l:
                    # Inside left rarefaction fan connecting left state to vacuum
                    h[i] = (1.0 / (9.0 * g)) * (self.u_l + 2.0 * c_l - s) ** 2
                    u[i] = self.u_l + (2.0 / 3.0) * (s - self.u_l + c_l)
                else:
                    h[i], u[i] = 0.0, 0.0
            return h, u

        # Middle vacuum: both sides wet but waves create a vacuum in the middle
        c_l = np.sqrt(g * self.h_l)
        c_r = np.sqrt(g * self.h_r)
        # Left rarefaction: head = u_l - c_l, tail = u_l + 2*c_l
        # Right rarefaction: tail = u_r - 2*c_r, head = u_r + c_r
        for i, s in enumerate(S):
            if s <= self.u_l - c_l:
                h[i], u[i] = self.h_l, self.u_l
            elif s <= self.u_l + 2.0 * c_l:
                h[i] = (1.0 / (9.0 * g)) * (self.u_l + 2.0 * c_l - s) ** 2
                u[i] = self.u_l + (2.0 / 3.0) * (s - self.u_l + c_l)
            elif s < self.u_r - 2.0 * c_r:
                h[i], u[i] = 0.0, 0.0
            elif s <= self.u_r + c_r:
                h[i] = (1.0 / (9.0 * g)) * (s - self.u_r + 2.0 * c_r) ** 2
                u[i] = self.u_r + (2.0 / 3.0) * (s - self.u_r - c_r)
            else:
                h[i], u[i] = self.h_r, self.u_r
        return h, u

    def __repr__(self) -> str:
        left = self.left_wave.value
        right = self.right_wave.value
        dry = " [DRY BED]" if self.is_dry_bed else ""
        return (f"RiemannSolution({left}-{right}{dry}, "
                f"h*={self.h_star:.6f}, u*={self.u_star:.6f}, "
                f"iters={self.iterations})")


def _f_k(h_star: float, h_k: float, g: float) -> float:
    """
    Evaluate the wave function f_k(h*, h_k) for side k (eq. 30).

    Rarefaction if h* < h_k, shock if h* >= h_k.
    """
    if h_star < h_k:
        # Rarefaction (eq. 28)
        return 2.0 * np.sqrt(g) * (np.sqrt(h_star) - np.sqrt(h_k))
    else:
        # Shock (eq. 18)
        return (h_star - h_k) * np.sqrt(0.5 * g * (1.0 / h_star + 1.0 / h_k))


def _df_k(h_star: float, h_k: float, g: float) -> float:
    """
    Derivative of f_k with respect to h* (for Newton iteration).
    """
    if h_star < h_k:
        # d/dh* of rarefaction branch
        return np.sqrt(g / h_star)
    else:
        # d/dh* of shock branch
        A = 0.5 * g * (1.0 / h_star + 1.0 / h_k)
        return np.sqrt(A / 2.0) * (2.0 - (h_star - h_k) / (2.0 * h_star)) \
               if A > 0 else 0.0


def _df_k_exact(h_star: float, h_k: float, g: float) -> float:
    """
    Exact derivative of f_k with respect to h* (for Newton iteration).
    """
    if h_star < h_k:
        # Rarefaction: f = 2*sqrt(g)*(sqrt(h*) - sqrt(h_k))
        # df/dh* = 2*sqrt(g) * 1/(2*sqrt(h*)) = sqrt(g/h*)
        return np.sqrt(g / h_star)
    else:
        # Shock: f = (h* - h_k) * sqrt(g/2 * (1/h* + 1/h_k))
        # Let Q = g/2 * (1/h* + 1/h_k)
        Q = 0.5 * g * (1.0 / h_star + 1.0 / h_k)
        sqrtQ = np.sqrt(Q)
        # dQ/dh* = -g/(2 h*^2)
        dQ = -0.5 * g / (h_star ** 2)
        # df/dh* = sqrt(Q) + (h* - h_k) * dQ / (2*sqrt(Q))
        return sqrtQ + (h_star - h_k) * dQ / (2.0 * sqrtQ)


def _phi(h_star: float, h_l: float, h_r: float, u_l: float, u_r: float, g: float) -> float:
    """
    Residual function phi(h*) = f_l(h*, h_l) + f_r(h*, h_r) + u_r - u_l = 0 (eq. 34).
    """
    return _f_k(h_star, h_l, g) + _f_k(h_star, h_r, g) + (u_r - u_l)


def _dphi(h_star: float, h_l: float, h_r: float, g: float) -> float:
    """
    Derivative of phi w.r.t. h*.
    """
    return _df_k_exact(h_star, h_l, g) + _df_k_exact(h_star, h_r, g)


def _initial_guess(h_l: float, h_r: float, u_l: float, u_r: float, g: float) -> float:
    """
    Compute initial guess for h* using the two-rarefaction approximation (eq. 35).

    This provides a good starting point for Newton's method.
    """
    c_l = np.sqrt(g * h_l)
    c_r = np.sqrt(g * h_r)

    # Two-rarefaction approximation (eq. 35)
    h0 = (1.0 / (16.0 * g)) * (u_l - u_r + 2.0 * (c_l + c_r)) ** 2

    # Clamp to avoid h* = 0 (which causes division issues in shock formula)
    h0 = max(h0, 1e-12)
    return h0


def solve(h_l: float, u_l: float, h_r: float, u_r: float,
          g: float = 9.81, tol: float = 1e-10, max_iter: int = 100) -> RiemannSolution:
    """
    Solve the exact Riemann problem for the 1D shallow water equations.

    Parameters
    ----------
    h_l : float
        Left water depth (>= 0).
    u_l : float
        Left velocity.
    h_r : float
        Right water depth (>= 0).
    u_r : float
        Right velocity.
    g : float
        Gravitational acceleration (default 9.81).
    tol : float
        Convergence tolerance for Newton iteration.
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    RiemannSolution
        Complete solution that can be sampled at any (x, t).

    Raises
    ------
    ValueError
        If depths are negative.
    RuntimeError
        If Newton iteration fails to converge.
    """
    if h_l < 0 or h_r < 0:
        raise ValueError(f"Water depths must be non-negative: h_l={h_l}, h_r={h_r}")

    c_l = np.sqrt(g * h_l) if h_l > 0 else 0.0
    c_r = np.sqrt(g * h_r) if h_r > 0 else 0.0

    # ===== Handle dry bed (vacuum) cases =====

    # Case 1: Both sides dry
    if h_l <= 0 and h_r <= 0:
        return RiemannSolution(
            h_star=0.0, u_star=0.0,
            h_l=h_l, u_l=u_l, h_r=h_r, u_r=u_r,
            left_wave=WaveType.RAREFACTION, right_wave=WaveType.RAREFACTION,
            is_dry_bed=True, g=g,
        )

    # Case 2: Left side dry (h_l = 0)
    if h_l <= 0:
        return RiemannSolution(
            h_star=0.0, u_star=u_r - 2.0 * c_r,
            h_l=h_l, u_l=u_l, h_r=h_r, u_r=u_r,
            left_wave=WaveType.RAREFACTION, right_wave=WaveType.RAREFACTION,
            is_dry_bed=True, g=g,
        )

    # Case 3: Right side dry (h_r = 0)
    if h_r <= 0:
        return RiemannSolution(
            h_star=0.0, u_star=u_l + 2.0 * c_l,
            h_l=h_l, u_l=u_l, h_r=h_r, u_r=u_r,
            left_wave=WaveType.RAREFACTION, right_wave=WaveType.RAREFACTION,
            is_dry_bed=True, g=g,
        )

    # Case 4: Middle vacuum generation
    # Vacuum forms when: u_r - u_l >= 2*(c_l + c_r)
    if u_r - u_l >= 2.0 * (c_l + c_r):
        return RiemannSolution(
            h_star=0.0, u_star=0.5 * (u_l + u_r),
            h_l=h_l, u_l=u_l, h_r=h_r, u_r=u_r,
            left_wave=WaveType.RAREFACTION, right_wave=WaveType.RAREFACTION,
            S_l_head=u_l - c_l, S_l_tail=u_l + 2.0 * c_l,
            S_r_tail=u_r - 2.0 * c_r, S_r_head=u_r + c_r,
            is_dry_bed=True, g=g,
        )

    # ===== Wet-bed Riemann problem: find h* via Newton iteration =====

    h_star = _initial_guess(h_l, h_r, u_l, u_r, g)
    iterations = 0

    for k in range(max_iter):
        iterations = k + 1
        phi_val = _phi(h_star, h_l, h_r, u_l, u_r, g)
        dphi_val = _dphi(h_star, h_l, h_r, g)

        if abs(dphi_val) < 1e-30:
            break

        dh = phi_val / dphi_val
        h_star_new = h_star - dh

        # Ensure positivity
        if h_star_new <= 0:
            h_star_new = h_star * 0.5

        if abs(dh) / (0.5 * (h_star + h_star_new) + 1e-30) < tol:
            h_star = h_star_new
            break

        h_star = h_star_new
    else:
        raise RuntimeError(
            f"Newton iteration did not converge after {max_iter} iterations. "
            f"h_l={h_l}, u_l={u_l}, h_r={h_r}, u_r={u_r}, last h*={h_star}"
        )

    # ===== Compute u* from h* (eq. 36) =====
    f_l = _f_k(h_star, h_l, g)
    f_r = _f_k(h_star, h_r, g)
    u_star = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)

    # ===== Determine wave types (section 2.4) =====
    left_wave = WaveType.SHOCK if h_star >= h_l else WaveType.RAREFACTION
    right_wave = WaveType.SHOCK if h_star >= h_r else WaveType.RAREFACTION

    # ===== Compute wave speeds =====
    sol = RiemannSolution(
        h_star=h_star, u_star=u_star,
        h_l=h_l, u_l=u_l, h_r=h_r, u_r=u_r,
        left_wave=left_wave, right_wave=right_wave,
        iterations=iterations, g=g,
    )

    # Left wave speeds
    if left_wave == WaveType.SHOCK:
        # Shock speed (eq. 14)
        q_l = np.sqrt(0.5 * g * h_star * (h_star + h_l)) / h_l
        sol.S_l = u_l - q_l
    else:
        # Rarefaction head and tail speeds (section 2.2)
        sol.S_l_head = u_l - c_l
        sol.S_l_tail = u_star - np.sqrt(g * h_star)

    # Right wave speeds
    if right_wave == WaveType.SHOCK:
        # Shock speed (eq. 16)
        q_r = np.sqrt(0.5 * g * h_star * (h_star + h_r)) / h_r
        sol.S_r = u_r + q_r
    else:
        # Rarefaction head and tail speeds (section 2.2)
        sol.S_r_tail = u_star + np.sqrt(g * h_star)
        sol.S_r_head = u_r + c_r

    return sol


def solve_profile(h_l: float, u_l: float, h_r: float, u_r: float,
                  x_min: float = -10.0, x_max: float = 10.0,
                  nx: int = 1000, t: float = 1.0,
                  g: float = 9.81) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the Riemann problem and return the solution profile.

    Parameters
    ----------
    h_l, u_l : float
        Left state (depth, velocity).
    h_r, u_r : float
        Right state (depth, velocity).
    x_min, x_max : float
        Spatial domain.
    nx : int
        Number of spatial points.
    t : float
        Time to evaluate at.
    g : float
        Gravitational acceleration.

    Returns
    -------
    x : ndarray
        Spatial coordinates.
    h : ndarray
        Water depth at each point.
    u : ndarray
        Velocity at each point.
    """
    sol = solve(h_l, u_l, h_r, u_r, g=g)
    x = np.linspace(x_min, x_max, nx)
    h, u = sol.sample(x, t)
    return x, h, u


# ============================================================
# Plotting utility
# ============================================================

def plot_solution(sol: RiemannSolution, t: float = 1.0,
                  x_min: float = -10.0, x_max: float = 10.0,
                  nx: int = 1000, title: str = None):
    """
    Plot the Riemann solution at time t.

    Parameters
    ----------
    sol : RiemannSolution
        Solution object from solve().
    t : float
        Time to plot.
    x_min, x_max : float
        Spatial domain.
    nx : int
        Number of points.
    title : str, optional
        Plot title.
    """
    import matplotlib.pyplot as plt

    x = np.linspace(x_min, x_max, nx)
    h, u = sol.sample(x, t)
    hu = h * u

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(x, h, 'b-', linewidth=1.5)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('h')
    axes[0].set_title('Water depth h(x, t)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, u, 'r-', linewidth=1.5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('u')
    axes[1].set_title('Velocity u(x, t)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, hu, 'g-', linewidth=1.5)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('hu')
    axes[2].set_title('Momentum hu(x, t)')
    axes[2].grid(True, alpha=0.3)

    if title is None:
        lw = sol.left_wave.value
        rw = sol.right_wave.value
        dry = " [dry bed]" if sol.is_dry_bed else ""
        title = (f"Riemann solution at t={t}: {lw}-{rw}{dry}\n"
                 f"(h_l={sol.h_l}, u_l={sol.u_l}, h_r={sol.h_r}, u_r={sol.u_r})")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================
# Main: run test cases from the paper
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Exact Riemann Solver for 1D Shallow Water Equations")
    print("=" * 70)

    g = 9.81

    # Test 1: Dam break (Section 1.2, Fig. 1-2)
    # h_l=2, h_r=1, u_l=u_r=0 → left rarefaction, right shock
    print("\n--- Test 1: Dam break (h_l=2, h_r=1, u_l=u_r=0) ---")
    sol1 = solve(h_l=2.0, u_l=0.0, h_r=1.0, u_r=0.0, g=g)
    print(sol1)
    print(f"  h* = {sol1.h_star:.6f}, u* = {sol1.u_star:.6f}")

    # Test 2: All-shock (Section 3.1, eq. 41)
    # h_l=h_r=1, u_l=2, u_r=0 → left shock, right shock
    print("\n--- Test 2: All-shock (h_l=h_r=1, u_l=2, u_r=0) ---")
    sol2 = solve(h_l=1.0, u_l=2.0, h_r=1.0, u_r=0.0, g=g)
    print(sol2)
    print(f"  h* = {sol2.h_star:.6f}, u* = {sol2.u_star:.6f}")

    # Test 3: All-rarefaction (Section 3.2, eq. 42)
    # h_l=h_r=1, u_l=0, u_r=2 → left rarefaction, right rarefaction
    print("\n--- Test 3: All-rarefaction (h_l=h_r=1, u_l=0, u_r=2) ---")
    sol3 = solve(h_l=1.0, u_l=0.0, h_r=1.0, u_r=2.0, g=g)
    print(sol3)
    print(f"  h* = {sol3.h_star:.6f}, u* = {sol3.u_star:.6f}")

    # Test 4: Symmetric dam break
    # h_l=2, h_r=1, u_l=0, u_r=0 → left rarefaction, right shock
    print("\n--- Test 4: Left shock, right rarefaction ---")
    sol4 = solve(h_l=1.0, u_l=0.0, h_r=2.0, u_r=0.0, g=g)
    print(sol4)
    print(f"  h* = {sol4.h_star:.6f}, u* = {sol4.u_star:.6f}")

    # Test 5: Dry bed on right
    print("\n--- Test 5: Dry bed on right (h_l=1, u_l=0, h_r=0, u_r=0) ---")
    sol5 = solve(h_l=1.0, u_l=0.0, h_r=0.0, u_r=0.0, g=g)
    print(sol5)

    # Test 6: Vacuum generation in the middle
    print("\n--- Test 6: Vacuum generation (h_l=h_r=1, u_l=-10, u_r=10) ---")
    sol6 = solve(h_l=1.0, u_l=-10.0, h_r=1.0, u_r=10.0, g=g)
    print(sol6)

    # Test 7: Extreme
    sol7 = solve(h_l=0.1, u_l=-93., h_r=1.1, u_r=-8.3, g=g)

    # Verify: u* = 1 for test 2 and test 3 (from the paper)
    print("\n--- Verification ---")
    print(f"  Test 2 u* ≈ 1.0: {abs(sol2.u_star - 1.0) < 0.01}")
    print(f"  Test 3 u* ≈ 1.0: {abs(sol3.u_star - 1.0) < 0.01}")

    # Sample and display solution profile for dam break
    print("\n--- Sampling dam-break solution at t=1.0 ---")
    x, h, u = solve_profile(h_l=2.0, u_l=0.0, h_r=1.0, u_r=0.0,
                             x_min=-10, x_max=10, nx=20, t=1.0, g=g)
    print(f"  {'x':>8s}  {'h':>10s}  {'u':>10s}")
    for xi, hi, ui in zip(x, h, u):
        print(f"  {xi:8.3f}  {hi:10.6f}  {ui:10.6f}")

    # Try plotting if matplotlib is available
    try:
        print("\n--- Plotting all test cases ---")
        for i, sol in enumerate([sol1, sol2, sol3, sol4, sol5, sol6, sol7], 1):
            plot_solution(sol, t=1.0, x_min=-100, x_max=100)
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
    except Exception as e:
        print(f"  (plotting failed: {e})")


