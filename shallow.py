import numpy as np
import matplotlib.pyplot as plt


def godunov_shallow_water(h, hu, dx, dt, iter=False, g=9.81):
    """
    Solve the 1D shallow water equations using the Godunov scheme.

    Parameters:
    - h: numpy array, water depth
    - hu: numpy array, momentum (h * u)
    - dx: float, spatial resolution
    - dt: float, time step
    - g: float, gravitational acceleration (default 9.81 m/s^2)

    Returns:
    - h_new: numpy array, updated water depth
    - hu_new: numpy array, updated momentum
    """

    def flux(h, hu):
        """Compute the physical flux for shallow water equations."""
        u = np.where(h > 0, hu / h, 0)  # Avoid division by zero
        return np.array([hu, hu * u + 0.5 * g * h ** 2])

    def riemann_solver(hL, huL, hR, huR):
        """Solve Riemann problem for shallow water equations."""
        uL = huL / hL if hL > 0 else 0
        uR = huR / hR if hR > 0 else 0

        # Compute fluxes for left and right states
        FL = flux(hL, huL)
        FR = flux(hR, huR)

        # Roe averages
        hRoe = max(0, 0.5 * (hL + hR))  # Ensure positivity of h
        if hL + hR > 0:
            uRoe = (np.sqrt(hL) * uL + np.sqrt(hR) * uR) / (np.sqrt(hL) + np.sqrt(hR))
        else:
            uRoe = 0
        cRoe = np.sqrt(g * hRoe)

        # Wave speeds
        sL = min(uL - np.sqrt(g * hL), uRoe - cRoe)
        sR = max(uR + np.sqrt(g * hR), uRoe + cRoe)

        # Flux computation
        if sL >= 0:
            return FL
        elif sR <= 0:
            return FR
        else:
            # Flux in the intermediate region
            return (sR * FL - sL * FR + sL * sR * (np.array([hR, huR]) - np.array([hL, huL]))) / (sR - sL)

    def riemann_solver_iter(hL, huL, hR, huR, g=9.81, tol=1e-6, max_iter=1000):
        """
        Exact Riemann solver for the shallow water equations using Newton's method.

        Parameters:
        - hL: float, water depth on the left
        - huL: float, momentum on the left
        - hR: float, water depth on the right
        - huR: float, momentum on the right
        - g: float, gravitational acceleration
        - tol: float, tolerance for Newton's method
        - max_iter: int, maximum number of iterations for Newton's method

        Returns:
        - Flux at the interface as a numpy array [F_h, F_hu]
        """
        # Compute velocities on the left and right
        uL = huL / hL if hL > 0 else 0
        uR = huR / hR if hR > 0 else 0

        # Function to solve for h_star
        def f(h_star):
            """Nonlinear equation for h_star."""
            termL = (h_star - hL) * (np.sqrt(0.5 * g * (h_star + hL)))
            termR = (h_star - hR) * (np.sqrt(0.5 * g * (h_star + hR)))
            return termL + termR + (uR - uL)

        def df(h_star):
            """Derivative of f(h_star)."""
            termL = np.sqrt(0.5 * g * (h_star + hL)) + 0.5 * g * (h_star - hL) / np.sqrt(0.5 * g * (h_star + hL))
            termR = np.sqrt(0.5 * g * (h_star + hR)) + 0.5 * g * (h_star - hR) / np.sqrt(0.5 * g * (h_star + hR))
            return termL + termR

        # Initial guess for h_star (arithmetic mean of hL and hR)
        h_star = 0.5 * (hL + hR)

        # Newton's method to find h_star
        for _ in range(max_iter):
            f_val = f(h_star)
            df_val = df(h_star)
            if abs(f_val) < tol:  # Convergence check
                break
            h_star -= f_val / df_val
        else:
            raise RuntimeError("Newton's method failed to converge for h_star")

        # Compute the intermediate velocity
        u_star = 0.5 * (uL + uR) - 0.5 * (np.sqrt(g / h_star) * (hR - hL))

        # Compute fluxes based on the solution
        F_h = h_star * u_star
        F_hu = h_star * u_star ** 2 + 0.5 * g * h_star ** 2

        return np.array([F_h, F_hu])

    # Number of cells
    n = len(h)

    # Compute fluxes at interfaces
    h_flux = np.zeros(n - 1)
    hu_flux = np.zeros(n - 1)
    for i in range(n - 1):
        if not iter:
            flux_i = riemann_solver(h[i], hu[i], h[i + 1], hu[i + 1])
        else:
            flux_i = riemann_solver_iter(h[i], hu[i], h[i + 1], hu[i + 1])
        h_flux[i], hu_flux[i] = flux_i

    # Update h and hu using flux differences
    h_new = h.copy()
    hu_new = hu.copy()
    for i in range(1, n - 1):
        h_new[i] -= dt / dx * (h_flux[i] - h_flux[i - 1])
        hu_new[i] -= dt / dx * (hu_flux[i] - hu_flux[i - 1])

    # Enforce positivity of h
    h_new = np.maximum(h_new, 0)
    return h_new, hu_new


# Simulation setup
L = 10.0  # Domain length (m)
nx = 100  # Number of spatial points
dx = L / nx  # Spatial resolution
g = 9.81  # Gravitational acceleration (m/s^2)
CFL = 0.5  # CFL condition
t_end = 0.8  # Simulation end time

# Initial conditions
h = np.ones(nx) * 1.0
h[nx // 2:] = 0.5  # Step in water depth
hu = np.zeros(nx)  # Momentum (initially zero)

# Updated time-stepping loop to compute solutions with both solvers
h_iter = h.copy()  # Initial condition for iterative solver
hu_iter = hu.copy()

# Initialize time
t = 0.0
h_history_approx = []  # Store water depth from approximate solver
h_history_iter = []    # Store water depth from iterative solver

while t < t_end:
    # Compute maximum wave speed for CFL condition
    u = np.where(h > 0, hu / h, 0)
    c = np.sqrt(g * h)
    dt = CFL * dx / (np.max(np.abs(u) + c))  # Adaptive time step

    # Update solution using approximate solver
    h, hu = godunov_shallow_water(h, hu, dx, dt, g=g)

    h_iter, hu_iter = godunov_shallow_water(h, hu, dx, dt, iter=True, g=g)

    # Advance time
    t += dt

    # Store data for plotting
    h_history_approx.append(h.copy())
    h_history_iter.append(h_iter.copy())

# Plot final results using subplots
x = np.linspace(0, L, nx)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot water depth (h)
axes[0].plot(x, h, label='Approximate Solver (h)', linestyle='-', color='blue')
axes[0].plot(x, h_iter, label='Iterative Solver (h)', linestyle='-', color='orange')
axes[0].set_ylabel('Water Depth (h)')
axes[0].legend()
axes[0].grid()
axes[0].set_title('Water Depth (h) Comparison')

# Plot momentum (hu)
axes[1].plot(x, hu, label='Approximate Solver (hu)', linestyle='-', color='blue')
axes[1].plot(x, hu_iter, label='Iterative Solver (hu)', linestyle='-', color='orange')
axes[1].set_xlabel('x (m)')
axes[1].set_ylabel('Momentum (hu)')
axes[1].legend()
axes[1].grid()
axes[1].set_title('Momentum (hu) Comparison')

# Adjust layout
plt.tight_layout()
plt.show()
