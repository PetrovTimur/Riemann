import numpy as np
import matplotlib.pyplot as plt


def godunov_shallow_water(h, hu, dx, dt, g=9.81):
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

    # Number of cells
    n = len(h)

    # Compute fluxes at interfaces
    h_flux = np.zeros(n - 1)
    hu_flux = np.zeros(n - 1)
    for i in range(n - 1):
        flux_i = riemann_solver(h[i], hu[i], h[i + 1], hu[i + 1])
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
CFL = 0.9  # CFL condition
t_end = 0.8  # Simulation end time

# Initial conditions
h = np.ones(nx) * 1.0
h[nx // 2:] = 0.5  # Step in water depth
hu = np.zeros(nx)  # Momentum (initially zero)

# Time-stepping loop
t = 0.0
while t < t_end:
    # Compute maximum wave speed for CFL condition
    u = np.where(h > 0, hu / h, 0)
    c = np.sqrt(g * h)
    dt = CFL * dx / (np.max(np.abs(u) + c))  # Adaptive time step

    # Update solution
    h, hu = godunov_shallow_water(h, hu, dx, dt, g=g)
    t += dt

# Plot the results
x = np.linspace(0, L, nx)
plt.plot(x, h, label='Water Depth')
# plt.plot(x, hu, label='Momentum')
plt.xlabel('x (m)')
plt.ylabel('h, hu')
plt.legend()
plt.title('Shallow Water Equations - Godunov Scheme')
plt.show()
