import numpy as np
import matplotlib.pyplot as plt
from shallow import GodunovSolver, CabaretSolver

# Simulation setup
L = 2000.0  # Domain length (m)
nx = 100  # Number of spatial points
dx = L / nx  # Spatial resolution
g = 9.8066  # Gravitational acceleration (m/s^2)
CFL = 0.5  # CFL condition

# Initial conditions
h = np.ones(nx)
h[:nx//2] = 2  # Step in water depth
# h[nx // 4:nx //2] = 10  # Step in water depth
u = np.zeros(nx)
hu = h * u  # Momentum (initially zero)
# hu[:nx // 5] = 2.5
print(h, hu, sep='\n')

# # Initialize time
# t = 0.0
# h_history_approx = []  # Store water depth from approximate solver
# h_history_iter = []    # Store water depth from iterative solver


t_end = 60  # Simulation end time
def compute(h, hu, solver):
    t = 0.0
    while t < t_end:
        # Compute maximum wave speed for CFL condition
        u = np.where(h > 0, hu / h, 0)
        c = np.sqrt(g * h)
        dt = CFL * dx / (np.max(np.abs(u) + c))  # Adaptive time step

        h, hu = solver.step(h, hu, dx, dt)

        # Advance time
        t += dt

    return h, hu

new_h = np.zeros((2 * h.shape[0] - 1))
new_hu = np.zeros((2 * hu.shape[0] - 1))

new_h[::2] = h
new_h[1::2] = (h[:-1] + h[1:]) // 2

new_hu[::2] = hu
new_hu[1::2] = (hu[:-1] + hu[1:]) // 2

h_newton, hu_newton = compute(h.copy(), hu.copy(), solver=GodunovSolver(solver_func='newton'))
h_cabaret, hu_cabaret = compute(new_h.copy(), new_hu.copy(), solver=CabaretSolver(solver_func='classic'))
# h_iter, hu_iter = compute(h_iter, hu_iter, solver=GodunovSolver(solver_func='iter', model=None))

# Plot final results using subplots
x = np.linspace(0, L, nx)

fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot water depth (h)
axes[0].plot(x, h_cabaret[::2], label='Cabaret Solver', linestyle='-', color='blue')
# axes[0].plot(x, h_iter, label='Iterative Solver (h)', linestyle='-', color='orange')
axes[0].plot(x, h_newton, label='Newton\'s Method', linestyle='-', color='green')
axes[0].set_ylabel('Water Depth (h)')
axes[0].legend()
axes[0].grid()
axes[0].set_title('Water Depth (h) Comparison')

# Plot momentum (hu)
axes[1].plot(x, hu_cabaret[::2], label='Cabaret Solver', linestyle='-', color='blue')
# axes[1].plot(x, hu_iter, label='Iterative Solver (hu)', linestyle='-', color='orange')
axes[1].plot(x, hu_newton, label='Newton\'s Method', linestyle='-', color='green')
axes[1].set_ylabel('Momentum (hu)')
axes[1].legend()
axes[1].grid()
axes[1].set_title('Momentum (hu) Comparison')

# Adjust layout
plt.tight_layout()


# Save the plot as a PNG file
plt.savefig('myplot.png')

plt.show()