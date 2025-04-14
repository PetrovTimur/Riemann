import numpy as np
from solvers import GodunovSolver, CabaretSolver


def compute(h, hu, solver, dx, t_end):
    # print(h, hu)
    t = 0.0
    CFL = 0.3
    g = 9.81
    while t < t_end:
        # Compute maximum wave speed for CFL condition
        u = np.where(h > 0, hu / h, 0)
        c = np.sqrt(g * h)
        dt = CFL * dx / np.max(np.abs(u) + c)
        h, hu = solver.step(h, hu, dx, dt)
        t += dt

    # print(t)
    # print(h)
    return h, hu

def insert_intermediate_points(h, hu):
    new_h = np.zeros((2 * h.shape[0] - 1))
    new_hu = np.zeros((2 * hu.shape[0] - 1))

    new_h[::2] = h
    # print(new_h.shape[0] // 2)
    new_h[new_h.shape[0] // 2] = (new_h[new_h.shape[0] // 2 - 2] + new_h[new_h.shape[0] // 2 + 2]) / 2
    new_h[1::2] = (h[:-1] + h[1:]) / 2


    # print(hu)
    new_hu[::2] = hu
    # print(hu[:-1], hu[1:])
    # print((hu[:-1] + hu[1:]) // 2)
    new_hu[new_hu.shape[0] // 2] = (new_hu[new_hu.shape[0] // 2 - 2] + new_hu[new_hu.shape[0] // 2 + 2]) / 2
    new_hu[1::2] = (hu[:-1] + hu[1:]) / 2


    return new_h, new_hu


def new_grid(h, hu):
    new_h = np.zeros(2 * len(h) + 1)
    new_hu = np.zeros(2 * len(hu) + 1)

    # Place original values at odd indexes
    new_h[1::2] = h
    new_hu[1::2] = hu

    # Calculate means for even indexes (interior points)
    new_h[2:-1:2] = (h[1:] + h[:-1]) / 2
    new_hu[2:-1:2] = (hu[1:] + hu[:-1]) / 2

    # Handle boundary points (first and last even indices)
    new_h[0] = h[0]  # First point gets value of first original point
    new_h[-1] = h[-1]  # Last point gets value of last original point
    new_hu[0] = hu[0]
    new_hu[-1] = hu[-1]

    return new_h, new_hu


def initial_conditions(nx, h_l, u_l, h_r, u_r):
    h = np.full(nx, h_r)
    h[:nx // 2] = h_l   # Step in water depth

    u = np.full(nx, u_r)
    u[:nx // 2] = u_l

    hu = h * u
    return h, hu

def run_simulation(L, nx, h_l, u_l, h_r, u_r, t_end):
    dx = L / nx
    h0, hu0 = initial_conditions(nx, h_l, u_l, h_r, u_r)

    # print(h0, hu0)

    new_h, new_hu = new_grid(h0, hu0)
    # # print(new_h, new_hu)
    solver = CabaretSolver(solver_func='classic')
    h_final, hu_final = compute(new_h.copy(), new_hu.copy(), solver, dx, t_end)
    return h_final, hu_final

    # solver = GodunovSolver(solver_func='newton')
    # h_final, hu_final = compute(h0.copy(), hu0.copy(), solver, dx, t_end)
    # return h_final, hu_final

def error_norm(a, b, dx):
    # return np.max(np.abs(a - b))
    return np.sqrt(np.sum(dx * np.abs(a - b) ** 2))