import numpy as np
import matplotlib.pyplot as plt
import torch

from newton import riemann_solver_newton


def flux(h, hu, g=9.81):
    """Compute the physical flux for shallow water equations."""
    u = np.where(h > 0, hu / h, 0)  # Avoid division by zero
    return np.array([hu, hu * u + 0.5 * g * h ** 2])

def riemann_solver(hL, huL, hR, huR, g=9.81):
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
        part = (sR * FL - sL * FR + sL * sR * (np.array([hR, huR]) - np.array([hL, huL]))) / (sR - sL)
        part = np.append(part, [hRoe, uRoe])
        return part

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

    return np.array([F_h, F_hu, h_star, u_star])

def riemann_solver_nn(hL, huL, hR, huR, model, g=9.81):
    inputs = torch.tensor([hL, huL, hR, huR, np.sqrt(g * hL), np.sqrt(g * hR)],
                          dtype=torch.float32)
    h_star, u_star = model(inputs.to('cuda')).cpu().detach().numpy()

    F_h = h_star * u_star
    F_hu = h_star * u_star ** 2 + 0.5 * h_star ** 2

    flux_i = [F_h, F_hu, h_star, u_star]

    return flux_i


class GodunovSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        # Acceptable solver options: 'default', 'iter', 'newton', 'nn'
        self.solver_func = solver_func
        print(self.solver_func)

    def step(self, h, hu, dx, dt):
        n = len(h)
        h_flux = np.zeros(n - 1)
        hu_flux = np.zeros(n - 1)
        hh = np.zeros(n - 1)
        uu = np.zeros(n - 1)

        for i in range(n - 1):
            if self.solver_func == 'classic':
                flux_i = riemann_solver(h[i], hu[i], h[i + 1], hu[i + 1])
            elif self.solver_func == 'iter':
                flux_i = riemann_solver_iter(h[i], hu[i], h[i + 1], hu[i + 1], self.g)
            elif self.solver_func == 'newton':
                flux_i = riemann_solver_newton(h[i], hu[i], h[i + 1], hu[i + 1])['flux']
            elif self.solver_func == 'nn':
                flux_i = riemann_solver_nn(h[i], hu[i], h[i + 1], hu[i + 1], self.model, self.g)

            h_flux[i], hu_flux[i], hh[i], uu[i] = flux_i

        h_new = h.copy()
        hu_new = hu.copy()

        for i in range(1, n - 1):
            h_new[i] -= dt / dx * (h_flux[i] - h_flux[i - 1])
            hu_new[i] -= dt / dx * (hu_flux[i] - hu_flux[i - 1])

        h_new = np.maximum(h_new, 0)

        return h_new, hu_new


class CabaretSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        # Acceptable solver options: 'default', 'iter', 'newton', 'nn'
        self.solver_func = solver_func
        print(self.solver_func)

    def step(self, h, hu, dx, dt):
        u = hu / h

        # Step 1
        h[1::2] -= dt / (2 * dx) * (hu[2::2] - hu[:-1:2])
        hu[1::2] -= dt / (2 * dx) * ((h[2::2] * (u[2::2]) ** 2 + 0.5 * self.g * h[2::2] ** 2) - (
        h[:-1:2] * (u[:-1:2]) ** 2 + 0.5 * self.g * h[:-1:2] ** 2))

        # print(h)

        if self.solver_func == 'classic':
            # Step 2
            u = hu / h
            for i in range(h.shape[0]):
                if u[i] > (self.g * h[i]) ** 0.5:
                    print('oops', i, u[i], h[i])

            # Calculate characteristics
            left_char = u - 2 * np.sqrt(self.g * h)
            right_char = u + 2 * np.sqrt(self.g * h)

            left_char_new = np.zeros_like(left_char)
            right_char_new = np.zeros_like(right_char)

            left_char_new[:-1:2] = 2 * left_char[1::2] - left_char[2::2]
            right_char_new[2::2] = 2 * right_char[1::2] - right_char[:-1:2]

            # print(left_char_new.shape)
            # Normalize characteristics
            for i in range(2, left_char_new.shape[0] - 1, 2):
                if left_char_new[i] < min(left_char[i + 2], left_char[i + 1], left_char[i]):
                    # print('left less then min', i, left_char[i + 2], left_char[i + 1], left_char[i], left_char_new[i])
                    left_char_new[i] = min(left_char[i + 2], left_char[i + 1], left_char[i])
                elif left_char_new[i] > max(left_char[i + 2], left_char[i + 1], left_char[i]):
                    # print('left greater then max', i, left_char[i + 2], left_char[i + 1], left_char[i], left_char_new[i])
                    left_char_new[i] = max(left_char[i + 2], left_char[i + 1], left_char[i])

                if right_char_new[i] < min(right_char[i - 2], right_char[i - 1], right_char[i]):
                    # print('right less then min', i, right_char[i - 2], right_char[i - 1], right_char[i], right_char_new[i])
                    right_char_new[i] = min(right_char[i - 2], right_char[i - 1], right_char[i])
                if right_char_new[i] > max(right_char[i - 2], right_char[i - 1], right_char[i]):
                    # print('right greater then max', i, right_char[i - 2], right_char[i - 1], right_char[i], right_char_new[i])
                    right_char_new[i] = max(right_char[i - 2], right_char[i - 1], right_char[i])

                # Calculate values
                u[2:-1:2] = (left_char_new[2:-1:2] + right_char_new[2:-1:2]) / 2
                h[2:-1:2] = ((right_char_new[2:-1:2] - left_char_new[2:-1:2]) / 4) ** 2 / self.g
                hu[2:-1:2] = h[2:-1:2] * u[2:-1:2]
        else:
            for i in range(1, h.shape[0] - 3, 2):
                fluxx = riemann_solver_nn(h[i], hu[i], h[i + 2], hu[i + 2], self.model, self.g)
                _, _, hh, uu = fluxx
                h[i + 1] = hh
                hu[i + 1] = hh * uu


        # print(min(h))

        # u = hu / h

        # Step 3
        h[1::2] -= dt / (2 * dx) * (hu[2::2] - hu[:-1:2])
        hu[1::2] -= dt / (2 * dx) * ((h[2::2] * (u[2::2]) ** 2 + 0.5 * self.g * h[2::2] ** 2) - (h[:-1:2] * (u[:-2:2]) ** 2 + 0.5 * self.g * h[:-2:2] ** 2))

        # print(min(h))

        return h, hu
