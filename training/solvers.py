import numpy as np
import torch
from scipy.optimize import fsolve, newton


def riemann_solver_newton(hL, huL, hR, huR, g = 9.8066, tol=1e-6, max_iter=1000):

    uL = huL / hL if hL > 0 else 0
    uR = huR / hR if hR > 0 else 0

    h_k = np.array([hL, hR])
    u_k = np.array([uL, uR])
    c_k = np.sqrt(g * h_k)
    #
    # # Check for dry bed / vacuum formation
    # # Vacuum occurs when: u_R - u_L > 2(c_L + c_R)
    # if u_k[1] - u_k[0] > 2 * c_k.sum():
    #     # Vacuum state: return dry bed in star region
    #     h_star = 0.0
    #     u_star = 0.5 * (u_k[0] + u_k[1])  # Average velocity (though meaningless in vacuum)
    #     return {
    #         "flux": np.array([0.0, 0.0, 0.0, 0.0]),
    #         "star": np.array([h_star, u_star]),
    #         "data": np.array([0]),  # No iterations for vacuum case
    #         "velocity": np.array([u_k[0] - c_k[0], None, u_k[1] + c_k[1], None]),
    #     }

    c = 0.25 * (u_k[0] - u_k[1]) + 0.5 * c_k.sum()

    grad = 1
    eps = 1e-15

    phi_k = np.array([0, 0])
    dphi_k = np.array([0, 0])


    k = 0
    while abs(grad) > tol and k < max_iter:
        k += 1
        s_k = c / c_k
        phi_k = np.where(s_k >= 1, ((c - c_k) * (s_k + 1) * np.sqrt(1 + s_k ** -2) / np.sqrt(2)), 2 * (c - c_k))
        dphi_k = np.where(s_k > 1, ((2 * s_k ** 2 + 1 + s_k ** -2) / (np.sqrt(2) * s_k * np.sqrt(1 + s_k ** -2))), 2)

        grad = (phi_k.sum() - (u_k[0] - u_k[1])) / (dphi_k.sum())
        c -= grad
        # print(f'k = {k}, F = {phi_k}, grad = {grad}')

    # print(k)

    h_star = c ** 2 / g
    u_star = 0.5 * (u_k.sum() + phi_k[1] - phi_k[0])
    # print(f'h = {h_star}, u = {u_star}')

    def compute_wave(u, h):
        D_L, D_R, D_starL, D_starR = None, None, None, None
        if u > 0:               # Contact to the right of the cell point
            if c > c_k[0]:      # Shock wave
                D_L = uL - c * np.sqrt(1 + (c / c_k[0]) ** 2) / np.sqrt(2)
                if D_L < 0:
                    H = h
                    U = u
                else:
                    H = hL
                    U = uL
                # print("Shock wave to the left: ", hL, uL, hR, uR, H, U)
            else:               # Rarefaction
                c_starL = c_k[0] + (uL - u) / 2
                D_starL = u - c_starL
                D_L = u - c_k[0]

                if D_starL < 0:
                    H = h
                    U = u
                elif D_L > 0:
                    H = hL
                    U = uL
                else:
                    c_star = (2 * c_k[0] + uL) / 3
                    U = c_star
                    H = c_star ** 2 / g
                # print("Rarefaction to the left: ", hL, uL, hR, uR, H, U)
        else:                   # Contact to the left of the cell point
            if c > c_k[1]:      # Shock wave
                D_R = uR + c * np.sqrt(1 + (c / c_k[1]) ** 2) / np.sqrt(2)
                if D_R > 0:
                    H = h
                    U = u
                else:
                    H = hR
                    U = uR
                # print("Shock wave to the right: ", hL, uL, hR, uR, H, U)
            else:               # Rarefaction
                c_starR = c_k[1] - (uR - u) / 2
                D_starR = u + c_starR
                D_R = u + c_k[1]

                if D_starR > 0:
                    H = h
                    U = u
                elif D_R < 0:
                    H = hR
                    U = uR
                else:
                    c_star = (2 * c_k[1] - uR) / 3
                    U = -c_star
                    H = c_star ** 2 / g
                # print("Rarefaction to the right: ", hL, uL, hR, uR, H, U, D_starR, D_R)
        return H, U, D_L, D_starL, D_R, D_starR

    def rarefaction(x, t, u, h):
        left = True
        if left:
            new_h = 1. / (9 * g) * (u + 2 * np.sqrt(g * h) - x / t) ** 2
            new_u = u + 2 / 3 * (x / t - u + np.sqrt(g * h))
        else:
            new_h = 1. / (9 * g) * (x / t - u +  2 * np.sqrt(g * h)) ** 2
            new_u = u + 2 / 3 * (x / t - u - np.sqrt(g * h))

        return new_h, new_u



        # if u + c < 0:  # Hydraulic jump (shock)
        #     return {
        #         "D": u - c * np.sqrt((1 + (c / c_star) ** 2) / 2),
        #         "H": h,
        #         "U": u
        #     }
        # elif u < c:  # Rarefaction wave
        #     c_star_local = c + (u - u_star) / 2
        #     return {
        #         "D": u - c_star_local,
        #         "H": c_star_local ** 2 / g,
        #         "U": u_star
        #     }
        # else:  # Contact discontinuity
        #     return {
        #         "D": u - c,
        #         "H": h,
        #         "U": u
        #     }

    # print(hL, uL, h_star, u_star, hR, uR)

    res = compute_wave(u_star, h_star)
    H, U = res[:2]
    D_L, D_starL, D_R, D_starR = res[2:]

    # wave_L = compute_wave(h_k[0], u_k[0], c_k[0], c)
    # wave_R = compute_wave(h_k[1], u_k[1], c_k[1], c)

    # h_star = H
    # u_star = U

    # h_star = 0.5 * (wave_R['H'] + wave_L['H'])
    # h_star = 0.5 * (wave_R['H'] + wave_L['H'])

    F_h = H * U
    F_hu = H * U ** 2 + 0.5 * g * H ** 2

    F_h_star = h_star * u_star
    F_hu_star = h_star * u_star ** 2 + 0.5 * g * h_star ** 2


    return {
        # "flux": np.array([F_h, F_hu, h_star, u_star]),
        # "flux": np.array([F_h_star, F_hu_star, h_star, u_star]),
        "flux": np.array([F_h, F_hu, H, U]),
        "star": np.array([h_star, u_star]),
        "data": np.array([k]),
        "velocity": np.array([D_L, D_starL, D_R, D_starR]),
        # "h_star": h_star,
        # "u_star": u_star,
        # "wave_left": wave_L,
        # "wave_right": wave_R
    }


def riemann_solver_scipy(hL, huL, hR, huR, g=9.8066, tol=1e-6, use_newton=True):
    """
    Riemann solver using scipy optimization routines.

    Parameters:
    -----------
    hL, huL : float
        Left state: water depth and momentum
    hR, huR : float
        Right state: water depth and momentum
    g : float
        Gravitational acceleration
    tol : float
        Convergence tolerance
    use_newton : bool
        If True, use scipy.optimize.newton (faster for 1D)
        If False, use scipy.optimize.fsolve (more robust)

    Returns:
    --------
    dict : Same format as riemann_solver_newton
    """

    # Compute velocities
    uL = huL / hL if hL > 0 else 0
    uR = huR / hR if hR > 0 else 0

    h_k = np.array([hL, hR])
    u_k = np.array([uL, uR])
    c_k = np.sqrt(g * h_k)

    # Check for dry bed / vacuum formation
    # Vacuum occurs when: u_R - u_L > 2(c_L + c_R)
    if u_k[1] - u_k[0] > 2 * c_k.sum():
        # Vacuum state: return dry bed in star region
        h_star = 0.0
        u_star = 0.5 * (u_k[0] + u_k[1])  # Average velocity (though meaningless in vacuum)
        return {
            "flux": np.array([0.0, 0.0, 0.0, 0.0]),
            "star": np.array([h_star, u_star]),
            "data": np.array([0]),  # No iterations for vacuum case
            "velocity": np.array([u_k[0] - c_k[0], None, u_k[1] + c_k[1], None]),
        }

    # Initial guess for c_star
    c_initial = 0.25 * (u_k[0] - u_k[1]) + 0.5 * c_k.sum()

    # Define the residual function
    def residual(c):
        """Residual function for finding c_star"""
        s_k = c / c_k
        phi_k = np.where(s_k >= 1,
                        ((c - c_k) * (s_k + 1) * np.sqrt(1 + s_k ** -2) / np.sqrt(2)),
                        2 * (c - c_k))
        return phi_k.sum() - (u_k[0] - u_k[1])

    # Define derivative for Newton's method (optional, for speed)
    def residual_derivative(c):
        """Derivative of residual w.r.t. c"""
        s_k = c / c_k
        dphi_k = np.where(s_k > 1,
                         ((2 * s_k ** 2 + 1 + s_k ** -2) / (np.sqrt(2) * s_k * np.sqrt(1 + s_k ** -2))),
                         2)
        return dphi_k.sum()

    # Solve using scipy
    try:
        if use_newton:
            # Use Newton's method with derivative
            c = newton(residual, c_initial, fprime=residual_derivative,
                      tol=tol, maxiter=1000)
        else:
            # Use fsolve (more robust, no derivative needed)
            c = fsolve(residual, c_initial, xtol=tol, maxfev=1000)[0]
    except (RuntimeError, ValueError):
        # Fallback: use fsolve if Newton fails
        c = fsolve(residual, c_initial, xtol=tol, maxfev=1000)[0]

    # Compute phi_k at solution
    s_k = c / c_k
    phi_k = np.where(s_k >= 1,
                    ((c - c_k) * (s_k + 1) * np.sqrt(1 + s_k ** -2) / np.sqrt(2)),
                    2 * (c - c_k))

    # Star region values
    h_star = c ** 2 / g
    u_star = 0.5 * (u_k.sum() + phi_k[1] - phi_k[0])

    def compute_wave(u, h):
        """Compute wave structure and velocities"""
        D_L, D_R, D_starL, D_starR = None, None, None, None
        if u > 0:  # Contact to the right
            if c > c_k[0]:  # Shock wave
                D_L = uL - c * np.sqrt(1 + (c / c_k[0]) ** 2) / np.sqrt(2)
                if D_L < 0:
                    H = h
                    U = u
                else:
                    H = hL
                    U = uL
            else:  # Rarefaction
                c_starL = c_k[0] + (uL - u) / 2
                D_starL = u - c_starL
                D_L = u - c_k[0]

                if D_starL < 0:
                    H = h
                    U = u
                elif D_L > 0:
                    H = hL
                    U = uL
                else:
                    c_star = (2 * c_k[0] + uL) / 3
                    U = c_star
                    H = c_star ** 2 / g
        else:  # Contact to the left
            if c > c_k[1]:  # Shock wave
                D_R = uR + c * np.sqrt(1 + (c / c_k[1]) ** 2) / np.sqrt(2)
                if D_R > 0:
                    H = h
                    U = u
                else:
                    H = hR
                    U = uR
            else:  # Rarefaction
                c_starR = c_k[1] - (uR - u) / 2
                D_starR = u + c_starR
                D_R = u + c_k[1]

                if D_starR > 0:
                    H = h
                    U = u
                elif D_R < 0:
                    H = hR
                    U = uR
                else:
                    c_star = (2 * c_k[1] - uR) / 3
                    U = -c_star
                    H = c_star ** 2 / g
        return H, U, D_L, D_starL, D_R, D_starR

    # Compute wave structure
    res = compute_wave(u_star, h_star)
    H, U = res[:2]
    D_L, D_starL, D_R, D_starR = res[2:]

    # Compute fluxes
    F_h = H * U
    F_hu = H * U ** 2 + 0.5 * g * H ** 2

    F_h_star = h_star * u_star
    F_hu_star = h_star * u_star ** 2 + 0.5 * g * h_star ** 2

    return {
        "flux": np.array([F_h, F_hu, H, U]),
        "star": np.array([h_star, u_star]),
        "data": np.array([0]),  # No iteration count in scipy version
        "velocity": np.array([D_L, D_starL, D_R, D_starR]),
    }


def flux(h, hu, g=9.81):
    u = np.where(h > 0, hu / h, 0)  # Avoid division by zero
    return np.array([hu, hu * u + 0.5 * g * h ** 2])

def riemann_solver_approx(hL, huL, hR, huR, g=9.81):
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

def riemann_solver_nn(hL, huL, hR, huR, model, g=9.8066):
    # inputs = torch.tensor([hL, huL, hR, huR, np.sqrt(g * hL), np.sqrt(g * hR)], dtype=torch.float32)
    inputs = torch.tensor([hL, huL, hR, huR], dtype=torch.float32)
    h_star, u_star = model(inputs.to('cuda')).cpu().detach().numpy()

    F_h = h_star * u_star
    F_hu = h_star * u_star ** 2 + 0.5 * g * h_star ** 2

    flux_i = [F_h, F_hu, h_star, u_star]

    return flux_i

def nn_solver(data, model):
    inputs = torch.tensor(data, dtype=torch.float32).cuda()
    I1, I2 = model.generate(inputs.unsqueeze(0)).squeeze(0).cpu().detach().numpy()

    return I1, I2


class BaseSolver:
    def __init__(self):
        pass

class GodunovSolver(BaseSolver):
    def __init__(self, solver_func='classic', model=None, g=9.81):
        super().__init__()
        self.g = g
        self.model = model
        self.solver_func = solver_func
        # print(self.solver_func)

    def step(self, h, hu, dx, dt):
        n = len(h)
        h_flux = np.zeros(n - 1)
        hu_flux = np.zeros(n - 1)
        hh = np.zeros(n - 1)
        uu = np.zeros(n - 1)

        for i in range(n - 1):
            if self.solver_func == 'approx':
                flux_i = riemann_solver_approx(h[i], hu[i], h[i + 1], hu[i + 1])
            elif self.solver_func == 'newton':
                flux_i = riemann_solver_newton(h[i], hu[i], h[i + 1], hu[i + 1])['flux']
            elif self.solver_func == 'nn':
                flux_i = riemann_solver_nn(h[i], hu[i], h[i + 1], hu[i + 1], self.model, self.g)

            h_flux[i], hu_flux[i], hh[i], uu[i] = flux_i

        h_new = h.copy().astype(np.float32)
        hu_new = hu.copy().astype(np.float32)


        for i in range(1, n - 1):
            # print(dt, dx, dt / dx * (h_flux[i] - h_flux[i - 1]))
            h_new[i] -= dt / dx * (h_flux[i] - h_flux[i - 1])
            hu_new[i] -= dt / dx * (hu_flux[i] - hu_flux[i - 1])

        h_new = np.maximum(h_new, 0)

        # print('heyyy', h_new)

        return h_new, hu_new


class CabaretSolver(BaseSolver):
    def __init__(self, solver_func='classic', g=9.81):
        super().__init__()
        self.g = g

        # Acceptable solver options: 'default', 'iter', 'newton', 'nn'
        self.solver_func = solver_func
        # print("Cabaret solver: ", self.solver_func)

        self.dx, self.dt = None, None
        self.h, self.hu, self.pos_char, self.neg_char = None, None, None, None

    def _correct_invariants(self, pos_char, neg_char, pos_char_new, neg_char_new):
        for i in range(2, neg_char_new.shape[0] - 1, 2):
            # lambda_pos = u[i + 1] + (self.g * h[i + 1]) ** 0.5
            # lambda_neg = u[i + 1] - (self.g * h[i + 1]) ** 0.5

            if pos_char_new[i] < min(pos_char[i - 2], pos_char[i - 1], pos_char[i]):
                # print('right < then min', i, pos_char[i - 2], pos_char[i - 1], pos_char[i], pos_char_new[i])
                pos_char_new[i] = min(pos_char[i - 2], pos_char[i - 1], pos_char[i])  # Все значения берем с n-го слоя
            if pos_char_new[i] > max(pos_char[i - 2], pos_char[i - 1], pos_char[i]):
                # print('right > then max', i, pos_char[i - 2], pos_char[i - 1], pos_char[i], pos_char_new[i])
                pos_char_new[i] = max(pos_char[i - 2], pos_char[i - 1], pos_char[i])

            if neg_char_new[i] < min(neg_char[i + 2], neg_char[i + 1], neg_char[i]):
                # print('left < then min', i, neg_char[i + 2], neg_char[i + 1], neg_char[i], neg_char_new[i])
                neg_char_new[i] = min(neg_char[i + 2], neg_char[i + 1], neg_char[i])
            elif neg_char_new[i] > max(neg_char[i + 2], neg_char[i + 1], neg_char[i]):
                # print('left > then max', i, neg_char[i + 2], neg_char[i + 1], neg_char[i], neg_char_new[i])
                neg_char_new[i] = max(neg_char[i + 2], neg_char[i + 1], neg_char[i])

            # neg_char_new[i] = np.clip(a=neg_char_new[i], a_max=max(neg_char[i + 2], neg_char[i + 1], neg_char[i]), a_min=min(neg_char[i + 2], neg_char[i + 1], neg_char[i]))
            # pos_char_new[i] = np.clip(a=pos_char_new[i], a_max=max(pos_char[i - 2], pos_char[i - 1], pos_char[i]), a_min=min(pos_char[i - 2], pos_char[i - 1], pos_char[i]))

        return pos_char_new, neg_char_new

    def _step1(self):
        self.u = self.hu / (self.h + 1e-12)

        # Инварианты на n-м слое по времени
        self.neg_char = self.u - 2 * np.sqrt(np.maximum(0, self.g * self.h))
        self.pos_char = self.u + 2 * np.sqrt(np.maximum(0, self.g * self.h))

        self.h[1::2] -= self.dt / (2 * self.dx) * (self.hu[2::2] - self.hu[:-1:2])
        self.hu[1::2] -= self.dt / (2 * self.dx) * ((self.h[2::2] * (self.u[2::2]) ** 2 + 0.5 * self.g * self.h[2::2] ** 2) - (self.h[:-1:2] * (self.u[:-1:2]) ** 2 + 0.5 * self.g * self.h[:-1:2] ** 2))
        # print(self.h)

    def _step2(self):
        self.u = self.hu / (self.h + 1e-12)     # TODO

        # В четных точках старые инварианты (n слой), в консервативных точках на полуцелом шаге по времени (n + 1/2)
        neg_char_new = self.u - 2 * np.sqrt(np.maximum(0, self.g * self.h))
        pos_char_new = self.u + 2 * np.sqrt(np.maximum(0, self.g * self.h))

        for i in range(2, neg_char_new.shape[0] - 1, 2):
            # lambda_left_neg = self.u[i - 1] - (self.g * self.h[i - 1]) ** 0.5
            # lambda_left_pos = self.u[i - 1] + (self.g * self.h[i - 1]) ** 0.5
            # lambda_right_neg = self.u[i + 1] - (self.g * self.h[i + 1]) ** 0.5
            # lambda_right_pos = self.u[i + 1] + (self.g * self.h[i + 1]) ** 0.5

            neg_char_new[i] = 2 * neg_char_new[i + 1] - self.neg_char[i + 2]
            pos_char_new[i] = 2 * pos_char_new[i - 1] - self.pos_char[i - 2]

            # print(i // 2, pos_char_new[i], neg_char_new[i])

        self._correct_invariants(self.pos_char, self.neg_char, pos_char_new, neg_char_new)

        for i in range(2, len(self.u) - 1, 2):
            self.u[i] = (neg_char_new[i] + pos_char_new[i]) / 2
            self.h[i] = ((pos_char_new[i] - neg_char_new[i]) / 4) ** 2 / self.g
            self.hu[i] = self.h[i] * self.u[i]

    def _step3(self):
        self.h[1::2] -= self.dt / (2 * self.dx) * (self.hu[2::2] - self.hu[:-1:2])
        self.hu[1::2] -= self.dt / (2 * self.dx) * ((self.h[2::2] * (self.u[2::2]) ** 2 + 0.5 * self.g * self.h[2::2] ** 2) - (self.h[:-1:2] * (self.u[:-1:2]) ** 2 + 0.5 * self.g * self.h[:-1:2] ** 2))

    def step(self, h, hu, dx, dt):
        self.h = h
        self.hu = hu
        self.dx = dx
        self.dt = dt
        # print('pre step 1')
        # print(self.h[18:23])
        # print(self.hu[18:23])
        self._step1()
        # print(dt, dx, self.h[1::2])
        # print('after step 1')
        # print(self.h[18:23])
        # print(self.hu[18:23])
        self._step2()
        # print(dt, dx, self.h[::2])
        # print('after step 2')
        # print(self.h[18:23])
        # print(self.hu[18:23])
        self._step3()
        # print('after step 3')
        # print(self.h[18:23])
        # print(self.hu[18:23])
        return self.h, self.hu



class CabaretSolverPlus(CabaretSolver):
    def __init__(self, model=None, g=9.806):
        super().__init__(g=g)

        self.h_combined_n = None
        self.hu_combined_n = None

        self.h_node_n = None
        self.hu_node_n = None
        self.u_node_n = None

        self.h_cell_n = None
        self.hu_cell_n = None

        self.h_cell_n_plus_half = None
        self.hu_cell_n_plus_half = None

        self.h_node_n_plus_1_char = None
        self.hu_node_n_plus_1_char = None

        self.h_cell_n_plus_1 = None
        self.hu_cell_n_plus_1 = None

        self.dx = None
        self.dt = None

        self.N_total_points = 0
        self.N_nodes = 0
        self.N_cells = 0

        self.model = model

    def F_m(self, H, u):
        return H * u

    def F_h(self, H, u):
        return H * u ** 2 + 0.5 * self.g * H ** 2

    def _step1(self):
        self.u_node_n = self.hu_node_n / (self.h_node_n + 1e-12)

        self.h_cell_n_plus_half = np.zeros(self.N_cells)
        self.hu_cell_n_plus_half = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            h_cell_n_curr = self.h_cell_n[i]
            hu_cell_n_curr = self.hu_cell_n[i]

            flux_m_i = self.F_m(self.h_node_n[i], self.u_node_n[i])
            flux_m_i_plus_1 = self.F_m(self.h_node_n[i + 1], self.u_node_n[i + 1])

            flux_h_i = self.F_h(self.h_node_n[i], self.u_node_n[i])
            flux_h_i_plus_1 = self.F_h(self.h_node_n[i + 1], self.u_node_n[i + 1])

            self.h_cell_n_plus_half[i] = h_cell_n_curr - 0.5 * self.dt / self.dx * (flux_m_i_plus_1 - flux_m_i)
            self.hu_cell_n_plus_half[i] = hu_cell_n_curr - 0.5 * self.dt / self.dx * (flux_h_i_plus_1 - flux_h_i)

    def _correct_invariants(self, extrapolated_invariant, min_bound, max_bound):
        return np.clip(extrapolated_invariant, min_bound, max_bound)

    def _step2(self):
        u_cell_n_plus_half = self.hu_cell_n_plus_half / (self.h_cell_n_plus_half + 1e-12)
        c_cell_n_plus_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

        I1_cell_n_plus_half = u_cell_n_plus_half + 2 * c_cell_n_plus_half
        I2_cell_n_plus_half = u_cell_n_plus_half - 2 * c_cell_n_plus_half

        lambda1_cell_n_plus_half = u_cell_n_plus_half + c_cell_n_plus_half
        lambda2_cell_n_plus_half = u_cell_n_plus_half - c_cell_n_plus_half

        u_node_n = self.hu_node_n / (self.h_node_n + 1e-12)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))

        I1_node_n = u_node_n + 2 * c_node_n
        I2_node_n = u_node_n - 2 * c_node_n

        u_cell_n = self.hu_cell_n / (self.h_cell_n + 1e-12)
        c_cell_n = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n))

        I1_cell_n = u_cell_n + 2 * c_cell_n
        I2_cell_n = u_cell_n - 2 * c_cell_n

        I1_node_n_plus_1 = np.zeros(self.N_nodes)
        I2_node_n_plus_1 = np.zeros(self.N_nodes)

        self.h_node_n_plus_1_char = np.zeros(self.N_nodes)
        self.hu_node_n_plus_1_char = np.zeros(self.N_nodes)

        for j in range(1, self.N_nodes - 1):
            lambda1_avg_at_node_j = 0.5 * (lambda1_cell_n_plus_half[j - 1] + lambda1_cell_n_plus_half[j])
            lambda2_avg_at_node_j = 0.5 * (lambda2_cell_n_plus_half[j - 1] + lambda2_cell_n_plus_half[j])

            if lambda1_avg_at_node_j > 0:
                I1_extrapolated = 2 * I1_cell_n_plus_half[j - 1] - I1_node_n[j - 1]
                relevant_cell_idx_I1 = j - 1
            else:
                I1_extrapolated = 2 * I1_cell_n_plus_half[j] - I1_node_n[j + 1]
                relevant_cell_idx_I1 = j

            if lambda2_avg_at_node_j > 0:
                I2_extrapolated = 2 * I2_cell_n_plus_half[j - 1] - I2_node_n[j - 1]
                relevant_cell_idx_I2 = j - 1
            else:
                I2_extrapolated = 2 * I2_cell_n_plus_half[j] - I2_node_n[j + 1]
                relevant_cell_idx_I2 = j

            if (relevant_cell_idx_I1 >= relevant_cell_idx_I2) and (self.model is not None):
                print('hhh: ', j, relevant_cell_idx_I1, relevant_cell_idx_I2, self.h_cell_n_plus_half[relevant_cell_idx_I1], self.h_cell_n_plus_half[relevant_cell_idx_I2])

                data = [I1_node_n[j - 1], I1_node_n[j], I1_node_n[j + 1], I2_node_n[j - 1], I2_node_n[j], I2_node_n[j + 1],
                        I1_cell_n_plus_half[j - 1], I1_cell_n_plus_half[j], I2_cell_n_plus_half[j - 1], I2_cell_n_plus_half[j],
                        I1_cell_n[j - 1], I1_cell_n[j], I2_cell_n[j - 1], I2_cell_n[j]]

                I1, I2 = nn_solver(data, self.model)
            # print(j, I1_extrapolated, I2_extrapolated)

            max_I1_bound = np.max([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])
            min_I1_bound = np.min([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])

            I1_node_n_plus_1[j] = self._correct_invariants(I1_extrapolated, min_I1_bound, max_I1_bound)

            max_I2_bound = np.max([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])
            min_I2_bound = np.min([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])

            I2_node_n_plus_1[j] = self._correct_invariants(I2_extrapolated, min_I2_bound, max_I2_bound)

            g1_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I1] + 1e-12)
            g2_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I2] + 1e-12)

            if (relevant_cell_idx_I1 >= relevant_cell_idx_I2) and (self.model is not None) and (self.h_node_n[j - 1] != self.h_node_n[j + 1]):
                I1_node_n_plus_1[j] = I1
                I2_node_n_plus_1[j] = I2

            if True:
            # if self.model is not None:
                self.h_node_n_plus_1_char[j] = ((I1_node_n_plus_1[j] - I2_node_n_plus_1[j]) / 4) ** 2 / self.g
                u_node_n_plus_1_at_j = (I1_node_n_plus_1[j] + I2_node_n_plus_1[j]) / 2
            else:
                self.h_node_n_plus_1_char[j] = (I1_node_n_plus_1[j] - I2_node_n_plus_1[j]) / (g1_star + g2_star)
                u_node_n_plus_1_at_j = (I1_node_n_plus_1[j] * g2_star + I2_node_n_plus_1[j] * g1_star) / (g1_star + g2_star)


            self.h_node_n_plus_1_char[j] = np.maximum(0.0, self.h_node_n_plus_1_char[j])
            self.hu_node_n_plus_1_char[j] = self.h_node_n_plus_1_char[j] * u_node_n_plus_1_at_j


        self.h_node_n_plus_1_char[0] = self.h_node_n[0]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n[0]
        self.h_node_n_plus_1_char[self.N_nodes - 1] = self.h_node_n[self.N_nodes - 1]
        self.hu_node_n_plus_1_char[self.N_nodes - 1] = self.hu_node_n[self.N_nodes - 1]

    def _step3(self):
        u_node_n_plus_1_char = self.hu_node_n_plus_1_char / (self.h_node_n_plus_1_char + 1e-12)

        self.h_cell_n_plus_1 = np.zeros(self.N_cells)
        self.hu_cell_n_plus_1 = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            flux_m_i_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_m_i_plus_1_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            flux_h_i_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_h_i_plus_1_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            self.h_cell_n_plus_1[i] = self.h_cell_n_plus_half[i] - \
                                      0.5 * self.dt / self.dx * (flux_m_i_plus_1_n_plus_1 - flux_m_i_n_plus_1)
            self.hu_cell_n_plus_1[i] = self.hu_cell_n_plus_half[i] - \
                                       0.5 * self.dt / self.dx * (flux_h_i_plus_1_n_plus_1 - flux_h_i_n_plus_1)

    def step(self, h_n_total_points, hu_n_total_points, dx, dt):
        self.h_combined_n = h_n_total_points.copy()
        self.hu_combined_n = hu_n_total_points.copy()
        self.dx = dx
        self.dt = dt

        self.N_total_points = len(self.h_combined_n)
        self.N_cells = (self.N_total_points - 1) // 2
        self.N_nodes = self.N_cells + 1

        self.h_node_n = self.h_combined_n[::2]
        self.hu_node_n = self.hu_combined_n[::2]
        self.h_cell_n = self.h_combined_n[1::2]
        self.hu_cell_n = self.hu_combined_n[1::2]

        self._step1()
        self._step2()
        # print(dt, dx, self.h_node_n_plus_1_char)
        self._step3()

        h_n_plus_1_total_points = np.zeros(self.N_total_points)
        hu_n_plus_1_total_points = np.zeros(self.N_total_points)

        # Fix border points
        self.h_cell_n_plus_1[0] = self.h_cell_n[0]
        self.hu_cell_n_plus_1[0] = self.hu_cell_n[0]

        self.h_cell_n_plus_1[-1] = self.h_cell_n[-1]
        self.hu_cell_n_plus_1[-1] = self.hu_cell_n[-1]

        # Fix border points
        self.h_node_n_plus_1_char[0] = self.h_node_n_plus_1_char[1]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n_plus_1_char[1]

        self.h_node_n_plus_1_char[-1] = self.h_node_n_plus_1_char[-2]
        self.hu_node_n_plus_1_char[-1] = self.hu_node_n_plus_1_char[-2]

        h_n_plus_1_total_points[1::2] = self.h_cell_n_plus_1
        hu_n_plus_1_total_points[1::2] = self.hu_cell_n_plus_1

        h_n_plus_1_total_points[::2] = self.h_node_n_plus_1_char
        hu_n_plus_1_total_points[::2] = self.hu_node_n_plus_1_char

        # print(self.h_node_n)
        # print(self.h_node_n_plus_1_char)

        return h_n_plus_1_total_points, hu_n_plus_1_total_points


class CabaretSolverPlusPlus(CabaretSolverPlus):
    def __init__(self, model=None, g=9.806):
        super().__init__(model=model, g=g)

    def _step2(self):
        u_cell_n_plus_half = self.hu_cell_n_plus_half / (self.h_cell_n_plus_half + 1e-12)
        c_cell_n_plus_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

        I1_cell_n_plus_half = u_cell_n_plus_half + 2 * c_cell_n_plus_half
        I2_cell_n_plus_half = u_cell_n_plus_half - 2 * c_cell_n_plus_half

        lambda1_cell_n_plus_half = u_cell_n_plus_half + c_cell_n_plus_half
        lambda2_cell_n_plus_half = u_cell_n_plus_half - c_cell_n_plus_half

        u_node_n = self.hu_node_n / (self.h_node_n + 1e-12)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))
        I1_node_n = u_node_n + 2 * c_node_n
        I2_node_n = u_node_n - 2 * c_node_n

        u_cell_n = self.hu_cell_n / (self.h_cell_n + 1e-12)
        c_cell_n = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n))
        I1_cell_n = u_cell_n + 2 * c_cell_n
        I2_cell_n = u_cell_n - 2 * c_cell_n

        I1_node_n_plus_1 = np.zeros(self.N_nodes)
        I2_node_n_plus_1 = np.zeros(self.N_nodes)

        self.h_node_n_plus_1_char = np.zeros(self.N_nodes)
        self.hu_node_n_plus_1_char = np.zeros(self.N_nodes)

        for j in range(1, self.N_nodes - 1):
            is_sonic_point_lambda1 = (
                        np.sign(lambda1_cell_n_plus_half[j - 1]) * np.sign(lambda1_cell_n_plus_half[j]) < 0)
            is_sonic_point_lambda2 = (
                        np.sign(lambda2_cell_n_plus_half[j - 1]) * np.sign(lambda2_cell_n_plus_half[j]) < 0)

            if is_sonic_point_lambda1:
                u_j_n_plus_half_interp = 0.5 * (u_cell_n_plus_half[j - 1] + u_cell_n_plus_half[j])
                c_j_n_plus_half_interp = 0.5 * (c_cell_n_plus_half[j - 1] + c_cell_n_plus_half[j])

                lambda1_to_use_for_direction = u_j_n_plus_half_interp + c_j_n_plus_half_interp
            else:
                lambda1_to_use_for_direction = 0.5 * (lambda1_cell_n_plus_half[j - 1] + lambda1_cell_n_plus_half[j])

            if lambda1_to_use_for_direction >= 0:
                I1_extrapolated = 2 * I1_cell_n_plus_half[j - 1] - I1_node_n[j - 1]
                relevant_cell_idx_I1 = j - 1
            else:
                I1_extrapolated = 2 * I1_cell_n_plus_half[j] - I1_node_n[j + 1]
                relevant_cell_idx_I1 = j

            lambda2_to_use_for_direction = 0.0
            if is_sonic_point_lambda2:
                u_j_n_plus_half_interp = 0.5 * (u_cell_n_plus_half[j - 1] + u_cell_n_plus_half[j])
                c_j_n_plus_half_interp = 0.5 * (c_cell_n_plus_half[j - 1] + c_cell_n_plus_half[j])
                lambda2_to_use_for_direction = u_j_n_plus_half_interp - c_j_n_plus_half_interp
            else:
                lambda2_to_use_for_direction = 0.5 * (lambda2_cell_n_plus_half[j - 1] + lambda2_cell_n_plus_half[j])

            if lambda2_to_use_for_direction >= 0:
                I2_extrapolated = 2 * I2_cell_n_plus_half[j - 1] - I2_node_n[j - 1]
                relevant_cell_idx_I2 = j - 1
            else:
                I2_extrapolated = 2 * I2_cell_n_plus_half[j] - I2_node_n[j + 1]
                relevant_cell_idx_I2 = j

            max_I1_bound = np.max([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])
            min_I1_bound = np.min([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])

            I1_node_n_plus_1[j] = self._correct_invariants(I1_extrapolated, min_I1_bound, max_I1_bound)

            max_I2_bound = np.max([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])
            min_I2_bound = np.min([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])

            I2_node_n_plus_1[j] = self._correct_invariants(I2_extrapolated, min_I2_bound, max_I2_bound)

            g1_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I1] + 1e-12)
            g2_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I2] + 1e-12)

            self.h_node_n_plus_1_char[j] = (I1_node_n_plus_1[j] - I2_node_n_plus_1[j]) / (g1_star + g2_star)
            u_node_n_plus_1_at_j = (I1_node_n_plus_1[j] * g2_star + I2_node_n_plus_1[j] * g1_star) / (g1_star + g2_star) / 2        # ?

            self.h_node_n_plus_1_char[j] = np.maximum(0.0, self.h_node_n_plus_1_char[j])
            self.hu_node_n_plus_1_char[j] = self.h_node_n_plus_1_char[j] * u_node_n_plus_1_at_j

        self.h_node_n_plus_1_char[0] = self.h_node_n[0]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n[0]
        self.h_node_n_plus_1_char[self.N_nodes - 1] = self.h_node_n[self.N_nodes - 1]
        self.hu_node_n_plus_1_char[self.N_nodes - 1] = self.hu_node_n[self.N_nodes - 1]

class CabaretSolverGT(CabaretSolverPlus):
    def __init__(self, model=None, g=9.806):
        super().__init__(model=model, g=g)

    def step(self, h_n_total_points, hu_n_total_points, dx, dt, riemann_h, riemann_u):
        self.h_combined_n = h_n_total_points.copy()
        self.hu_combined_n = hu_n_total_points.copy()
        self.dx = dx
        self.dt = dt

        self.N_total_points = len(self.h_combined_n)
        self.N_cells = (self.N_total_points - 1) // 2
        self.N_nodes = self.N_cells + 1

        self.h_node_n = self.h_combined_n[::2]
        self.hu_node_n = self.hu_combined_n[::2]
        self.h_cell_n = self.h_combined_n[1::2]
        self.hu_cell_n = self.hu_combined_n[1::2]

        self._step1()
        # self._step2()

        # print(h_n_total_points.shape)
        # print(riemann_h.shape)
        # print(self.h_cell_n.shape)
        # print(riemann_h.shape, self.h_node_n_plus_1_char.shape)
        # print(riemann_h)
        # print(riemann_u)
        self.h_node_n_plus_1_char = riemann_h[::2]
        self.hu_node_n_plus_1_char = self.h_node_n_plus_1_char * riemann_u[::2]

        # print(dt, dx, self.h_node_n_plus_1_char)
        self._step3()

        h_n_plus_1_total_points = np.zeros(self.N_total_points)
        hu_n_plus_1_total_points = np.zeros(self.N_total_points)

        # Fix border points
        self.h_cell_n_plus_1[0] = self.h_cell_n[0]
        self.hu_cell_n_plus_1[0] = self.hu_cell_n[0]

        self.h_cell_n_plus_1[-1] = self.h_cell_n[-1]
        self.hu_cell_n_plus_1[-1] = self.hu_cell_n[-1]

        # Fix border points
        self.h_node_n_plus_1_char[0] = self.h_node_n_plus_1_char[1]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n_plus_1_char[1]

        self.h_node_n_plus_1_char[-1] = self.h_node_n_plus_1_char[-2]
        self.hu_node_n_plus_1_char[-1] = self.hu_node_n_plus_1_char[-2]

        h_n_plus_1_total_points[1::2] = self.h_cell_n_plus_1
        hu_n_plus_1_total_points[1::2] = self.hu_cell_n_plus_1

        h_n_plus_1_total_points[::2] = self.h_node_n_plus_1_char
        hu_n_plus_1_total_points[::2] = self.hu_node_n_plus_1_char

        # print(self.h_node_n)
        # print(self.h_node_n_plus_1_char)

        return h_n_plus_1_total_points, hu_n_plus_1_total_points

class CabaretSolverNN(CabaretSolver):
    def __init__(self, model, g=9.81, softmax=False):
        super().__init__(g=g)

        self.model = model
        self.softmax = softmax

    def _step2(self):
        self.u = self.hu / self.h

        neg_char_new = self.u - 2 * np.sqrt(self.g * self.h)
        pos_char_new = self.u + 2 * np.sqrt(self.g * self.h)

        for i in range(1, self.h.shape[0] - 3, 2):
            hL = self.h[i]
            hR = self.h[i + 2]
            huL = self.hu[i]
            huR = self.hu[i + 2]

            # Compute u and c values
            uL = huL / hL
            uR = huR / hR
            cL = np.sqrt(self.g * hL)
            cR = np.sqrt(self.g * hR)

            if (-cL <= uL <= cL) and (-cR <= uR <= cR):
                # use classic
                neg_char_new[i] = 2 * neg_char_new[i + 1] - self.neg_char[i + 2]
                pos_char_new[i] = 2 * pos_char_new[i - 1] - self.pos_char[i - 2]

            else:
                # print('using nn', i)
                j = i + 1

                I1_node_n = self.pos_char[::2]
                I2_node_n = self.neg_char[::2]

                I1_cell_n = self.pos_char[1::2]
                I2_cell_n = self.neg_char[1::2]

                I1_cell_n_plus_half = pos_char_new[1::2]
                I2_cell_n_plus_half = neg_char_new[1::2]

                j = j // 2

                data = [I1_node_n[j - 1], I1_node_n[j], I1_node_n[j + 1],
                        I2_node_n[j - 1], I2_node_n[j], I2_node_n[j + 1],
                        I1_cell_n_plus_half[j - 1], I1_cell_n_plus_half[j],
                        I2_cell_n_plus_half[j - 1], I2_cell_n_plus_half[j],
                        I1_cell_n[j - 1], I1_cell_n[j], I2_cell_n[j - 1], I2_cell_n[j]]

                if ((uR < -cR) and (uL < cL)) or ((uL < -cL) and (uR < cR)):
                    u_cell_n_plus_half = (I1_cell_n_plus_half + I2_cell_n_plus_half) / 2
                    h_cell_n_plus_half = ((I1_cell_n_plus_half - I2_cell_n_plus_half) / 4) ** 2 / self.g

                    u_node_n = (I1_node_n + I2_node_n) / 2
                    h_node_n = ((I1_node_n - I2_node_n) / 4) ** 2 / self.g

                    u_cell_n = (I1_cell_n + I2_cell_n) / 2
                    h_cell_n = ((I1_cell_n - I2_cell_n) / 4) ** 2 / self.g

                    # Flip heights and velocity sign
                    h_cell_n[j - 1], h_cell_n[j] = h_cell_n[j], h_cell_n[j - 1]
                    u_cell_n[j - 1], u_cell_n[j] = u_cell_n[j], u_cell_n[j - 1]
                    u_cell_n *= -1

                    h_node_n[j - 1], h_node_n[j + 1] = h_node_n[j + 1], h_node_n[j - 1]
                    u_node_n[j - 1], u_node_n[j + 1] = u_node_n[j + 1], u_node_n[j - 1]
                    u_node_n *= -1

                    h_cell_n_plus_half[j - 1], h_cell_n_plus_half[j] = h_cell_n_plus_half[j], h_cell_n_plus_half[j - 1]
                    u_cell_n_plus_half[j - 1], u_cell_n_plus_half[j] = u_cell_n_plus_half[j], u_cell_n_plus_half[j - 1]
                    u_cell_n_plus_half *= -1

                    # New invariants
                    I1_node_n_flipped = u_node_n + 2 * np.sqrt(self.g * h_node_n)
                    I2_node_n_flipped = u_node_n - 2 * np.sqrt(self.g * h_node_n)

                    I1_cell_n_plus_half_flipped = u_cell_n_plus_half + 2 * np.sqrt(self.g * h_cell_n_plus_half)
                    I2_cell_n_plus_half_flipped = u_cell_n_plus_half - 2 * np.sqrt(self.g * h_cell_n_plus_half)

                    I1_cell_n_flipped = u_cell_n + 2 * np.sqrt(self.g * h_cell_n)
                    I2_cell_n_flipped = u_cell_n - 2 * np.sqrt(self.g * h_cell_n)

                    data = [I1_node_n_flipped[j - 1], I1_node_n_flipped[j], I1_node_n_flipped[j + 1],
                        I2_node_n_flipped[j - 1], I2_node_n_flipped[j], I2_node_n_flipped[j + 1],
                        I1_cell_n_plus_half_flipped[j - 1], I1_cell_n_plus_half_flipped[j],
                        I2_cell_n_plus_half_flipped[j - 1], I2_cell_n_plus_half_flipped[j],
                        I1_cell_n_flipped[j - 1], I1_cell_n_flipped[j], I2_cell_n_flipped[j - 1], I2_cell_n_flipped[j]]

                I1, I2 = nn_solver(data, self.model)

                if ((uR < -cR) and (uL < cL)) or ((uL < -cL) and (uR < cR)):
                    u_new = (I1 + I2) / 2
                    h_new = ((I1 - I2) / 4) ** 2 / self.g

                    u_new *= -1

                    I1 = u_new + 2 * np.sqrt(self.g * h_new)
                    I2 = u_new - 2 * np.sqrt(self.g * h_new)


                # neg_char_new[i + 1] = uu - 2 * np.sqrt(self.g * hh)
                # pos_char_new[i + 1] = uu + 2 * np.sqrt(self.g * hh)

                j = j * 2

                pos_char_new[j] = I1
                neg_char_new[j] = I2

            # print(i, self.h[i], self.hu[i], self.h[i + 2], self.hu[i + 2], self.h[i + 1], self.hu[i + 1])

        self._correct_invariants(self.pos_char, self.neg_char, pos_char_new, neg_char_new)

        for i in range(2, len(self.u) - 1, 2):
            self.u[i] = (neg_char_new[i] + pos_char_new[i]) / 2
            self.h[i] = ((pos_char_new[i] - neg_char_new[i]) / 4) ** 2 / self.g

            # self.h[i] = np.clip(self.h[i], a_min=min(self.h[i - 1], self.h[i + 1]), a_max=max(self.h[i - 1], self.h[i + 1]))

            self.hu[i] = self.h[i] * self.u[i]


class CabaretSolverImproved(CabaretSolverPlus):
    """CABARET solver with locally implicit sonic point processing.

    Implements the algorithm from Afanasiev & Goloviznin:
    - Equations (15)-(16): Implicit transfer of invariants along characteristics
      at sonic points using 2nd-order Lagrange interpolation
    - Equation (19): Non-linear correction (monotonization) at sonic points
    - Standard CABARET (4)-(6) at non-sonic points
    """

    def __init__(self, model=None, g=9.81, newton_tol=1e-8, newton_max_iter=100, dry_eps=1e-5, monotonize=True):
        super().__init__(model=model, g=g)
        self.newton_tol = newton_tol
        self.newton_max_iter = newton_max_iter
        self.dry_eps = dry_eps
        self.monotonize = monotonize

    @staticmethod
    def _safe_velocity(h, hu, eps):
        """Compute u = hu/h, returning 0 where h < eps."""
        return np.where(h > eps, hu / np.maximum(h, eps), 0.0)

    def _apply_dry_cell(self, h, hu):
        """Dry cell treatment: clip h >= 0, zero out momentum where h < dry_eps."""
        h = np.maximum(h, 0.0)
        hu = np.where(h < self.dry_eps, 0.0, hu)
        return h, hu

    def _step1(self):
        """Phase 1 with dry cell treatment."""
        self.u_node_n = self._safe_velocity(self.h_node_n, self.hu_node_n, self.dry_eps)

        self.h_cell_n_plus_half = np.zeros(self.N_cells)
        self.hu_cell_n_plus_half = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            h_cell_n_curr = self.h_cell_n[i]
            hu_cell_n_curr = self.hu_cell_n[i]

            flux_m_i = self.F_m(self.h_node_n[i], self.u_node_n[i])
            flux_m_i_plus_1 = self.F_m(self.h_node_n[i + 1], self.u_node_n[i + 1])

            flux_h_i = self.F_h(self.h_node_n[i], self.u_node_n[i])
            flux_h_i_plus_1 = self.F_h(self.h_node_n[i + 1], self.u_node_n[i + 1])

            self.h_cell_n_plus_half[i] = h_cell_n_curr - 0.5 * self.dt / self.dx * (flux_m_i_plus_1 - flux_m_i)
            self.hu_cell_n_plus_half[i] = hu_cell_n_curr - 0.5 * self.dt / self.dx * (flux_h_i_plus_1 - flux_h_i)

        # Dry cell treatment after conservative update
        self.h_cell_n_plus_half, self.hu_cell_n_plus_half = self._apply_dry_cell(
            self.h_cell_n_plus_half, self.hu_cell_n_plus_half)

    def _step3(self):
        """Phase 3 with dry cell treatment."""
        u_node_n_plus_1_char = self._safe_velocity(
            self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char, self.dry_eps)

        self.h_cell_n_plus_1 = np.zeros(self.N_cells)
        self.hu_cell_n_plus_1 = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            flux_m_i_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_m_i_plus_1_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            flux_h_i_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_h_i_plus_1_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            self.h_cell_n_plus_1[i] = self.h_cell_n_plus_half[i] - \
                                      0.5 * self.dt / self.dx * (flux_m_i_plus_1_n_plus_1 - flux_m_i_n_plus_1)
            self.hu_cell_n_plus_1[i] = self.hu_cell_n_plus_half[i] - \
                                       0.5 * self.dt / self.dx * (flux_h_i_plus_1_n_plus_1 - flux_h_i_n_plus_1)

        # Dry cell treatment after conservative update
        self.h_cell_n_plus_1, self.hu_cell_n_plus_1 = self._apply_dry_cell(
            self.h_cell_n_plus_1, self.hu_cell_n_plus_1)

    def _evaluate_lagrange_I1(self, u, c, j):
        """Evaluate the Lagrange polynomial P2(xi_hat) for invariant I1.

        Constructs a 2nd-order interpolating polynomial (eq 13) through:
          - (I1)^{n+1/2}_{j-1} at xi = 0          (left cell center)
          - (I1)^n_j           at xi = xi_bar_1    (node, carried by characteristic from t_n)
          - (I1)^{n+1/2}_{j}   at xi = dx          (right cell center)

        Evaluates at xi_hat_1, the foot of the characteristic from (x_j, t_{n+1})
        going backward to t_{n+1/2} (eq 14).

        Args:
            u, c: unknown velocity and wave speed at node j, time n+1
            j: node index
        """
        eps = 1e-12
        dx = self.dx
        dt = self.dt
        dry = self.dry_eps

        # Eigenvalue at node j, time n
        u_j_n = self._safe_velocity(self.h_node_n[j:j+1], self.hu_node_n[j:j+1], dry)[0]
        c_j_n = np.sqrt(self.g * max(0.0, self.h_node_n[j]))
        lambda1_n = u_j_n + c_j_n

        # xi_bar_1: foot of characteristic from (x_j, t_n) forward to t_{n+1/2}
        # eq (12): xi_bar_1 = 0.5 * h_{i-1/2} + 0.5 * tau * lambda1^n
        xi_bar = 0.5 * dx + 0.5 * dt * lambda1_n

        # xi_hat_1: foot of characteristic from (x_j, t_{n+1}) backward to t_{n+1/2}
        # eq (14): xi_hat_1 = 0.5 * h_{i-1/2} - 0.5 * tau * (u + c)
        xi_hat = 0.5 * dx - 0.5 * dt * (u + c)

        # I1 values at the three interpolation points
        u_cell_half = self._safe_velocity(self.h_cell_n_plus_half, self.hu_cell_n_plus_half, dry)
        c_cell_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))
        u_node_n = self._safe_velocity(self.h_node_n, self.hu_node_n, dry)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))

        f0 = u_cell_half[j - 1] + 2.0 * c_cell_half[j - 1]  # I1 at left cell
        f1 = u_node_n[j] + 2.0 * c_node_n[j]                  # I1 at node j
        f2 = u_cell_half[j] + 2.0 * c_cell_half[j]            # I1 at right cell

        # Lagrange basis polynomials evaluated at xi_hat (eq 13)
        # L0 at xi=0:       L0 = (xi_hat - xi_bar)(xi_hat - dx) / [(0 - xi_bar)(0 - dx)]
        # L1 at xi=xi_bar:  L1 = (xi_hat)(xi_hat - dx)          / [(xi_bar)(xi_bar - dx)]
        # L2 at xi=dx:      L2 = (xi_hat)(xi_hat - xi_bar)      / [(dx)(dx - xi_bar)]
        denom0 = xi_bar * dx
        denom1 = xi_bar * (xi_bar - dx)
        denom2 = dx * (dx - xi_bar)

        # Safeguard denominators
        denom0 = denom0 if abs(denom0) > eps else np.copysign(eps, denom0) if denom0 != 0 else eps
        denom1 = denom1 if abs(denom1) > eps else np.copysign(eps, denom1) if denom1 != 0 else eps
        denom2 = denom2 if abs(denom2) > eps else np.copysign(eps, denom2) if denom2 != 0 else eps

        L0 = (xi_hat - xi_bar) * (xi_hat - dx) / denom0
        L1 = xi_hat * (xi_hat - dx) / denom1
        L2 = xi_hat * (xi_hat - xi_bar) / denom2

        return f0 * L0 + f1 * L1 + f2 * L2

    def _evaluate_lagrange_I2(self, u, c, j):
        """Evaluate the Lagrange polynomial P2(xi_hat) for invariant I2.

        Same structure as _evaluate_lagrange_I1 but for the I2 characteristic,
        using lambda2 = u - c and I2 = u - 2c (eq 16).
        """
        eps = 1e-12
        dx = self.dx
        dt = self.dt
        dry = self.dry_eps

        # Eigenvalue at node j, time n
        u_j_n = self._safe_velocity(self.h_node_n[j:j+1], self.hu_node_n[j:j+1], dry)[0]
        c_j_n = np.sqrt(self.g * max(0.0, self.h_node_n[j]))
        lambda2_n = u_j_n - c_j_n

        # xi_bar_2: foot of characteristic from (x_j, t_n) forward
        xi_bar = 0.5 * dx + 0.5 * dt * lambda2_n

        # xi_hat_2: foot of characteristic from (x_j, t_{n+1}) backward
        xi_hat = 0.5 * dx - 0.5 * dt * (u - c)

        # I2 values at the three interpolation points
        u_cell_half = self._safe_velocity(self.h_cell_n_plus_half, self.hu_cell_n_plus_half, dry)
        c_cell_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))
        u_node_n = self._safe_velocity(self.h_node_n, self.hu_node_n, dry)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))

        f0 = u_cell_half[j - 1] - 2.0 * c_cell_half[j - 1]  # I2 at left cell
        f1 = u_node_n[j] - 2.0 * c_node_n[j]                  # I2 at node j
        f2 = u_cell_half[j] - 2.0 * c_cell_half[j]            # I2 at right cell

        # Lagrange basis polynomials evaluated at xi_hat
        denom0 = xi_bar * dx
        denom1 = xi_bar * (xi_bar - dx)
        denom2 = dx * (dx - xi_bar)

        denom0 = denom0 if abs(denom0) > eps else np.copysign(eps, denom0) if denom0 != 0 else eps
        denom1 = denom1 if abs(denom1) > eps else np.copysign(eps, denom1) if denom1 != 0 else eps
        denom2 = denom2 if abs(denom2) > eps else np.copysign(eps, denom2) if denom2 != 0 else eps

        L0 = (xi_hat - xi_bar) * (xi_hat - dx) / denom0
        L1 = xi_hat * (xi_hat - dx) / denom1
        L2 = xi_hat * (xi_hat - xi_bar) / denom2

        return f0 * L0 + f1 * L1 + f2 * L2

    def _compute_residual_F1(self, u, c, j):
        """Residual F1 = (u + 2c) - P2(xi_hat_1) for equation (15)."""
        return (u + 2.0 * c) - self._evaluate_lagrange_I1(u, c, j)

    def _compute_residual_F2(self, u, c, j):
        """Residual F2 = (u - 2c) - P2(xi_hat_2) for equation (16)."""
        return (u - 2.0 * c) - self._evaluate_lagrange_I2(u, c, j)

    def _get_initial_guess(self, j):
        """Get initial guess for Newton's method using the averaging method (eq 11).

        Uses the average of invariants from neighboring cells as starting point,
        which typically gives convergence in 2-3 Newton iterations.
        """
        dry = self.dry_eps
        u_cell = self._safe_velocity(self.h_cell_n_plus_half, self.hu_cell_n_plus_half, dry)
        c_cell = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

        I1_avg = 0.5 * ((u_cell[j - 1] + 2 * c_cell[j - 1]) + (u_cell[j] + 2 * c_cell[j]))
        I2_avg = 0.5 * ((u_cell[j - 1] - 2 * c_cell[j - 1]) + (u_cell[j] - 2 * c_cell[j]))

        u_guess = 0.5 * (I1_avg + I2_avg)
        c_guess = 0.25 * (I1_avg - I2_avg)
        c_guess = max(c_guess, 1e-6)

        return u_guess, c_guess

    def _solve_implicit_case_A(self, j, I2_value):
        """Case A (eq 17): Sonic only in I1.

        Use constraint u - 2c = I2_value to reduce to one equation F1(c) = 0.
        """
        _, c_guess = self._get_initial_guess(j)
        eps = 1e-12

        def residual_c(c_val):
            c_val = max(c_val, eps)
            u_val = I2_value + 2.0 * c_val
            return self._compute_residual_F1(u_val, c_val, j)

        try:
            c_sol = newton(residual_c, c_guess, tol=self.newton_tol, maxiter=self.newton_max_iter)
            c_sol = max(c_sol, eps)
        except (RuntimeError, ValueError):
            try:
                c_sol = fsolve(residual_c, c_guess, xtol=self.newton_tol, maxfev=self.newton_max_iter * 10)[0]
                c_sol = max(c_sol, eps)
            except (RuntimeError, ValueError):
                c_sol = c_guess

        u_sol = I2_value + 2.0 * c_sol
        return u_sol, c_sol

    def _solve_implicit_case_B(self, j, I1_value):
        """Case B (eq 18): Sonic only in I2.

        Use constraint u + 2c = I1_value to reduce to one equation F2(c) = 0.
        """
        _, c_guess = self._get_initial_guess(j)
        eps = 1e-12

        def residual_c(c_val):
            c_val = max(c_val, eps)
            u_val = I1_value - 2.0 * c_val
            return self._compute_residual_F2(u_val, c_val, j)

        try:
            c_sol = newton(residual_c, c_guess, tol=self.newton_tol, maxiter=self.newton_max_iter)
            c_sol = max(c_sol, eps)
        except (RuntimeError, ValueError):
            try:
                c_sol = fsolve(residual_c, c_guess, xtol=self.newton_tol, maxfev=self.newton_max_iter * 10)[0]
                c_sol = max(c_sol, eps)
            except (RuntimeError, ValueError):
                c_sol = c_guess

        u_sol = I1_value - 2.0 * c_sol
        return u_sol, c_sol

    def _solve_implicit_case_C(self, j):
        """Case C: Sonic in both I1 and I2.

        Solve the full 2x2 system F1(u, c) = 0, F2(u, c) = 0 (eqs 15-16).
        """
        u_guess, c_guess = self._get_initial_guess(j)
        eps = 1e-12

        def residuals(x):
            u_val, c_val = x
            c_val = max(c_val, eps)
            F1 = self._compute_residual_F1(u_val, c_val, j)
            F2 = self._compute_residual_F2(u_val, c_val, j)
            return [F1, F2]

        try:
            solution = fsolve(residuals, [u_guess, c_guess],
                              xtol=self.newton_tol, maxfev=self.newton_max_iter * 10)
            u_sol = solution[0]
            c_sol = max(solution[1], eps)
        except (RuntimeError, ValueError):
            u_sol, c_sol = u_guess, c_guess

        return u_sol, c_sol

    def _step2(self):
        """Characteristic phase with locally implicit sonic point processing.

        For each internal node:
        1. Detect sonic points (eigenvalue sign change across adjacent cells)
        2. Compute provisional invariants:
           - Non-sonic: standard extrapolation (eq 4)
           - Sonic: implicit Lagrange interpolation along characteristics (eqs 15-18)
        3. Monotonize using maximum principle:
           - Non-sonic: standard bounds (eq 5)
           - Sonic: wider bounds using both adjacent cells (eq 19)
        4. Recover h, u from corrected invariants (eq 6)
        """
        eps = 1e-12
        dry = self.dry_eps

        # Cell center values at n+1/2
        u_cell_half = self._safe_velocity(self.h_cell_n_plus_half, self.hu_cell_n_plus_half, dry)
        c_cell_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

        I1_cell_half = u_cell_half + 2.0 * c_cell_half
        I2_cell_half = u_cell_half - 2.0 * c_cell_half

        lambda1_cell_half = u_cell_half + c_cell_half
        lambda2_cell_half = u_cell_half - c_cell_half

        # Node values at n
        u_node_n = self._safe_velocity(self.h_node_n, self.hu_node_n, dry)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))

        I1_node_n = u_node_n + 2.0 * c_node_n
        I2_node_n = u_node_n - 2.0 * c_node_n

        # Initialize arrays
        I1_node_new = np.zeros(self.N_nodes)
        I2_node_new = np.zeros(self.N_nodes)

        self.h_node_n_plus_1_char = np.zeros(self.N_nodes)
        self.hu_node_n_plus_1_char = np.zeros(self.N_nodes)

        for j in range(1, self.N_nodes - 1):
            # --- Step 1: Sonic Point Detection (Section 5) ---
            sonic_I1 = (lambda1_cell_half[j - 1] * lambda1_cell_half[j] <= 0)
            sonic_I2 = (lambda2_cell_half[j - 1] * lambda2_cell_half[j] <= 0)

            # --- Step 2 & 3: Compute provisional (tilde) invariants ---
            if sonic_I1 and sonic_I2:
                # Case C: Both invariants sonic - solve full implicit system
                u_impl, c_impl = self._solve_implicit_case_C(j)
                I1_tilde = u_impl + 2.0 * c_impl
                I2_tilde = u_impl - 2.0 * c_impl

            elif sonic_I1:
                # Case A: Only I1 is sonic
                # Standard extrapolation for I2 (eq 4)
                if lambda2_cell_half[j - 1] >= 0:
                    I2_tilde = 2.0 * I2_cell_half[j - 1] - I2_node_n[j - 1]
                else:
                    I2_tilde = 2.0 * I2_cell_half[j] - I2_node_n[j + 1]
                # Implicit solve for I1 with I2 constraint (eq 17)
                u_impl, c_impl = self._solve_implicit_case_A(j, I2_tilde)
                I1_tilde = u_impl + 2.0 * c_impl

            elif sonic_I2:
                # Case B: Only I2 is sonic
                # Standard extrapolation for I1 (eq 4)
                if lambda1_cell_half[j - 1] >= 0:
                    I1_tilde = 2.0 * I1_cell_half[j - 1] - I1_node_n[j - 1]
                else:
                    I1_tilde = 2.0 * I1_cell_half[j] - I1_node_n[j + 1]
                # Implicit solve for I2 with I1 constraint (eq 18)
                u_impl, c_impl = self._solve_implicit_case_B(j, I1_tilde)
                I2_tilde = u_impl - 2.0 * c_impl

            else:
                # No sonic point: Standard CABARET extrapolation (eq 4)
                if lambda1_cell_half[j - 1] >= 0:
                    I1_tilde = 2.0 * I1_cell_half[j - 1] - I1_node_n[j - 1]
                else:
                    I1_tilde = 2.0 * I1_cell_half[j] - I1_node_n[j + 1]

                if lambda2_cell_half[j - 1] >= 0:
                    I2_tilde = 2.0 * I2_cell_half[j - 1] - I2_node_n[j - 1]
                else:
                    I2_tilde = 2.0 * I2_cell_half[j] - I2_node_n[j + 1]

            # --- Step 4: Monotonization (Non-Linear Correction) ---
            # Each invariant gets bounds based on whether IT is sonic,
            # not whether the node has any sonic point.
            # Sonic invariant (eq 19): both cell centers + node value
            # Non-sonic invariant (eq 5): relevant cell + its two bounding nodes
            if sonic_I1:
                min_I1 = min(I1_cell_half[j - 1], I1_node_n[j], I1_cell_half[j])
                max_I1 = max(I1_cell_half[j - 1], I1_node_n[j], I1_cell_half[j])
            else:
                if lambda1_cell_half[j - 1] >= 0:
                    idx1 = j - 1
                else:
                    idx1 = j
                min_I1 = min(I1_node_n[idx1], I1_cell_half[idx1], I1_node_n[idx1 + 1])
                max_I1 = max(I1_node_n[idx1], I1_cell_half[idx1], I1_node_n[idx1 + 1])

            if sonic_I2:
                min_I2 = min(I2_cell_half[j - 1], I2_node_n[j], I2_cell_half[j])
                max_I2 = max(I2_cell_half[j - 1], I2_node_n[j], I2_cell_half[j])
            else:
                if lambda2_cell_half[j - 1] >= 0:
                    idx2 = j - 1
                else:
                    idx2 = j
                min_I2 = min(I2_node_n[idx2], I2_cell_half[idx2], I2_node_n[idx2 + 1])
                max_I2 = max(I2_node_n[idx2], I2_cell_half[idx2], I2_node_n[idx2 + 1])

            if self.monotonize:
                I1_node_new[j] = np.clip(I1_tilde, min_I1, max_I1)
                I2_node_new[j] = np.clip(I2_tilde, min_I2, max_I2)
            else:
                I1_node_new[j] = I1_tilde
                I2_node_new[j] = I2_tilde

            # --- Step 5: Recover h, u from invariants (eq 6) ---
            self.h_node_n_plus_1_char[j] = ((I1_node_new[j] - I2_node_new[j]) / 4.0) ** 2 / self.g
            u_new = (I1_node_new[j] + I2_node_new[j]) / 2.0

            self.h_node_n_plus_1_char[j] = max(0.0, self.h_node_n_plus_1_char[j])
            self.hu_node_n_plus_1_char[j] = self.h_node_n_plus_1_char[j] * u_new

        # Boundary conditions
        self.h_node_n_plus_1_char[0] = self.h_node_n[0]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n[0]
        self.h_node_n_plus_1_char[-1] = self.h_node_n[-1]
        self.hu_node_n_plus_1_char[-1] = self.hu_node_n[-1]

        # Dry cell treatment for nodes
        self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char = self._apply_dry_cell(
            self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char)



class RiemannSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        self.solver_func = solver_func
        # print(self.solver_func)

    def solve(self, x, t: float, h_l: float, u_l: float, h_r: float, u_r: float):
        # res = riemann_solver_newton(h_l, h_l * u_l, h_r, h_r * u_r)
        res = riemann_solver_scipy(h_l, h_l * u_l, h_r, h_r * u_r)
        # print(res)
        h_star, u_star = res['star']
        D_L, D_starL, D_R, D_starR = res['velocity']

        if h_star >= h_l:
            # Left shock wave
            D_L = u_l - 1 / h_l * np.sqrt(self.g / 2 * (h_l * h_star * (h_l + h_star)))
            D_starL = None
        else:
            # Left rarefaction
            D_L = u_l - np.sqrt(self.g * h_l)
            D_starL = u_star - np.sqrt(self.g * h_star)

        if h_star >= h_r:
            # Right shock wave
            D_R = u_r + 1 / h_r * np.sqrt(self.g / 2 * (h_r * h_star * (h_r + h_star)))
            D_starR = None
        else:
            # Right rarefaction
            D_R = u_r + np.sqrt(self.g * h_r)
            D_starR = u_star + np.sqrt(self.g * h_star)

        left_star_boundary = D_starL if D_starL is not None else D_L
        right_star_boundary = D_starR if D_starR is not None else D_R

        xi = x / t

        def h_profile(xi):
            h = np.empty_like(xi)

            # Region I: Left constant state (xi < D_L)
            cond1 = xi < D_L

            # Region II: Left rarefaction fan (if applicable)
            cond2 = (D_starL is not None) & (xi >= D_L) & (xi < left_star_boundary)

            # Region III: Star region (between left and right star boundaries)
            cond3 = (xi >= left_star_boundary) & (xi < right_star_boundary)

            # Region IV: Right rarefaction fan (if applicable)
            cond4 = (D_starR is not None) & (xi >= right_star_boundary) & (xi < D_R)

            # Region V: Right constant state (xi >= D_R)
            cond5 = xi >= D_R

            h[cond1] = h_l

            if D_starL is not None:
                # For left rarefaction fan, use similarity solution:
                h[cond2] = ((u_l + 2 * np.sqrt(self.g * h_l) - xi[cond2]) / (3 * np.sqrt(self.g))) ** 2
            else:
                # If no rarefaction, there is no fan region (shock)
                h[cond2] = h_l

            h[cond3] = h_star

            if D_starR is not None:
                # For right rarefaction fan, use similarity solution:
                h[cond4] = ((xi[cond4] - u_r + 2 * np.sqrt(self.g * h_r)) / (3 * np.sqrt(self.g))) ** 2
            else:
                h[cond4] = h_r

            h[cond5] = h_r

            return h

        def u_profile(xi):
            u = np.empty_like(xi)

            # Region I: Left constant state
            cond1 = xi < D_L

            # Region II: Left rarefaction fan
            cond2 = (D_starL is not None) & (xi >= D_L) & (xi < left_star_boundary)

            # Region III: Star region
            cond3 = (xi >= left_star_boundary) & (xi < right_star_boundary)

            # Region IV: Right rarefaction fan
            cond4 = (D_starR is not None) & (xi >= right_star_boundary) & (xi < D_R)

            # Region V: Right constant state
            cond5 = xi >= D_R

            u[cond1] = u_l

            if D_starL is not None:
                # For left rarefaction fan:
                u[cond2] = u_l + (2 / 3) * (xi[cond2] - (u_l - np.sqrt(self.g * h_l)))
            else:
                u[cond2] = u_l

            u[cond3] = u_star

            if D_starR is not None:
                # For right rarefaction fan:
                u[cond4] = u_r - (2 / 3) * ((u_r + np.sqrt(self.g * h_r)) - xi[cond4])
            else:
                u[cond4] = u_r

            u[cond5] = u_r

            return u

        h_vals = h_profile(xi)
        u_vals = u_profile(xi)

        return {
            "vals": (h_vals, u_vals),
            "bounds": [D_L, D_starL, D_R, D_starR, left_star_boundary, right_star_boundary]
        }
