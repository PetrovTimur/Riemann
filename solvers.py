import numpy as np
import torch


def riemann_solver_newton(hL, huL, hR, huR, g = 9.8066, tol=1e-6, max_iter=1000):

    uL = huL / hL if hL > 0 else 0
    uR = huR / hR if hR > 0 else 0

    h_k = np.array([hL, hR])
    u_k = np.array([uL, uR])
    c_k = np.sqrt(g * h_k)

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


class Solver:
    def __init__(self):
        pass

class GodunovSolver(Solver):
    def __init__(self, solver_func='classic', model=None, g=9.81):
        super().__init__()
        self.g = g
        self.model = model
        self.solver_func = solver_func
        print(self.solver_func)

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


class CabaretSolver(Solver):
    def __init__(self, solver_func='classic', g=9.81):
        super().__init__()
        self.g = g

        # Acceptable solver options: 'default', 'iter', 'newton', 'nn'
        self.solver_func = solver_func
        print("Cabaret solver: ", self.solver_func)

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
        self.u = self.hu / (self.h + 1e-12)

        # В четных точках старые инварианты (n слой), в консервативных точках на полуцелом шаге по времени (n + 1/2)
        neg_char_new = self.u - 2 * np.sqrt(np.maximum(0, self.g * self.h))
        pos_char_new = self.u + 2 * np.sqrt(np.maximum(0, self.g * self.h))

        for i in range(2, neg_char_new.shape[0] - 1, 2):
            # lambda_left_neg = self.u[i - 1] - (self.g * self.h[i - 1]) ** 0.5
            # lambda_left_pos = self.u[i - 1] + (self.g * self.h[i - 1]) ** 0.5
            # lambda_right_neg = self.u[i + 1] - (self.g * self.h[i + 1]) ** 0.5
            # lambda_right_pos = self.u[i + 1] + (self.g * self.h[i + 1]) ** 0.5

            # Нейр
            if self.solver_func == 'selective_nn' and ((np.sign(lambda_left_neg) != np.sign(lambda_right_neg)) or (
                    np.sign(lambda_left_pos) != np.sign(lambda_right_pos))):
                # print('test')
                fluxx = riemann_solver_nn(self.h[i - 1], self.hu[i - 1], self.h[i + 1], self.hu[i + 1], self.model,
                                          self.g)
                _, _, hh, uu = fluxx
                neg_char_new[i] = uu - 2 * (np.maximum(1e-12, self.g * hh)) ** 0.5
                pos_char_new[i] = uu + 2 * (np.maximum(1e-12, self.g * hh)) ** 0.5

            # Обычная линейная экстраполяция
            else:
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
    def __init__(self, model, g=9.81):
        """
        Initializes the CABARET solver with gravitational acceleration.
        Assumes a flat bottom (b=0) for shallow water equations.

        Args:
            g (float): Acceleration due to gravity (default: 9.81 m/s^2).
        """
        super().__init__(g=g)

        # Internal state variables to hold data for the current time step.
        # 'combined' arrays contain both nodal (even indices) and cell-centered (odd indices) values.
        self.h_combined_n = None
        self.hu_combined_n = None

        # Derived views/copies for easier indexing of nodal and cell-centered values at time 'n'.
        self.h_node_n = None  # Nodal H at time n (H_i^n)
        self.hu_node_n = None  # Nodal Hu at time n (Hu_i^n)
        self.u_node_n = None  # Nodal u at time n (u_i^n)

        self.h_cell_n = None  # Cell-centered H at time n (H_i+1/2^n)
        self.hu_cell_n = None  # Cell-centered Hu at time n (Hu_i+1/2^n)

        # Intermediate results for the CABARET scheme steps:
        self.h_cell_n_plus_half = None  # H_i+1/2^(n+1/2) - conservative variables at cell centers, intermediate time
        self.hu_cell_n_plus_half = None  # Hu_i+1/2^(n+1/2) - conservative variables at cell centers, intermediate time

        self.h_node_n_plus_1_char = None  # H_i^(n+1) - provisional nodal values from characteristic step
        self.hu_node_n_plus_1_char = None  # Hu_i^(n+1) - provisional nodal values from characteristic step

        self.h_cell_n_plus_1 = None  # H_i+1/2^(n+1) - final conservative variables at cell centers, new time
        self.hu_cell_n_plus_1 = None  # Hu_i+1/2^(n+1) - final conservative variables at cell centers, new time

        # Spatial and temporal discretizations parameters:
        self.dx = None  # Single float for uniform cell width.
        self.dt = None  # Single float for time step.

        # Dimensions of the grid (derived from input array lengths):
        self.N_total_points = 0  # Total number of points in the combined array.
        # N_total_points = 2 * N_cells + 1.
        self.N_nodes = 0  # Number of nodal points (N_nodes = N_cells + 1).
        self.N_cells = 0  # Number of computational cells.

    def F_m(self, H, u):
        """Mass flux function F_m = Hu."""
        return H * u

    def F_h(self, H, u):
        """Momentum flux function F_h = Hu^2 + g/2 H^2."""
        return H * u ** 2 + 0.5 * self.g * H ** 2

    def _step1(self):
        """
        Step 1: Compute intermediate conservative variables at cell centers (n+1/2).
        Uses an explicit conservative scheme (Eq. 3 from the paper, simplified for b=0).
        Inputs: self.h_node_n, self.hu_node_n (at time n, nodal), self.h_cell_n, self.hu_cell_n (at time n, cell-centered).
        Outputs: Populates self.h_cell_n_plus_half, self.hu_cell_n_plus_half (at time n+1/2, cell-centered).
        """
        # Calculate u_node at time n for flux computation.
        # Add a small epsilon to avoid division by zero if H is zero.
        self.u_node_n = self.hu_node_n / (self.h_node_n + 1e-12)

        self.h_cell_n_plus_half = np.zeros(self.N_cells)
        self.hu_cell_n_plus_half = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            # H_i+1/2^n and Hu_i+1/2^n are directly available from the odd indices of the input combined array.
            h_cell_n_curr = self.h_cell_n[i]
            hu_cell_n_curr = self.hu_cell_n[i]

            # Fluxes at nodes i and i+1 using nodal values at time n.
            flux_m_i = self.F_m(self.h_node_n[i], self.u_node_n[i])
            flux_m_i_plus_1 = self.F_m(self.h_node_n[i + 1], self.u_node_n[i + 1])

            flux_h_i = self.F_h(self.h_node_n[i], self.u_node_n[i])
            flux_h_i_plus_1 = self.F_h(self.h_node_n[i + 1], self.u_node_n[i + 1])

            # Apply explicit update for conservative variables (based on Eq. 3 with b=0).
            self.h_cell_n_plus_half[i] = h_cell_n_curr - 0.5 * self.dt / self.dx * (flux_m_i_plus_1 - flux_m_i)
            self.hu_cell_n_plus_half[i] = hu_cell_n_curr - 0.5 * self.dt / self.dx * (flux_h_i_plus_1 - flux_h_i)

    def _correct_invariants(self, extrapolated_invariant, min_bound, max_bound):
        """
        Applies nonlinear correction to an extrapolated Riemann invariant (Chapter 6).
        This clips the extrapolated value to be within the min/max bounds of the surrounding points.

        Args:
            extrapolated_invariant (float): The Riemann invariant value after extrapolation.
            min_bound (float): The minimum value among relevant points at time 'n'.
            max_bound (float): The maximum value among relevant points at time 'n'.

        Returns:
            float: The corrected (clipped) Riemann invariant.
        """
        return np.clip(extrapolated_invariant, min_bound, max_bound)

    def _step2(self):
        """
        Step 2: Compute provisional nodal flux variables (H_i^(n+1), Hu_i^(n+1)) using the characteristic method.
        This step integrates:
        1. Transsonic point handling (Chapter 8, using averaged characteristic speeds for extrapolation direction).
        2. Nonlinear correction (Chapter 6, applying max/min principle to Riemann invariants).

        Inputs: self.h_cell_n_plus_half, self.hu_cell_n_plus_half (at n+1/2, cell-centered),
                self.h_node_n, self.hu_node_n (at n, nodal), self.h_cell_n, self.hu_cell_n (at n, cell-centered).
        Outputs: Populates self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char (provisional nodal values at time n+1).
        """
        # Calculate u, c, I1, I2 (Riemann invariants) at cell centers at n+1/2.
        u_cell_n_plus_half = self.hu_cell_n_plus_half / (self.h_cell_n_plus_half + 1e-12)
        c_cell_n_plus_half = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n_plus_half))

        # Using Riemann invariants as defined in the paper (below Eq. 7, page 4): I1 = u + c, I2 = u - c
        I1_cell_n_plus_half = u_cell_n_plus_half + c_cell_n_plus_half
        I2_cell_n_plus_half = u_cell_n_plus_half - c_cell_n_plus_half

        # Characteristic speeds (lambda values) are still defined as u +/- c (as in Eq. 2 of the paper).
        # These are numerically equivalent to the chosen Riemann invariants I1, I2 in this paper's formulation.
        lambda1_cell_n_plus_half = u_cell_n_plus_half + c_cell_n_plus_half
        lambda2_cell_n_plus_half = u_cell_n_plus_half - c_cell_n_plus_half

        # Calculate u, c, I1, I2 at nodes at time n.
        u_node_n = self.hu_node_n / (self.h_node_n + 1e-12)
        c_node_n = np.sqrt(self.g * np.maximum(0.0, self.h_node_n))
        # Using Riemann invariants as defined in the paper for extrapolation.
        I1_node_n = u_node_n + c_node_n
        I2_node_n = u_node_n - c_node_n

        # Calculate u, c, I1, I2 at cell centers at time n (for nonlinear correction bounds).
        u_cell_n = self.hu_cell_n / (self.h_cell_n + 1e-12)
        c_cell_n = np.sqrt(self.g * np.maximum(0.0, self.h_cell_n))
        # Using Riemann invariants as defined in the paper for bounds.
        I1_cell_n = u_cell_n + c_cell_n
        I2_cell_n = u_cell_n - c_cell_n

        # Initialize arrays for new nodal Riemann invariants at n+1.
        I1_node_n_plus_1 = np.zeros(self.N_nodes)
        I2_node_n_plus_1 = np.zeros(self.N_nodes)

        self.h_node_n_plus_1_char = np.zeros(self.N_nodes)
        self.hu_node_n_plus_1_char = np.zeros(self.N_nodes)

        # Apply extrapolation for interior nodes (j from 1 to N_nodes-2).
        for j in range(1, self.N_nodes - 1):
            # --- Chapter 8: Transsonic Point Handling (Extrapolation Direction) ---
            # Determine effective characteristic speed at node j by averaging adjacent cell values (Chapter 8, Eq. 13).
            # Cell j-1 is to the left of node j, and Cell j is to the right of node j.
            lambda1_avg_at_node_j = 0.5 * (lambda1_cell_n_plus_half[j - 1] + lambda1_cell_n_plus_half[j])
            lambda2_avg_at_node_j = 0.5 * (lambda2_cell_n_plus_half[j - 1] + lambda2_cell_n_plus_half[j])

            # Extrapolate I1 based on averaged lambda1 (Eq. from Chapter 5, simplified).
            if lambda1_avg_at_node_j > 0:  # Characteristic comes from the left (cell j-1).
                I1_extrapolated = 2 * I1_cell_n_plus_half[j - 1] - I1_node_n[j - 1]
                relevant_cell_idx_I1 = j - 1  # For nonlinear correction, bounds come from this cell
            else:  # Characteristic comes from the right (cell j).
                I1_extrapolated = 2 * I1_cell_n_plus_half[j] - I1_node_n[j + 1]
                relevant_cell_idx_I1 = j  # For nonlinear correction, bounds come from this cell

            # Extrapolate I2 based on averaged lambda2.
            if lambda2_avg_at_node_j > 0:  # Characteristic comes from the left (cell j-1).
                I2_extrapolated = 2 * I2_cell_n_plus_half[j - 1] - I2_node_n[j - 1]
                relevant_cell_idx_I2 = j - 1  # For nonlinear correction, bounds come from this cell
            else:  # Characteristic comes from the right (cell j).
                I2_extrapolated = 2 * I2_cell_n_plus_half[j] - I2_node_n[j + 1]
                relevant_cell_idx_I2 = j  # For nonlinear correction, bounds come from this cell

            if relevant_cell_idx_I1 >= relevant_cell_idx_I2:
                print('hhh: ', j)

            # print(j, I1_extrapolated, I2_extrapolated)

            # --- Chapter 6: Nonlinear Correction of Riemann Invariants ---
            # Calculate bounds for I1 from the relevant cell (based on Chapter 6 text and formulas).
            # The bounds for a cell_idx (e.g., cell i+1/2) are derived from values at node i, cell i+1/2, node i+1.
            # Using I1_node_n[relevant_cell_idx_I1], I1_cell_n[relevant_cell_idx_I1], I1_node_n[relevant_cell_idx_I1+1]
            # to determine the min/max range for the current cell's neighborhood.
            max_I1_bound = np.max([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])
            min_I1_bound = np.min([I1_node_n[relevant_cell_idx_I1],
                                   I1_cell_n[relevant_cell_idx_I1],
                                   I1_node_n[relevant_cell_idx_I1 + 1]])

            # Apply correction to I1 using the dedicated function.
            I1_node_n_plus_1[j] = self._correct_invariants(I1_extrapolated, min_I1_bound, max_I1_bound)

            # Calculate bounds for I2 from the relevant cell.
            max_I2_bound = np.max([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])
            min_I2_bound = np.min([I2_node_n[relevant_cell_idx_I2],
                                   I2_cell_n[relevant_cell_idx_I2],
                                   I2_node_n[relevant_cell_idx_I2 + 1]])

            # Apply correction to I2 using the dedicated function.
            I2_node_n_plus_1[j] = self._correct_invariants(I2_extrapolated, min_I2_bound, max_I2_bound)

            # --- Chapter 7: Invert Riemann invariants to get u and H at the provisional nodes n+1 (Eq. 12) ---
            # G_k* = g / c_k*, where c_k* is the wave speed from the cell the characteristic k originated from.
            # We need to ensure c_cell_n_plus_half[relevant_cell_idx_I1/I2] is not zero.
            g1_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I1] + 1e-12)
            g2_star = self.g / (c_cell_n_plus_half[relevant_cell_idx_I2] + 1e-12)

            # Reconstruct H at node j (H_j^(n+1))
            # self.h_node_n_plus_1_char[j] = (I1_node_n_plus_1[j] - I2_node_n_plus_1[j]) / (g1_star + g2_star)
            # Reconstruct u at node j (u_j^(n+1))
            # u_node_n_plus_1_at_j = (I1_node_n_plus_1[j] * g2_star + I2_node_n_plus_1[j] * g1_star) / (g1_star + g2_star)
            self.h_node_n_plus_1_char[j] = ((I1_node_n_plus_1[j] - I2_node_n_plus_1[j]) / 2) ** 2 / self.g
            u_node_n_plus_1_at_j = (I1_node_n_plus_1[j] + I2_node_n_plus_1[j]) / 2

            # Ensure H is not negative due to numerical errors; set a small positive floor before computing Hu.
            self.h_node_n_plus_1_char[j] = np.maximum(0.0, self.h_node_n_plus_1_char[j])
            self.hu_node_n_plus_1_char[j] = self.h_node_n_plus_1_char[j] * u_node_n_plus_1_at_j


        # Apply boundary conditions to the provisional nodal values (at Node 0 and Node N_nodes-1).
        # For this example, we assume fixed boundary conditions, copying from the initial state at 'n'.
        # These provisional boundary values will be used directly in the final output.
        self.h_node_n_plus_1_char[0] = self.h_node_n[0]
        self.hu_node_n_plus_1_char[0] = self.hu_node_n[0]
        self.h_node_n_plus_1_char[self.N_nodes - 1] = self.h_node_n[self.N_nodes - 1]
        self.hu_node_n_plus_1_char[self.N_nodes - 1] = self.hu_node_n[self.N_nodes - 1]

    def _step3(self):
        """
        Step 3: Compute final conservative variables at cell centers (n+1).
        Uses an implicit conservative scheme (Eq. 4 from the paper, simplified for b=0).
        Inputs: self.h_cell_n_plus_half, self.hu_cell_n_plus_half (at n+1/2, cell-centered),
                self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char (provisional at n+1, nodal).
        Outputs: Populates self.h_cell_n_plus_1, self.hu_cell_n_plus_1 (final cell-centered values at n+1).
        """
        # Calculate u_node at n+1 from the provisional nodal values.
        # Add a small epsilon to avoid division by zero if H is zero.
        u_node_n_plus_1_char = self.hu_node_n_plus_1_char / (self.h_node_n_plus_1_char + 1e-12)

        self.h_cell_n_plus_1 = np.zeros(self.N_cells)
        self.hu_cell_n_plus_1 = np.zeros(self.N_cells)

        for i in range(self.N_cells):
            # Fluxes at nodes i and i+1 using the provisional nodal values at n+1.
            flux_m_i_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_m_i_plus_1_n_plus_1 = self.F_m(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            flux_h_i_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i], u_node_n_plus_1_char[i])
            flux_h_i_plus_1_n_plus_1 = self.F_h(self.h_node_n_plus_1_char[i + 1], u_node_n_plus_1_char[i + 1])

            # Apply implicit update for conservative variables (based on Eq. 4 with b=0).
            self.h_cell_n_plus_1[i] = self.h_cell_n_plus_half[i] - \
                                      0.5 * self.dt / self.dx * (flux_m_i_plus_1_n_plus_1 - flux_m_i_n_plus_1)
            self.hu_cell_n_plus_1[i] = self.hu_cell_n_plus_half[i] - \
                                       0.5 * self.dt / self.dx * (flux_h_i_plus_1_n_plus_1 - flux_h_i_n_plus_1)

    def step(self, h_n_total_points, hu_n_total_points, dx, dt):
        """
        Performs one time step for the CABARET scheme, updating values from time 'n' to 'n+1'.

        Args:
            h_n_total_points (np.array): Combined array of H values at current time 'n'.
                                         - Even indices (0, 2, 4, ...): H_i^n (nodal values).
                                         - Odd indices (1, 3, 5, ...): H_i+1/2^n (cell-centered values).
            hu_n_total_points (np.array): Combined array of Hu values at current time 'n'.
                                          Same indexing convention as h_n_total_points.
            dx (float): The uniform cell width for the computational grid.
            dt (float): The time step for the current iteration.

        Returns:
            tuple: (h_n_plus_1_total_points, hu_n_plus_1_total_points)
                   Updated combined H and Hu arrays at time 'n+1', following the same
                   indexing convention as the input.
        """
        # Store input values and determine grid dimensions for the current time step.
        self.h_combined_n = h_n_total_points.copy()
        self.hu_combined_n = hu_n_total_points.copy()
        self.dx = dx
        self.dt = dt

        self.N_total_points = len(self.h_combined_n)
        # Assuming N_total_points = 2 * N_cells + 1 (e.g., 3 points -> 1 cell, 5 points -> 2 cells)
        self.N_cells = (self.N_total_points - 1) // 2
        self.N_nodes = self.N_cells + 1  # Total nodes is N_cells + 1

        # Extract nodal and cell-centered values from the combined input array for clarity in steps.
        self.h_node_n = self.h_combined_n[::2]  # H_0^n, H_1^n, ..., H_N_nodes-1^n
        self.hu_node_n = self.hu_combined_n[::2]  # Hu_0^n, Hu_1^n, ..., Hu_N_nodes-1^n
        self.h_cell_n = self.h_combined_n[1::2]  # H_0+1/2^n, H_1+1/2^n, ..., H_N_cells-1+1/2^n
        self.hu_cell_n = self.hu_combined_n[1::2]  # Hu_0+1/2^n, Hu_1+1/2^n, ..., Hu_N_cells-1+1/2^n

        # Execute the three main steps of the CABARET algorithm.
        self._step1()  # Computes self.h_cell_n_plus_half, self.hu_cell_n_plus_half
        self._step2()  # Computes self.h_node_n_plus_1_char, self.hu_node_n_plus_1_char (provisional nodal values with nonlinear correction)
        # print(dt, dx, self.h_node_n_plus_1_char)
        self._step3()  # Computes self.h_cell_n_plus_1, self.hu_cell_n_plus_1 (final cell-centered values)

        # Assemble the final combined output array for time 'n+1'.
        h_n_plus_1_total_points = np.zeros(self.N_total_points)
        hu_n_plus_1_total_points = np.zeros(self.N_total_points)

        # Populate odd indices with the final cell-centered values computed in _step3.
        # These correspond to H_i+1/2^(n+1) and Hu_i+1/2^(n+1).
        h_n_plus_1_total_points[1::2] = self.h_cell_n_plus_1
        hu_n_plus_1_total_points[1::2] = self.hu_cell_n_plus_1

        # Populate even indices with the final nodal values.
        # These come directly from the "potokovye peremennye na novom vremennom sloe" computed in _step2.
        h_n_plus_1_total_points[::2] = self.h_node_n_plus_1_char
        hu_n_plus_1_total_points[::2] = self.hu_node_n_plus_1_char

        return h_n_plus_1_total_points, hu_n_plus_1_total_points


class CabaretSolverNN(CabaretSolver):
    def __init__(self, model, g=9.81):
        super().__init__(g=g)

        self.model = model

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
                print('using nn', i)
                #Use nn
                # fluxx = riemann_solver_nn(self.h[i], self.hu[i], self.h[i + 2], self.hu[i + 2], self.model, self.g)
                fluxx = riemann_solver_newton(self.h[i], self.hu[i], self.h[i + 2], self.hu[i + 2])['flux']
                _, _, hh, uu = fluxx
                # self.h[i + 1] = hh
                # self.hu[i + 1] = hh * uu
                neg_char_new[i + 1] = uu - 2 * np.sqrt(self.g * hh)
                pos_char_new[i + 1] = uu + 2 * np.sqrt(self.g * hh)

            # print(i, self.h[i], self.hu[i], self.h[i + 2], self.hu[i + 2], self.h[i + 1], self.hu[i + 1])

        self._correct_invariants(self.pos_char, self.neg_char, pos_char_new, neg_char_new)

        for i in range(2, len(self.u) - 1, 2):
            self.u[i] = (neg_char_new[i] + pos_char_new[i]) / 2
            self.h[i] = ((pos_char_new[i] - neg_char_new[i]) / 4) ** 2 / self.g

            # self.h[i] = np.clip(self.h[i], a_min=min(self.h[i - 1], self.h[i + 1]), a_max=max(self.h[i - 1], self.h[i + 1]))

            self.hu[i] = self.h[i] * self.u[i]


class CabaretSolverImproved(CabaretSolver):
    def __init__(self, model, g=9.81):
        super().__init__(g=g)

    def _step2(self):
        self.u = self.hu / self.h

        # В четных точках старые инварианты (n слой), в консервативных точках на полуцелом шаге по времени (n + 1/2)
        neg_char_new = self.u - 2 * np.sqrt(self.g * self.h)
        pos_char_new = self.u + 2 * np.sqrt(self.g * self.h)

        for i in range(2, neg_char_new.shape[0] - 1, 2):
            lambda_left_neg = self.u[i - 1] - (self.g * self.h[i - 1]) ** 0.5
            lambda_left_pos = self.u[i - 1] + (self.g * self.h[i - 1]) ** 0.5
            lambda_right_neg = self.u[i + 1] - (self.g * self.h[i + 1]) ** 0.5
            lambda_right_pos = self.u[i + 1] + (self.g * self.h[i + 1]) ** 0.5

            # Кабаре с обработкой звуковых точек, не дописан, не используется
            if (np.sign(lambda_left_neg) != np.sign(lambda_right_neg) or np.sign(lambda_left_pos) != np.sign(lambda_right_pos)):
                c_middle = ((self.g * self.h[i - 1]) ** 0.5 + (self.g * self.h[i + 1]) ** 0.5) / 2
                u_middle = (self.u[i - 1] + self.u[i + 1]) / 2
                lambda_middle_neg = u_middle - c_middle
                lambda_middle_pos = u_middle + c_middle
                neg_char_middle = u_middle - 2 * c_middle
                pos_char_middle = u_middle + 2 * c_middle
                if np.sign(lambda_left_neg) != np.sign(lambda_right_neg):
                    neg_char_new[i] = 2 * neg_char_middle - self.neg_char[i]
                else:
                    neg_char_new[i] = 2 * self.neg_char[i + 1] - self.neg_char[i + 2]
                if np.sign(lambda_left_pos) != np.sign(lambda_right_pos):
                    pos_char_new[i] = 2 * pos_char_middle - self.pos_char[i]
                else:
                    pos_char_new[i] = 2 * self.pos_char[i - 1] - self.pos_char[i - 2]
            else:
                neg_char_new[i] = 2 * neg_char_new[i + 1] - self.neg_char[i + 2]
                pos_char_new[i] = 2 * pos_char_new[i - 1] - self.pos_char[i - 2]

        self._correct_invariants(self.pos_char, self.neg_char, pos_char_new, neg_char_new)

        for i in range(2, len(self.u) - 1, 2):
            self.u[i] = (neg_char_new[i] + pos_char_new[i]) / 2
            self.h[i] = ((pos_char_new[i] - neg_char_new[i]) / 4) ** 2 / self.g
            self.hu[i] = self.h[i] * self.u[i]




class RiemannSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        self.solver_func = solver_func
        # print(self.solver_func)

    def solve(self, x, t, h_l, u_l, h_r, u_r):
        res = riemann_solver_newton(h_l, h_l * u_l, h_r, h_r * u_r)
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
