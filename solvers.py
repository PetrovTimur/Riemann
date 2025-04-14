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

    # Apply wave type calculations for left and right sides
    # wave_L = compute_wave(h_k[0], u_k[0], c_k[0], c)
    # wave_R = compute_wave(h_k[1], u_k[1], c_k[1], c)

    # h_star = H
    # u_star = U

    # h_star = 0.5 * (wave_R['H'] + wave_L['H'])
    # h_star = 0.5 * (wave_R['H'] + wave_L['H'])

    F_h = H * U
    F_hu = H * U ** 2 + 0.5 * g * H ** 2



    return {
        "flux": np.array([F_h, F_hu, h_star, u_star]),
        "star": np.array([h_star, u_star]),
        "data": np.array([k]),
        "velocity": np.array([D_L, D_starL, D_R, D_starR]),
        # "h_star": h_star,
        # "u_star": u_star,
        # "wave_left": wave_L,
        # "wave_right": wave_R
    }

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

        h_new = h.copy().astype(np.float32)
        hu_new = hu.copy().astype(np.float32)


        for i in range(1, n - 1):
            # print(dt, dx, dt / dx * (h_flux[i] - h_flux[i - 1]))
            h_new[i] -= dt / dx * (h_flux[i] - h_flux[i - 1])
            hu_new[i] -= dt / dx * (hu_flux[i] - hu_flux[i - 1])

        h_new = np.maximum(h_new, 0)

        # print('heyyy', h_new)

        return h_new, hu_new


class CabaretSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        # Acceptable solver options: 'default', 'iter', 'newton', 'nn'
        self.solver_func = solver_func
        print(self.solver_func)
        self.f = open('train.txt', 'w+')

    def step(self, h, hu, dx, dt):
        u = hu / h

        # h_left = h[0]
        # h_right = h[-1]
        # hu_left = hu[0]
        # hu_right = hu[-1]
        # print('1', h_left, h_right, h_left, h_right)

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
                    self.f.write(f"{h[i]}, {hu[i]}, {dx}, {dt}\n")

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

        # h[0] = h_left
        # h[-1] = h_right
        # hu[0] = hu_left
        # hu[-1] = hu_right
        # print('3', h_left, h_right, h_left, h_right)

        # Step 3
        h[1::2] -= dt / (2 * dx) * (hu[2::2] - hu[:-1:2])
        hu[1::2] -= dt / (2 * dx) * ((h[2::2] * (u[2::2]) ** 2 + 0.5 * self.g * h[2::2] ** 2) - (h[:-1:2] * (u[:-2:2]) ** 2 + 0.5 * self.g * h[:-2:2] ** 2))

        # print(min(h))

        return h, hu


class RiemannSolver:
    def __init__(self, solver_func='classic', model=None, g=9.81):
        self.g = g
        self.model = model
        self.solver_func = solver_func
        print(self.solver_func)

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
        # Similarly, for the right wave:
        right_star_boundary = D_starR if D_starR is not None else D_R

        # Create spatial grid and similarity variable xi = x/t
        # x = np.linspace(-a, a, 500)
        xi = x / t

        def h_profile(xi):
            h = np.empty_like(xi)

            # Region I: Left constant state (xi < D_L)
            cond1 = xi < D_L

            # Region II: Left rarefaction fan (if applicable)
            # Only defined if D_starL is not None.
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
                # h = [ (u_l + 2*sqrt(g*h_l) - xi)/(3*sqrt(g)) ]^2
                h[cond2] = ((u_l + 2 * np.sqrt(self.g * h_l) - xi[cond2]) / (3 * np.sqrt(self.g))) ** 2
            else:
                # If no rarefaction, there is no fan region (shock)
                h[cond2] = h_l  # Not really used.

            h[cond3] = h_star

            if D_starR is not None:
                # For right rarefaction fan, use similarity solution:
                # h = [ (xi - u_r + 2*sqrt(g*h_r))/(3*sqrt(g)) ]^2
                h[cond4] = ((xi[cond4] - u_r + 2 * np.sqrt(self.g * h_r)) / (3 * np.sqrt(self.g))) ** 2
            else:
                h[cond4] = h_r  # Not really used.

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
                # u = u_l + (2/3)*(xi - (u_l - sqrt(g*h_l)))
                u[cond2] = u_l + (2 / 3) * (xi[cond2] - (u_l - np.sqrt(self.g * h_l)))
            else:
                u[cond2] = u_l  # Not really used.

            u[cond3] = u_star

            if D_starR is not None:
                # For right rarefaction fan:
                # u = u_r - (2/3)*((u_r + sqrt(g*h_r)) - xi)
                u[cond4] = u_r - (2 / 3) * ((u_r + np.sqrt(self.g * h_r)) - xi[cond4])
            else:
                u[cond4] = u_r  # Not really used.

            u[cond5] = u_r

            return u

        h_vals = h_profile(xi)
        u_vals = u_profile(xi)

        return {"vals": (h_vals, u_vals),
                "bounds": [D_L, D_starL, D_R, D_starR, left_star_boundary, right_star_boundary]
                }
