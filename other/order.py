import numpy as np
import matplotlib.pyplot as plt
from training.solvers import RiemannSolver
from simulate import run_simulation, error_norm

def plot():
    solver = RiemannSolver()
    n = 80
    L = 10
    x = np.linspace(-L, L, 2 * n + 1, endpoint=True)
    t = 2

    h_l = 1
    h_r = 0.206612
    u_l = 0
    u_r = 3.416828

    # h_l = 1
    # h_r = 1
    # u_l = -0.1
    # u_r = 0.1

    res = solver.solve(x, t, h_l, u_l, h_r, u_r)
    h_vals, u_vals = res['vals']
    D_L, D_starL, D_R, D_starR, left_star_boundary, right_star_boundary = res['bounds']


    fig = plt.figure(figsize=(10, 8))
    axes = fig.subplots(nrows=2, ncols=1)

    axes[0].plot(x, h_vals, 'b-', linewidth=2, label='Analytical solution')

    # Границы регионов решения
    # axes[0].axvline(D_L * t, color='k', linestyle='--', label='Left wave head')
    # if D_starL is not None:
    #     axes[0].axvline(left_star_boundary * t, color='c', linestyle='--', label='Left rarefaction tail')
    # axes[0].axvline(right_star_boundary * t, color='m', linestyle='--', label='Star region right boundary')
    # if D_starR is not None:
    #     axes[0].axvline(D_R * t, color='g', linestyle='--', label='Right rarefaction head')
    # else:
    #     axes[0].axvline(D_R * t, color='k', linestyle='--', label='Right shock speed')
    axes[0].set_title(f'Height Profiles at t = {t}')


    axes[1].plot(x, u_vals, 'r-', linewidth=2, label='Analytical solution')
    axes[1].set_title(f'Speed Profiles at t = {t}')

    # Границы регионов решения
    # axes[1].axvline(D_L * t, color='k', linestyle='--', label='Left wave head')
    # if D_starL is not None:
    #     axes[1].axvline(left_star_boundary * t, color='c', linestyle='--', label='Left rarefaction tail')
    # axes[1].axvline(right_star_boundary * t, color='m', linestyle='--', label='Star region right boundary')
    # if D_starR is not None:
    #     axes[1].axvline(D_R * t, color='g', linestyle='--', label='Right rarefaction head')
    # else:
    #     axes[1].axvline(D_R * t, color='k', linestyle='--', label='Right shock speed')


    resolutions = [n, 2 * n]
    h_results = []
    u_results = []

    solvers = ['classic', 'classic_improved', 'generic_nn', 'selective_nn']

    for nx in resolutions:
        result = run_simulation(2 * L, nx, h_l, u_l, h_r, u_r, solver=solvers[2], t_end=t)
        h_results.append(result[0])
        u_results.append(result[1] / result[0])
        # print(h_results[-1].shape, np.all(h_results[-1] == h_results[-1][::-1]))
        axes[0].plot(np.linspace(-L, L, 2 * nx + 1, endpoint=True), h_results[-1], label=f"Cabaret with {nx} points", linestyle='--', marker='o', markersize=4)
        axes[1].plot(np.linspace(-L, L, 2 * nx + 1, endpoint=True), u_results[-1], label=f"Cabaret with {nx} points", linestyle='--', marker='o', markersize=4)
        # axes[0].plot(np.linspace(-L, L, nx, endpoint=True), result[0][1::2], label=f"Cabaret with {nx} points", linestyle='--')
        # axes[1].plot(np.linspace(-L, L, nx, endpoint=True), result[1][1::2] / result[0][1::2], label=f"Cabaret with {nx} points", linestyle='--')



    h_n = h_results[0][1::2]
    error1 = error_norm(h_n, h_vals[1::2], 2 * L / resolutions[0])

    # axes[0].plot(np.linspace(-L, L, resolutions[0], endpoint=True), np.abs(h_vals[1::2] - h_n), label="Error at " + str(resolutions[0]), linestyle='--')

    h_2n = (h_results[1][1:-2:2] + h_results[1][3::2])[::2] * 0.5
    error2 = error_norm(h_2n, h_vals[1::2], 2 * L / resolutions[1])

    # axes[0].plot(np.linspace(-L, L, resolutions[0], endpoint=True), np.abs(h_vals[1::2] - h_2n), label="Error at " + str(resolutions[1]), linestyle='--')


    order = np.log2(error1 / error2)

    # print(h_vals[:10], h_results[1][1:10:2], h_results[0][1:10:2])


    print("Error between n and analytical:", error1)
    print("Error between 2n and analytical:", error2)
    print("Estimated order of convergence:", order)


    axes[0].legend()
    axes[0].grid(True)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot()