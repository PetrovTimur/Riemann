import numpy as np
import matplotlib.pyplot as plt
from shallow import RiemannSolver, CabaretSolver, GodunovSolver
from solver import run_simulation, initial_conditions, error_norm

def plot():
    sovler = RiemannSolver()
    n = 100
    L = 10
    x = np.linspace(-L, L, n + 1, endpoint=True)
    t = 0.5
    h_l = 1
    h_r = 1
    u_l = -0.1
    u_r = 0.1
    res = sovler.solve(x, t, h_l, u_l, h_r, u_r)
    h_vals, u_vals = res['vals']
    D_L, D_starL, D_R, D_starR, left_star_boundary, right_star_boundary = res['bounds']


    fig = plt.figure(figsize=(10, 8))
    axes = fig.subplots(nrows=2, ncols=1)

    # Plot water depth
    # plt.subplot(2, 1, 1)
    axes[0].plot(x, h_vals, 'b-', linewidth=2, label='Water depth, h(x,t)')

    # Mark the wave boundaries with vertical dashed lines.
    # axes[0].axvline(D_L * t, color='k', linestyle='--', label='Left wave head')
    # if D_starL is not None:
    #     axes[0].axvline(left_star_boundary * t, color='c', linestyle='--', label='Left rarefaction tail')
    # axes[0].axvline(right_star_boundary * t, color='m', linestyle='--', label='Star region right boundary')
    # if D_starR is not None:
    #     axes[0].axvline(D_R * t, color='g', linestyle='--', label='Right rarefaction head')
    # else:
    #     axes[0].axvline(D_R * t, color='k', linestyle='--', label='Right shock speed')
    # axes[0].title(f'Shallow Water Riemann Problem Profiles at t = {t}')
    # axes[0].ylabel('h(x,t)')


    # Plot velocity profile
    # plt.subplot(2, 1, 2)
    axes[1].plot(x, u_vals, 'r-', linewidth=2, label='Velocity, u(x,t)')

    # axes[1].axvline(D_L * t, color='k', linestyle='--', label='Left wave head')
    # if D_starL is not None:
    #     axes[1].axvline(left_star_boundary * t, color='c', linestyle='--', label='Left rarefaction tail')
    # axes[1].axvline(right_star_boundary * t, color='m', linestyle='--', label='Star region right boundary')
    # if D_starR is not None:
    #     axes[1].axvline(D_R * t, color='g', linestyle='--', label='Right rarefaction head')
    # else:
    #     axes[1].axvline(D_R * t, color='k', linestyle='--', label='Right shock speed')
    # axes[1].xlabel('x')
    # axes[1].ylabel('u(x,t)')


    resolutions = [n, 2 * n]
    results = []

    # Run simulation on different grid resolutions.
    for nx in resolutions:
        h_result = run_simulation(L, nx, h_l, u_l, h_r, u_r, t_end=t)
        results.append(h_result)
        # print(results)
        axes[0].plot(np.linspace(-L, L, nx + 1, endpoint=True), h_result[0], label=str(nx), linestyle='-.',)
        axes[1].plot(np.linspace(-L, L, nx + 1, endpoint=True), h_result[1], label=str(nx), linestyle='-.',)

    # For the medium grid: pick points with step 2 to get n values.
    # h_med_common = results[1][::2]
    error1 = error_norm(results[0][0], h_vals, L / resolutions[0])
    # print(results[0][:20], results[1][:20], results[2][:20], h_med_common[:20])

    # axes[0][1].plot(np.linspace(-L, L, resolutions[0] + 1, endpoint=True), np.abs(results[0] - h_med_common),
    #                 label=str(resolutions[0]), linestyle='--')

    # h_fine_common = results[1][::2]
    error2 = error_norm(results[1][0][::2], h_vals, L / resolutions[0])

    # axes[0][1].plot(np.linspace(-L, L, resolutions[1] + 1, endpoint=True), np.abs(results[1] - h_fine_common),
    #                 label=str(resolutions[1]), linestyle='--')

    # plt.show()

    # Estimate order of convergence.
    order = np.log2(error1 / error2)


    print("Error between n and 2n grids:", error1)
    # print("Error between 2n and 4n grids:", error2)
    print("Estimated order of convergence:", order)


    axes[0].legend()
    axes[0].grid(True)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot()