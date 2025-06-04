import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from solvers import CabaretSolver, GodunovSolver, CabaretSolverNN, CabaretSolverPlusPlus, RiemannSolver
from models import load_nn


class Simulation:
    def __init__(self, config: dict):
        self.L = config['L']
        self.nx = config['nx']
        self.h_l = config['h_l']
        self.u_l = config['u_l']
        self.h_r = config['h_r']
        self.u_r = config['u_r']
        self.t_end = config['t_end']
        self.dx = 2 * self.L / self.nx
        print(self.dx)
        self.solver = config['solver']

        self.h, self.hu = self.initial_conditions()

        if isinstance(self.solver, CabaretSolver):
            self.h, self.hu = self.new_grid(self.h, self.hu)

        self.x = np.linspace(-self.L, self.L, len(self.h), endpoint=True)
        print(len(self.x))
        self.hu_final = None
        self.h_final = None
        self.h_rollout = []
        self.hu_rollout = []

    def initial_conditions(self):
        if isinstance(self.solver, GodunovSolver):
            self.nx += 1

        h = np.full(self.nx, self.h_r, dtype=np.float32)
        h[:self.nx // 2] = self.h_l
        u = np.full(self.nx, self.u_r, dtype=np.float32)
        u[:self.nx // 2] = self.u_l

        if isinstance(self.solver, GodunovSolver):
            i = self.nx // 2
            h[i] = (h[i - 1] + h[i + 1]) / 2
            u[i] = (u[i - 1] + u[i + 1]) / 2

        hu = h * u
        return h, hu

    def new_grid(self, h, hu):
        new_h = np.zeros(2 * len(h) + 1, dtype=np.float32)
        new_hu = np.zeros(2 * len(hu) + 1, dtype=np.float32)
        new_h[1::2] = h
        new_hu[1::2] = hu

        new_h[2:-1:2] = (h[1:] + h[:-1]) / 2
        new_hu[2:-1:2] = (hu[1:] + hu[:-1]) / 2

        new_h[0] = h[0]
        new_h[-1] = h[-1]
        new_hu[0] = hu[0]
        new_hu[-1] = hu[-1]

        new_h[1::2] = (new_h[2::2] + new_h[:-2:2]) / 2
        new_hu[1::2] = (new_hu[2::2] + new_hu[:-2:2]) / 2

        # new_h[2:-1:2] = (new_h[1:-3:2] + new_h[3:-1:2]) / 2
        # new_hu[2:-1:2] = (new_hu[1:-3:2] + new_hu[3:-1:2]) / 2

        return new_h, new_hu

    def run(self):
        t = 0.0
        CFL = 0.3
        g = 9.81
        h = self.h.copy()
        hu = self.hu.copy()
        self.h_rollout = [h.copy()]
        self.hu_rollout = [hu.copy()]
        while t < self.t_end:
            # u = np.where(h > 0, hu / h, 0)
            # c = np.sqrt(g * h)
            # dt = CFL * self.dx / np.max(np.abs(u + c))
            # dt = min(dt, CFL * self.dx / np.max(np.abs(u - c)))

            u_vals = np.where(h > 1e-12, hu / h, 0.0)  # Use a small epsilon for robustness
            c_vals = np.sqrt(g * np.maximum(0.0, h))  # Ensure H is non-negative for sqrt

            # Calculate characteristic speeds
            lambda1_vals = u_vals + c_vals
            lambda2_vals = u_vals - c_vals

            # Find the maximum absolute characteristic speed in the domain
            # Concatenate and take max absolute value for robustness across all points and both waves
            max_abs_speed = np.max(np.concatenate((np.abs(lambda1_vals), np.abs(lambda2_vals))))

            dt = CFL * self.dx / max_abs_speed

            h, hu = self.solver.step(h.copy(), hu.copy(), self.dx, dt)
            t += dt
            self.h_rollout.append(h.copy())
            self.hu_rollout.append(hu.copy())
        self.h_final = h
        self.hu_final = hu
        return h, hu

    def plot(self):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        x = np.linspace(-self.L, self.L, len(self.h), endpoint=True)
        axes[0].plot(x, self.h, label='Initial h', linestyle='--', marker='o')
        axes[0].plot(x, self.h_final, label='Final h', linewidth=2)
        axes[0].set_title('Water Height h')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(x, self.hu, label='Initial hu', linestyle='--', marker='o')
        axes[1].plot(x, self.hu_final, label='Final hu', linewidth=2)
        axes[1].set_title('Momentum hu')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_animation(self):
        matplotlib.use("TkAgg")

        rollout_len = len(self.h_rollout)
        x = np.linspace(-self.L, self.L, len(self.h), endpoint=True)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(bottom=0.18)

        # Compute good y-limits for all time steps
        h_all = np.array(self.h_rollout)
        hu_all = np.array(self.hu_rollout)
        h_min, h_max = np.min(h_all), np.max(h_all)
        hu_min, hu_max = np.min(hu_all), np.max(hu_all)
        h_pad = 0.05 * (h_max - h_min) if h_max > h_min else 0.1
        hu_pad = 0.05 * (hu_max - hu_min) if hu_max > hu_min else 0.1

        h_line, = axes[0].plot(x, self.h_rollout[0], label='h', color='b', marker='o')
        axes[0].set_title('Water Height h')
        axes[0].set_ylim(h_min - h_pad, h_max + h_pad)
        axes[0].grid(True)

        hu_line, = axes[1].plot(x, self.hu_rollout[0], label='hu', color='r', marker='o')
        axes[1].set_title('Momentum hu')
        axes[1].set_ylim(hu_min - hu_pad, hu_max + hu_pad)
        axes[1].grid(True)

        ax_slider = plt.axes((0.15, 0.05, 0.7, 0.04))
        slider = Slider(ax_slider, 'Step', 0, rollout_len - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            h_line.set_ydata(self.h_rollout[idx])
            hu_line.set_ydata(self.hu_rollout[idx])
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

def plot_comparison(sims, labels=None, plot_solution=True, riemann_kwargs=None, plot_u=False):
    if not isinstance(sims, (list, tuple)):
        sims = [sims]
    n = len(sims)
    if labels is None:
        labels = [f'Solver {i+1}' for i in range(n)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # h comparison
    for sim, label in zip(sims, labels):
        # If CabaretSolver, plot only odd points
        if isinstance(sim.solver, CabaretSolver):
            x_plot = sim.x[1::2]
            h_plot = sim.h_final[1::2]
        else:
            x_plot = sim.x
            h_plot = sim.h_final
        axes[0].plot(x_plot, h_plot, label=f'Final h ({label})')
    axes[0].set_title('Water Height h')
    axes[0].legend()
    axes[0].grid(True)

    # hu or u comparison
    for sim, label in zip(sims, labels):
        if isinstance(sim.solver, CabaretSolver):
            x_plot = sim.x[1::2]
            h_plot = sim.h_final[1::2]
            hu_plot = sim.hu_final[1::2]
        else:
            x_plot = sim.x
            h_plot = sim.h_final
            hu_plot = sim.hu_final
        if plot_u:
            u_plot = np.where(h_plot > 1e-12, hu_plot / h_plot, 0.0)
            axes[1].plot(x_plot, u_plot, label=f'Final u ({label})')
        else:
            axes[1].plot(x_plot, hu_plot, label=f'Final hu ({label})')
    axes[1].set_title('Velocity u' if plot_u else 'Momentum hu')
    axes[1].legend()
    axes[1].grid(True)

    # Add Riemann solution if requested
    if plot_solution:
        sim0 = sims[0]
        x = sim0.x
        t = sim0.t_end if hasattr(sim0, 't_end') else 2.0
        h_l = sim0.h_l if hasattr(sim0, 'h_l') else 1.0
        h_r = sim0.h_r if hasattr(sim0, 'h_r') else 0.2
        u_l = sim0.u_l if hasattr(sim0, 'u_l') else 0.0
        u_r = sim0.u_r if hasattr(sim0, 'u_r') else 0.0
        L = sim0.L if hasattr(sim0, 'L') else 20
        if riemann_kwargs is not None:
            h_l = riemann_kwargs.get('h_l', h_l)
            h_r = riemann_kwargs.get('h_r', h_r)
            u_l = riemann_kwargs.get('u_l', u_l)
            u_r = riemann_kwargs.get('u_r', u_r)
            t = riemann_kwargs.get('t', t)
            L = riemann_kwargs.get('L', L)
        x_riem = np.linspace(-L, L, len(x), endpoint=True)
        solver = RiemannSolver()
        res = solver.solve(x_riem, t, h_l, u_l, h_r, u_r)
        h_riem, u_riem = res['vals']
        hu_riem = h_riem * u_riem
        axes[0].plot(x_riem, h_riem, label='True h', linestyle=':', color='k')
        if plot_u:
            axes[1].plot(x_riem, u_riem, label='True u', linestyle=':', color='k')
        else:
            axes[1].plot(x_riem, hu_riem, label='True hu', linestyle=':', color='k')
        axes[0].legend()
        axes[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    model = load_nn('generic_nn', path='checkpoints/model_ood_1.pt')

    config = {
        'L': 10,
        'nx': 20,

        # 'h_l': 1.0,
        # 'h_r': 0.206612,
        # 'u_l': 0,
        # 'u_r': 3.416828,

        'h_l': 1.0,
        'h_r': 2.0,
        'u_l': 0,
        'u_r': 0,

        # 'solver': GodunovSolver(solver_func='newton', model=None),
        'solver': CabaretSolverNN(model=model),
        't_end': 2.0
    }
    sim1 = Simulation(config)
    sim1.run()
    # sim.plot()
    # sim1.plot_animation()

    config['solver'] = GodunovSolver(solver_func='newton', model=None)
    sim2 = Simulation(config)
    sim2.run()
    # sim2.plot_animation()


    config['solver'] = CabaretSolverPlusPlus(model=None)
    # config['t_end'] = 0.2
    sim3 = Simulation(config)
    sim3.run()
    sim3.plot_animation()
    # sim3.plot()


    # plot_comparison([sim1, sim2], labels=['Cabaret', 'Godunov'], plot_solution=True)
    plot_comparison([sim1, sim2, sim3], labels=['Cabaret', 'Godunov', 'Cabaret++'], plot_solution=True, plot_u=True)
