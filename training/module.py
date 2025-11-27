import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from solvers import BaseSolver
from simulation import Simulation, plot_comparison

from metrics.toro import toro_tests

class BaseModule(nn.Module):
    def __init__(
        self,
        solver: BaseSolver,
        model: nn.Module,
    ):
        super().__init__()

        # store components
        self.solver = solver
        self.model = model

        # resolve device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model.to(self.device)

        # simple flag
        self._initialized = True

    def forward(self, data):
        return self.model(data)

    def loss(self, pred, data):
        loss_dict = {}

        loss_dict.update(self.model.loss(pred, data))

        total_loss = 0
        for k, v in loss_dict.items():
            total_loss += v.mean()

        return total_loss
    
    def metrics(self):
        
        # For now just run closed-loop

        config = toro_tests[0]

        config["solver"] = self.solver
        sim = Simulation(config)
        sim.run()

        return {}
    
    def visualize(self):
        image_dict = {}

        for i, test in enumerate(toro_tests):
            config = toro_tests[i]
            config["solver"] = self.solver
            sim = Simulation(config)
            sim.run()

            fig = plot_comparison([sim], plot_solution=True)
            fig.canvas.draw()

            w, h = fig.canvas.get_width_height()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = data.reshape(h, w, 3)

            image_dict.update(
                {
                    f"tests/toro_{i}": img,
                }
            )

            plt.close(fig)

        return image_dict

    def step(self, h, hu, dx, dt):
        """Optional helper to call solver one step if available.
        h, hu: numpy arrays (or tensors convertible to numpy) of shape [N]
        Returns updated (h_new, hu_new) or raises if solver missing.
        """

        # Accept torch tensors too
        if isinstance(h, torch.Tensor):
            h = h.detach().cpu().numpy()
        if isinstance(hu, torch.Tensor):
            hu = hu.detach().cpu().numpy()
        h_new, hu_new = self.solver.step(h, hu, dx, dt)
        return h_new, hu_new

