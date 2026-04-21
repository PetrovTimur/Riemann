import argparse
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from training.metrics.toro import toro_tests
from training.simulation import Simulation


def load_module(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    config_module = OmegaConf.create(ckpt["config"])
    module = instantiate(config_module.module)
    module.load_state_dict(ckpt["state_dict"], strict=False)
    module.to(device)
    module.eval()

    module.solver.model.load_state_dict(module.model.state_dict())
    module.solver.model.to(device)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", default="outputs/weight_dumps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    module = load_module(args.ckpt, args.device)
    model = module.solver.model
    model.record_weights = True

    for i, config in enumerate(toro_tests):
        config = dict(config)
        model.weight_history = []
        model.logit_history = []
        config["solver"] = module.solver

        sim = Simulation(config)
        sim.run()

        if len(model.weight_history) == 0:
            print(f"toro_{i}: no NN calls recorded")
            continue

        weights = torch.cat(model.weight_history, dim=0).numpy()
        logits = torch.cat(model.logit_history, dim=0).numpy()
        out_path = os.path.join(args.out_dir, f"toro_{i}.npz")
        np.savez(out_path, weights=weights, logits=logits)
        print(f"toro_{i}: saved weights {weights.shape} logits {logits.shape} to {out_path}")


if __name__ == "__main__":
    main()
