import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_file(npz_path, out_path):
    data = np.load(npz_path)
    weights = data["weights"]  # [N, 2, 7]
    logits = data["logits"] if "logits" in data.files else None

    n_calls, n_groups, n_feats = weights.shape
    n_cols = 2 if logits is not None else 1

    fig, axes = plt.subplots(n_groups, n_cols, figsize=(6 * n_cols, 4 * n_groups), sharex=True, squeeze=False)

    group_names = ["I1", "I2"]
    lmax = np.max(np.abs(logits)) if logits is not None else None

    for g in range(n_groups):
        gname = group_names[g] if g < len(group_names) else f"group {g}"

        ax = axes[g][0]
        im = ax.imshow(
            weights[:, g, :].T,
            aspect="auto", origin="lower", vmin=0.0, vmax=1.0,
            cmap="viridis", interpolation="nearest",
        )
        ax.set_ylabel("feature index")
        ax.set_title(f"{gname} softmax weights")
        ax.set_yticks(range(n_feats))
        fig.colorbar(im, ax=ax)

        if logits is not None:
            ax = axes[g][1]
            im = ax.imshow(
                logits[:, g, :].T,
                aspect="auto", origin="lower", vmin=-lmax, vmax=lmax,
                cmap="coolwarm", interpolation="nearest",
            )
            ax.set_title(f"{gname} pre-softmax logits")
            ax.set_yticks(range(n_feats))
            fig.colorbar(im, ax=ax)

    axes[-1][0].set_xlabel("NN call index")
    if logits is not None:
        axes[-1][1].set_xlabel("NN call index")
    fig.suptitle(os.path.basename(npz_path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"{npz_path} -> {out_path}  (weights {weights.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", default="outputs/weight_dumps")
    parser.add_argument("--out-dir", default="outputs/weight_dumps/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz_paths = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if not npz_paths:
        print(f"no .npz files found in {args.in_dir}")
        return

    for npz_path in npz_paths:
        name = os.path.splitext(os.path.basename(npz_path))[0]
        out_path = os.path.join(args.out_dir, f"{name}.png")
        plot_file(npz_path, out_path)


if __name__ == "__main__":
    main()
