import os
import csv
from typing import List, Callable, Iterable, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from hydra.core.hydra_config import HydraConfig
from training.module import BaseModule


class CSVDataset(Dataset):
    """CSV dataset using only integer cutoffs (both required).
    Specify:
      - feature_cols: int F (number of leading columns to use as features)
      - target_cols: int T (number of columns immediately following features to use as targets)
    Features are columns [0..F-1]; Targets are columns [F..F+T-1].
    """

    def __init__(
        self,
        path: str,
        feature_cols: int,
        target_cols: int,
    ):
        self.path = path
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            self.headers: List[str] = list(reader.fieldnames)
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"Empty CSV: {path}")

        ncols = len(self.headers)
        F = feature_cols
        T = target_cols
        if F <= 0:
            raise ValueError("feature_cols must be > 0")
        if T <= 0:
            raise ValueError("target_cols must be > 0")
        if F + T > ncols:
            raise ValueError(
                f"feature_cols + target_cols exceeds number of columns: F={F} T={T} ncols={ncols}"
            )

        self.feature_count = F
        self.target_count = T
        self.feature_idx = list(range(0, F))
        self.target_idx = list(range(F, F + T))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        feat_vals = [float(row[self.headers[i]]) for i in self.feature_idx]
        targ_vals = [float(row[self.headers[i]]) for i in self.target_idx]
        features = torch.tensor(feat_vals, dtype=torch.float32)
        targets = torch.tensor(targ_vals, dtype=torch.float32)
        return features, targets


class Trainer:
    def __init__(
        self,
        config: DictConfig,
        module: BaseModule,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], torch.optim.Optimizer],
        train_table: str,
        val_table: str,
        feature_cols: int,
        target_cols: int,
        num_pseudo_epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        validation_frequency_epochs: int,
        ckpt_frequency_epochs: int,
        checkpoint_path: Optional[str] = None,
    ):
        self.module = module
        self.num_pseudo_epochs = num_pseudo_epochs
        self.steps_per_epoch = steps_per_epoch
        self.train_table = train_table
        self.val_table = val_table
        self.batch_size = batch_size
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.validation_frequency_epochs = validation_frequency_epochs
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.ckpt_frequency_epochs = ckpt_frequency_epochs

        self.device = (
            next(self.module.parameters()).device
            if any(p.requires_grad for p in self.module.parameters())
            else torch.device("cpu")
        )

        # datasets & loaders (now using required integer cutoffs)
        self.train_ds = CSVDataset(
            self.train_table, self.feature_cols, self.target_cols
        )
        self.val_ds = CSVDataset(self.val_table, self.feature_cols, self.target_cols)
        self.train_loader = DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False
        )

        self.optimizer = optimizer(self.module.parameters())

        self.checkpoint_dir = HydraConfig.get().runtime.output_dir

        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        log_dir = os.path.join(self.checkpoint_dir, "logs")
        self.writer = SummaryWriter(log_dir)

        self.module.solver.model.to(self.device)

    def _checkpoint_payload(self) -> dict:
        cfg_container = None
        if isinstance(self.config, DictConfig):
            cfg_container = OmegaConf.to_container(self.config, resolve=True)
        return {"config": cfg_container, "state_dict": self.module.state_dict()}

    def _save_checkpoint(self, epoch: int):
        ckpt_dir = os.path.join(self.checkpoint_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        fname = f"ckpt_epoch_{epoch:04d}.pt"
        path = os.path.join(ckpt_dir, fname)
        torch.save(self._checkpoint_payload(), path)
        print(f"[Trainer] Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        abs_path = (
            path if os.path.isabs(path) else os.path.join(self.checkpoint_dir, path)
        )
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Checkpoint not found: {abs_path}")
        ckpt_obj = torch.load(abs_path, map_location=self.device)
        if not isinstance(ckpt_obj, dict) or "state_dict" not in ckpt_obj:
            raise ValueError(
                "Unsupported checkpoint format. Expected dict with keys: 'config' and 'state_dict'"
            )
        self.module.load_state_dict(ckpt_obj["state_dict"], strict=False)
        print(f"[Trainer] Loaded module state_dict from {abs_path}")

    def train(self):
        for pseudo_epoch in range(self.num_pseudo_epochs):
            self._train_epoch(pseudo_epoch)
            if self._is_validation_epoch(pseudo_epoch):
                self._val_epoch(pseudo_epoch)
                self.visualize(pseudo_epoch)
            if self._is_checkpoint_epoch(pseudo_epoch):
                self._save_checkpoint(pseudo_epoch)

    def _train_epoch(self, epoch: int):
        self.module.train()
        step = 0
        total_loss = 0.0
        for features, targets in self.train_loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            data = {"feats": features, "targets": targets}

            preds = self.module(data)
            loss = self.module.loss(preds, data)
            # print(loss, loss.shape)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            step += 1
            if step >= self.steps_per_epoch:
                break

        avg = total_loss / max(step, 1)

        self.writer.add_scalar("train_loss", avg, epoch)
        print(f"Epoch {epoch} train steps={step} loss={avg:.4f}")

    def _val_epoch(self, epoch: int) -> dict:
        """Run validation for a single epoch and return metrics.
        Visualization is handled separately via visualize_epoch().
        """
        self.module.eval()
        total_loss = 0.0
        step = 0

        self.module.solver.model.load_state_dict(self.module.model.state_dict())

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                data = {"feats": features, "targets": targets}

                preds = self.module(data)
                loss = self.module.loss(preds, data)
                total_loss += float(loss.item())

                step += 1

            metrics = self.module.metrics()

        avg = total_loss / max(step, 1)

        # log scalar only; visualization is separate
        self.writer.add_scalar("val_loss", avg, epoch)
        print(f"Epoch {epoch} val steps={step} loss={avg:.4f} metrics={metrics}")

        return {
            "val_loss": avg,
            "val_steps": step,
            **({} if metrics is None else metrics),
        }

    def visualize(self, epoch: int):
        """Log images to tensorboard for the given epoch using module.visualize()."""
        image_dict = self.module.visualize()
        for key, image in image_dict.items():
            self.writer.add_image(key, image, epoch, dataformats="HWC")

    def eval(self) -> dict:
        """Evaluate using the same logic as val_epoch without duplicating code."""
        metrics = self._val_epoch(epoch=self.num_pseudo_epochs)
        print(
            f"[Trainer] Eval full validation set steps={metrics['val_steps']} loss={metrics['val_loss']:.4f}"
        )
        return metrics

    def _is_validation_epoch(self, epoch: int):
        return (
            epoch == 0
            or epoch == self.num_pseudo_epochs - 1
            or (epoch + 1) % self.validation_frequency_epochs == 0
        )

    def _is_checkpoint_epoch(self, epoch: int):
        return (
            epoch == 0
            or epoch == self.num_pseudo_epochs - 1
            or (epoch + 1) % self.ckpt_frequency_epochs == 0
        )
