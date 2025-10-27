"""Training logger utilities for PINN-PBM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib.pyplot as plt


@dataclass
class TrainingLogger:
    """Collects loss metrics during training and renders diagnostic plots."""

    epochs: List[int] = field(default_factory=list)
    losses: Dict[str, List[float]] = field(
        default_factory=lambda: {"total": [], "physics": [], "data": []}
    )

    def log_epoch(self, epoch: int, total_loss: float, physics_loss: float, data_loss: float) -> None:
        self.epochs.append(epoch)
        self.losses["total"].append(float(total_loss))
        self.losses["physics"].append(float(physics_loss))
        self.losses["data"].append(float(data_loss))

    def plot_losses(self) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.semilogy(self.epochs, self.losses["total"], label="Total Loss")
        ax1.semilogy(self.epochs, self.losses["physics"], label="Physics Loss")
        ax1.semilogy(self.epochs, self.losses["data"], label="Data Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (log scale)")
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.4)

        ax2.plot(self.epochs, self.losses["total"], label="Total Loss")
        ax2.plot(self.epochs, self.losses["physics"], label="Physics Loss")
        ax2.plot(self.epochs, self.losses["data"], label="Data Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, ls="--", alpha=0.4)

        fig.tight_layout()
        return fig

    def to_dict(self) -> Dict[str, List[float]]:
        return {"epochs": self.epochs, **self.losses}
