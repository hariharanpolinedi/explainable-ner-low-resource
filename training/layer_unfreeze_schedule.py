"""
Layer-Wise Unfreezing Schedule Utility for Fine-Tuning (Phase 9).

Controls progressive parameter unfreezing during per-language fine-tuning:
- Lower layers: lexical word embeddings, char-CNN morphological representations, language embeddings, and BiLSTM contextual encoder.
- Upper layers: multi-head self-attention block, linear emission projection, and linear-chain CRF.

Supports:
- 'phased': Lower layers frozen during early epochs (0 to unfreeze_epoch - 1), all layers unfrozen thereafter.
- 'immediate': All layers unfrozen from epoch 0.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore


class LayerUnfreezeScheduler:
    """
    Manages parameter grad states across training epochs based on the selected schedule.
    """

    def __init__(
        self,
        model: nn.Module,
        schedule: str = "phased",
        unfreeze_epoch: int = 4,
    ) -> None:
        self.model = model
        self.schedule = schedule.lower().strip()
        self.unfreeze_epoch = unfreeze_epoch

        if self.schedule not in {"phased", "immediate"}:
            raise ValueError(f"Unknown unfreezing schedule '{schedule}'. Must be 'phased' or 'immediate'.")

        # Partition model sub-modules into lower and upper layer groups
        self.lower_modules = [
            getattr(model, "word_embedding", None),
            getattr(model, "char_cnn", None),
            getattr(model, "lang_embedding", None),
            getattr(model, "encoder", None),
        ]
        self.upper_modules = [
            getattr(model, "attention", None),
            getattr(model, "fc_shared", None),
            getattr(model, "crf_shared", None),
            getattr(model, "fc_heads", None),
            getattr(model, "crf_heads", None),
        ]

    def _set_module_grad(self, module: nn.Module | None, requires_grad: bool) -> None:
        if module is not None:
            for p in module.parameters():
                p.requires_grad = requires_grad

    def step_epoch(self, epoch: int) -> dict[str, Any]:
        """
        Configure layer parameter gradients at the beginning of an epoch.

        Args:
            epoch: 1-indexed current epoch number.

        Returns:
            Dictionary detailing layer freezing status.
        """
        if self.schedule == "immediate":
            # All layers trainable
            for mod in self.lower_modules + self.upper_modules:
                self._set_module_grad(mod, True)
            status = {
                "epoch": epoch,
                "schedule": self.schedule,
                "lower_layers_frozen": False,
                "upper_layers_frozen": False,
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            }
            return status

        # Phased schedule: 1-indexed epoch < unfreeze_epoch has lower layers frozen
        # e.g., if unfreeze_epoch=4, epochs 1, 2, 3 have lower layers frozen
        lower_frozen = (epoch < self.unfreeze_epoch)

        for mod in self.lower_modules:
            self._set_module_grad(mod, not lower_frozen)
        for mod in self.upper_modules:
            self._set_module_grad(mod, True)

        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())

        return {
            "epoch": epoch,
            "schedule": self.schedule,
            "lower_layers_frozen": lower_frozen,
            "upper_layers_frozen": False,
            "trainable_params": trainable_count,
            "total_params": total_count,
        }
