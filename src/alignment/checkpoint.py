"""Adapter-only checkpoint persistence for alignment training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

from .config import AlignmentConfig
from .types import TrainerState


class AlignmentCheckpointManager:
    """Save and restore LoRA, optimizer, scheduler, progress, and RNG state."""

    def __init__(
        self,
        output_dir: str | Path,
        accelerator: Accelerator,
        config: AlignmentConfig,
    ) -> None:
        """Configure checkpoint storage for one distributed run.

        Args:
            output_dir: Training directory containing ``checkpoints``.
            accelerator: Distributed runtime used for barriers and unwrapping.
            config: Resolved run configuration persisted with checkpoints.

        Returns:
            Nothing. No files are created until :meth:`save` is called.
        """
        self.root = Path(output_dir).expanduser() / "checkpoints"
        self.accelerator = accelerator
        self.config = config

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        state: TrainerState,
    ) -> Path:
        """Write one resumable adapter-only distributed checkpoint.

        Args:
            model: Prepared PEFT model containing the active adapter.
            optimizer: Optimizer over active adapter parameters.
            scheduler: Learning-rate scheduler exposing ``state_dict``.
            state: Completed global and micro-step counters.

        Returns:
            Path to ``checkpoint-{global_step}``.
        """
        checkpoint_dir = self.root / f"checkpoint-{state.global_step}"
        if self.accelerator.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                checkpoint_dir / "adapter",
                safe_serialization=True,
            )
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                checkpoint_dir / "training.pt",
            )
            (checkpoint_dir / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": state.global_step,
                        "micro_step": state.micro_step,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (checkpoint_dir / "config.yaml").write_text(
                yaml.safe_dump(self.config.to_dict(), sort_keys=False),
                encoding="utf-8",
            )
        self.accelerator.wait_for_everyone()
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(self.accelerator.device)
            if self.accelerator.device.type == "cuda"
            else None,
        }
        torch.save(
            rng_state,
            checkpoint_dir / f"rng-rank-{self.accelerator.process_index}.pt",
        )
        self.accelerator.wait_for_everyone()
        return checkpoint_dir

    def load(
        self,
        path: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> TrainerState:
        """Restore one adapter-only checkpoint into initialized components.

        Args:
            path: Checkpoint directory produced by :meth:`save`.
            model: Prepared PEFT model with a compatible default adapter.
            optimizer: Initialized optimizer to restore.
            scheduler: Initialized scheduler to restore.

        Returns:
            Restored trainer counters.

        Raises:
            FileNotFoundError: If required checkpoint files are missing.
        """
        checkpoint_dir = Path(path).expanduser()
        state_path = checkpoint_dir / "trainer_state.json"
        training_path = checkpoint_dir / "training.pt"
        adapter_dir = checkpoint_dir / "adapter"
        if not state_path.is_file() or not training_path.is_file() or not adapter_dir.is_dir():
            raise FileNotFoundError(f"Incomplete alignment checkpoint: {checkpoint_dir}")

        unwrapped = self.accelerator.unwrap_model(model)
        adapter_state = load_peft_weights(str(adapter_dir), device=str(self.accelerator.device))
        set_peft_model_state_dict(unwrapped, adapter_state, adapter_name="default")
        training_state = torch.load(
            training_path,
            map_location=self.accelerator.device,
            weights_only=False,
        )
        optimizer.load_state_dict(training_state["optimizer"])
        scheduler.load_state_dict(training_state["scheduler"])

        rng_path = checkpoint_dir / f"rng-rank-{self.accelerator.process_index}.pt"
        if rng_path.is_file():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng["cuda"] is not None and self.accelerator.device.type == "cuda":
                torch.cuda.set_rng_state(rng["cuda"], self.accelerator.device)

        values = json.loads(state_path.read_text(encoding="utf-8"))
        self.accelerator.wait_for_everyone()
        return TrainerState(
            global_step=int(values["global_step"]),
            micro_step=int(values["micro_step"]),
        )
