"""Prompt-only datasets for online image-generation alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.geneval.metadata import load_metadata_rows

from .types import PromptBatch


class PromptFileDataset(Dataset[tuple[str, dict[str, Any] | None]]):
    """Load plain prompts or structured GenEval JSONL training rows."""

    def __init__(self, path: str | Path) -> None:
        """Read and validate a prompt file.

        Args:
            path: UTF-8 text file containing one prompt per non-empty line.

        Returns:
            Nothing. Parsed prompts are retained in memory.

        Raises:
            FileNotFoundError: If the prompt file does not exist.
            ValueError: If it contains no non-empty prompts.
        """
        prompt_path = Path(path).expanduser()
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
        if prompt_path.suffix.lower() == ".jsonl":
            metadata = tuple(load_metadata_rows(prompt_path))
            self.items = tuple((row["prompt"], row) for row in metadata)
        else:
            prompts = tuple(
                line.strip()
                for line in prompt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            self.items = tuple((prompt, None) for prompt in prompts)
        if not self.items:
            raise ValueError(f"Prompt file is empty: {prompt_path}")

    def __len__(self) -> int:
        """Return the number of available prompt conditions.

        Returns:
            Number of parsed non-empty lines.
        """
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, dict[str, Any] | None]:
        """Return one prompt and optional metadata row by dataset index.

        Args:
            index: Zero-based prompt index.

        Returns:
            Prompt string and optional structured metadata at ``index``.
        """
        return self.items[index]


def collate_prompts(
    items: list[tuple[str, dict[str, Any] | None]],
) -> PromptBatch:
    """Convert dataset rows into the framework prompt container.

    Args:
        items: Non-empty prompt/metadata pairs.

    Returns:
        Prompt batch with immutable ordered prompt storage.
    """
    prompts, metadata = zip(*items)
    if all(row is None for row in metadata):
        return PromptBatch(prompts=tuple(prompts))
    if any(row is None for row in metadata):
        raise ValueError("Prompt batches cannot mix structured and plain rows")
    return PromptBatch(
        prompts=tuple(prompts),
        metadata=tuple(metadata),  # type: ignore[arg-type]
    )
