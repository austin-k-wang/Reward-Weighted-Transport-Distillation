#!/usr/bin/env python3
"""Score prompt-image JSONL samples with one unified reward evaluator."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rewards import create_reward_from_config  # noqa: E402


def resolve_dtype(name: str) -> torch.dtype:
    """Resolve a command-line floating dtype.

    Args:
        name: One of ``float32``, ``float16``, or ``bfloat16`` and their common
            aliases.

    Returns:
        Corresponding PyTorch floating dtype.

    Raises:
        ValueError: If ``name`` is unsupported.
    """
    values = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def load_reward_spec(path: Path) -> dict[str, Any]:
    """Load one reward mapping from YAML.

    Args:
        path: YAML path containing either the reward mapping directly or under
            a top-level ``reward`` key.

    Returns:
        Mutable reward configuration dictionary.

    Raises:
        TypeError: If the selected YAML value is not a mapping.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    value = raw.get("reward", raw) if isinstance(raw, dict) else raw
    if not isinstance(value, dict):
        raise TypeError("Reward configuration must be a YAML mapping")
    return dict(value)


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSON objects from a JSONL file.

    Args:
        path: Input line-delimited JSON file.

    Returns:
        Ordered list of sample dictionaries.

    Raises:
        ValueError: If no samples exist or any row is not an object.
    """
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"No samples found in {path}")
    return rows


def load_image_tensor(path: Path) -> torch.Tensor:
    """Load one RGB image as a canonical SANA-range tensor.

    Args:
        path: Image file readable by Pillow.

    Returns:
        CPU float32 tensor shaped ``[3,H,W]`` in ``[-1,1]``.
    """
    with Image.open(path) as image:
        pixels = torch.from_numpy(np.array(image.convert("RGB"), copy=True))
    return pixels.permute(2, 0, 1).float().div(127.5).sub(1.0)


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_dir: Path,
    reward: object,
    batch_size: int,
    prompt_field: str = "prompt",
    image_field: str = "image",
) -> list[dict[str, Any]]:
    """Score JSONL rows in bounded image batches.

    Args:
        rows: Ordered sample mappings containing prompt and image path fields.
        input_dir: Base directory used to resolve relative image paths.
        reward: Canonical evaluator exposing ``score``.
        batch_size: Maximum images loaded and scored together.
        prompt_field: Mapping key containing each prompt string.
        image_field: Mapping key containing each image path.

    Returns:
        Copies of input rows with one floating ``reward`` field appended.

    Raises:
        ValueError: If ``batch_size`` is not positive or images in one batch
            have incompatible shapes.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output: list[dict[str, Any]] = []
    progress = tqdm(range(0, len(rows), batch_size), desc="Scoring samples", unit="batch")
    for start in progress:
        chunk = rows[start : start + batch_size]
        prompts = [str(row[prompt_field]) for row in chunk]
        image_tensors = []
        for row in chunk:
            image_path = Path(str(row[image_field])).expanduser()
            if not image_path.is_absolute():
                image_path = input_dir / image_path
            image_tensors.append(load_image_tensor(image_path))
        shapes = {tuple(image.shape) for image in image_tensors}
        if len(shapes) != 1:
            raise ValueError(f"Images in one score batch must share shape, got {sorted(shapes)}")
        images = torch.stack(image_tensors)
        metadata = [row.get("metadata") for row in chunk]
        if all(value is None for value in metadata):
            metadata = None
        values = reward.score(  # type: ignore[attr-defined]
            prompts,
            images,
            batch_size=batch_size,
            metadata=metadata,
        )
        for row, value in zip(chunk, values.tolist()):
            output.append({**dict(row), "reward": float(value)})
    return output


def write_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write score rows as line-delimited JSON.

    Args:
        rows: Ordered serializable mappings.
        path: Destination JSONL file whose parent is created.

    Returns:
        Nothing after all rows are durably closed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone reward scoring parser.

    Returns:
        Argument parser for reward YAML, sample JSONL, device, dtype, and output.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Reward YAML.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--image-field", default="image")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load a reward once, score every input row, and persist results and logs.

    Args:
        argv: Optional command-line token sequence.

    Returns:
        Process status zero after successful scoring.
    """
    args = build_parser().parse_args(argv)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_jsonl.with_suffix(".log")),
        ],
        force=True,
    )
    spec = load_reward_spec(args.config)
    reward = create_reward_from_config(
        spec,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
    )
    rows = load_rows(args.input_jsonl)
    scored = score_rows(
        rows,
        input_dir=args.input_jsonl.parent,
        reward=reward,
        batch_size=args.batch_size,
        prompt_field=args.prompt_field,
        image_field=args.image_field,
    )
    write_rows(scored, args.output_jsonl)
    values = torch.tensor([row["reward"] for row in scored], dtype=torch.float32)
    logging.info(
        "Scored %d samples: mean=%.6f std=%.6f",
        values.numel(),
        values.mean().item(),
        values.std(unbiased=False).item(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
