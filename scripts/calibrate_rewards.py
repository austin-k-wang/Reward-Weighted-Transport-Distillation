#!/usr/bin/env python3
"""Score a representative sample set and save fixed reward calibration."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.score_samples import (  # noqa: E402
    load_reward_spec,
    load_rows,
    resolve_dtype,
    score_rows,
    write_rows,
)
from src.rewards import create_reward_from_config  # noqa: E402


def calibration_statistics(values: torch.Tensor) -> dict[str, float | int]:
    """Compute fixed global calibration and diagnostic quantiles.

    Args:
        values: One-dimensional finite reward tensor containing at least two
            representative samples.

    Returns:
        JSON-compatible statistics including ``mean`` and population-standard
        deviation ``scale`` for RWTD or composite configuration.

    Raises:
        ValueError: If values are not one-dimensional, finite, sufficiently
            numerous, or have zero variance.
    """
    values = values.detach().float().cpu()
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("Calibration requires at least two scalar rewards")
    if not torch.isfinite(values).all():
        raise ValueError("Calibration rewards must all be finite")
    scale = values.std(unbiased=False)
    if scale <= 0:
        raise ValueError("Calibration rewards must have nonzero variance")
    quantiles = torch.quantile(
        values,
        torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99]),
    )
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "scale": scale.item(),
        "minimum": values.min().item(),
        "maximum": values.max().item(),
        "q01": quantiles[0].item(),
        "q05": quantiles[1].item(),
        "q50": quantiles[2].item(),
        "q95": quantiles[3].item(),
        "q99": quantiles[4].item(),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the calibration command-line parser.

    Returns:
        Parser accepting the same model/sample options as standalone scoring
        plus calibration and optional scored-row destinations.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Reward YAML.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--scored-jsonl", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--image-field", default="image")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Score the calibration corpus and persist stable global statistics.

    Args:
        argv: Optional command-line token sequence.

    Returns:
        Process status zero after scoring and serialization.
    """
    args = build_parser().parse_args(argv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_json.with_suffix(".log")),
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
    if args.scored_jsonl is not None:
        write_rows(scored, args.scored_jsonl)
    values = torch.tensor([row["reward"] for row in scored], dtype=torch.float32)
    statistics = calibration_statistics(values)
    artifact = {
        "provider": spec.get("provider"),
        "input_jsonl": str(args.input_jsonl.resolve()),
        **statistics,
    }
    args.output_json.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    logging.info(
        "Saved calibration for %d samples: mean=%.6f scale=%.6f",
        statistics["count"],
        statistics["mean"],
        statistics["scale"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
