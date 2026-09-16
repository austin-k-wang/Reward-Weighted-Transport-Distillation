"""Unit tests for the lightweight parts of the quick visualization script."""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from src import pickscore as pickscore_module
from src import rewards as rewards_module


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/quick_viz.py"
SPEC = importlib.util.spec_from_file_location("quick_viz", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
quick_viz = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quick_viz)


def test_load_prompts_reads_prompt_column_and_limit(tmp_path: Path) -> None:
    """Verify TSV parsing preserves order and excludes metadata columns.

    Args:
        tmp_path: Pytest-provided temporary directory used to hold a small TSV
            fixture with three prompt rows.

    Returns:
        Nothing. The test asserts that the first two prompt strings are
        returned when the requested limit is two.
    """
    prompt_file = tmp_path / "prompts.tsv"
    prompt_file.write_text(
        "Prompt\tCategory\tNote\nfirst image\tA\tone\nsecond image\tB\ttwo\nthird image\tC\tthree\n",
        encoding="utf-8",
    )

    assert quick_viz.load_prompts(prompt_file, 2) == ["first image", "second image"]


def test_load_prompts_supports_plain_text(tmp_path: Path) -> None:
    """Verify line-oriented prompt files skip empty lines.

    Args:
        tmp_path: Pytest-provided temporary directory used to hold a plain
            text prompt fixture.

    Returns:
        Nothing. The test asserts that whitespace is stripped and blank lines
        are omitted.
    """
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text(" first prompt \n\nsecond prompt\n", encoding="utf-8")

    assert quick_viz.load_prompts(prompt_file, 16) == ["first prompt", "second prompt"]


def test_load_prompts_rejects_nonpositive_limit(tmp_path: Path) -> None:
    """Verify invalid prompt counts fail before reading an input file.

    Args:
        tmp_path: Pytest-provided temporary directory used to construct a path
            that does not need to exist for this validation case.

    Returns:
        Nothing. The test asserts that a zero limit raises ``ValueError``.
    """
    with pytest.raises(ValueError, match="greater than zero"):
        quick_viz.load_prompts(tmp_path / "unused.tsv", 0)


def test_local_huggingface_snapshot_resolves_main_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify project-local Hugging Face snapshots resolve through refs/main.

    Args:
        monkeypatch: Pytest fixture replacing the quick-viz project root.
        tmp_path: Temporary root containing a minimal cache structure.

    Returns:
        Nothing. The resolved absolute snapshot path is asserted.
    """
    monkeypatch.setattr(quick_viz, "PROJECT_ROOT", tmp_path)
    repository = (
        tmp_path
        / "models/hf-cache/hub/models--openai--clip-vit-large-patch14"
    )
    (repository / "refs").mkdir(parents=True)
    (repository / "refs/main").write_text("revision-1\n", encoding="utf-8")
    snapshot = repository / "snapshots/revision-1"
    snapshot.mkdir(parents=True)

    assert quick_viz.local_huggingface_snapshot(
        "openai/clip-vit-large-patch14"
    ) == str(snapshot)


def test_save_grid_normalizes_shape_device_and_dtype(tmp_path: Path) -> None:
    """Verify image batches become one CPU-compatible float32 PNG grid.

    Args:
        tmp_path: Pytest-provided temporary directory receiving the generated
            grid artifact.

    Returns:
        Nothing. The test checks the returned ``(N, C, H, W)`` shape and that
        CPU bfloat16 inputs are successfully serialized as a PNG.
    """
    images = [
        torch.full((1, 3, 8, 8), -1.0, dtype=torch.bfloat16),
        torch.full((1, 3, 8, 8), 1.0, dtype=torch.bfloat16),
    ]
    output_file = tmp_path / "grid.png"

    shape = quick_viz.save_grid(images, output_file)

    assert shape == (2, 3, 8, 8)
    assert output_file.is_file()
    assert output_file.stat().st_size > 0


def test_score_and_print_images_reports_each_score(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify PickScore values are printed in prompt order with their mean.

    Args:
        monkeypatch: Pytest fixture replacing the heavyweight PickScore model
            with a deterministic mock.
        capsys: Pytest fixture capturing standard output for assertions.

    Returns:
        Nothing. The test checks returned values, printed per-image scores,
        the aggregate mean, and conversion from ``(1, C, H, W)`` image items
        to the scorer's expected ``(C, H, W)`` tensors.
    """
    scorer = Mock()
    scorer.score.return_value = torch.tensor([1.25, 2.75])
    scorer_class = Mock(return_value=scorer)
    monkeypatch.setattr(pickscore_module, "PickScore", scorer_class)
    images = [
        torch.zeros((1, 3, 8, 8)),
        torch.ones((1, 3, 8, 8)),
    ]

    scores = quick_viz.score_and_print_images(
        prompts=["first prompt", "second prompt"],
        images=images,
        model_name_or_path="fake-model",
        processor_name_or_path="fake-processor",
        batch_size=2,
        logger=logging.getLogger("test_quick_viz"),
    )

    output = capsys.readouterr().out
    assert scores == [1.25, 2.75]
    assert "[0000] PickScore=1.250000 | first prompt" in output
    assert "[0001] PickScore=2.750000 | second prompt" in output
    assert "Mean PickScore=2.000000" in output
    scored_images = scorer.score.call_args.kwargs["images"]
    assert [tuple(image.shape) for image in scored_images] == [(3, 8, 8), (3, 8, 8)]


def test_score_all_rewards_writes_aligned_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify all five wrappers are scored and serialized in prompt order.

    Args:
        monkeypatch: Pytest fixture replacing heavyweight reward constructors.
        tmp_path: Temporary directory receiving the score JSON artifact.

    Returns:
        Nothing. The test checks provider names, aligned values, and the
        persisted sidecar generated beside the requested grid.
    """
    constructed: list[Mock] = []

    def fake_constructor(*args: object, **kwargs: object) -> Mock:
        """Construct one deterministic fake reward evaluator.

        Args:
            *args: Positional model configuration accepted and ignored.
            **kwargs: Keyword model configuration accepted and ignored.

        Returns:
            Mock evaluator returning two fixed scalar rewards.
        """
        del args, kwargs
        evaluator = Mock()
        evaluator.score.return_value = torch.tensor([1.0, 2.0])
        constructed.append(evaluator)
        return evaluator

    for name in (
        "PickScoreReward",
        "HPSv21Reward",
        "LAIONAestheticReward",
        "CLIPReward",
        "ImageRewardEvaluator",
    ):
        monkeypatch.setattr(rewards_module, name, fake_constructor)

    output_file = tmp_path / "grid.png"
    results = quick_viz.score_all_rewards(
        prompts=["first", "second"],
        images=[
            torch.zeros((1, 3, 8, 8)),
            torch.ones((1, 3, 8, 8)),
        ],
        pickscore_model="pickscore",
        pickscore_processor="processor",
        hps_base_checkpoint=tmp_path / "hps-base.bin",
        hps_checkpoint=tmp_path / "hps.pt",
        clip_model="clip",
        aesthetic_checkpoint=tmp_path / "aesthetic.pth",
        imagereward_root=tmp_path / "imagereward",
        imagereward_tokenizer="bert-tokenizer",
        batch_size=2,
        output_file=output_file,
        logger=logging.getLogger("test_quick_viz_all_rewards"),
    )

    assert set(results) == {
        "PickScore",
        "HPSv2",
        "Aesthetics",
        "CLIP",
        "ImageReward",
    }
    assert all(values == [1.0, 2.0] for values in results.values())
    assert len(constructed) == 5
    assert output_file.with_suffix(".scores.json").is_file()
