"""Unit tests for deterministic SANA-Sprint Parti-Prompts evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/eval_parti_prompt.py"
SPEC = importlib.util.spec_from_file_location("eval_parti_prompt", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class FakeAccelerator:
    """Provide the single-rank Accelerator interface used by the evaluator."""

    device = torch.device("cpu")
    process_index = 0
    num_processes = 1
    is_main_process = True
    is_local_main_process = True

    def wait_for_everyone(self) -> None:
        """Represent a no-op synchronization barrier for one process.

        Args:
            None.

        Returns:
            Nothing because the fake runtime has only one rank.
        """


class FakeSanaPipeline:
    """Return one deterministic image tensor and record generator seeds."""

    def __init__(self) -> None:
        """Initialize an empty list of observed PyTorch seeds.

        Args:
            None.

        Returns:
            Nothing. Calls append their generator seed to ``seeds``.
        """
        self.seeds: list[int] = []

    def __call__(
        self,
        *,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Generate a zero-valued fake SANA image batch.

        Args:
            prompt: Prompt accepted for API compatibility.
            height: Output tensor height.
            width: Output tensor width.
            guidance_scale: Guidance accepted for API compatibility.
            num_inference_steps: Step count accepted for API compatibility.
            generator: Seeded generator whose initial seed is recorded.

        Returns:
            Float tensor with shape ``(1, 3, height, width)`` in ``[-1, 1]``.
        """
        del prompt, guidance_scale, num_inference_steps
        self.seeds.append(generator.initial_seed())
        return torch.zeros((1, 3, height, width), dtype=torch.float32)


class FakeReward:
    """Return fixed raw rewards while validating canonical scorer inputs."""

    def score(
        self,
        prompts: list[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Any = None,
    ) -> torch.Tensor:
        """Score two fake image tensors with fixed rewards.

        Args:
            prompts: One prompt per generated image.
            images: CPU image tensor shaped ``(2,3,32,32)``.
            batch_size: Positive scoring batch size.
            metadata: Optional metadata accepted and ignored.

        Returns:
            Raw reward tensor ``[20, 22]`` with shape ``(2,)``.
        """
        del metadata
        assert len(prompts) == images.shape[0] == 2
        assert tuple(images.shape) == (2, 3, 32, 32)
        assert batch_size == 2
        return torch.tensor([20.0, 22.0])


def test_sample_seed_is_stable_and_unique() -> None:
    """Verify global prompt/sample seeds are deterministic and distinct.

    Args:
        None.

    Returns:
        Nothing. The test compares repeat calls and neighboring coordinates.
    """
    seed = evaluation.sample_seed(7, 3, 42)

    assert seed == evaluation.sample_seed(7, 3, 42)
    assert seed != evaluation.sample_seed(7, 4, 42)
    assert seed != evaluation.sample_seed(8, 3, 42)


def test_shard_prompt_indices_cover_split_without_overlap() -> None:
    """Verify strided rank shards cover every prompt exactly once.

    Args:
        None.

    Returns:
        Nothing. The test asserts three rank shards are disjoint and complete.
    """
    shards = [
        evaluation.shard_prompt_indices(8, process_index, 3)
        for process_index in range(3)
    ]

    assert sorted(index for shard in shards for index in shard) == list(range(8))
    assert len({index for shard in shards for index in shard}) == 8


def test_load_parti_prompts_removes_header_and_applies_cap(tmp_path: Path) -> None:
    """Verify TSV loading reads the first column in order with a smoke cap.

    Args:
        tmp_path: Pytest directory receiving a small Parti-style TSV fixture.

    Returns:
        Nothing. The test checks header removal and one-prompt truncation.
    """
    prompt_file = tmp_path / "parti.tsv"
    prompt_file.write_text(
        "Prompt\tCategory\nfirst\tA\nsecond\tB\n",
        encoding="utf-8",
    )

    assert evaluation.load_parti_prompts(prompt_file, max_prompts=1) == ["first"]


def test_load_parti_prompts_randomizes_deterministically_before_cap(
    tmp_path: Path,
) -> None:
    """Verify seeded prompt selection is stable and differs across seeds.

    Args:
        tmp_path: Pytest directory receiving a small Parti-style TSV fixture.

    Returns:
        Nothing. The test compares repeated and differently seeded selections.
    """
    prompt_file = tmp_path / "parti.tsv"
    prompt_file.write_text(
        "Prompt\nfirst\nsecond\nthird\nfourth\n",
        encoding="utf-8",
    )

    selected = evaluation.load_parti_prompts(
        prompt_file,
        max_prompts=2,
        randomize=True,
        random_seed=7,
    )

    assert selected == evaluation.load_parti_prompts(
        prompt_file,
        max_prompts=2,
        randomize=True,
        random_seed=7,
    )
    assert selected != evaluation.load_parti_prompts(
        prompt_file,
        max_prompts=2,
        randomize=True,
        random_seed=8,
    )


def test_parser_defaults_to_local_sana_auxiliary_snapshots() -> None:
    """Verify auxiliary defaults use a complete cache or canonical repository.

    Args:
        None.

    Returns:
        Nothing. The test checks that defaults never select an arbitrary or
        malformed auxiliary model source.
    """
    args = evaluation.build_parser().parse_args([])

    assert (
        Path(args.text_encoder_path).is_dir()
        or args.text_encoder_path == "Efficient-Large-Model/gemma-2-2b-it"
    )
    assert (
        Path(args.vae_path).is_dir()
        or args.vae_path == "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
    )


def test_parser_accepts_random_evaluation_settings() -> None:
    """Verify the CLI represents a requested randomized 200-by-20 run.

    Args:
        None.

    Returns:
        Nothing. The test checks prompt selection and generation values.
    """
    args = evaluation.build_parser().parse_args(
        [
            "--num-prompts",
            "200",
            "--randomize-prompts",
            "--prompt-order-seed",
            "17",
            "--images-per-prompt",
            "20",
        ]
    )

    assert args.max_prompts == 200
    assert args.randomize_prompts
    assert args.prompt_order_seed == 17
    assert args.images_per_prompt == 20


def test_local_snapshot_rejects_missing_weight_shards(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify an indexed Hugging Face snapshot is usable only when complete.

    Args:
        monkeypatch: Pytest fixture redirecting the evaluator's project root.
        tmp_path: Pytest directory receiving a minimal Hugging Face cache tree.

    Returns:
        Nothing. The test checks rejection before, and acceptance after, the
        indexed model shard is created.
    """
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    repository_dir = (
        tmp_path
        / "models/hf-cache/hub"
        / "models--organization--model"
    )
    snapshot = repository_dir / "snapshots/revision"
    snapshot.mkdir(parents=True)
    (repository_dir / "refs").mkdir()
    (repository_dir / "refs/main").write_text("revision\n", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"weight":"model-00001-of-00001.safetensors"}}\n',
        encoding="utf-8",
    )

    assert evaluation.local_huggingface_snapshot("organization/model") is None

    (snapshot / "model-00001-of-00001.safetensors").touch()
    assert evaluation.local_huggingface_snapshot("organization/model") == str(snapshot)


def test_load_sana_pipeline_applies_peft_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify optional checkpoint adapters replace the native pipeline model.

    Args:
        monkeypatch: Pytest fixture replacing heavyweight SANA and PEFT loaders
            with call-recording mocks.
        tmp_path: Pytest directory representing an existing PEFT adapter.

    Returns:
        Nothing. The test checks adapter loading arguments and the frozen
        pipeline model installed for evaluation.
    """
    pipeline = MagicMock()
    base_model = MagicMock()
    wrapped_model = MagicMock()
    pipeline.model = base_model
    pipeline.eval.return_value = pipeline
    pipeline.requires_grad_.return_value = pipeline
    pipeline_class = MagicMock(return_value=pipeline)
    adapter_loader = MagicMock(return_value=wrapped_model)
    monkeypatch.setattr(evaluation, "SanaSprintPipeline", pipeline_class)
    monkeypatch.setattr(evaluation.PeftModel, "from_pretrained", adapter_loader)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()

    loaded = evaluation.load_sana_pipeline(
        tmp_path / "config.yaml",
        tmp_path / "base.pth",
        device=torch.device("cpu"),
        max_timesteps=1.5708,
        adapter_path=adapter_path,
    )

    assert loaded is pipeline
    adapter_loader.assert_called_once_with(
        base_model,
        str(adapter_path.resolve()),
        is_trainable=False,
    )
    assert pipeline.model is wrapped_model
    pipeline.requires_grad_.assert_called_once_with(False)


def test_evaluate_prompts_returns_rewards_and_seed_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify the single-rank loop generates, scores, and aggregates records.

    Args:
        monkeypatch: Pytest fixture replacing distributed object gathering
            with an identity operation.
        tmp_path: Output directory used with image saving disabled.

    Returns:
        Nothing. The test checks global statistics, tensor generation count,
        and deterministic per-sample seed recording.
    """
    monkeypatch.setattr(evaluation, "gather_object", lambda records: records)
    pipe = FakeSanaPipeline()

    summary = evaluation.evaluate_prompts(
        prompts=["test prompt"],
        pipe=pipe,
        scorers={"pickscore": FakeReward()},
        accelerator=FakeAccelerator(),
        output_dir=tmp_path,
        images_per_prompt=2,
        score_batch_size=2,
        resolution=32,
        num_inference_steps=1,
        guidance_scale=4.5,
        base_seed=42,
        save_images=False,
    )

    metrics = summary["reward_metrics"]["pickscore"]
    assert metrics["average_reward"] == 21.0
    assert metrics["reward_std"] == 1.0
    assert summary["num_generations"] == 2
    record = summary["results"][0]
    assert record["rewards"]["pickscore"] == [20.0, 22.0]
    assert record["seeds"] == [
        evaluation.sample_seed(0, 0, 42),
        evaluation.sample_seed(0, 1, 42),
    ]
    assert pipe.seeds == record["seeds"]
