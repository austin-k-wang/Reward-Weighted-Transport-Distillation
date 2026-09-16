"""Mocked tests for differentiable native SANA-Sprint policy rollouts."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.alignment.generators.sana_sprint import SanaSprintPolicy


class FakeTokens:
    """Tensor container matching the tokenizer output methods used by SANA."""

    def __init__(self, count: int, length: int) -> None:
        """Create deterministic token IDs and masks.

        Args:
            count: Prompt batch size.
            length: Token sequence length.

        Returns:
            Nothing. Public tensors are initialized on CPU.
        """
        self.input_ids = torch.ones(count, length, dtype=torch.long)
        self.attention_mask = torch.ones(count, length, dtype=torch.long)

    def to(self, device: torch.device) -> "FakeTokens":
        """Move token tensors to a target device and return this container.

        Args:
            device: Destination device.

        Returns:
            This mutated token container.
        """
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


class FakeTokenizer:
    """Provide fixed-length tokenization for policy tests."""

    def encode(self, text: str) -> list[int]:
        """Return one token per whitespace-delimited input word.

        Args:
            text: Input CHI prompt.

        Returns:
            Deterministic token ID list.
        """
        return list(range(max(1, len(text.split()))))

    def __call__(self, prompts: list[str], **kwargs: object) -> FakeTokens:
        """Tokenize a prompt batch to the requested padded length.

        Args:
            prompts: Prompt strings.
            **kwargs: Tokenizer options containing ``max_length``.

        Returns:
            Fixed token tensor container.
        """
        return FakeTokens(len(prompts), int(kwargs["max_length"]))


class FakeTextEncoder(torch.nn.Module):
    """Map token IDs to small deterministic caption embeddings."""

    def __init__(self) -> None:
        """Initialize one frozen-compatible scalar parameter."""
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        """Expand IDs into four-channel token embeddings.

        Args:
            input_ids: Token IDs shaped ``[B,T]``.
            attention_mask: Unused mask shaped ``[B,T]``.

        Returns:
            One tuple element shaped ``[B,T,4]``.
        """
        del attention_mask
        return (input_ids.float().unsqueeze(-1).expand(-1, -1, 4) * self.scale,)


class FakeSanaModel(torch.nn.Module):
    """Expose one adapter parameter and a base-policy disable context."""

    def __init__(self) -> None:
        """Initialize frozen base and trainable adapter parameters."""
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.tensor(0.5))
        self.adapter_disabled = False
        self.last_data_info: dict[str, torch.Tensor] = {}

    @contextmanager
    def disable_adapter(self):
        """Temporarily switch the fake model to frozen-base behavior."""
        previous = self.adapter_disabled
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = previous

    def forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        captions: torch.Tensor,
        *,
        data_info: dict[str, torch.Tensor],
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return latent-scaled fake trigflow velocity.

        Args:
            latents: Normalized SCM latents.
            timestep: One trigflow timestep per image.
            captions: Caption embeddings.
            data_info: Image shape, aspect ratio, and CFG tensors.
            mask: Caption attention masks.

        Returns:
            Velocity tensor with the same shape as ``latents``.
        """
        del timestep, captions, mask
        self.last_data_info = data_info
        value = self.base if self.adapter_disabled else self.base + self.adapter
        return latents * value


class FakeScheduler:
    """Perform a deterministic one-step update without stochastic noise."""

    def set_timesteps(self, *, device: torch.device, **kwargs: object) -> None:
        """Create one active and one terminal timestep.

        Args:
            device: Tensor device.
            **kwargs: Unused scheduler settings.

        Returns:
            Nothing. ``timesteps`` is set to ``[1,0]``.
        """
        del kwargs
        self.timesteps = torch.tensor([1.0, 0.0], device=device)

    def step(
        self,
        model_output: torch.Tensor,
        timeindex: int,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        *,
        return_dict: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Subtract model velocity from the current sample.

        Args:
            model_output: Predicted velocity.
            timeindex: Unused step index.
            timestep: Unused active timestep.
            sample: Current latent tensor.
            return_dict: Must be false for tuple output.

        Returns:
            Identical previous-sample and predicted-clean tensors.
        """
        del timeindex, timestep
        assert not return_dict
        denoised = sample - model_output
        return denoised, denoised


class FakePipeline(torch.nn.Module):
    """Own the minimal native pipeline attributes used by the policy adapter."""

    def __init__(self) -> None:
        """Initialize fake model, VAE, tokenizer, and Sprint configuration."""
        super().__init__()
        self.model = FakeSanaModel()
        self.vae = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.text_encoder = FakeTextEncoder()
        self.tokenizer = FakeTokenizer()
        self.weight_dtype = torch.float32
        self.vae_dtype = torch.float32
        self.config = SimpleNamespace(
            max_timesteps=1.0,
            intermediate_timesteps=None,
            timesteps=None,
            vae=SimpleNamespace(
                vae_downsample_rate=2,
                vae_latent_dim=3,
                vae_type="fake",
            ),
            scheduler=SimpleNamespace(sigma_data=0.5),
            text_encoder=SimpleNamespace(chi_prompt=[], model_max_length=3),
        )


def fake_decode(
    name: str,
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode fake latents with a frozen differentiable convolution.

    Args:
        name: Unused VAE type.
        vae: Frozen convolutional decoder.
        latents: Latents shaped ``[N,3,H,W]``.

    Returns:
        Images shaped like ``latents``.
    """
    del name
    return vae(latents)


def test_policy_switches_adapter_modes_and_preserves_live_gradients() -> None:
    """Verify matched one-step rollouts only differentiate active adapters.

    Returns:
        Nothing. The test checks shapes, CFG metadata, graph boundaries, and
        trainable parameter gradients.
    """
    pipeline = FakePipeline()
    policy = SanaSprintPolicy(
        pipeline,
        resolution=4,
        guidance_scale=4.5,
        rollout_chunk_size=1,
        decode_chunk_size=1,
        vae_gradient_checkpointing=True,
        scheduler_cls=FakeScheduler,
        vae_decode_fn=fake_decode,
    )
    noise = policy.sample_initial_noise(2)

    current = policy.rollout(
        ["prompt"],
        samples_per_prompt=2,
        trainable=True,
        initial_noise=noise,
    )
    reference = policy.rollout(
        ["prompt"],
        samples_per_prompt=2,
        trainable=False,
        initial_noise=noise,
    )
    current.images.square().mean().backward()

    assert current.images.shape == (2, 3, 2, 2)
    assert current.images.requires_grad
    assert not reference.images.requires_grad
    assert pipeline.model.adapter.grad is not None
    assert pipeline.model.base.grad is None
    torch.testing.assert_close(
        pipeline.model.last_data_info["cfg_scale"],
        torch.tensor([4.5]),
    )


def test_policy_caches_prompt_embeddings_and_discards_text_encoder(
    tmp_path: Path,
) -> None:
    """Verify cached Gemma outputs replace the resident text encoder.

    Args:
        tmp_path: Temporary directory receiving the persistent cache artifact.

    Returns:
        Nothing. In-memory and disk-reloaded embeddings must match direct
        encoding, and an uncached prompt after Gemma removal must fail.
    """
    pipeline = FakePipeline()
    policy = SanaSprintPolicy(
        pipeline,
        resolution=4,
        guidance_scale=4.5,
        rollout_chunk_size=1,
        decode_chunk_size=1,
        vae_gradient_checkpointing=True,
        scheduler_cls=FakeScheduler,
        vae_decode_fn=fake_decode,
    )
    prompts = ("first prompt", "second prompt")
    expected_embeddings, expected_masks = policy.encode_prompts(prompts)
    cache_path = tmp_path / "gemma-embeddings.pt"

    policy.cache_prompt_embeddings(
        prompts,
        batch_size=1,
        cache_path=cache_path,
        cache_signature="fake-v1",
    )
    cached_embeddings, cached_masks = policy.encode_prompts(prompts)

    assert policy.text_encoder is None
    assert pipeline.text_encoder is None
    torch.testing.assert_close(cached_embeddings, expected_embeddings)
    torch.testing.assert_close(cached_masks, expected_masks)
    try:
        policy.encode_prompts(("uncached prompt",))
    except KeyError:
        pass
    else:
        raise AssertionError("Expected an uncached prompt to be rejected")

    reloaded_pipeline = FakePipeline()
    with torch.no_grad():
        reloaded_pipeline.text_encoder.scale.fill_(99)
    reloaded_policy = SanaSprintPolicy(
        reloaded_pipeline,
        resolution=4,
        guidance_scale=4.5,
        rollout_chunk_size=1,
        decode_chunk_size=1,
        vae_gradient_checkpointing=True,
        scheduler_cls=FakeScheduler,
        vae_decode_fn=fake_decode,
    )
    reloaded_policy.cache_prompt_embeddings(
        prompts,
        batch_size=1,
        cache_path=cache_path,
        cache_signature="fake-v1",
    )
    disk_embeddings, disk_masks = reloaded_policy.encode_prompts(prompts)

    assert cache_path.is_file()
    assert reloaded_policy.text_encoder is None
    torch.testing.assert_close(disk_embeddings, expected_embeddings)
    torch.testing.assert_close(disk_masks, expected_masks)


def test_vae_gradient_checkpoint_recomputes_decode_during_backward() -> None:
    """Verify checkpointed VAE decoding is recomputed during backpropagation.

    Returns:
        Nothing. The fake decoder call count must increase during backward while
        preserving gradients to the active SANA adapter.
    """
    pipeline = FakePipeline()
    calls = 0

    def counting_decode(
        name: str,
        vae: torch.nn.Module,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode fake latents while recording forward invocations.

        Args:
            name: Unused VAE type.
            vae: Frozen fake convolutional VAE.
            latents: Latent tensor shaped ``[N,3,H,W]``.

        Returns:
            Decoded tensor with the same shape as ``latents``.
        """
        nonlocal calls
        calls += 1
        del name
        return vae(latents)

    policy = SanaSprintPolicy(
        pipeline,
        resolution=4,
        guidance_scale=4.5,
        rollout_chunk_size=1,
        decode_chunk_size=1,
        vae_gradient_checkpointing=True,
        scheduler_cls=FakeScheduler,
        vae_decode_fn=counting_decode,
    )
    rollout = policy.rollout(
        ("prompt",),
        samples_per_prompt=1,
        trainable=True,
    )
    forward_calls = calls

    rollout.images.square().mean().backward()

    assert forward_calls == 1
    assert calls == 2
    assert pipeline.model.adapter.grad is not None
