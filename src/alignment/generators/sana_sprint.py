"""Differentiable one-step adapter for the native SANA-Sprint pipeline."""

from __future__ import annotations

import gc
import logging
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model
from torch.utils.checkpoint import checkpoint
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import AlignmentConfig
from ..types import RolloutResult


ROOT = Path(__file__).resolve().parents[3]
SANA_ROOT = ROOT / "Sana"
logger = logging.getLogger(__name__)


def _load_sana_symbols() -> tuple[type[Any], type[Any], Any, Any, Any]:
    """Import native SANA components after making the vendored tree visible.

    Returns:
        Tuple containing ``SanaSprintPipeline``, ``SCMScheduler``,
        ``get_vae``, ``vae_decode``, and ``set_grad_checkpoint``.
    """
    if str(SANA_ROOT) not in sys.path:
        sys.path.insert(0, str(SANA_ROOT))
    from app.sana_sprint_pipeline import SanaSprintPipeline
    from diffusion import SCMScheduler
    from diffusion.model.builder import get_vae, vae_decode
    from diffusion.model.utils import set_grad_checkpoint

    return SanaSprintPipeline, SCMScheduler, get_vae, vae_decode, set_grad_checkpoint


def _resolve_model_source(source: str, *, local_files_only: bool) -> str:
    """Resolve a local path or cached Hugging Face repository snapshot.

    Args:
        source: Existing local directory or Hugging Face repository ID.
        local_files_only: Whether network downloads are forbidden.

    Returns:
        Existing local path or snapshot directory accepted by model loaders.
    """
    path = Path(source).expanduser()
    if path.exists():
        return str(path)
    if local_files_only:
        return snapshot_download(repo_id=source, local_files_only=True)
    return source


class SanaSprintPolicy(nn.Module):
    """Expose live-LoRA and frozen-base one-step SANA-Sprint populations."""

    def __init__(
        self,
        pipeline: nn.Module,
        *,
        resolution: int,
        guidance_scale: float,
        rollout_chunk_size: int,
        decode_chunk_size: int,
        vae_gradient_checkpointing: bool,
        scheduler_cls: type[Any],
        vae_decode_fn: Any,
    ) -> None:
        """Wrap initialized native SANA components for differentiable rollout.

        Args:
            pipeline: Loaded native pipeline owning model, VAE, tokenizer,
                text encoder, and parsed Sprint config.
            resolution: Square output resolution, divisible by the VAE stride.
            guidance_scale: Embedded Sprint classifier-free guidance value.
            rollout_chunk_size: Maximum model-forward population chunk.
            decode_chunk_size: Maximum VAE decode population chunk.
            vae_gradient_checkpointing: Whether to recompute each VAE decode
                chunk during backward instead of retaining decoder activations.
            scheduler_cls: Native SCM scheduler class or a compatible fake.
            vae_decode_fn: Native differentiable VAE decode function.

        Returns:
            Nothing. The adapter borrows all modules from ``pipeline``.
        """
        super().__init__()
        self.pipeline = pipeline
        self.model = pipeline.model
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.config = pipeline.config
        self.resolution = int(resolution)
        self.guidance_scale = float(guidance_scale)
        self.rollout_chunk_size = int(rollout_chunk_size)
        self.decode_chunk_size = int(decode_chunk_size)
        self.vae_gradient_checkpointing = bool(vae_gradient_checkpointing)
        self.scheduler_cls = scheduler_cls
        self.vae_decode_fn = vae_decode_fn
        self._prompt_embedding_cache: dict[
            str,
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

        self.vae.eval().requires_grad_(False)
        self.text_encoder.eval().requires_grad_(False)

    @classmethod
    def from_config(
        cls,
        config: AlignmentConfig,
        *,
        device: torch.device,
    ) -> "SanaSprintPolicy":
        """Load SANA-Sprint, attach LoRA, and construct the policy adapter.

        Args:
            config: Validated alignment model and LoRA configuration.
            device: Process-local accelerator device.

        Returns:
            Policy with a frozen 1.6B base and trainable attention adapters.

        Raises:
            ValueError: If PEFT does not match any requested target modules.
        """
        pipeline_cls, scheduler_cls, get_vae_fn, vae_decode_fn, set_grad_checkpoint = (
            _load_sana_symbols()
        )

        class LocalAssetPipeline(pipeline_cls):
            """Native pipeline variant that resolves large assets explicitly."""

            def build_vae(self, native_config: Any) -> nn.Module:
                """Load the configured VAE from a local/cache-resolved source."""
                source = config.model.vae_path or native_config.vae_pretrained
                resolved = _resolve_model_source(
                    source,
                    local_files_only=config.model.local_files_only,
                )
                return get_vae_fn(
                    native_config.vae_type,
                    resolved,
                    self.device,
                ).to(self.vae_dtype)

            def build_text_encoder(self, native_config: Any) -> tuple[Any, nn.Module]:
                """Load Gemma tokenizer/decoder without implicit Hub API calls."""
                source = config.model.text_encoder_path
                if source is None:
                    return super().build_text_encoder(native_config)
                resolved = _resolve_model_source(
                    source,
                    local_files_only=config.model.local_files_only,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    resolved,
                    local_files_only=config.model.local_files_only,
                )
                tokenizer.padding_side = "right"
                text_encoder = (
                    AutoModelForCausalLM.from_pretrained(
                        resolved,
                        torch_dtype=torch.bfloat16,
                        local_files_only=config.model.local_files_only,
                    )
                    .get_decoder()
                    .to(self.device)
                )
                return tokenizer, text_encoder

        pipeline = LocalAssetPipeline(config.model.config_path, device=device)
        pipeline.config.max_timesteps = config.model.max_timesteps
        pipeline.from_pretrained(config.model.checkpoint_path)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.model.requires_grad_(False)

        peft_config = PeftLoraConfig(
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            init_lora_weights=config.lora.init_weights,
            target_modules=list(config.lora.target_modules),
        )
        pipeline.model = get_peft_model(pipeline.model, peft_config)
        trainable = [name for name, parameter in pipeline.model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError(
                "LoRA target_modules matched no SANA-Sprint modules: "
                f"{config.lora.target_modules}"
            )
        if config.model.gradient_checkpointing:
            set_grad_checkpoint(pipeline.model)
        pipeline.model.train()
        logger.info("Attached LoRA to %d trainable tensors", len(trainable))
        return cls(
            pipeline,
            resolution=config.model.resolution,
            guidance_scale=config.model.guidance_scale,
            rollout_chunk_size=config.model.rollout_chunk_size,
            decode_chunk_size=config.model.decode_chunk_size,
            vae_gradient_checkpointing=config.model.vae_gradient_checkpointing,
            scheduler_cls=scheduler_cls,
            vae_decode_fn=vae_decode_fn,
        )

    @property
    def device(self) -> torch.device:
        """Return the process-local model device.

        Returns:
            Device containing SANA model parameters and generated tensors.
        """
        return next(self.model.parameters()).device

    @property
    def weight_dtype(self) -> torch.dtype:
        """Return the native SANA model execution dtype.

        Returns:
            Pipeline dtype, normally bfloat16 for Sprint 1.6B.
        """
        return self.pipeline.weight_dtype

    def set_prepared_model(self, model: nn.Module) -> None:
        """Install a model returned by ``Accelerator.prepare``.

        Args:
            model: Distributed or unwrapped PEFT model to use for subsequent
                rollouts.

        Returns:
            Nothing. Both policy and native pipeline references are updated.
        """
        self.model = model
        self.pipeline.model = model

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only trainable LoRA parameters.

        Returns:
            Non-empty list of adapter parameters accepted by the optimizer.
        """
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _adapter_owner(self) -> nn.Module:
        """Resolve the PEFT module through an optional distributed wrapper.

        Returns:
            Unwrapped model exposing PEFT adapter context managers.
        """
        return getattr(self.model, "module", self.model)

    def sample_initial_noise(
        self,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample initial SCM latents for square SANA generation.

        Args:
            count: Number of independent particles.
            generator: Optional device-local random generator.

        Returns:
            Tensor shaped ``[count,C,H/32,W/32]`` with standard deviation
            ``sigma_data``.
        """
        stride = int(self.config.vae.vae_downsample_rate)
        latent_size = self.resolution // stride
        sigma_data = float(self.config.scheduler.sigma_data)
        return torch.randn(
            count,
            int(self.config.vae.vae_latent_dim),
            latent_size,
            latent_size,
            generator=generator,
            device=self.device,
        ) * sigma_data

    def encode_prompts(
        self,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts using Sprint's CHI prefix and Gemma token slicing.

        Args:
            prompts: One text condition per generated image.

        Returns:
            Detached caption embeddings shaped ``[N,1,T,D]`` and attention
            masks shaped ``[N,T]``.
        """
        if not prompts:
            raise ValueError("At least one prompt is required")
        if self._prompt_embedding_cache:
            missing = [
                prompt
                for prompt in dict.fromkeys(prompts)
                if prompt not in self._prompt_embedding_cache
            ]
            if missing:
                raise KeyError(
                    "Prompt embedding cache does not contain "
                    f"{len(missing)} requested prompts"
                )
            embeddings = torch.stack(
                [self._prompt_embedding_cache[prompt][0] for prompt in prompts],
            ).to(device=self.device, dtype=self.weight_dtype)
            masks = torch.stack(
                [self._prompt_embedding_cache[prompt][1] for prompt in prompts],
            ).to(device=self.device)
            return embeddings, masks
        return self._encode_prompts_uncached(prompts)

    def _encode_prompts_uncached(
        self,
        prompts: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts directly with the resident Gemma decoder.

        Args:
            prompts: Non-empty prompt strings to encode as one batch.

        Returns:
            Detached caption embeddings shaped ``[N,1,T,D]`` and masks shaped
            ``[N,T]`` on the policy device.

        Raises:
            RuntimeError: If Gemma has already been discarded after caching.
        """
        if self.text_encoder is None:
            raise RuntimeError("Gemma was discarded after prompt caching")
        with torch.no_grad():
            chi_lines = self.config.text_encoder.chi_prompt
            if chi_lines:
                chi_prompt = "\n".join(chi_lines)
                prompts_all = [chi_prompt + prompt for prompt in prompts]
                chi_tokens = len(self.tokenizer.encode(chi_prompt))
                max_length = (
                    chi_tokens + int(self.config.text_encoder.model_max_length) - 2
                )
            else:
                prompts_all = list(prompts)
                max_length = int(self.config.text_encoder.model_max_length)
            tokens = self.tokenizer(
                prompts_all,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device=self.device)
            model_length = int(self.config.text_encoder.model_max_length)
            select_index = [0] + list(range(-model_length + 1, 0))
            embeddings = self.text_encoder(
                tokens.input_ids,
                tokens.attention_mask,
            )[0][:, None][:, :, select_index].to(self.weight_dtype)
            masks = tokens.attention_mask[:, select_index]
        return embeddings.detach(), masks.detach()

    @torch.no_grad()
    def cache_prompt_embeddings(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        cache_path: str | Path | None = None,
        cache_signature: str = "",
        rebuild: bool = False,
    ) -> Path | None:
        """Load or compute a finite CPU prompt cache and discard Gemma.

        Args:
            prompts: Complete training and optional evaluation prompt set.
                Duplicate strings are encoded only once.
            batch_size: Maximum number of unique prompts per Gemma forward.
            cache_path: Optional persistent PyTorch cache artifact. When it
                contains the exact prompt sequence and signature, Gemma
                inference is skipped.
            cache_signature: Stable description of the encoder and prompt
                preprocessing configuration used to validate an artifact.
            rebuild: Whether to ignore and replace an existing artifact.

        Returns:
            Resolved persistent cache path, or ``None`` when disk persistence
            is disabled. CPU tensors remain in memory for rollout lookup.

        Raises:
            ValueError: If no prompts are supplied or ``batch_size`` is invalid.
            RuntimeError: If Gemma has already been discarded.
        """
        if batch_size < 1:
            raise ValueError("Prompt encoding batch size must be positive")
        unique_prompts = tuple(dict.fromkeys(prompts))
        if not unique_prompts:
            raise ValueError("At least one prompt is required for caching")
        resolved_cache = (
            Path(cache_path).expanduser().resolve()
            if cache_path is not None
            else None
        )
        if resolved_cache is not None and resolved_cache.is_file() and not rebuild:
            payload = torch.load(
                resolved_cache,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(payload, dict):
                payload = {}
            cached_prompts = tuple(payload.get("prompts", ()))
            embeddings = payload.get("embeddings")
            masks = payload.get("masks")
            valid = (
                payload.get("version") == 1
                and payload.get("signature") == cache_signature
                and cached_prompts == unique_prompts
                and isinstance(embeddings, torch.Tensor)
                and isinstance(masks, torch.Tensor)
                and embeddings.shape[0] == len(unique_prompts)
                and masks.shape[0] == len(unique_prompts)
            )
            if valid:
                self._prompt_embedding_cache = {
                    prompt: (embeddings[index], masks[index])
                    for index, prompt in enumerate(unique_prompts)
                }
                self._discard_text_encoder()
                logger.info(
                    "Loaded %d Gemma prompt embeddings from %s",
                    len(unique_prompts),
                    resolved_cache,
                )
                return resolved_cache
            logger.warning(
                "Ignoring incompatible Gemma prompt cache at %s",
                resolved_cache,
            )
        if self.text_encoder is None:
            raise RuntimeError("Gemma is unavailable for prompt caching")

        cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        starts = range(0, len(unique_prompts), batch_size)
        progress = tqdm(
            starts,
            total=(len(unique_prompts) + batch_size - 1) // batch_size,
            desc="Caching Gemma prompt embeddings",
            unit="batch",
        )
        for start in progress:
            prompt_batch = unique_prompts[start : start + batch_size]
            embeddings, masks = self._encode_prompts_uncached(prompt_batch)
            for index, prompt in enumerate(prompt_batch):
                cache[prompt] = (
                    embeddings[index].detach().cpu(),
                    masks[index].detach().cpu(),
                )
        self._prompt_embedding_cache = cache
        if resolved_cache is not None:
            resolved_cache.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = resolved_cache.with_name(
                f".{resolved_cache.name}.tmp"
            )
            torch.save(
                {
                    "version": 1,
                    "signature": cache_signature,
                    "prompts": unique_prompts,
                    "embeddings": torch.stack(
                        [cache[prompt][0] for prompt in unique_prompts],
                    ),
                    "masks": torch.stack(
                        [cache[prompt][1] for prompt in unique_prompts],
                    ),
                },
                temporary_path,
            )
            temporary_path.replace(resolved_cache)
            logger.info("Saved Gemma prompt embeddings to %s", resolved_cache)
        self._discard_text_encoder()
        logger.info(
            "Cached %d unique prompt embeddings on CPU and discarded Gemma",
            len(cache),
        )
        return resolved_cache

    def _discard_text_encoder(self) -> None:
        """Remove Gemma module references and release its CUDA allocation.

        Returns:
            Nothing. The tokenizer and CPU embedding cache remain available.
        """
        self.pipeline.text_encoder = None
        self.text_encoder = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def rollout(
        self,
        prompts: Sequence[str],
        *,
        samples_per_prompt: int,
        trainable: bool,
        initial_noise: torch.Tensor | None = None,
    ) -> RolloutResult:
        """Generate one live-LoRA or frozen-base one-step population.

        Args:
            prompts: Independent text conditions.
            samples_per_prompt: Number of particles generated per condition.
            trainable: True to retain the graph through active LoRA, false to
                disable adapters and execute under ``torch.no_grad``.
            initial_noise: Optional pre-scaled SCM noise shaped
                ``[len(prompts)*samples_per_prompt,C,H/32,W/32]``.

        Returns:
            Generated latent/image population with repeated prompt alignment.

        Raises:
            ValueError: If prompts, sample count, or supplied noise disagree.
        """
        if not prompts or samples_per_prompt < 1:
            raise ValueError("rollout requires prompts and a positive sample count")
        repeated = tuple(
            prompt for prompt in prompts for _ in range(samples_per_prompt)
        )
        count = len(repeated)
        noise = self.sample_initial_noise(count) if initial_noise is None else initial_noise
        if noise.shape[0] != count:
            raise ValueError(
                f"Expected {count} initial noise samples, got {noise.shape[0]}"
            )
        noise = noise.to(device=self.device)
        caption_embs, masks = self.encode_prompts(repeated)

        adapter_context = nullcontext()
        if not trainable:
            disable_adapter = getattr(self._adapter_owner(), "disable_adapter", None)
            adapter_context = disable_adapter() if callable(disable_adapter) else nullcontext()
        gradient_context = nullcontext() if trainable else torch.no_grad()
        with adapter_context, gradient_context:
            denoised = self._one_step(noise, caption_embs, masks)
            images = self._decode(denoised)
        return RolloutResult(
            name="current" if trainable else "reference",
            prompts=repeated,
            initial_noise=noise,
            denoised_latents=denoised,
            images=images,
            extras={"trainable": trainable},
        )

    def _one_step(
        self,
        noise: torch.Tensor,
        caption_embs: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Run the exact native one-step SCM/trigflow update in chunks.

        Args:
            noise: Pre-scaled SCM latent tensor shaped ``[N,C,H,W]``.
            caption_embs: Gemma conditions shaped ``[N,1,T,D]``.
            masks: Gemma attention masks shaped ``[N,T]``.

        Returns:
            Denoised latent tensor shaped like ``noise`` with live gradients
            when active LoRA is selected.
        """
        scheduler = self.scheduler_cls()
        scheduler.set_timesteps(
            num_inference_steps=1,
            max_timesteps=self.config.max_timesteps,
            intermediate_timesteps=self.config.intermediate_timesteps,
            timesteps=self.config.timesteps,
            device=self.device,
        )
        timestep = scheduler.timesteps[0]
        sigma_data = float(self.config.scheduler.sigma_data)
        chunks: list[torch.Tensor] = []
        for start in range(0, noise.shape[0], self.rollout_chunk_size):
            latent_chunk = noise[start : start + self.rollout_chunk_size]
            embedding_chunk = caption_embs[start : start + self.rollout_chunk_size]
            mask_chunk = masks[start : start + self.rollout_chunk_size]
            count = latent_chunk.shape[0]
            data_info = {
                "img_hw": torch.tensor(
                    [[self.resolution, self.resolution]],
                    device=self.device,
                    dtype=torch.float32,
                ).repeat(count, 1),
                "aspect_ratio": torch.ones(count, 1, device=self.device),
                "cfg_scale": torch.full(
                    (count,),
                    self.guidance_scale,
                    device=self.device,
                    dtype=torch.float32,
                ),
            }
            model_output = sigma_data * self.model(
                latent_chunk / sigma_data,
                timestep.expand(count),
                embedding_chunk,
                data_info=data_info,
                mask=mask_chunk,
            )
            _, pred_x0 = scheduler.step(
                model_output,
                0,
                timestep,
                latent_chunk,
                return_dict=False,
            )
            chunks.append(pred_x0)
        return torch.cat(chunks, dim=0)

    def _decode(self, denoised: torch.Tensor) -> torch.Tensor:
        """Decode denoised Sprint latents while preserving live gradients.

        Args:
            denoised: SCM prediction shaped ``[N,C,H,W]`` and scaled by
                ``sigma_data``.

        Returns:
            Concatenated images shaped ``[N,3,resolution,resolution]`` in the
            native approximate ``[-1,1]`` range.
        """
        sigma_data = float(self.config.scheduler.sigma_data)
        decoded = []
        for chunk in denoised.split(self.decode_chunk_size):
            def decode_chunk(values: torch.Tensor) -> torch.Tensor:
                """Decode one scaled latent chunk through the frozen VAE.

                Args:
                    values: Denoised latent chunk shaped ``[N,C,H,W]``.

                Returns:
                    Decoded image tensor shaped
                    ``[N,3,resolution,resolution]``.
                """
                return self.vae_decode_fn(
                    self.config.vae.vae_type,
                    self.vae,
                    (values / sigma_data).to(self.pipeline.vae_dtype),
                )
            if (
                self.vae_gradient_checkpointing
                and torch.is_grad_enabled()
                and chunk.requires_grad
            ):
                decoded.append(
                    checkpoint(
                        decode_chunk,
                        chunk,
                        use_reentrant=False,
                    )
                )
            else:
                decoded.append(decode_chunk(chunk))
        return torch.cat(decoded, dim=0)
