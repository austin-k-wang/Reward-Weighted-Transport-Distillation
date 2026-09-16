"""Convenient batched inference wrapper for the PickScore v1 reward model."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


ImageInput = str | Path | Image.Image | torch.Tensor


class PickScore:
    """Score prompt-image pairs with the PickScore v1 CLIP-H reward model.

    The wrapper accepts image paths, PIL images, and tensors. A prompt string
    is automatically broadcast when scoring multiple images. Tensor inputs
    may have shape ``(C, H, W)``, ``(H, W, C)``, or ``(N, C, H, W)`` and are
    interpreted using ``tensor_value_range``.
    """

    DEFAULT_MODEL = "yuvalkirstain/PickScore_v1"
    DEFAULT_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        processor_name_or_path: str | Path = DEFAULT_PROCESSOR,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        tensor_value_range: tuple[float, float] = (-1.0, 1.0),
        logger: logging.Logger | None = None,
    ) -> None:
        """Load PickScore and its image/text processor.

        Args:
            model_name_or_path: Hugging Face model ID or local directory for
                PickScore weights.
            processor_name_or_path: Hugging Face model ID or local directory
                containing the LAION CLIP-H processor and tokenizer.
            device: Inference device. Defaults to CUDA when available and CPU
                otherwise.
            dtype: Floating-point dtype used by model parameters and floating
                processor inputs. Float32 is the reference PickScore dtype.
            tensor_value_range: Inclusive ``(minimum, maximum)`` represented
                by floating image tensors. SANA outputs use ``(-1, 1)``.
            logger: Optional logger. If omitted, a module logger is used.

        Returns:
            Nothing. The initialized instance owns an evaluation-only,
            frozen PickScore model and its processor.

        Raises:
            ValueError: If ``tensor_value_range`` is invalid.
            RuntimeError: If CUDA is explicitly requested but unavailable.
        """
        range_min, range_max = tensor_value_range
        if range_max <= range_min:
            raise ValueError("tensor_value_range maximum must exceed its minimum")

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PickScore but is unavailable")

        self.dtype = dtype
        self.tensor_value_range = tensor_value_range
        self.logger = logger or logging.getLogger(__name__)

        self.logger.info(
            "Loading PickScore model=%s processor=%s device=%s dtype=%s",
            model_name_or_path,
            processor_name_or_path,
            self.device,
            self.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(str(processor_name_or_path))
        self.model = AutoModel.from_pretrained(str(model_name_or_path))
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=self.dtype)

    @torch.inference_mode()
    def score(
        self,
        prompts: str | Sequence[str],
        images: ImageInput | Sequence[ImageInput],
        batch_size: int = 8,
        show_progress: bool = True,
        metadata: Sequence[Mapping[str, object] | None] | None = None,
    ) -> torch.Tensor:
        """Compute raw PickScore logits for corresponding prompt-image pairs.

        Args:
            prompts: One prompt to broadcast across every image, or one prompt
                per image.
            images: One image or a sequence of images. A rank-4 tensor is
                expanded along its leading batch dimension. Individual tensor
                images must have shape ``(C, H, W)`` or ``(H, W, C)``.
            batch_size: Maximum number of prompt-image pairs processed in one
                forward pass.
            show_progress: Whether to display a tqdm batch progress bar.
            metadata: Optional structured rows accepted for reward-interface
                compatibility and ignored by PickScore.

        Returns:
            A CPU float32 tensor of shape ``(N,)`` containing unnormalized
            PickScore logits. These scores are suitable for comparing systems
            over the same prompt set.

        Raises:
            ValueError: If inputs are empty, pair counts differ, or
                ``batch_size`` is not positive.
            TypeError: If an image input type is unsupported.
        """
        del metadata
        image_items = self._normalize_images(images)
        prompt_items = self._normalize_prompts(prompts, len(image_items))
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        scores = []
        starts = range(0, len(image_items), batch_size)
        progress = tqdm(
            starts,
            total=(len(image_items) + batch_size - 1) // batch_size,
            desc="Computing PickScore",
            unit="batch",
            disable=not show_progress,
        )
        for start in progress:
            batch_images = [
                self._to_pil_image(image)
                for image in image_items[start : start + batch_size]
            ]
            batch_prompts = prompt_items[start : start + batch_size]

            image_inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
            )
            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = self._move_inputs(image_inputs)
            text_inputs = self._move_inputs(text_inputs)

            image_features = self._feature_tensor(
                self.model.get_image_features(**image_inputs)
            )
            text_features = self._feature_tensor(
                self.model.get_text_features(**text_inputs)
            )
            image_features = F.normalize(image_features.float(), dim=-1)
            text_features = F.normalize(text_features.float(), dim=-1)
            batch_scores = self.model.logit_scale.float().exp() * (
                image_features * text_features
            ).sum(dim=-1)
            scores.append(batch_scores.cpu())

        result = torch.cat(scores).to(dtype=torch.float32)
        self.logger.info(
            "Scored %d prompt-image pairs; mean raw PickScore=%.6f",
            result.numel(),
            result.mean().item(),
        )
        return result

    @torch.inference_mode()
    def preference_probabilities(
        self,
        prompt: str,
        images: ImageInput | Sequence[ImageInput],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Compute relative preference probabilities for one prompt's images.

        Args:
            prompt: Prompt shared by all candidate images.
            images: One or more candidate images accepted by :meth:`score`.
            batch_size: Maximum candidates processed in one forward pass.
            show_progress: Whether to display a tqdm batch progress bar.

        Returns:
            A CPU float32 tensor of shape ``(N,)`` whose values sum to one.
            Values are the softmax of raw PickScore logits across candidates.
        """
        scores = self.score(
            prompts=prompt,
            images=images,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return torch.softmax(scores, dim=0)

    def __call__(
        self,
        prompts: str | Sequence[str],
        images: ImageInput | Sequence[ImageInput],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """Call :meth:`score` using the same prompt, image, and batch options.

        Args:
            prompts: One broadcast prompt or one prompt per image.
            images: Image paths, PIL images, or image tensors.
            batch_size: Maximum number of pairs per model forward pass.
            show_progress: Whether to display the tqdm progress bar.

        Returns:
            A CPU float32 raw-score tensor with shape ``(N,)``.
        """
        return self.score(prompts, images, batch_size, show_progress)

    def _normalize_images(
        self,
        images: ImageInput | Sequence[ImageInput],
    ) -> list[ImageInput]:
        """Convert a single image or image batch into a non-empty Python list.

        Args:
            images: A supported single image, a rank-4 tensor batch with shape
                ``(N, C, H, W)``, or a sequence of supported images.

        Returns:
            A non-empty list containing individual image inputs.

        Raises:
            ValueError: If a tensor has an unsupported rank or no images are
                provided.
            TypeError: If ``images`` is not a supported input or sequence.
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                image_items: list[ImageInput] = [images]
            elif images.ndim == 4:
                image_items = list(images)
            else:
                raise ValueError(
                    "Image tensors must have shape (C, H, W), (H, W, C), "
                    f"or (N, C, H, W); got {tuple(images.shape)}"
                )
        elif isinstance(images, (str, Path, Image.Image)):
            image_items = [images]
        elif isinstance(images, Sequence):
            image_items = list(images)
        else:
            raise TypeError(f"Unsupported images input type: {type(images).__name__}")

        if not image_items:
            raise ValueError("At least one image is required")
        return image_items

    def _normalize_prompts(
        self,
        prompts: str | Sequence[str],
        image_count: int,
    ) -> list[str]:
        """Broadcast or validate prompts against the image count.

        Args:
            prompts: One prompt string or a sequence containing one string per
                image.
            image_count: Number of normalized image inputs.

        Returns:
            A list of exactly ``image_count`` prompt strings.

        Raises:
            ValueError: If prompt and image counts differ or any prompt is not
                a string.
        """
        if isinstance(prompts, str):
            prompt_items = [prompts] * image_count
        else:
            prompt_items = list(prompts)

        if len(prompt_items) != image_count:
            raise ValueError(
                f"Expected one prompt per image, got {len(prompt_items)} "
                f"prompts and {image_count} images"
            )
        if any(not isinstance(prompt, str) for prompt in prompt_items):
            raise ValueError("Every prompt must be a string")
        return prompt_items

    def _to_pil_image(self, image: ImageInput) -> Image.Image:
        """Convert one supported image input to an owned RGB PIL image.

        Args:
            image: Image path, PIL image, or rank-3 tensor. Floating tensors
                are rescaled from ``tensor_value_range`` to ``[0, 255]``;
                uint8 tensors are used directly.

        Returns:
            An RGB ``PIL.Image.Image`` detached from source files and tensors.

        Raises:
            FileNotFoundError: If an image path does not exist.
            ValueError: If a tensor does not have one, three, or four channels.
            TypeError: If ``image`` has an unsupported type.
        """
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB").copy()
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported image type: {type(image).__name__}")

        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Individual image tensor must be rank 3, got {tensor.ndim}")
        if tensor.shape[0] not in (1, 3, 4) and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        if tensor.shape[0] not in (1, 3, 4):
            raise ValueError(
                "Image tensor channel count must be 1, 3, or 4; "
                f"got shape {tuple(tensor.shape)}"
            )

        if tensor.dtype == torch.uint8:
            pixels = tensor
        else:
            range_min, range_max = self.tensor_value_range
            pixels = (
                tensor.float()
                .sub(range_min)
                .div(range_max - range_min)
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
            )

        array = pixels.permute(1, 2, 0).numpy()
        if array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")

    def _move_inputs(self, inputs: Any) -> dict[str, torch.Tensor]:
        """Move processor tensors to the model device and compatible dtype.

        Args:
            inputs: Mapping-like Transformers processor output. Floating
                tensors, such as pixel values, are converted to model dtype;
                integer token IDs and masks retain their dtype.

        Returns:
            A dictionary mapping input names to tensors on ``self.device``.
            Tensor shapes are unchanged.
        """
        moved = {}
        for name, value in inputs.items():
            if torch.is_floating_point(value):
                moved[name] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[name] = value.to(device=self.device)
        return moved

    @staticmethod
    def _feature_tensor(features: Any) -> torch.Tensor:
        """Extract an embedding tensor across Transformers return conventions.

        Args:
            features: Tensor returned by ``get_image_features`` or
                ``get_text_features``, or an output object exposing
                ``pooler_output``.

        Returns:
            Embedding tensor with shape ``(N, D)``.

        Raises:
            TypeError: If no tensor representation can be extracted.
        """
        if isinstance(features, torch.Tensor):
            return features
        pooled = getattr(features, "pooler_output", None)
        if isinstance(pooled, torch.Tensor):
            return pooled
        raise TypeError(f"Unsupported model feature output: {type(features).__name__}")
