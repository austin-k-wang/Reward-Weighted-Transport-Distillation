"""Simple frozen DINOv2 image feature encoder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


ImageInput = str | Path | Image.Image | torch.Tensor
FeatureSelection = str | Sequence[str]
SUPPORTED_FEATURES = ("cls", "patch_mean", "patch_std", "patch")


class DINOv2FeatureEncoder(nn.Module):
    """Extract frozen DINOv2 features from paths, PIL images, or tensors.

    ``encode`` is the convenient inference API and accepts all supported image
    types. ``forward`` accepts a tensor in ``tensor_value_range`` and preserves
    gradients with respect to that tensor, which makes it suitable for RWTD.
    By default both methods return one final-layer CLS embedding per image.
    """

    DEFAULT_MODEL = (
        Path(__file__).resolve().parents[1] / "models" / "facebook-dinov2-base"
    )

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        tensor_value_range: tuple[float, float] = (-1.0, 1.0),
        default_features: FeatureSelection = ("cls",),
        local_files_only: bool = True,
    ) -> None:
        """Load and freeze a DINOv2 model and its image processor.

        Args:
            model_name_or_path: Hugging Face model ID or local checkpoint
                directory containing DINOv2 weights and processor metadata.
            device: Model device. Defaults to CUDA when available, otherwise
                CPU.
            dtype: Floating-point dtype for model parameters and image inputs.
            tensor_value_range: Inclusive value range represented by floating
                tensor images. Generated SANA/SDXL images commonly use
                ``(-1, 1)``.
            default_features: Ordered feature names concatenated by
                :meth:`forward` and :meth:`encode`. Supported names are
                ``cls``, ``patch_mean``, ``patch_std``, and ``patch``.
            local_files_only: Whether Hugging Face loading must avoid network
                access.

        Returns:
            Nothing. The instance owns a frozen evaluation-mode DINOv2 model.

        Raises:
            ValueError: If the tensor range or feature selection is invalid.
            RuntimeError: If CUDA is requested but unavailable.
        """
        super().__init__()
        range_min, range_max = tensor_value_range
        if range_max <= range_min:
            raise ValueError("tensor_value_range maximum must exceed its minimum")

        resolved_device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DINOv2 but is unavailable")

        self.tensor_value_range = (float(range_min), float(range_max))
        self.default_features = self._validate_features(default_features)
        self.processor = AutoImageProcessor.from_pretrained(
            str(model_name_or_path),
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            str(model_name_or_path),
            local_files_only=local_files_only,
        )
        self.model.eval().requires_grad_(False)
        self.model.to(device=resolved_device, dtype=dtype)

        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32)
        std = torch.tensor(self.processor.image_std, dtype=torch.float32)
        self.register_buffer("image_mean", mean.view(1, -1, 1, 1), persistent=False)
        self.register_buffer("image_std", std.view(1, -1, 1, 1), persistent=False)

    @property
    def device(self) -> torch.device:
        """Return the device containing the DINOv2 model parameters.

        Returns:
            Device used for model execution and differentiable tensor inputs.
        """
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the floating-point dtype of the DINOv2 model.

        Returns:
            Parameter dtype used for model execution.
        """
        return next(self.model.parameters()).dtype

    def forward(
        self,
        images: torch.Tensor,
        features: FeatureSelection | None = None,
        *,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Extract concatenated differentiable features from image tensors.

        Args:
            images: Image tensor shaped ``(C,H,W)`` or ``(B,C,H,W)``. Floating
                values are interpreted using ``tensor_value_range``; uint8
                values are interpreted in ``[0,255]``.
            features: Ordered feature blocks to flatten and concatenate.
                Defaults to ``default_features``.
            normalize: Whether to L2-normalize the final feature vector.

        Returns:
            Feature tensor shaped ``(B,D_total)`` on the model device. For
            DINOv2-Base, the default CLS output has shape ``(B,768)``.

        Raises:
            ValueError: If the input shape or feature names are invalid.
        """
        selected = self._validate_features(
            self.default_features if features is None else features
        )
        values = self.vector_features(images, selected)
        result = torch.cat([values[name].flatten(start_dim=1) for name in selected], dim=1)
        if normalize:
            result = F.normalize(result.float(), dim=-1)
        return result

    def vector_features(
        self,
        images: torch.Tensor,
        features: FeatureSelection | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract named final-layer DINOv2 token feature blocks.

        Args:
            images: Image tensor shaped ``(C,H,W)`` or ``(B,C,H,W)`` in the
                configured tensor range.
            features: Requested blocks from ``cls``, ``patch_mean``,
                ``patch_std``, and ``patch``. Defaults to ``default_features``.

        Returns:
            Ordered dictionary of tensors. ``cls``, ``patch_mean``, and
            ``patch_std`` have shape ``(B,1,D)``; ``patch`` has shape
            ``(B,N,D)``. Gradients to floating input images are preserved.
        """
        selected = self._validate_features(
            self.default_features if features is None else features
        )
        pixel_values = self._preprocess_tensor(images)
        outputs = self.model(pixel_values=pixel_values)
        hidden = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("DINOv2 did not return last_hidden_state shaped (B,N,D)")

        cls_token = hidden[:, :1]
        patch_tokens = hidden[:, 1:]
        available = {
            "cls": cls_token,
            "patch_mean": patch_tokens.mean(dim=1, keepdim=True),
            "patch_std": patch_tokens.std(dim=1, correction=0, keepdim=True),
            "patch": patch_tokens,
        }
        return {name: available[name] for name in selected}

    @torch.inference_mode()
    def encode(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        features: FeatureSelection | None = None,
        batch_size: int = 8,
        normalize: bool = False,
    ) -> torch.Tensor:
        """Conveniently encode image paths, PIL images, or image tensors.

        Args:
            images: One image, a sequence of images, or a rank-4 tensor batch.
                Floating tensors use ``tensor_value_range``.
            features: Ordered feature blocks to concatenate. The default is
                the final-layer CLS token.
            batch_size: Maximum number of images per DINOv2 forward pass.
            normalize: Whether to L2-normalize each returned feature vector.

        Returns:
            CPU float32 features shaped ``(N,D_total)``.

        Raises:
            ValueError: If no images are supplied or ``batch_size`` is invalid.
            TypeError: If an image type is unsupported.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        selected = self._validate_features(
            self.default_features if features is None else features
        )
        image_items = self._normalize_images(images)
        encoded: list[torch.Tensor] = []
        for start in range(0, len(image_items), batch_size):
            batch = [
                self._to_pil_image(image)
                for image in image_items[start : start + batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(
                device=self.device,
                dtype=self.dtype,
            )
            outputs = self.model(pixel_values=pixel_values)
            hidden = getattr(outputs, "last_hidden_state", None)
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise TypeError("DINOv2 did not return last_hidden_state shaped (B,N,D)")
            cls_token = hidden[:, :1]
            patch_tokens = hidden[:, 1:]
            available = {
                "cls": cls_token,
                "patch_mean": patch_tokens.mean(dim=1, keepdim=True),
                "patch_std": patch_tokens.std(dim=1, correction=0, keepdim=True),
                "patch": patch_tokens,
            }
            result = torch.cat(
                [available[name].flatten(start_dim=1) for name in selected],
                dim=1,
            )
            if normalize:
                result = F.normalize(result.float(), dim=-1)
            encoded.append(result.float().cpu())
        return torch.cat(encoded, dim=0)

    def _preprocess_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """Convert tensor images to normalized DINOv2 model inputs.

        Args:
            images: Tensor shaped ``(C,H,W)`` or ``(B,C,H,W)`` with three RGB
                channels. Floating values use ``tensor_value_range`` and uint8
                values use ``[0,255]``.

        Returns:
            Normalized tensor shaped ``(B,3,H_model,W_model)`` on the model
            device and in the model dtype.

        Raises:
            ValueError: If the tensor rank or channel count is unsupported.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Expected image tensor with shape (3,H,W) or (B,3,H,W), "
                f"got {tuple(images.shape)}"
            )

        images = images.to(device=self.device)
        if images.dtype == torch.uint8:
            images = images.float().div(255)
        else:
            range_min, range_max = self.tensor_value_range
            images = images.float()
            images = images.sub(range_min).div(range_max - range_min).clamp(0, 1)

        crop_size = getattr(self.processor, "crop_size", None) or {}
        height = self._size_value(crop_size, "height", fallback=224)
        width = self._size_value(crop_size, "width", fallback=height)
        resize_size = getattr(self.processor, "size", None) or {}
        shortest_edge = self._size_value(
            resize_size,
            "shortest_edge",
            fallback=min(height, width),
        )
        source_height, source_width = images.shape[-2:]
        scale = shortest_edge / min(source_height, source_width)
        resized_height = max(height, int(round(source_height * scale)))
        resized_width = max(width, int(round(source_width * scale)))
        if images.shape[-2:] != (resized_height, resized_width):
            images = F.interpolate(
                images,
                size=(resized_height, resized_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        top = (resized_height - height) // 2
        left = (resized_width - width) // 2
        images = images[:, :, top : top + height, left : left + width]
        images = images.to(dtype=self.dtype)
        mean = self.image_mean.to(device=images.device, dtype=images.dtype)
        std = self.image_std.to(device=images.device, dtype=images.dtype)
        return (images - mean) / std

    def _normalize_images(
        self,
        images: ImageInput | Sequence[ImageInput],
    ) -> list[ImageInput]:
        """Expand a supported single image or tensor batch into a list.

        Args:
            images: Single path, PIL image, rank-3 tensor, rank-4 tensor batch,
                or a sequence containing supported individual images.

        Returns:
            Non-empty list of individual image inputs.

        Raises:
            ValueError: If an image tensor rank is unsupported or input is
                empty.
            TypeError: If the top-level input type is unsupported.
        """
        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                items: list[ImageInput] = [images]
            elif images.ndim == 4:
                items = list(images)
            else:
                raise ValueError(
                    "Image tensors must have shape (C,H,W) or (N,C,H,W), "
                    f"got {tuple(images.shape)}"
                )
        elif isinstance(images, (str, Path, Image.Image)):
            items = [images]
        elif isinstance(images, Sequence):
            items = list(images)
        else:
            raise TypeError(f"Unsupported images input type: {type(images).__name__}")
        if not items:
            raise ValueError("At least one image is required")
        return items

    def _to_pil_image(self, image: ImageInput) -> Image.Image:
        """Convert one supported image input into an RGB PIL image.

        Args:
            image: Image path, PIL image, or rank-3 tensor. Floating tensors
                are converted from ``tensor_value_range`` and uint8 tensors
                are interpreted directly.

        Returns:
            Owned RGB PIL image suitable for the Hugging Face processor.

        Raises:
            ValueError: If a tensor is not rank three or has invalid channels.
            TypeError: If the image input type is unsupported.
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

    @staticmethod
    def _validate_features(features: FeatureSelection) -> tuple[str, ...]:
        """Validate and freeze an ordered DINOv2 feature selection.

        Args:
            features: Sequence containing supported feature block names.

        Returns:
            Non-empty tuple preserving the requested order.

        Raises:
            ValueError: If no features are selected or a name is unsupported.
        """
        selected = (features,) if isinstance(features, str) else tuple(features)
        if not selected:
            raise ValueError("At least one DINOv2 feature must be selected")
        invalid = [name for name in selected if name not in SUPPORTED_FEATURES]
        if invalid:
            raise ValueError(
                f"Unsupported DINOv2 features {invalid}; "
                f"choose from {SUPPORTED_FEATURES}"
            )
        return selected

    @staticmethod
    def _size_value(size: Any, key: str, *, fallback: int) -> int:
        """Read one positive image-size component from processor metadata.

        Args:
            size: Scalar or mapping-like processor size configuration.
            key: Mapping key such as ``height`` or ``width``.
            fallback: Value used when the requested component is absent.

        Returns:
            Positive integer image size.
        """
        if hasattr(size, "get"):
            value = size.get(key) or size.get("shortest_edge") or fallback
        else:
            value = size or fallback
        return int(value)
