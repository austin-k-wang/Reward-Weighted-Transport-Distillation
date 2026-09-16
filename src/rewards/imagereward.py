"""ImageReward evaluator with self-contained Transformers compatibility shims."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from .base import Metadata, as_reward_tensor, images_to_pil, iter_batch_slices, validate_reward_inputs


def _compat_apply_chunking(
    forward_fn: Callable[..., torch.Tensor],
    chunk_size: int,
    chunk_dim: int,
    *input_tensors: torch.Tensor,
) -> torch.Tensor:
    """Apply a forward function in equal tensor chunks for old ImageReward code.

    Args:
        forward_fn: Callable accepting one corresponding chunk from each tensor.
        chunk_size: Number of elements per chunk, or zero to disable chunking.
        chunk_dim: Tensor dimension split into chunks.
        *input_tensors: Equally shaped input tensors consumed by ``forward_fn``.

    Returns:
        Concatenated forward outputs along ``chunk_dim``, with the same leading
        structure as an unchunked call.

    Raises:
        ValueError: If tensor shapes differ or the dimension is not divisible by
            the requested chunk size.
    """
    if not input_tensors:
        raise ValueError("at least one input tensor is required")
    if any(tensor.shape != input_tensors[0].shape for tensor in input_tensors[1:]):
        raise ValueError("all input tensors must have the same shape")
    if chunk_size <= 0:
        return forward_fn(*input_tensors)
    dimension = input_tensors[0].shape[chunk_dim]
    if dimension % chunk_size:
        raise ValueError("chunk dimension must be divisible by chunk_size")
    chunks = [tensor.split(chunk_size, dim=chunk_dim) for tensor in input_tensors]
    outputs = [forward_fn(*parts) for parts in zip(*chunks)]
    return torch.cat(outputs, dim=chunk_dim)


def _compat_find_pruneable_heads(
    heads: Sequence[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.Tensor]:
    """Find attention heads and flattened parameter indices retained by pruning.

    Args:
        heads: Requested original attention-head indices to remove.
        n_heads: Current number of attention heads.
        head_size: Number of channels in each attention head.
        already_pruned_heads: Original head indices removed by prior operations.

    Returns:
        A pair containing newly pruned original head indices and a one-dimensional
        long tensor selecting all channels that remain.
    """
    mask = torch.ones(n_heads, head_size)
    selected = set(heads) - already_pruned_heads
    for head in selected:
        adjusted = head - sum(previous < head for previous in already_pruned_heads)
        mask[adjusted] = 0
    indices = mask.reshape(-1).contiguous().eq(1).nonzero().reshape(-1)
    return selected, indices


def _compat_prune_linear_layer(
    layer: torch.nn.Linear,
    index: torch.Tensor,
    dim: int = 0,
) -> torch.nn.Linear:
    """Clone selected rows or columns from a linear layer for BERT pruning.

    Args:
        layer: Source linear transformation.
        index: One-dimensional indices retained along ``dim``.
        dim: Weight dimension to prune, either output dimension zero or input
            dimension one.

    Returns:
        A new linear layer on the source device with copied trainable weights
        and the corresponding bias.
    """
    selected = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, selected).detach().clone()
    bias = None
    if layer.bias is not None:
        bias = (
            layer.bias.detach().clone()
            if dim == 1
            else layer.bias.index_select(0, selected).detach().clone()
        )
    size = list(layer.weight.shape)
    size[dim] = selected.numel()
    result = torch.nn.Linear(size[1], size[0], bias=bias is not None).to(
        device=layer.weight.device, dtype=layer.weight.dtype
    )
    with torch.no_grad():
        result.weight.copy_(weight)
        if bias is not None:
            result.bias.copy_(bias)
    return result


def _install_imagereward_compatibility(
    local_files_only: bool,
    tokenizer_name_or_path: str | Path = "bert-base-uncased",
) -> None:
    """Install APIs expected by ImageReward when using recent Transformers.

    Args:
        local_files_only: Whether the patched BERT tokenizer loader must avoid
            Hugging Face network access.
        tokenizer_name_or_path: Hugging Face BERT tokenizer ID or complete
            local snapshot directory used by ImageReward's BLIP encoder.

    Returns:
        Nothing. Missing legacy symbols and tokenizer behavior are patched in
        loaded dependency modules process-wide.
    """
    import transformers.modeling_utils as modeling_utils

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = _compat_apply_chunking
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = _compat_find_pruneable_heads
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = _compat_prune_linear_layer

    import ImageReward.models.BLIP.blip_pretrain as blip_pretrain

    def init_tokenizer() -> Any:
        """Build the BLIP BERT tokenizer required by ImageReward.

        Returns:
            A BERT tokenizer with ImageReward's decoder and encoder tokens.
        """
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            str(tokenizer_name_or_path), local_files_only=local_files_only
        )
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    blip_pretrain.init_tokenizer = init_tokenizer

    from transformers import PreTrainedModel

    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = property(
            lambda self: getattr(self, "_tied_weights_keys", [])
        )


class ImageRewardEvaluator:
    """Evaluate aligned prompt-image pairs with ImageReward v1."""

    def __init__(
        self,
        model_name_or_path: str | Path = "ImageReward-v1.0",
        *,
        checkpoint_root: str | Path | None = None,
        tokenizer_name_or_path: str | Path = "bert-base-uncased",
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        local_files_only: bool = False,
    ) -> None:
        """Load ImageReward with compatibility for current Transformers releases.

        Args:
            model_name_or_path: ImageReward model name or local checkpoint path.
            checkpoint_root: Optional directory used by ImageReward for checkpoints.
            tokenizer_name_or_path: Hugging Face BERT tokenizer ID or local
                snapshot containing ``vocab.txt``.
            device: Inference device, defaulting to CUDA when available.
            dtype: Floating model parameter dtype.
            local_files_only: Whether tokenizer loading must avoid network access.

        Returns:
            Nothing. The instance owns a frozen ImageReward model.
        """
        _install_imagereward_compatibility(
            local_files_only,
            tokenizer_name_or_path=tokenizer_name_or_path,
        )
        import ImageReward as image_reward

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        load_kwargs: dict[str, object] = {"device": str(self.device)}
        if checkpoint_root is not None:
            load_kwargs["download_root"] = str(checkpoint_root)
        self.model = image_reward.load(str(model_name_or_path), **load_kwargs)
        self.model.eval().to(dtype=dtype)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def score(
        self,
        prompts: Sequence[str],
        images: torch.Tensor,
        *,
        batch_size: int,
        metadata: Metadata = None,
    ) -> torch.Tensor:
        """Return one ImageReward score for each aligned prompt-image pair.

        Args:
            prompts: Prompt strings aligned with the images.
            images: Floating image tensor shaped ``[N,3,H,W]`` in ``[-1,1]``.
            batch_size: Maximum number of pairs per ImageReward call.
            metadata: Optional aligned metadata, accepted but ignored.

        Returns:
            Detached CPU float32 ImageReward scores shaped ``[N]``.
        """
        count = validate_reward_inputs(prompts, images, batch_size, metadata)
        pil_images = images_to_pil(images)
        chunks: list[torch.Tensor] = []
        for indices in iter_batch_slices(count, batch_size):
            chunk_prompts = list(prompts[indices])
            _, values = self.model.inference_rank(chunk_prompts, pil_images[indices])
            tensor = torch.as_tensor(values, dtype=torch.float32)
            chunk_count = len(chunk_prompts)
            if tensor.numel() == chunk_count * chunk_count:
                tensor = tensor.reshape(chunk_count, chunk_count).diagonal()
            chunks.append(as_reward_tensor(tensor, chunk_count))
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float32)
