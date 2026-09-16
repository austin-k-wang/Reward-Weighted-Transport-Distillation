#!/usr/bin/env python
"""Persistent rank-local GenEval detector and color reward service."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.geneval.metadata import COLORS, validate_metadata
from src.geneval.protocol import ProtocolError, receive_frame, send_frame
from src.geneval.scoring import (
    ImageScore,
    aggregate_diagnostics,
    score_detections,
)

LOGGER = logging.getLogger("geneval_reward_server")


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute inclusive-coordinate intersection over union.

    Parameters:
        box_a: First bounding box with shape ``[>=4]``.
        box_b: Second bounding box with shape ``[>=4]``.

    Returns:
        Intersection-over-union scalar in ``[0,1]``.
    """

    def area(box: Sequence[float]) -> float:
        """Compute inclusive pixel area for one ``[x1,y1,x2,y2]`` box."""

        return max(float(box[2] - box[0] + 1), 0.0) * max(
            float(box[3] - box[1] + 1), 0.0
        )

    intersection = area(
        [
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ]
    )
    union = area(box_a) + area(box_b) - intersection
    return intersection / union if union else 0.0


class GenEvalBackend:
    """Persistent Mask2Former and OpenCLIP inference backend."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load detector, color model, transforms, and class names once.

        Parameters:
            args: Parsed server arguments containing model paths, thresholds,
                device, and crop behavior.

        Returns:
            Ready backend retaining all GPU models for subsequent requests.
        """

        import mmdet
        import open_clip
        import torch
        from clip_benchmark.metrics import zeroshot_classification as zsc
        from mmdet.apis import inference_detector, init_detector

        self.torch = torch
        self.zsc = zsc
        self.inference_detector = inference_detector
        mmdet_file = mmdet.__file__
        if mmdet_file is None:
            raise RuntimeError("Could not resolve the installed mmdet package path.")
        config = args.model_config or str(
            Path(mmdet_file).resolve().parent.parent
            / "configs"
            / "mask2former"
            / "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
        )
        checkpoint = (
            Path(args.model_path)
            / f"{args.detector_model}.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"GenEval detector checkpoint not found: {checkpoint}")
        self.detector = init_detector(config, str(checkpoint), device=args.device)
        (
            self.clip_model,
            _,
            self.transform,
        ) = open_clip.create_model_and_transforms(
            args.clip_model, pretrained="openai", device=args.device
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(args.clip_model)
        with Path(args.object_names).open(encoding="utf-8") as handle:
            self.class_names = [line.strip() for line in handle if line.strip()]
        self.device = args.device
        self.threshold = args.threshold
        self.counting_threshold = args.counting_threshold
        self.max_objects = args.max_objects
        self.max_overlap = args.max_overlap
        self.position_threshold = args.position_threshold
        self.background_color = args.background_color
        self.crop_objects = args.crop_objects
        self.color_batch_size = args.color_batch_size
        self.color_classifiers: dict[str, Any] = {}

    def _classifier(self, class_name: str) -> Any:
        """Build or retrieve the official zero-shot color classifier.

        Parameters:
            class_name: Detector object class used in CLIP text templates.

        Returns:
            Torch classifier matrix shaped ``[embedding_dim,10]``.
        """

        classifier = self.color_classifiers.get(class_name)
        if classifier is None:
            original_tqdm = self.zsc.tqdm
            self.zsc.tqdm = lambda iterator, *args, **kwargs: iterator
            try:
                classifier = self.zsc.zero_shot_classifier(
                    self.clip_model,
                    self.tokenizer,
                    list(COLORS),
                    [
                        f"a photo of a {{c}} {class_name}",
                        f"a photo of a {{c}}-colored {class_name}",
                        "a photo of a {c} object",
                    ],
                    self.device,
                )
            finally:
                self.zsc.tqdm = original_tqdm
            self.color_classifiers[class_name] = classifier
        return classifier

    def _extract_objects(
        self,
        result: Any,
    ) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[np.ndarray, Any]]]]:
        """Convert one MMDetection result into sorted class records.

        Parameters:
            result: MMDetection output for one image, containing bbox arrays and
                optional segmentation masks.

        Returns:
            Pair of class-to-box arrays ``[N,5]`` and aligned box/mask records.
        """

        bboxes = result[0] if isinstance(result, tuple) else result
        masks = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        detections: dict[str, np.ndarray] = {}
        objects: dict[str, list[tuple[np.ndarray, Any]]] = {}
        for class_index, class_name in enumerate(self.class_names):
            class_boxes = np.asarray(bboxes[class_index], dtype=np.float32)
            if class_boxes.size == 0:
                continue
            ordering = np.argsort(class_boxes[:, 4])[::-1].tolist()
            kept: list[int] = []
            while ordering and len(kept) < self.max_objects:
                best = ordering.pop(0)
                kept.append(best)
                if self.max_overlap < 1.0:
                    ordering = [
                        candidate
                        for candidate in ordering
                        if _compute_iou(
                            class_boxes[best], class_boxes[candidate]
                        )
                        < self.max_overlap
                    ]
            detections[class_name] = class_boxes[kept, :5]
            objects[class_name] = [
                (
                    class_boxes[index, :5],
                    None if masks is None else masks[class_index][index],
                )
                for index in kept
            ]
        return detections, objects

    def _crop_object(
        self, image: np.ndarray, box: np.ndarray, mask: Any
    ) -> Any:
        """Transform one detected object crop for OpenCLIP.

        Parameters:
            image: RGB uint8 image with shape ``[H,W,3]``.
            box: Detection box with shape ``[5]``.
            mask: Optional segmentation mask with shape ``[H,W]``.

        Returns:
            OpenCLIP image tensor with shape ``[3,input_h,input_w]``.
        """

        source = Image.fromarray(image, mode="RGB")
        if self.background_color == "original":
            blank = source.copy()
        else:
            blank = Image.new("RGB", source.size, color=self.background_color)
        if mask is not None:
            mask_array = np.asarray(mask)
            if mask_array.shape != image.shape[:2]:
                raise ValueError(
                    f"segmentation mask shape {mask_array.shape} does not match "
                    f"image shape {image.shape[:2]}"
                )
            mask_image = Image.fromarray(
                (mask_array.astype(np.uint8) * 255), mode="L"
            )
            source = Image.composite(source, blank, mask_image)
        if self.crop_objects:
            source = source.crop(tuple(float(value) for value in box[:4]))
        return self.transform(source)

    def _color_probabilities(
        self,
        images: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        objects: Sequence[Mapping[str, list[tuple[np.ndarray, Any]]]],
    ) -> list[dict[int, np.ndarray]]:
        """Batch all requested object crops through OpenCLIP.

        Parameters:
            images: RGB uint8 batch with shape ``[B,H,W,3]``.
            rows: Validated metadata rows with shape ``[B]``.
            objects: Aligned detector object records for each image.

        Returns:
            Per-image mappings from include index to color probabilities with
            shape ``[requested_count,10]``.
        """

        tasks: list[tuple[int, int, int, str, Any]] = []
        output: list[dict[int, np.ndarray]] = [dict() for _ in rows]
        for image_index, row in enumerate(rows):
            threshold = (
                self.counting_threshold
                if row["tag"] == "counting"
                else self.threshold
            )
            for include_index, clause in enumerate(row["include"]):
                if "color" not in clause:
                    continue
                candidates = [
                    item
                    for item in objects[image_index].get(clause["class"], [])
                    if float(item[0][4]) > threshold
                ][: clause["count"]]
                output[image_index][include_index] = np.zeros(
                    (len(candidates), len(COLORS)), dtype=np.float32
                )
                for object_index, (box, mask) in enumerate(candidates):
                    transformed = self._crop_object(images[image_index], box, mask)
                    tasks.append(
                        (
                            image_index,
                            include_index,
                            object_index,
                            clause["class"],
                            transformed,
                        )
                    )
        for start in range(0, len(tasks), self.color_batch_size):
            batch_tasks = tasks[start : start + self.color_batch_size]
            tensors = self.torch.stack([task[4] for task in batch_tasks]).to(
                self.device
            )
            with self.torch.no_grad(), self.torch.cuda.amp.autocast():
                features = self.torch.nn.functional.normalize(
                    self.clip_model.encode_image(tensors), dim=-1
                )
                for task_index, (
                    image_index,
                    include_index,
                    object_index,
                    class_name,
                    _,
                ) in enumerate(batch_tasks):
                    logits = (
                        100.0
                        * features[task_index].float()
                        @ self._classifier(class_name).float()
                    )
                    probabilities = self.torch.softmax(logits, dim=-1)
                    output[image_index][include_index][object_index] = (
                        probabilities.detach().cpu().numpy()
                    )
        return output

    def warmup(self) -> None:
        """Warm detector and OpenCLIP kernels before accepting requests.

        Parameters:
            None.

        Returns:
            ``None`` after one synthetic detector and color forward pass.
        """

        image = np.full((256, 256, 3), 127, dtype=np.uint8)
        self.inference_detector(self.detector, image[..., ::-1].copy())
        tensor = self.transform(Image.fromarray(image)).unsqueeze(0).to(self.device)
        classifier = self._classifier("car")
        with self.torch.no_grad(), self.torch.cuda.amp.autocast():
            features = self.torch.nn.functional.normalize(
                self.clip_model.encode_image(tensor), dim=-1
            )
            _ = features.float() @ classifier.float()
        self.torch.cuda.synchronize()

    def score(
        self,
        images: np.ndarray,
        rows: Sequence[Mapping[str, Any]],
        *,
        reward_mode: str,
        binary_bonus: float,
    ) -> list[ImageScore]:
        """Run batched detection, batched color classification, and scoring.

        Parameters:
            images: RGB uint8 images with shape ``[B,H,W,3]``.
            rows: One structured GenEval metadata row per image.
            reward_mode: ``binary``, ``dense``, or ``hybrid``.
            binary_bonus: Official-correctness bonus for hybrid rewards.

        Returns:
            Per-image score records with shape ``[B]``.
        """

        detector_inputs = [image[..., ::-1].copy() for image in images]
        raw_results = self.inference_detector(self.detector, detector_inputs)
        extracted = [self._extract_objects(result) for result in raw_results]
        detections = [item[0] for item in extracted]
        objects = [item[1] for item in extracted]
        colors = self._color_probabilities(images, rows, objects)
        height, width = images.shape[1:3]
        scores = [
            score_detections(
                row,
                detections[index],
                image_size=(height, width),
                color_probabilities=colors[index],
                reward_mode=reward_mode,
                binary_bonus=binary_bonus,
                detection_threshold=(
                    self.counting_threshold
                    if row["tag"] == "counting"
                    else self.threshold
                ),
                position_threshold=self.position_threshold,
            )
            for index, row in enumerate(rows)
        ]
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        return scores


def _score_request(
    backend: GenEvalBackend,
    header: Mapping[str, Any],
    payload: bytes,
) -> tuple[list[float], dict[str, float | int]]:
    """Validate and execute one protocol score request.

    Parameters:
        backend: Loaded persistent inference backend.
        header: Decoded request header.
        payload: Raw contiguous uint8 RGB image bytes.

    Returns:
        Pair of rewards shaped ``[B]`` and request diagnostics.
    """

    shape = header.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        )
        or shape[-1] != 3
    ):
        raise ValueError("shape must be [B,H,W,3] with positive integer dimensions")
    if header.get("dtype") != "uint8":
        raise ValueError("only uint8 image payloads are supported")
    expected_bytes = int(np.prod(shape, dtype=np.int64))
    if len(payload) != expected_bytes:
        raise ValueError(
            f"payload has {len(payload)} bytes; shape requires {expected_bytes}"
        )
    metadata = header.get("metadata")
    if not isinstance(metadata, list) or len(metadata) != shape[0]:
        raise ValueError(f"metadata must contain exactly {shape[0]} rows")
    rows = [validate_metadata(row) for row in metadata]
    images = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
    scores = backend.score(
        images,
        rows,
        reward_mode=str(header.get("reward_mode", "hybrid")),
        binary_bonus=float(header.get("binary_bonus", 0.25)),
    )
    return [score.reward for score in scores], aggregate_diagnostics(scores)


def serve(args: argparse.Namespace) -> None:
    """Load models and serve persistent framed requests until shutdown.

    Parameters:
        args: Parsed server configuration.

    Returns:
        ``None`` after graceful signal or explicit shutdown.
    """

    backend = GenEvalBackend(args)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    if not args.no_warmup:
        LOGGER.info("warming detector and color classifier")
        backend.warmup()
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    should_stop = False

    def request_stop(signum: int, _frame: object) -> None:
        """Mark the server for shutdown after a termination signal."""

        nonlocal should_stop
        LOGGER.info("received signal %s; stopping", signum)
        should_stop = True
        listener.close()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    request_count = 0
    try:
        listener.bind(str(socket_path))
        os.chmod(socket_path, 0o600)
        listener.listen(1)
        listener.settimeout(1.0)
        LOGGER.info("ready socket=%s pid=%d", socket_path, os.getpid())
        while not should_stop:
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if should_stop:
                    break
                raise
            with connection:
                connection.settimeout(args.timeout)
                while not should_stop:
                    request_id: object = None
                    try:
                        frame = receive_frame(
                            connection,
                            max_header_bytes=args.max_header_bytes,
                            max_payload_bytes=args.max_payload_bytes,
                        )
                        request_id = frame.header.get("request_id")
                        operation = frame.header.get("op")
                        if operation == "health":
                            send_frame(
                                connection,
                                {
                                    "request_id": request_id,
                                    "ok": True,
                                    "ready": True,
                                    "pid": os.getpid(),
                                    "request_count": request_count,
                                },
                            )
                        elif operation == "warmup":
                            started = time.perf_counter()
                            backend.warmup()
                            send_frame(
                                connection,
                                {
                                    "request_id": request_id,
                                    "ok": True,
                                    "warmup_latency_ms": (
                                        time.perf_counter() - started
                                    )
                                    * 1000.0,
                                },
                            )
                        elif operation == "shutdown":
                            send_frame(
                                connection,
                                {"request_id": request_id, "ok": True},
                            )
                            should_stop = True
                        elif operation == "score":
                            started = time.perf_counter()
                            rewards, diagnostics = _score_request(
                                backend, frame.header, frame.payload
                            )
                            request_count += 1
                            latency_ms = (time.perf_counter() - started) * 1000.0
                            diagnostics.update(
                                {
                                    "latency_ms": latency_ms,
                                    "server_request_count": request_count,
                                }
                            )
                            send_frame(
                                connection,
                                {
                                    "request_id": request_id,
                                    "ok": True,
                                    "rewards": rewards,
                                    "diagnostics": diagnostics,
                                },
                            )
                            LOGGER.info(
                                "request=%d batch=%d latency_ms=%.1f "
                                "reward_mean=%.4f reward_std=%.4f "
                                "dense_mean=%.4f correct_mean=%.4f",
                                request_count,
                                len(rewards),
                                latency_ms,
                                diagnostics.get("reward_mean", 0.0),
                                diagnostics.get("reward_std", 0.0),
                                diagnostics.get("dense_mean", 0.0),
                                diagnostics.get("official_correct_mean", 0.0),
                            )
                        else:
                            raise ValueError(f"unsupported operation: {operation!r}")
                    except (ProtocolError, OSError):
                        break
                    except Exception as exc:
                        LOGGER.exception("request failed")
                        try:
                            send_frame(
                                connection,
                                {
                                    "request_id": request_id,
                                    "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                            )
                        except OSError:
                            break
    finally:
        listener.close()
        if socket_path.exists():
            socket_path.unlink()
        LOGGER.info("server stopped after %d score requests", request_count)


def build_parser() -> argparse.ArgumentParser:
    """Build the persistent GenEval server argument parser.

    Parameters:
        None.

    Returns:
        Configured parser for model, socket, and threshold options.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-config")
    parser.add_argument("--object-names", required=True)
    parser.add_argument(
        "--detector-model",
        default="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco",
    )
    parser.add_argument("--clip-model", default="ViT-L-14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--counting-threshold", type=float, default=0.9)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--max-overlap", type=float, default=1.0)
    parser.add_argument("--position-threshold", type=float, default=0.1)
    parser.add_argument("--background-color", default="#999")
    parser.add_argument(
        "--no-crop-objects", action="store_false", dest="crop_objects"
    )
    parser.set_defaults(crop_objects=True)
    parser.add_argument("--color-batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-header-bytes", type=int, default=1 << 20)
    parser.add_argument("--max-payload-bytes", type=int, default=1 << 30)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """Parse arguments, configure logs, and run the scorer service.

    Parameters:
        None.

    Returns:
        ``None`` after the service exits.
    """

    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    serve(args)


if __name__ == "__main__":
    main()
