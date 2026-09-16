"""Pure GenEval correctness and dense-reward computations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .metadata import COLORS, validate_metadata

DEFAULT_DETECTION_THRESHOLD = 0.3


@dataclass(frozen=True)
class ImageScore:
    """Reward and diagnostics for one image.

    Parameters:
        reward: Selected binary, dense, or hybrid scalar reward.
        dense: Mean partial-credit score in ``[0,1]``.
        official_correct: Exact GenEval binary correctness.
        clause_scores: Named partial-credit component values in ``[0,1]``.

    Returns:
        Immutable image-level score record.
    """

    reward: float
    dense: float
    official_correct: bool
    clause_scores: dict[str, tuple[float, ...]]


def _boxes_for_class(
    detections: Mapping[str, np.ndarray],
    class_name: str,
) -> np.ndarray:
    """Normalize and confidence-sort detections for one object class.

    Parameters:
        detections: Mapping from class names to arrays shaped ``[N,>=5]`` with
            ``x1,y1,x2,y2,confidence`` columns.
        class_name: Requested detector class.

    Returns:
        Float32 array shaped ``[N,5]`` sorted by descending confidence.
    """

    boxes = np.asarray(
        detections.get(class_name, np.empty((0, 5), dtype=np.float32)),
        dtype=np.float32,
    )
    if boxes.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 5:
        raise ValueError(
            f"detections for {class_name!r} must have shape [N,>=5], got {boxes.shape}"
        )
    boxes = boxes[:, :5]
    return boxes[np.argsort(-boxes[:, 4])]


def _presence_credit(boxes: np.ndarray, count: int) -> float:
    """Compute soft confidence credit for a requested object count.

    Parameters:
        boxes: Confidence-sorted detections shaped ``[N,5]``.
        count: Number of required objects.

    Returns:
        Mean top-``count`` detector confidence with missing objects scored zero.
    """

    scores = np.zeros(count, dtype=np.float32)
    available = min(count, len(boxes))
    if available:
        scores[:available] = np.clip(boxes[:available, 4], 0.0, 1.0)
    return float(scores.mean())


def _absence_credit(boxes: np.ndarray, forbidden_count: int) -> float:
    """Compute soft credit for suppressing a forbidden extra object.

    Parameters:
        boxes: Confidence-sorted detections shaped ``[N,5]``.
        forbidden_count: First object rank whose presence violates the clause.

    Returns:
        One minus confidence at the forbidden rank, or one when absent.
    """

    index = forbidden_count - 1
    if index >= len(boxes):
        return 1.0
    return float(1.0 - np.clip(boxes[index, 4], 0.0, 1.0))


def _position_values(
    source_boxes: np.ndarray,
    target_boxes: np.ndarray,
    relation: str,
    image_size: tuple[int, int],
    position_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute official booleans and dense margins for a spatial relation.

    Parameters:
        source_boxes: Source object boxes shaped ``[S,5]``.
        target_boxes: Reference object boxes shaped ``[T,5]``.
        relation: One of left/right/above/below.
        image_size: Image ``(height,width)`` retained for API clarity and
            validation; official GenEval normalizes by pairwise center offset.
        position_threshold: Fraction of summed object dimensions ignored before
            assigning a directional relation.

    Returns:
        Pair ``(official, dense)`` with flattened shapes ``[S*T]``. Official
        entries are booleans; dense values lie in ``[0,1]``.
    """

    del image_size
    values: list[float] = []
    official: list[bool] = []
    for source in source_boxes:
        for target in target_boxes:
            source_center = np.asarray(
                [(source[0] + source[2]) / 2.0, (source[1] + source[3]) / 2.0]
            )
            target_center = np.asarray(
                [(target[0] + target[2]) / 2.0, (target[1] + target[3]) / 2.0]
            )
            source_dims = np.asarray(
                [abs(source[2] - source[0]), abs(source[3] - source[1])]
            )
            target_dims = np.asarray(
                [abs(target[2] - target[0]), abs(target[3] - target[1])]
            )
            offset = source_center - target_center
            norm = float(np.linalg.norm(offset))
            revised = (
                np.maximum(
                    np.abs(offset) - position_threshold * (source_dims + target_dims),
                    0.0,
                )
                * np.sign(offset)
            )
            direction = revised / norm if norm > 0 else np.zeros(2, dtype=np.float32)
            dx, dy = float(direction[0]), float(direction[1])
            metric = {
                "left of": -dx,
                "right of": dx,
                "above": -dy,
                "below": dy,
            }.get(relation)
            if metric is None:
                raise ValueError(f"unsupported position relation: {relation}")
            official.append(metric > 0.5)
            values.append(float(np.clip((metric + 1.0) / 2.0, 0.0, 1.0)))
    return np.asarray(official), np.asarray(values, dtype=np.float32)


def score_detections(
    metadata: Mapping[str, Any],
    detections: Mapping[str, np.ndarray],
    *,
    image_size: tuple[int, int],
    color_probabilities: Mapping[int, np.ndarray] | None = None,
    reward_mode: str = "hybrid",
    binary_bonus: float = 0.25,
    detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    position_threshold: float = 0.1,
) -> ImageScore:
    """Score detector and color-classifier outputs against GenEval metadata.

    Parameters:
        metadata: Structured GenEval row for this image.
        detections: Class-to-box mapping where each array has shape ``[N,5]``.
        image_size: Image ``(height,width)``.
        color_probabilities: Optional mapping from include-clause index to
            desired object color distributions shaped ``[count,10]``.
        reward_mode: ``binary``, ``dense``, or ``hybrid``.
        binary_bonus: Exact-correctness bonus added in hybrid mode.
        detection_threshold: Official object confidence threshold.
        position_threshold: Official relative-position dimension tolerance.

    Returns:
        :class:`ImageScore` with exact official parity and dense clause credit.
    """

    row = validate_metadata(metadata)
    if reward_mode not in {"binary", "dense", "hybrid"}:
        raise ValueError("reward_mode must be binary, dense, or hybrid")
    if binary_bonus < 0 or not np.isfinite(binary_bonus):
        raise ValueError("binary_bonus must be finite and non-negative")
    color_probabilities = color_probabilities or {}
    selected: dict[int, np.ndarray] = {}
    official = True
    presence_scores: list[float] = []
    absence_scores: list[float] = []
    color_scores: list[float] = []
    position_scores: list[float] = []

    for include_index, clause in enumerate(row["include"]):
        boxes = _boxes_for_class(detections, clause["class"])
        presence_scores.append(_presence_credit(boxes, clause["count"]))
        valid = boxes[boxes[:, 4] > detection_threshold][: clause["count"]]
        selected[include_index] = valid
        if len(valid) < clause["count"]:
            official = False
        color = clause.get("color")
        if color is not None:
            probabilities = np.asarray(
                color_probabilities.get(
                    include_index,
                    np.empty((0, len(COLORS)), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            desired_index = COLORS.index(color)
            desired = np.zeros(clause["count"], dtype=np.float32)
            usable = min(clause["count"], len(probabilities))
            if probabilities.ndim != 2 or (
                len(probabilities) and probabilities.shape[1] != len(COLORS)
            ):
                raise ValueError(
                    f"color probabilities for include[{include_index}] must "
                    f"have shape [N,{len(COLORS)}]"
                )
            if usable:
                desired[:usable] = np.clip(
                    probabilities[:usable, desired_index], 0.0, 1.0
                )
                official &= bool(
                    np.all(
                        np.argmax(probabilities[:usable], axis=1) == desired_index
                    )
                )
            if usable < clause["count"]:
                official = False
            color_scores.append(float(desired.mean()))

    for clause in row.get("exclude", []):
        boxes = _boxes_for_class(detections, clause["class"])
        absence_scores.append(_absence_credit(boxes, clause["count"]))
        if np.count_nonzero(boxes[:, 4] > detection_threshold) >= clause["count"]:
            official = False

    for include_index, clause in enumerate(row["include"]):
        position = clause.get("position")
        if position is None:
            continue
        relation, target_index = position
        source_boxes = selected[include_index]
        target_boxes = selected[target_index]
        if len(source_boxes) == 0 or len(target_boxes) == 0:
            official = False
            position_scores.append(0.0)
            continue
        correct_values, dense_values = _position_values(
            source_boxes,
            target_boxes,
            relation,
            image_size,
            position_threshold,
        )
        official &= bool(np.all(correct_values))
        position_scores.append(float(dense_values.mean()))

    all_scores = presence_scores + absence_scores + color_scores + position_scores
    dense = float(np.mean(all_scores)) if all_scores else 0.0
    if reward_mode == "binary":
        reward = float(official)
    elif reward_mode == "dense":
        reward = dense
    else:
        reward = dense + binary_bonus * float(official)
    return ImageScore(
        reward=float(reward),
        dense=dense,
        official_correct=bool(official),
        clause_scores={
            "presence": tuple(presence_scores),
            "absence": tuple(absence_scores),
            "color": tuple(color_scores),
            "position": tuple(position_scores),
        },
    )


def aggregate_diagnostics(scores: Sequence[ImageScore]) -> dict[str, float | int]:
    """Aggregate image scores into request-level logging diagnostics.

    Parameters:
        scores: Sequence of per-image score records.

    Returns:
        JSON-compatible means for dense/binary reward and each clause family.
    """

    if not scores:
        return {
            "count": 0,
            "official_correct_mean": 0.0,
            "dense_mean": 0.0,
        }
    diagnostics: dict[str, float | int] = {
        "count": len(scores),
        "reward_mean": float(np.mean([score.reward for score in scores])),
        "reward_std": float(np.std([score.reward for score in scores])),
        "reward_min": float(np.min([score.reward for score in scores])),
        "reward_max": float(np.max([score.reward for score in scores])),
        "official_correct_mean": float(
            np.mean([score.official_correct for score in scores])
        ),
        "dense_mean": float(np.mean([score.dense for score in scores])),
    }
    for name in ("presence", "absence", "color", "position"):
        values = [
            value
            for score in scores
            for value in score.clause_scores.get(name, ())
        ]
        if values:
            diagnostics[f"{name}_mean"] = float(np.mean(values))
    return diagnostics
