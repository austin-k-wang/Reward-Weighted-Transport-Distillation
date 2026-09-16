"""Validation and semantic signatures for structured GenEval prompts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

GENEVAL_TAGS = frozenset(
    {"single_object", "two_object", "counting", "colors", "position", "color_attr"}
)
COLORS = (
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
)
POSITIONS = frozenset({"left of", "right of", "above", "below"})


class MetadataError(ValueError):
    """Raised when a structured GenEval metadata row is invalid."""


def _validate_clause(clause: object, *, row_index: int, field: str) -> dict[str, Any]:
    """Validate one GenEval include/exclude clause.

    Parameters:
        clause: Candidate mapping with ``class``, ``count``, and optional
            ``color``/``position`` keys.
        row_index: Clause index used in actionable validation errors.
        field: Parent field name, either ``include`` or ``exclude``.

    Returns:
        A normalized copy of the clause.
    """

    if not isinstance(clause, Mapping):
        raise MetadataError(f"{field}[{row_index}] must be an object")
    allowed = {"class", "count", "color", "position"}
    unknown = set(clause) - allowed
    if unknown:
        raise MetadataError(f"{field}[{row_index}] has unknown keys: {sorted(unknown)}")
    class_name = clause.get("class")
    count = clause.get("count")
    if not isinstance(class_name, str) or not class_name.strip():
        raise MetadataError(f"{field}[{row_index}].class must be a non-empty string")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise MetadataError(f"{field}[{row_index}].count must be a positive integer")
    normalized: dict[str, Any] = {"class": class_name.strip(), "count": count}
    color = clause.get("color")
    if color is not None:
        if color not in COLORS:
            raise MetadataError(
                f"{field}[{row_index}].color must be one of {list(COLORS)}"
            )
        normalized["color"] = color
    position = clause.get("position")
    if position is not None:
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or position[0] not in POSITIONS
            or not isinstance(position[1], int)
            or isinstance(position[1], bool)
        ):
            raise MetadataError(
                f"{field}[{row_index}].position must be [relation, include_index]"
            )
        normalized["position"] = [position[0], position[1]]
    return normalized


def validate_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one GenEval prompt metadata row.

    Parameters:
        row: Mapping containing ``prompt``, ``tag``, ``include``, and optional
            ``exclude`` clauses.

    Returns:
        A JSON-compatible normalized metadata dictionary. Clause ordering is
        preserved because position references use include-list indices.
    """

    if not isinstance(row, Mapping):
        raise MetadataError("metadata row must be an object")
    prompt = row.get("prompt")
    tag = row.get("tag")
    include = row.get("include")
    exclude = row.get("exclude", [])
    if not isinstance(prompt, str) or not prompt.strip():
        raise MetadataError("prompt must be a non-empty string")
    if tag not in GENEVAL_TAGS:
        raise MetadataError(f"tag must be one of {sorted(GENEVAL_TAGS)}")
    if not isinstance(include, list) or not include:
        raise MetadataError("include must be a non-empty list")
    if not isinstance(exclude, list):
        raise MetadataError("exclude must be a list")
    normalized_include = [
        _validate_clause(value, row_index=index, field="include")
        for index, value in enumerate(include)
    ]
    normalized_exclude = [
        _validate_clause(value, row_index=index, field="exclude")
        for index, value in enumerate(exclude)
    ]
    for clause_index, clause in enumerate(normalized_include):
        position = clause.get("position")
        if position is not None:
            target = position[1]
            if target < 0 or target >= len(normalized_include) or target == clause_index:
                raise MetadataError(
                    f"include[{clause_index}].position target {target} is invalid"
                )
    expected_counts = {
        "single_object": 1,
        "two_object": 2,
        "counting": 1,
        "colors": 1,
        "position": 2,
        "color_attr": 2,
    }
    if len(normalized_include) != expected_counts[tag]:
        raise MetadataError(
            f"{tag} rows require {expected_counts[tag]} include clauses"
        )
    if tag == "position":
        positioned = [clause for clause in normalized_include if "position" in clause]
        if len(positioned) != 1:
            raise MetadataError("position rows require exactly one position clause")
    if tag in {"colors", "color_attr"} and any(
        "color" not in clause for clause in normalized_include
    ):
        raise MetadataError(f"{tag} rows require color on every include clause")
    normalized: dict[str, Any] = {
        "prompt": prompt.strip(),
        "tag": tag,
        "include": normalized_include,
    }
    if normalized_exclude:
        normalized["exclude"] = normalized_exclude
    return normalized


def canonical_signature(row: Mapping[str, Any]) -> str:
    """Build a semantic signature used to reject evaluation-set leakage.

    Parameters:
        row: Valid or unvalidated GenEval metadata row.

    Returns:
        Stable JSON signature. Two-object and color-attribution bindings are
        symmetric, while inverse-equivalent position descriptions (for
        example, ``A right of B`` and ``B left of A``) share a signature.
    """

    metadata = validate_metadata(row)
    tag = metadata["tag"]
    clauses = metadata["include"]
    if tag == "position":
        source_index = next(
            index for index, clause in enumerate(clauses) if "position" in clause
        )
        source = clauses[source_index]
        relation, target_index = source["position"]
        subject = source["class"]
        target = clauses[target_index]["class"]
        if relation == "right of":
            subject, target, relation = target, subject, "left of"
        elif relation == "below":
            subject, target, relation = target, subject, "above"
        value: object = [tag, relation, subject, target]
    elif tag == "two_object":
        value = [tag, sorted(clause["class"] for clause in clauses)]
    elif tag == "color_attr":
        value = [
            tag,
            sorted((clause["class"], clause["color"]) for clause in clauses),
        ]
    else:
        include_key = sorted(
            (
                clause["class"],
                clause["count"],
                clause.get("color"),
            )
            for clause in clauses
        )
        exclude_key = sorted(
            (clause["class"], clause["count"], clause.get("color"))
            for clause in metadata.get("exclude", [])
        )
        value = [tag, include_key, exclude_key]
    return json.dumps(value, separators=(",", ":"), sort_keys=False)


def load_metadata_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate GenEval metadata from a JSONL file.

    Parameters:
        path: Input JSONL path containing one metadata object per line.

    Returns:
        List of normalized rows in file order.
    """

    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                rows.append(validate_metadata(value))
            except (json.JSONDecodeError, MetadataError) as exc:
                raise MetadataError(f"{path}:{line_number}: {exc}") from exc
    if not rows:
        raise MetadataError(f"{path}: no metadata rows found")
    return rows


def select_metadata_subset(
    rows: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Select a reproducible random subset while preserving source order.

    Parameters:
        rows: Validated GenEval rows in official file order.
        count: Number of distinct rows to select.
        seed: Non-negative seed controlling the sampled row indices.

    Returns:
        ``(source_index, row)`` pairs sorted by source index. The source index
        identifies each row in the complete official evaluation set.

    Raises:
        ValueError: If ``count`` is outside ``[1,len(rows)]`` or ``seed`` is
            negative.
    """
    if count < 1 or count > len(rows):
        raise ValueError(
            f"count must be between 1 and {len(rows)}, got {count}"
        )
    if seed < 0:
        raise ValueError("seed must be non-negative")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [(index, rows[index]) for index in indices]


def signatures(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect canonical signatures from metadata rows.

    Parameters:
        rows: Iterable of GenEval metadata mappings.

    Returns:
        Set containing one canonical signature for every distinct row.
    """

    return {canonical_signature(row) for row in rows}
