"""Validate exact offline KServe tokenization and translate probabilities to spans."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pii_engine.lib.remote_models import KSERVE_MODEL_PINS, KSERVE_TOKENIZER_SHA256
from pii_engine.services.analyzer import EntityMatch
from pii_engine.services.remote_http import RemoteAnalysisError

if TYPE_CHECKING:
    from pii_engine.config.remote import RemoteModel


def load_tokenizer(model: RemoteModel) -> tuple[Any, dict[str, str]]:
    """Require a checksum-pinned tokenizer-only directory from the selected revision."""
    if len(model.languages) != 1 or KSERVE_MODEL_PINS.get(model.languages[0]) != (
        model.upstream,
        model.revision,
    ):
        raise ValueError("KServe model does not match a supported immutable model pin")
    root = model.tokenizer_path
    if root is None or not {"config.json", "tokenizer_config.json"}.issubset(
        model.tokenizer_sha256
    ):
        raise ValueError("KServe tokenizer configuration is incomplete")
    expected = KSERVE_TOKENIZER_SHA256[model.languages[0]]
    if not expected.items() <= model.tokenizer_sha256.items():
        raise ValueError("KServe tokenizer digests do not match the supported immutable revision")
    _verify_files(root, model.tokenizer_sha256)
    config = json.loads((root / "config.json").read_bytes())
    labels = config.get("id2label")
    if (
        not isinstance(labels, dict)
        or not labels
        or set(labels) != {str(i) for i in range(len(labels))}
    ):
        raise ValueError("KServe class labels are incomplete")
    for label in labels.values():
        if not isinstance(label, str) or (
            label != "O" and (label[:2] not in {"B-", "I-"} or label[2:] not in model.label_mapping)
        ):
            raise ValueError("KServe class label has no explicit policy mapping")
    return _create_tokenizer(root), labels


def _create_tokenizer(root: Path) -> Any:  # noqa: ANN401
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        root,
        local_files_only=True,
        trust_remote_code=False,
        do_lower_case=False,
    )
    if not tokenizer.is_fast:
        raise ValueError("KServe alignment requires a fast offset-capable tokenizer")
    return tokenizer


def _verify_files(root: Path, digests: dict[str, str]) -> None:
    for name, digest in digests.items():
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("tokenizer files must be directly below their immutable root")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("tokenizer file is missing or is a symlink")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("tokenizer file digest does not match its deployment pin")
    if {path.name for path in root.iterdir()} != set(digests):
        raise ValueError("tokenizer directory contains unverified files")


def token_chunks(text: str, tokenizer: Any) -> Iterator[tuple[int, str]]:  # noqa: ANN401
    """Cover all characters with overlapping windows, rechecking retokenized sizes."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(start, end) for start, end in encoded["offset_mapping"] if start < end]
    pending = []
    for index in range(0, len(offsets), 384):
        start = 0 if index == 0 else offsets[index][0]
        end = len(text) if index + 448 >= len(offsets) else offsets[index + 448][0]
        pending.append((start, text[start:end]))
        if end == len(text):
            break
    if not offsets:
        pending = [(0, text)]
    pending.reverse()
    while pending:
        offset, chunk = pending.pop()
        encoded = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False,
            return_offsets_mapping=True,
        )
        if len(encoded["input_ids"]) <= 512:
            yield offset, chunk
            continue
        offsets = [end for start, end in encoded["offset_mapping"] if start < end]
        # Reserve space for special tokens and retokenization at window boundaries.
        boundary = offsets[min(448, len(offsets) - 1)]
        if boundary <= 64 or boundary >= len(chunk):
            raise RemoteAnalysisError("tokenizer cannot produce bounded complete chunks")
        pending.append((offset + boundary - 64, chunk[boundary - 64 :]))
        pending.append((offset, chunk[:boundary]))


def decode_predictions(
    value: object,
    text: str,
    offsets: list[tuple[int, int]],
    labels: dict[str, str],
    model: RemoteModel,
) -> list[EntityMatch]:
    """Validate every class and token before aggregating BIO evidence."""
    if not isinstance(value, dict) or "error" in value:
        raise RemoteAnalysisError("KServe response is not an object")
    predictions = value.get("predictions")
    if (
        not isinstance(predictions, list)
        or len(predictions) != 1
        or not isinstance(predictions[0], list)
        or len(predictions[0]) != len(offsets)
    ):
        raise RemoteAnalysisError("KServe token coverage is incomplete")
    matches: list[EntityMatch] = []
    previous = "O"
    for probabilities, (start, end) in zip(predictions[0], offsets, strict=True):
        class_id, score = _winning_class(probabilities, labels)
        label = labels[class_id]
        if not 0 <= start <= end <= len(text):
            raise RemoteAnalysisError("tokenizer offsets are invalid")
        if start == end:
            previous = "O"
            continue
        if label == "O":
            previous = "O"
            continue
        entity = model.label_mapping[label[2:]]
        if (
            label.startswith("I-")
            and previous[2:] == label[2:]
            and matches
            and matches[-1].entity_type == entity
            and matches[-1].end <= start
        ):
            last = matches.pop()
            matches.append(EntityMatch(entity, last.start, end, max(last.score, score), model.name))
        else:
            matches.append(EntityMatch(entity, start, end, score, model.name))
        previous = label
    return matches


def _winning_class(probabilities: object, labels: dict[str, str]) -> tuple[str, float]:
    if not isinstance(probabilities, dict) or probabilities.keys() != labels.keys():
        raise RemoteAnalysisError("KServe class coverage is invalid")
    values = list(probabilities.values())
    if any(
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
        for score in values
    ):
        raise RemoteAnalysisError("KServe class probabilities are invalid")
    if abs(sum(values) - 1) > max(0.01, len(labels) * 0.0001):
        raise RemoteAnalysisError("KServe class probabilities are not normalized")
    winner = max(labels, key=lambda key: probabilities[key])
    return winner, float(probabilities[winner])
