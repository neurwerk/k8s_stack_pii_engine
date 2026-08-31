"""Exhaustively traverse only schema-approved model-visible text leaves."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from pii_engine.models.contracts import (
    SUPPORTED_REQUEST_ADAPTER,
    JsonValue,
    McpJsonValue,
    McpRequest,
    OpenAIChatRequest,
    OpenAIResponsesRequest,
    SupportedRequest,
)
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError

type PathPart = str | int
_JSON_STRING_BOUNDARY = "__PII_ENGINE_JSON_STRING__"
_MAX_MCP_META_NODES = 4_096
# A 5 MiB JSON body cannot encode more nodes than one-byte values plus delimiters.
_MAX_ADMITTED_JSON_NODES = 2_621_440


@dataclass(frozen=True)
class TextLeaf:
    """Reference one independently analyzed string in a validated request."""

    path: tuple[PathPart, ...]
    text: str
    _replacement_path: tuple[PathPart, ...] | None = None


def _json_text_leaves(
    value: JsonValue, path: tuple[PathPart, ...], depth: int, max_depth: int
) -> list[TextLeaf]:
    """Return every string from a schema-designated JSON payload."""
    if depth > max_depth:
        raise InvalidAnalysisRequestError("request nesting exceeds the configured limit")
    if isinstance(value, str):
        return [TextLeaf(path, value)]
    if isinstance(value, list):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _json_text_leaves(item, (*path, index), depth + 1, max_depth)
        ]
    if isinstance(value, dict):
        return [
            leaf
            for key, item in value.items()
            for leaf in _json_text_leaves(item, (*path, key), depth + 1, max_depth)
        ]
    return []


def iter_text_leaves(request: SupportedRequest, max_depth: int = 32) -> list[TextLeaf]:
    """Return all and only model-visible text locations for a supported request."""
    if isinstance(request, McpRequest):
        validate_request_structure(request, max_depth)
        return _json_text_leaves(request.params.arguments, ("params", "arguments"), 0, max_depth)
    data = request.model_dump(mode="python", by_alias=True, exclude_none=True)
    if isinstance(request, OpenAIChatRequest):
        return _chat_leaves(data, max_depth)
    if isinstance(request, OpenAIResponsesRequest):
        return _responses_leaves(data, max_depth)
    raise TypeError("unsupported request model")


def validate_request_structure(request: SupportedRequest, max_depth: int) -> None:
    """Bound MCP control structures before cache access, analysis, or serialization."""
    if not isinstance(request, McpRequest):
        return
    _validate_json_structure(request.params.arguments, max_depth, max_nodes=None)
    _validate_json_structure(
        request.params.meta,
        max_depth,
        max_nodes=_MAX_MCP_META_NODES,
    )


def _validate_json_structure(
    value: McpJsonValue | None,
    max_depth: int,
    *,
    max_nodes: int | None,
) -> None:
    """Enforce depth before node count without traversing scalar children."""
    if value is None:
        return
    nodes = 1
    containers_checked = 0
    pending: list[tuple[McpJsonValue, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > max_depth:
            raise InvalidAnalysisRequestError("request nesting exceeds the configured limit")
        if isinstance(current, list):
            children = current
        elif isinstance(current, dict):
            children = current.values()
        else:
            continue
        containers_checked += 1
        if containers_checked > _MAX_ADMITTED_JSON_NODES:
            raise InvalidAnalysisRequestError("request structure exceeds the validation limit")
        child_count = len(children)
        nodes += child_count
        if child_count and depth == max_depth:
            raise InvalidAnalysisRequestError("request nesting exceeds the configured limit")
        pending.extend((item, depth + 1) for item in children if isinstance(item, (list, dict)))
    if max_nodes is not None and nodes > max_nodes:
        raise AnalysisRequestTooLargeError("MCP metadata contains too many JSON nodes")


def replace_text_leaves(
    request: SupportedRequest, replacements: dict[tuple[PathPart, ...], str]
) -> SupportedRequest:
    """Apply leaf-local replacements and revalidate the complete strict request."""
    leaves_by_path = {leaf.path: leaf for leaf in iter_text_leaves(request)}
    allowed_paths = leaves_by_path.keys()
    if unsupported_paths := replacements.keys() - allowed_paths:
        raise InvalidAnalysisRequestError(
            f"replacement path is not model-visible text: {min(unsupported_paths)!r}"
        )
    data = request.model_dump(mode="python", by_alias=True, exclude_none=True)
    encoded: dict[tuple[PathPart, ...], dict[tuple[PathPart, ...], str]] = {}
    for path, replacement in replacements.items():
        storage_path = leaves_by_path[path]._replacement_path or path
        if _JSON_STRING_BOUNDARY not in storage_path:
            _set_path(data, storage_path, replacement)
            continue
        boundary = storage_path.index(_JSON_STRING_BOUNDARY)
        root_path = storage_path[:boundary]
        encoded.setdefault(root_path, {})[storage_path[boundary + 1 :]] = replacement
    for root_path, nested_replacements in encoded.items():
        raw = _get_path(data, root_path)
        if not isinstance(raw, str):
            raise TypeError("JSON-string replacement root is not a string")
        parsed = cast(JsonValue, json.loads(raw))
        for nested_path, replacement in nested_replacements.items():
            if nested_path:
                _set_path(parsed, nested_path, replacement)
            else:
                parsed = replacement
        _set_path(
            data,
            root_path,
            json.dumps(parsed, ensure_ascii=False, separators=(",", ":")),
        )
    return SUPPORTED_REQUEST_ADAPTER.validate_python(data)


def _chat_leaves(data: dict[str, object], max_depth: int) -> list[TextLeaf]:
    leaves: list[TextLeaf] = []
    for index, message in enumerate(_list_of_dicts(data.get("messages"))):
        content = message.get("content")
        if isinstance(content, str):
            leaves.append(TextLeaf(("messages", index, "content"), content))
        elif isinstance(content, list):
            for part_index, part in enumerate(_list_of_dicts(content)):
                text = part.get("text")
                if isinstance(text, str):
                    leaves.append(
                        TextLeaf(("messages", index, "content", part_index, "text"), text)
                    )
        for call_index, call in enumerate(_list_of_dicts(message.get("tool_calls"))):
            function = call.get("function")
            if isinstance(function, dict):
                leaves.extend(
                    _argument_text_leaves(
                        function.get("arguments"),
                        ("messages", index, "tool_calls", call_index, "function", "arguments"),
                        max_depth,
                    )
                )
    leaves.extend(_tool_definition_leaves(data.get("tools"), max_depth))
    response_format = data.get("response_format")
    if isinstance(response_format, dict):
        leaves.extend(
            _schema_description_leaves(response_format, ("response_format",), 0, max_depth)
        )
    return leaves


def _responses_leaves(data: dict[str, object], max_depth: int) -> list[TextLeaf]:
    leaves: list[TextLeaf] = []
    instructions = data.get("instructions")
    if isinstance(instructions, str):
        leaves.append(TextLeaf(("instructions",), instructions))
    value = data.get("input")
    if isinstance(value, str):
        leaves.append(TextLeaf(("input",), value))
    elif isinstance(value, list):
        for index, item in enumerate(_list_of_dicts(value)):
            item_type = item.get("type")
            if item_type == "message":
                for part_index, part in enumerate(_list_of_dicts(item.get("content"))):
                    text = part.get("text")
                    if isinstance(text, str):
                        leaves.append(
                            TextLeaf(("input", index, "content", part_index, "text"), text)
                        )
            elif item_type == "function_call":
                leaves.extend(
                    _argument_text_leaves(
                        item.get("arguments"), ("input", index, "arguments"), max_depth
                    )
                )
            elif item_type == "function_call_output":
                leaves.extend(
                    _json_text_leaves(item.get("output"), ("input", index, "output"), 0, max_depth)
                )
    leaves.extend(_tool_definition_leaves(data.get("tools"), max_depth))
    leaves.extend(_response_text_format_leaves(data.get("text"), max_depth))
    return leaves


def _argument_text_leaves(
    value: JsonValue | object, path: tuple[PathPart, ...], max_depth: int
) -> list[TextLeaf]:
    """Parse standard stringified function arguments before leaf traversal."""
    if not isinstance(value, str):
        return _json_text_leaves(cast(JsonValue, value), path, 0, max_depth)
    try:
        parsed = cast(JsonValue, json.loads(value))
    except json.JSONDecodeError as exc:
        raise InvalidAnalysisRequestError("function arguments are not valid JSON") from exc
    leaves = _json_text_leaves(parsed, path, 0, max_depth)
    return [
        TextLeaf(
            leaf.path,
            leaf.text,
            (*path, _JSON_STRING_BOUNDARY, *leaf.path[len(path) :]),
        )
        for leaf in leaves
    ]


def _tool_definition_leaves(value: object, max_depth: int) -> list[TextLeaf]:
    leaves: list[TextLeaf] = []
    for index, tool in enumerate(_list_of_dicts(value)):
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        description = function.get("description")
        if isinstance(description, str):
            leaves.append(TextLeaf(("tools", index, "function", "description"), description))
        parameters = function.get("parameters")
        if isinstance(parameters, dict):
            leaves.extend(
                _schema_description_leaves(
                    parameters, ("tools", index, "function", "parameters"), 0, max_depth
                )
            )
    return leaves


def _response_text_format_leaves(value: object, max_depth: int) -> list[TextLeaf]:
    """Return model-visible prose from a Responses text format configuration."""
    if not isinstance(value, dict) or not isinstance(response_format := value.get("format"), dict):
        return []
    return _schema_description_leaves(response_format, ("text", "format"), 0, max_depth)


def _schema_description_leaves(
    value: JsonValue, path: tuple[PathPart, ...], depth: int, max_depth: int
) -> list[TextLeaf]:
    """Traverse JSON Schema prose while excluding protocol identifiers and enums."""
    if depth > max_depth:
        raise InvalidAnalysisRequestError("tool schema nesting exceeds the configured limit")
    if not isinstance(value, dict):
        return []
    leaves: list[TextLeaf] = []
    for key, item in value.items():
        if key in {"description", "title", "default", "examples"}:
            leaves.extend(_json_text_leaves(item, (*path, key), depth + 1, max_depth))
        else:
            leaves.extend(_nested_schema_leaves(key, item, path, depth, max_depth))
    return leaves


def _nested_schema_leaves(
    key: str,
    item: JsonValue,
    path: tuple[PathPart, ...],
    depth: int,
    max_depth: int,
) -> list[TextLeaf]:
    """Traverse only schema-bearing containers while excluding identifiers."""
    if key in {"schema", "json_schema", "items"}:
        return _schema_description_leaves(item, (*path, key), depth + 1, max_depth)
    if key in {"properties", "$defs"} and isinstance(item, dict):
        return [
            leaf
            for child_key, child in item.items()
            for leaf in _schema_description_leaves(
                child, (*path, key, child_key), depth + 1, max_depth
            )
        ]
    if key in {"allOf", "anyOf", "oneOf"} and isinstance(item, list):
        return [
            leaf
            for index, child in enumerate(item)
            for leaf in _schema_description_leaves(child, (*path, key, index), depth + 1, max_depth)
        ]
    return []


def _set_path(root: object, path: tuple[PathPart, ...], value: str) -> None:
    current = root
    for part in path[:-1]:
        current = _path_child(current, part)
    last = path[-1]
    if isinstance(current, dict) and isinstance(last, str) and isinstance(current.get(last), str):
        current[last] = value
        return
    if isinstance(current, list) and isinstance(last, int) and isinstance(current[last], str):
        current[last] = value
        return
    raise TypeError("replacement path is not a string leaf")


def _get_path(root: object, path: tuple[PathPart, ...]) -> object:
    current = root
    for part in path:
        current = _path_child(current, part)
    return current


def _path_child(current: object, part: PathPart) -> object:
    """Return one validated mutable traversal child."""
    if isinstance(current, dict) and isinstance(part, str):
        return current[part]
    if isinstance(current, list) and isinstance(part, int):
        return current[part]
    raise TypeError("invalid replacement path")


def _list_of_dicts(value: object) -> list[dict[str, JsonValue]]:
    """Return an already validated list in a type-checker-friendly form."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]
