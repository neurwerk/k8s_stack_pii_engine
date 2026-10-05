"""Thin legacy adapters around canonical shared request extraction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

from neurwerk_request_segments import ExtractedRequest, ExtractionLimitError, extract_request
from neurwerk_request_segments import TextSegment as SharedTextSegment
from neurwerk_request_segments.models import EngineResponsesRequest

from pii_engine.models.contracts import (
    DocumentAnalyzeRequest,
    LimitDetail,
    McpJsonValue,
    McpRequest,
    OpenAIResponsesRequest,
    SegmentRequest,
    SupportedRequest,
    TextSegment,
    ToolDefinition,
)
from pii_engine.services.errors import AnalysisRequestTooLargeError, InvalidAnalysisRequestError

if TYPE_CHECKING:
    from pii_engine.services.policy import PolicyResult

type PathPart = str | int
_MAX_MCP_META_NODES = 4_096
_MAX_ADMITTED_JSON_NODES = 2_621_440


@dataclass(frozen=True)
class TextLeaf:
    """Retain the legacy leaf view for local compatibility callers."""

    path: tuple[PathPart, ...]
    text: str


@dataclass(frozen=True)
class LegacyResponsesExtraction:
    """Translate only the v1 nested tool envelope around canonical extraction."""

    extracted: ExtractedRequest
    nested_tools: frozenset[int]

    @property
    def segments(self) -> list[SharedTextSegment]:
        """Expose canonical segment identities and text."""
        return self.extracted.segments

    @property
    def request_kind(self) -> Literal["responses"]:
        """Identify the Responses endpoint."""
        return "responses"

    @property
    def attachments_present(self) -> bool:
        """Expose the canonical attachment finding."""
        return self.extracted.attachments_present

    def diagnostic_path(self, segment_id: str) -> tuple[PathPart, ...]:
        """Restore the original nested tool path for legacy diagnostics."""
        path = self.extracted.diagnostic_path(segment_id)
        if len(path) > 2 and path[0] == "tools" and path[1] in self.nested_tools:
            return (*path[:2], "function", *path[2:])
        return path

    def rebuild(self, segments: Sequence[SharedTextSegment]) -> OpenAIResponsesRequest:
        """Rebuild transformed text before restoring each original tool envelope."""
        data = self.extracted.rebuild(segments).model_dump(by_alias=True, exclude_unset=True)
        for index in self.nested_tools:
            tool = data["tools"][index]
            data["tools"][index] = {
                "type": tool["type"],
                "function": {key: value for key, value in tool.items() if key != "type"},
            }
        return OpenAIResponsesRequest.model_validate(data)


type LegacyExtraction = ExtractedRequest | LegacyResponsesExtraction


def _extract(request: SupportedRequest, max_depth: int = 32) -> LegacyExtraction:
    validate_request_structure(request, max_depth)
    try:
        if isinstance(request, OpenAIResponsesRequest):
            data = request.model_dump(by_alias=True, exclude_unset=True)
            nested = frozenset(
                index
                for index, tool in enumerate(request.tools or [])
                if isinstance(tool, ToolDefinition)
            )
            for index in nested:
                tool = data["tools"][index]
                data["tools"][index] = {"type": tool["type"], **tool["function"]}
            canonical = EngineResponsesRequest.model_validate(data)
            return LegacyResponsesExtraction(
                extract_request(canonical, max_depth=max_depth), nested
            )
        return extract_request(request, max_depth=max_depth)
    except ExtractionLimitError as exc:
        raise AnalysisRequestTooLargeError(
            "request nesting exceeds the configured limit",
            limit=LimitDetail(
                component="request_segments",
                stage="inspection",
                reason=exc.reason,
                measured=exc.measured,
                maximum=exc.maximum,
                unit="levels" if exc.reason == "depth" else "items",
                exact=exc.exact,
            ),
        ) from None
    except (TypeError, ValueError) as exc:
        raise InvalidAnalysisRequestError("request text extraction failed") from exc


def legacy_segments(
    request: SupportedRequest | DocumentAnalyzeRequest,
    max_depth: int = 32,
    *,
    request_scoped: bool = False,
) -> tuple[SegmentRequest, LegacyExtraction]:
    """Translate a provider request once at the legacy boundary."""
    document = request if isinstance(request, DocumentAnalyzeRequest) else None
    model = cast(SupportedRequest, document.request if document is not None else request)
    extracted = _extract(model, max_depth)
    return SegmentRequest(
        request_kind=extracted.request_kind,
        scope="request" if request_scoped or document is not None else "session",
        segments=[TextSegment(id=item.id, text=item.text) for item in extracted.segments],
        attachments_present=extracted.attachments_present,
        text_pii_enabled=document.text_pii_enabled if document is not None else True,
        visual_findings=document.visual_findings if document is not None else None,
    ), extracted


def restore_legacy_result(result: PolicyResult, extracted: LegacyExtraction) -> None:
    """Reconstruct legacy output and diagnostic paths locally."""
    if result.segments is not None:
        result.request = cast(
            SupportedRequest,
            extracted.rebuild(
                [SharedTextSegment(id=item.id, text=item.text) for item in result.segments]
            ),
        )
    paths = {item.id: extracted.diagnostic_path(item.id) for item in extracted.segments}
    bounded_paths = {
        segment_id: tuple(
            min(part, 10_000_000) if isinstance(part, int) else part[:128] for part in path[:64]
        )
        for segment_id, path in paths.items()
    }
    result.diagnostics_truncated |= any(
        bounded_paths[item.segment_id] != paths[item.segment_id]
        for item in [*result.logical_detections, *result.effective_regions]
    )
    result.logical_detections = [
        replace(item, path=bounded_paths[item.segment_id]) for item in result.logical_detections
    ]
    result.effective_regions = [
        replace(item, path=bounded_paths[item.segment_id]) for item in result.effective_regions
    ]


def iter_text_leaves(request: SupportedRequest, max_depth: int = 32) -> list[TextLeaf]:
    """Expose canonical extraction as the legacy path-based view."""
    extracted = _extract(request, max_depth)
    return [TextLeaf(extracted.diagnostic_path(item.id), item.text) for item in extracted.segments]


def replace_text_leaves(
    request: SupportedRequest, replacements: dict[tuple[PathPart, ...], str]
) -> SupportedRequest:
    """Rebuild only canonical text locations, including encoded argument leaves."""
    extracted = _extract(request)
    paths = {item.id: extracted.diagnostic_path(item.id) for item in extracted.segments}
    if replacements.keys() - set(paths.values()):
        raise InvalidAnalysisRequestError("replacement path is not model-visible text")
    return cast(
        SupportedRequest,
        extracted.rebuild(
            [
                SharedTextSegment(id=item.id, text=replacements.get(paths[item.id], item.text))
                for item in extracted.segments
            ]
        ),
    )


def validate_request_structure(request: SupportedRequest, max_depth: int) -> None:
    """Bound legacy MCP control metadata before cache access or serialization."""
    if isinstance(request, McpRequest):
        _validate_json_structure(request.params.arguments, max_depth, max_nodes=None)
        _validate_json_structure(request.params.meta, max_depth, max_nodes=_MAX_MCP_META_NODES)


def _validate_json_structure(
    value: McpJsonValue | None,
    max_depth: int,
    *,
    max_nodes: int | None,
) -> None:
    if value is None:
        return
    nodes = 1
    containers_checked = 0
    pending: list[tuple[McpJsonValue, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > max_depth:
            _depth_limit(depth, max_depth)
        if isinstance(current, list):
            children = current
        elif isinstance(current, dict):
            children = current.values()
        else:
            continue
        containers_checked += 1
        if containers_checked > _MAX_ADMITTED_JSON_NODES:
            raise AnalysisRequestTooLargeError(
                "request structure exceeds the validation limit",
                limit=LimitDetail(
                    stage="inspection",
                    reason="nodes",
                    measured=containers_checked,
                    maximum=_MAX_ADMITTED_JSON_NODES,
                    unit="items",
                    exact=False,
                ),
            )
        child_count = len(children)
        nodes += child_count
        if child_count and depth == max_depth:
            _depth_limit(depth + 1, max_depth)
        pending.extend((item, depth + 1) for item in children if isinstance(item, (list, dict)))
    if max_nodes is not None and nodes > max_nodes:
        raise AnalysisRequestTooLargeError(
            "MCP metadata contains too many JSON nodes",
            limit=LimitDetail(
                stage="inspection",
                reason="nodes",
                measured=nodes,
                maximum=max_nodes,
                unit="items",
                exact=True,
            ),
        )


def _depth_limit(measured: int, maximum: int) -> None:
    raise AnalysisRequestTooLargeError(
        "request nesting exceeds the configured limit",
        limit=LimitDetail(
            stage="inspection",
            reason="depth",
            measured=measured,
            maximum=maximum,
            unit="levels",
            exact=False,
        ),
    )
