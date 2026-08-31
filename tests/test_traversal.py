"""Test exhaustive schema-aware request traversal and mutation."""

from __future__ import annotations

import json

import pytest

from pii_engine.config.settings import Settings
from pii_engine.models.contracts import (
    McpRequest,
    OpenAIChatRequest,
    OpenAIResponsesRequest,
)
from pii_engine.runtime import EngineRuntime
from pii_engine.services.traversal import iter_text_leaves, replace_text_leaves


def test_chat_traversal_includes_messages_parts_tools_and_results() -> None:
    """Every supported chat text location is independently visible."""
    request = OpenAIChatRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"query": "a@example.com"}},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "tool result"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find a person",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string", "description": "PII query"}},
                        },
                    },
                }
            ],
        }
    )
    assert [leaf.text for leaf in iter_text_leaves(request)] == [
        "hello",
        "a@example.com",
        "tool result",
        "Find a person",
        "PII query",
    ]


def test_responses_nested_payloads_are_traversed() -> None:
    """Responses function output strings cannot bypass policy."""
    responses = OpenAIResponsesRequest.model_validate(
        {
            "model": "test",
            "instructions": "system text",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "1",
                    "output": {"nested": ["secret", {"value": "a@example.com"}]},
                }
            ],
        }
    )
    assert [leaf.text for leaf in iter_text_leaves(responses)] == [
        "system text",
        "secret",
        "a@example.com",
    ]


def test_mcp_traverses_and_replaces_only_recursive_argument_values() -> None:
    """MCP protocol strings, keys, metadata, and non-strings remain immutable."""
    request = McpRequest.model_validate(
        {
            "jsonrpc": "2.0",
            "id": "id@example.com",
            "method": "tools/call",
            "params": {
                "name": "NL123456789B01",
                "arguments": {
                    "a@example.com": "first@example.com",
                    "nested": [
                        {"value": "second@example.com", "enabled": True},
                        12,
                        None,
                    ],
                },
                "_meta": {"trace": "meta@example.com"},
            },
        }
    )

    leaves = iter_text_leaves(request)

    assert [(leaf.path, leaf.text) for leaf in leaves] == [
        (("params", "arguments", "a@example.com"), "first@example.com"),
        (("params", "arguments", "nested", 0, "value"), "second@example.com"),
    ]
    transformed = replace_text_leaves(
        request,
        {
            leaves[0].path: "********",
            leaves[1].path: "########",
        },
    )
    assert isinstance(transformed, McpRequest)
    assert transformed.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "jsonrpc": "2.0",
        "id": "id@example.com",
        "method": "tools/call",
        "params": {
            "name": "NL123456789B01",
            "arguments": {
                "a@example.com": "********",
                "nested": [{"value": "########", "enabled": True}, 12, None],
            },
            "_meta": {"trace": "meta@example.com"},
        },
    }


def test_nested_tool_argument_is_sanitized() -> None:
    """Validated nested strings are replaced without changing protocol identifiers."""
    request = OpenAIChatRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "a@example.com"},
                            },
                        }
                    ],
                }
            ],
        }
    )
    runtime = EngineRuntime(Settings(allow_test_analyzer=True, enforce_client_identity=False))
    result = runtime.policy.analyze(request)
    assert isinstance(result.request, OpenAIChatRequest)
    assert result.request.messages[0].tool_calls[0].function.arguments == {"query": "*************"}


def test_stringified_tool_arguments_remain_valid_protocol_json() -> None:
    """Standard OpenAI argument strings are parsed leaf-wise and serialized back."""
    request = OpenAIChatRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": '{"query":"a@example.com","limit":2}',
                            },
                        }
                    ],
                }
            ],
        }
    )
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    result = runtime.policy.analyze(request)
    assert isinstance(result.request, OpenAIChatRequest)
    arguments = result.request.messages[0].tool_calls[0].function.arguments
    assert isinstance(arguments, str)
    assert json.loads(arguments) == {"query": "*************", "limit": 2}


def test_response_format_schema_prose_is_traversed_and_sanitized() -> None:
    """Structured-output descriptions and defaults cannot carry uninspected PII."""
    request = OpenAIChatRequest.model_validate(
        {
            "model": "test",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "result",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Use a@example.com",
                            }
                        },
                    },
                },
            },
        }
    )
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))
    result = runtime.policy.analyze(request)
    assert isinstance(result.request, OpenAIChatRequest)
    schema = result.request.response_format
    assert schema is not None
    assert "a@example.com" not in json.dumps(schema)


def test_responses_text_format_traverses_only_model_visible_schema_prose() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "test",
            "input": "Return JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "NL123456789B01",
                    "description": "Format for format@example.com",
                    "schema": {
                        "title": "Customer a@example.com",
                        "type": "object",
                        "properties": {
                            "a@example.com": {
                                "type": "string",
                                "description": "Value for b@example.com",
                                "default": "c@example.com",
                                "examples": ["d@example.com"],
                                "enum": ["e@example.com"],
                            }
                        },
                        "required": ["a@example.com"],
                    },
                    "strict": True,
                },
                "verbosity": "low",
            },
        }
    )

    leaves = iter_text_leaves(request)

    assert [leaf.path for leaf in leaves] == [
        ("input",),
        ("text", "format", "description"),
        ("text", "format", "schema", "title"),
        (
            "text",
            "format",
            "schema",
            "properties",
            "a@example.com",
            "description",
        ),
        ("text", "format", "schema", "properties", "a@example.com", "default"),
        ("text", "format", "schema", "properties", "a@example.com", "examples", 0),
    ]


def test_responses_text_format_replacement_preserves_complete_control_shape() -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "test",
            "input": "Return JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "customer_result",
                    "description": "Format for a@example.com",
                    "schema": {
                        "title": "Customer a@example.com",
                        "type": "object",
                        "properties": {
                            "email": {
                                "type": "string",
                                "description": "Value for a@example.com",
                                "default": "a@example.com",
                                "examples": ["a@example.com"],
                                "enum": ["a@example.com"],
                            }
                        },
                        "required": ["email"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
                "verbosity": "high",
            },
        }
    )
    runtime = EngineRuntime(Settings(allow_test_analyzer=True))

    result = runtime.policy.analyze(request)

    assert isinstance(result.request, OpenAIResponsesRequest)
    assert result.request.text is not None
    transformed = result.request.text.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert transformed == {
        "format": {
            "type": "json_schema",
            "name": "customer_result",
            "description": "Format for *************",
            "schema": {
                "title": "Customer *************",
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "Value for *************",
                        "default": "*************",
                        "examples": ["*************"],
                        "enum": ["a@example.com"],
                    }
                },
                "required": ["email"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        "verbosity": "high",
    }


@pytest.mark.parametrize(
    "path",
    [
        ("text", "format", "type"),
        ("text", "format", "name"),
        ("text", "format", "schema", "properties", "email", "type"),
        ("text", "format", "schema", "properties", "email", "enum", 0),
    ],
)
def test_responses_text_format_protocol_strings_cannot_be_replaced(
    path: tuple[str | int, ...],
) -> None:
    request = OpenAIResponsesRequest.model_validate(
        {
            "model": "test",
            "input": "Return JSON",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "customer_result",
                    "schema": {
                        "type": "object",
                        "properties": {"email": {"type": "string", "enum": ["a@example.com"]}},
                    },
                }
            },
        }
    )

    with pytest.raises(ValueError, match="not model-visible text"):
        replace_text_leaves(request, {path: "mutated"})


def test_invalid_stringified_tool_arguments_fail_closed() -> None:
    """Malformed serialized arguments never bypass JSON-aware traversal."""
    request = OpenAIChatRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{invalid"},
                        }
                    ],
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="valid JSON"):
        iter_text_leaves(request)


def test_unicode_offsets_remain_local_to_each_field() -> None:
    """Combining characters and non-ASCII text around PII survive leaf-local replacement."""
    content = "Préface e\u0301 東京 a@example.com fin"
    request = OpenAIChatRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": content}]}
    )
    runtime = EngineRuntime(Settings(allow_test_analyzer=True, enforce_client_identity=False))
    result = runtime.policy.analyze(request)
    assert isinstance(result.request, OpenAIChatRequest)
    transformed = result.request.messages[0].content
    assert transformed == "Préface e\u0301 東京 " + "*" * 13 + " fin"


def test_nested_payload_depth_is_bounded() -> None:
    """Deep MCP arguments fail before policy analysis can forward the request."""
    nested: object = "secret"
    for _depth in range(5):
        nested = {"value": nested}
    request = McpRequest.model_validate(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "lookup", "arguments": nested},
        }
    )
    with pytest.raises(ValueError, match="nesting"):
        iter_text_leaves(request, max_depth=2)
