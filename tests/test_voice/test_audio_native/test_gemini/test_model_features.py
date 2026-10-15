"""Offline tests for Gemini Live model-specific feature gates."""

from unittest.mock import Mock

import pytest

from tau2.voice.audio_native.gemini.provider import GeminiLiveProvider


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gemini-2.5-flash-native-audio-preview-12-2025", True),
        ("gemini-3.1-flash-live-preview", False),
        ("gemini-3.8-live-extended-thinking", False),
        ("models/gemini-3.8-live-extended-thinking", False),
    ],
)
def test_proactive_audio_support(model: str, expected: bool) -> None:
    assert GeminiLiveProvider._supports_proactive_audio(model) is expected


@pytest.mark.parametrize("use_raw_json_schema", [True, False])
def test_extended_thinking_tools_are_non_blocking(use_raw_json_schema: bool) -> None:
    """The required behavior must survive the SDK's Live setup conversion."""
    from google.genai import _live_converters, types

    tool = Mock()
    tool.openai_schema = {
        "function": {
            "name": "lookup_account",
            "description": "Look up an account",
            "parameters": {"type": "object", "properties": {}},
        }
    }
    provider = GeminiLiveProvider(
        api_key="test-key",
        model="gemini-3.8-live-extended-thinking",
        use_raw_json_schema=use_raw_json_schema,
    )
    declaration = provider._format_tools_for_api([tool])[0]
    assert declaration.behavior == types.Behavior.NON_BLOCKING

    class _DummyClient:
        vertexai = False

    params = types.LiveConnectParameters(
        model=provider.model,
        config=types.LiveConnectConfig(
            tools=[types.Tool(function_declarations=[declaration])]
        ),
    ).model_dump(exclude_none=True)
    setup = _live_converters._LiveConnectParameters_to_mldev(_DummyClient(), params)
    converted = setup["setup"]["tools"][0]["functionDeclarations"][0]
    assert converted.behavior == types.Behavior.NON_BLOCKING
