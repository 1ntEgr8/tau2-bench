"""Voice setup and reviewed provisioning preserve their SDK request semantics."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tau2.data_model.voice import VoiceCreateRequest, VoiceDesignRequest
from tau2.voice.elevenlabs_operations import create_voice, design_voice


@pytest.fixture
def client():
    client = Mock()
    client.text_to_voice.design.return_value = SimpleNamespace(
        previews=[
            SimpleNamespace(
                generated_voice_id="first",
                audio_base_64="YQ==",
                media_type="audio/mpeg",
            ),
            SimpleNamespace(
                generated_voice_id="second",
                audio_base_64="Yg==",
                media_type="audio/mpeg",
            ),
        ]
    )
    client.text_to_voice.create.return_value = SimpleNamespace(voice_id="saved")
    client.voices.get_all.return_value = SimpleNamespace(voices=[])
    return client


def test_shared_operations_preserve_parameters_and_never_retry_creation(client):
    previews = design_voice(
        client,
        VoiceDesignRequest(
            description="full brief",
            text="reviewed text",
            model="design-model",
            seed=31,
            loudness=0.4,
            guidance_scale=29,
        ),
    )
    assert [p.generated_voice_id for p in previews] == ["first", "second"]
    client.text_to_voice.design.assert_called_once_with(
        voice_description="full brief",
        text="reviewed text",
        model_id="design-model",
        seed=31,
        loudness=0.4,
        guidance_scale=29,
        auto_generate_text=False,
    )
    client.text_to_voice.create.side_effect = TimeoutError("uncertain creation")
    with pytest.raises(TimeoutError):
        create_voice(
            client,
            VoiceCreateRequest(
                name="account-name",
                description="short label",
                generated_voice_id="first",
            ),
        )
    client.text_to_voice.create.assert_called_once_with(
        voice_name="account-name",
        voice_description="short label",
        generated_voice_id="first",
    )


def test_setup_preserves_account_name_short_description_and_existing_ids(
    client, monkeypatch
):
    import elevenlabs

    from tau2.voice.scripts import setup_voices

    persona = SimpleNamespace(
        name="example",
        display_name="Example",
        prompt="long acoustic brief",
        short_description="short account description",
    )
    monkeypatch.setattr(setup_voices, "english_pack_personas", lambda: [persona])
    monkeypatch.setattr(elevenlabs, "ElevenLabs", lambda: client)
    client.voices.get_all.return_value = SimpleNamespace(
        voices=[SimpleNamespace(name="tau2_example", voice_id="old")]
    )
    result = setup_voices.setup_voices(seed=17, model="chosen", force=True)
    assert result == {"example": "saved"}
    client.voices.delete.assert_called_once_with(voice_id="old")
    client.text_to_voice.design.assert_called_once_with(
        voice_description="long acoustic brief",
        text=setup_voices.SAMPLE_TEXT,
        model_id="chosen",
        seed=17,
        loudness=setup_voices.VOICE_DESIGN_LOUDNESS,
        guidance_scale=setup_voices.VOICE_DESIGN_GUIDANCE_SCALE,
        auto_generate_text=False,
    )
    client.text_to_voice.create.assert_called_once_with(
        voice_name="tau2_example",
        voice_description="short account description",
        generated_voice_id="first",
    )


def test_factory_keeps_reviewed_text_and_selected_preview(client):
    from tau2.multilingual.factory.voice_assets import (
        ElevenLabsVoiceProvider,
        VoiceDesignConfig,
        VoiceSpec,
    )

    provider = ElevenLabsVoiceProvider.__new__(ElevenLabsVoiceProvider)
    provider.client = client
    preview = provider.preview(
        VoiceSpec(
            description="reviewed brief", audition_text="reviewed audition " * 10
        ),
        VoiceDesignConfig(
            model="reviewed-model", seed=8, loudness=0.1, guidance_scale=40
        ),
    )
    assert preview.generated_voice_id == "first"
    client.text_to_voice.design.assert_called_once_with(
        voice_description="reviewed brief",
        text="reviewed audition " * 10,
        model_id="reviewed-model",
        seed=8,
        loudness=0.1,
        guidance_scale=40,
        auto_generate_text=False,
    )
