"""Reviewed runtime prompt fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def english_prompt_content():
    return json.loads(
        (Path(__file__).parent / "fixtures/reviewed_prompt_content.json").read_text()
    )["en"]
