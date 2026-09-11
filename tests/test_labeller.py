"""Unit tests for the Labeller class."""

from typing import Literal

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from laurium.decoder_models.labeller import Labeller

LABELS = {"low": "Low priority", "high": "High priority"}


def _make_labeller(response: str) -> Labeller:
    """Build a Labeller with a deterministic mocked chat response."""
    return Labeller(
        LABELS,
        FakeListChatModel(responses=[response]),
        acronyms={"SLA": "service level agreement"},
    )


def test_prompt_contains_labels_descriptions_and_acronyms() -> None:
    """The rendered prompt contains label and acronym guidance."""
    labeller = _make_labeller('{"label": "low", "explanation": "urgent"}')

    prompt_text = labeller.prompt.format(text="The request is urgent")

    assert 'low" - Low priority' in prompt_text
    assert 'high" - High priority' in prompt_text
    assert "SLA - service level agreement" in prompt_text
    assert 'label: str - one of the labels ("low", "high")' in prompt_text
    assert "The request is urgent" in prompt_text


@pytest.mark.parametrize("label", ["low", "high"])
def test_label_accepts_each_provided_label(label: str) -> None:
    """Every configured label is accepted by the output schema."""
    labeller = _make_labeller(
        f'{{"label": "{label}", "explanation": "valid"}}'
    )

    result = labeller.label("A case note")

    assert result == {"label": label, "explanation": "valid"}
    assert (
        labeller.pydantic_model.model_fields["label"].annotation
        is Literal["low", "high"]
    )


def test_label_rejects_label_outside_configured_labels() -> None:
    """An output label absent from the configuration is rejected."""
    labeller = _make_labeller('{"label": "medium", "explanation": "invalid"}')

    with pytest.raises(OutputParserException):
        labeller.label("A case note")


def test_prompt_omits_acronym_section_when_acronyms_are_not_provided() -> None:
    """No acronym guidance is rendered when no acronyms are supplied."""
    labeller = Labeller(
        LABELS,
        FakeListChatModel(
            responses=['{"label": "low", "explanation": "valid"}']
        ),
    )

    assert "You may find the following acronyms helpful" not in labeller.prompt
