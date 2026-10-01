"""Unit tests for the Labeller class."""

import json
from typing import Literal

import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from laurium.decoder_models.labeller import Labeller

###############################################################################
# Constructor tests
###############################################################################


@pytest.mark.parametrize(
    "labels",
    [
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        {"low": "Low priority", "medium": "Medium priority", "high": "High"},
    ],
)
def test_constructor(labels: dict[str, str]) -> None:
    """The constructor initializes with the correct attributes."""
    llm = FakeListChatModel(responses=[])

    labeller = Labeller(labels, llm)

    assert labeller.llm is llm
    assert labeller.labels == labels

    # Check schema has been correctly set up
    assert labeller.schema_dtypes == {
        "label": Literal[tuple(labels)],
        "explanation": str,
    }
    assert labeller.schema_desc["explanation"] == (
        "The explanation for why this label was chosen."
    )

    # Check that Pydantic model has the correct field descriptions
    assert (
        labeller.pydantic_model.model_fields["label"].annotation
        == Literal[tuple(labels)]
    )
    assert (
        labeller.pydantic_model.model_fields["explanation"].description
        == "The explanation for why this label was chosen."
    )

    # Check null result has been created
    assert labeller.failed_result_template == {
        "label": pd.NA,
        "explanation": pd.NA,
    }


###############################################################################
# Prompt tests
###############################################################################


@pytest.mark.parametrize(
    "labels",
    [
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        {"low": "Low priority", "medium": "Medium priority", "high": "High"},
    ],
)
def test_labels_in_prompt(labels: dict[str, str]) -> None:
    """Verify that prompt contains the labels and their descriptions."""
    llm = FakeListChatModel(responses=[])

    labeller = Labeller(labels, llm)
    example_prompt = labeller.prompt.format(text="Sample text for labelling")
    for label, desc in labels.items():
        assert f'"{label}" - {desc}' in example_prompt
        assert f'"{label}"' in example_prompt


def test_acronyms_in_prompt() -> None:
    """Verify that acronym guidance appears in the prompt when provided."""
    labeller = Labeller(
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        FakeListChatModel(
            responses=['{"label": "positive", "explanation": "good"}']
        ),
        acronyms={"SLA": "service level agreement"},
    )

    example_prompt = labeller.prompt.format(text="Sample text for labelling")
    assert "You may find the following acronyms helpful" in example_prompt
    assert "SLA - service level agreement" in example_prompt


def test_acronyms_omitted_from_prompt_when_not_provided() -> None:
    """Verify that acronym guidance is omitted when none are provided."""
    labeller = Labeller(
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        FakeListChatModel(
            responses=['{"label": "positive", "explanation": "good"}']
        ),
    )

    example_prompt = labeller.prompt.format(text="Sample text for labelling")
    assert "You may find the following acronyms helpful" not in example_prompt


def test_base_prompt_in_prompt() -> None:
    """Verify that the base prompt appears in the prompt."""
    base_prompt = "You are an expert at labelling customer feedback."
    labeller = Labeller(
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        FakeListChatModel(
            responses=['{"label": "positive", "explanation": "good"}']
        ),
        base_prompt,
    )

    assert labeller.prompt is not None
    example_prompt = labeller.prompt.format(text="Sample text for labelling")
    assert base_prompt in example_prompt


def test_keywords_in_prompt() -> None:
    """Verify that keywords appear in the prompt."""
    labeller = Labeller(
        {"positive": "Positive sentiment", "negative": "Negative sentiment"},
        FakeListChatModel(
            responses=['{"label": "positive", "explanation": "good"}']
        ),
        keywords=["keyword", "another_keyword"],
    )

    assert labeller.prompt is not None
    example_prompt = labeller.prompt.format(text="Sample text for labelling")
    assert "keyword" in example_prompt
    assert "another_keyword" in example_prompt


###############################################################################
# Single extraction tests
###############################################################################


@pytest.mark.parametrize(
    "labels,llm_response",
    [
        (
            {"positive": "Positive sentiment", "negative": "Negative"},
            '{"label": "positive", "explanation": "good news"}',
        ),
        (
            {"low": "Low priority", "medium": "Medium", "high": "High"},
            '{"label": "high", "explanation": "urgent request"}',
        ),
    ],
)
def test_label_good(labels: dict[str, str], llm_response: str) -> None:
    """Test that the labeller correctly parses valid LLM output."""
    llm = FakeListChatModel(responses=[llm_response])

    labeller = Labeller(labels, llm, keywords=["keyword"])
    label = labeller.label("Sample text for labelling")
    assert label == json.loads(llm_response)


@pytest.mark.parametrize(
    "labels,llm_responses",
    [
        (
            {"positive": "Positive sentiment", "negative": "Negative"},
            [
                '{"label": "neutral", "explanation": "unclear"}',
                "Sure! Here's the label:\n"
                '{"label": "positive", "explanation": "good"}',
                "{label: positive, explanation: good}",
                "I'm sorry, I cannot provide a valid JSON response.",
            ],
        ),
        (
            {"low": "Low priority", "medium": "Medium", "high": "High"},
            [
                '{"label": "high"}',
                '{"explanation": "urgent request"}',
                '{"label": null, "explanation": "urgent request"}',
            ],
        ),
    ],
)
def test_label_bad(labels: dict[str, str], llm_responses: list[str]) -> None:
    """Test error raised for bad LLM output."""
    llm = FakeListChatModel(responses=llm_responses)
    labeller = Labeller(labels, llm, keywords=["keyword"])

    for _ in llm_responses:
        with pytest.raises(OutputParserException):
            labeller.label("Unparseable response")


###############################################################################
# Batch extraction tests
###############################################################################

# Test batch labelling (good, bad, error recovery, retries, ignore errors)


@pytest.mark.parametrize(
    "labels,llm_responses",
    [
        (
            {"positive": "Positive sentiment", "negative": "Negative"},
            [
                '{"label": "positive", "explanation": "good"}',
                '{"label": "negative", "explanation": "bad"}',
                '{"label": "positive", "explanation": "great"}',
            ],
        ),
        (
            {"low": "Low priority", "medium": "Medium", "high": "High"},
            ['{"label": "high", "explanation": "urgent"}'] * 3,
        ),
    ],
)
def test_batch_label_good(
    labels: dict[str, str], llm_responses: list[str]
) -> None:
    """Test that the labeller correctly parses valid batch LLM output."""
    llm = FakeListChatModel(responses=llm_responses)

    labeller = Labeller(labels, llm, keywords=["keyword"])
    result = labeller.batch_label(
        pd.DataFrame(
            {"text": ["Sample text for labelling"] * len(llm_responses)}
        ),
        "text",
    )

    assert result.shape[0] == len(llm_responses)
    assert "label" in result.columns
    assert "explanation" in result.columns
    for row, llm_response in zip(
        result.to_dict("records"), llm_responses, strict=True
    ):
        row.pop("text")  # Remove the input text column before comparison
        assert row == json.loads(llm_response)


@pytest.mark.parametrize(
    "labels,llm_responses",
    [
        (
            {"positive": "Positive sentiment", "negative": "Negative"},
            [
                '{"label": "neutral", "explanation": "unclear"}',
                "Sure! Here's the label:\n"
                '{"label": "positive", "explanation": "good"}',
                "{label: positive, explanation: good}",
                "I'm sorry, I cannot provide a valid JSON response.",
            ],
        ),
        (
            {"low": "Low priority", "medium": "Medium", "high": "High"},
            [
                '{"label": "high"}',
                '{"explanation": "urgent request"}',
                '{"label": null, "explanation": "urgent request"}',
            ],
        ),
    ],
)
def test_batch_all_bad_llm_response(
    labels: dict[str, str], llm_responses: list[str]
) -> None:
    """Test handling of mal-formatted LLM output."""
    llm = FakeListChatModel(responses=llm_responses)
    labeller = Labeller(labels, llm, keywords=["keyword"])

    result = labeller.batch_label(
        pd.DataFrame({"text": ["Unparseable response"]}),
        "text",
    )

    null_results = result.isnull()
    for col in ["label", "explanation"]:
        assert col in result.columns
        assert null_results[col].all()
