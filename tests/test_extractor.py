"""Unit tests for the Extractor class."""

import itertools
import json
import typing
from typing import Literal
from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pytest_mock.plugin import MockerFixture

from laurium.decoder_models.extractor import Extractor

###############################################################################
# Constructor tests
###############################################################################


@pytest.mark.parametrize(
    "fields,schema_dtypes,schema_desc",
    [
        (
            ["label"],
            [Literal["positive", "neutral", "negative"]],
            ["classification label"],
        ),
        (
            ["port", "ip_addr", "flagged"],
            [int, str, bool],
            ["Port number", "IP address", "Whether the activity is flagged"],
        ),
    ],
)
def test_constructor(
    fields: list[str],
    schema_dtypes: list[type],
    schema_desc: list[str],
) -> None:
    """The constructor initializes with the correct attributes."""
    llm = FakeListChatModel(responses=[])

    schema = {
        field: (dtype, desc)
        for field, dtype, desc in zip(
            fields, schema_dtypes, schema_desc, strict=True
        )
    }

    extractor = Extractor(schema, llm)

    assert extractor.llm is llm

    # Check schema has been correctly set up
    assert extractor.schema_dtypes == {
        field: dtype
        for field, dtype in zip(fields, schema_dtypes, strict=True)
    }
    assert extractor.schema_desc == {
        field: desc for field, desc in zip(fields, schema_desc, strict=True)
    }

    # Check that Pydantic model has the correct field descriptions
    for field, desc in zip(fields, schema_desc, strict=True):
        assert extractor.pydantic_model.model_fields[field].description == desc

    # Check null result has been created
    assert extractor.failed_result_template == {
        field: pd.NA for field in fields
    }


def test_constructor_creates_llm_from_config(mocker: MockerFixture) -> None:
    """Dictionary configuration is passed to the LLM factory."""
    llm = FakeListChatModel(responses=['{"label": "positive"}'])
    create_llm = mocker.patch(
        "laurium.decoder_models.llm.create_llm", return_value=llm
    )

    extractor = Extractor(
        {"label": (str, "classification label")},
        {"llm_platform": "ollama", "model_name": "test-model"},
        keywords=[],
    )

    assert extractor.llm is llm
    create_llm.assert_called_once_with(
        llm_platform="ollama", model_name="test-model"
    )


def test_constructor_rejects_unsupported_llm() -> None:
    """Unsupported LLM values fail with a useful error."""
    with pytest.raises(ValueError, match="llm must be either a dict"):
        Extractor({"label": (str, "classification label")}, object())


###############################################################################
# Prompt tests
###############################################################################


@pytest.mark.parametrize(
    "schema",
    [
        {
            "label": (
                Literal["positive", "neutral", "negative"],
                "classification label",
            ),
        },
        {
            "port": (int, "Port number"),
            "ip_addr": (str, "IP address"),
            "flagged": (bool, "Whether the activity is flagged"),
        },
    ],
)
def test_schema_in_prompt(schema: dict[str, tuple[type, str]]) -> None:
    """Verify that prompt contains the schema."""
    llm = FakeListChatModel(responses=[])

    extractor = Extractor(schema, llm, keywords=["keyword"])
    example_prompt = extractor.prompt.format(text="Sample text for extraction")
    for field, (dtype, desc) in schema.items():
        if typing.get_origin(dtype) is Literal:
            for allowed_value in typing.get_args(dtype):
                assert allowed_value in example_prompt
        else:
            assert f"<{dtype.__name__}>" in example_prompt
        assert f"{field}: {desc}" in example_prompt


def test_keywords_in_prompt() -> None:
    """Verify that keywords appear in the prompt."""
    extractor = Extractor(
        {"label": (str, "classification label")},
        FakeListChatModel(responses=['{"label": "positive"}']),
        keywords=["keyword", "another_keyword"],
    )

    assert extractor.prompt is not None
    example_prompt = extractor.prompt.format(text="Sample text for extraction")
    assert "keyword" in example_prompt
    assert "another_keyword" in example_prompt


def test_system_message_in_prompt() -> None:
    """Verify that the system message appears in the prompt."""
    system_message = "You are an expert at providing classification labels."
    extractor = Extractor(
        {"label": (str, "classification label")},
        FakeListChatModel(responses=['{"label": "positive"}']),
        system_message,
    )

    assert extractor.prompt is not None
    example_prompt = extractor.prompt.format(text="Sample text for extraction")
    assert system_message in example_prompt


###############################################################################
# Single extraction tests
###############################################################################


@pytest.mark.parametrize(
    "schema,llm_response",
    [
        (
            {
                "label": (
                    Literal["positive", "neutral", "negative"],
                    "classification label",
                ),
            },
            '{"label": "positive"}',
        ),
        (
            {
                "port": (int, "Port number"),
                "ip_addr": (str, "IP address"),
                "flagged": (bool, "Whether the activity is flagged"),
            },
            '{"port": 8080, "ip_addr": "127.0.0.1", "flagged": false}',
        ),
    ],
)
def test_label_good(
    schema: dict[str, tuple[type, str]], llm_response: str
) -> None:
    """Test that the extractor correctly parses valid LLM output."""
    llm = FakeListChatModel(responses=[llm_response])

    extractor = Extractor(schema, llm, keywords=["keyword"])
    label = extractor.label("Sample text for extraction")
    assert label == json.loads(llm_response)


@pytest.mark.parametrize(
    "schema,llm_responses",
    [
        (
            {
                "label": (
                    Literal["positive", "neutral", "negative"],
                    "classification label",
                ),
            },
            [
                '{"label": "good"}',
                'Sure! Here\'s the necessary JSON:\n{"label": "positive"}',
                "{label: positive}",
                "I'm sorry, I cannot provide a valid JSON response.",
            ],
        ),
        (
            {
                "port": (int, "Port number"),
                "ip_addr": (str, "IP address"),
                "flagged": (bool, "Whether the activity is flagged"),
            },
            [
                '{"port": null, "ip_addr": "127.0.0.1", "flagged": true}',
                '{"ip_addr": "127.0.0.1", "flagged": true}',
                '{"port": 8080, "ip_addr": "127.0.0.1", "flagged": "alert"}',
            ],
        ),
    ],
)
def test_label_bad(
    schema: dict[str, tuple[type, str]], llm_responses: list[str]
) -> None:
    """Test error raised for bad LLM output."""
    llm = FakeListChatModel(responses=llm_responses)
    extractor = Extractor(schema, llm, keywords=["keyword"])

    for _ in llm_responses:
        with pytest.raises(OutputParserException):
            extractor.label("Unparseable response")


###############################################################################
# Batch extraction tests
###############################################################################


@pytest.mark.parametrize(
    "schema,llm_responses",
    [
        (
            {
                "label": (
                    Literal["positive", "neutral", "negative"],
                    "classification label",
                ),
            },
            [
                '{"label": "positive"}',
                '{"label": "neutral"}',
                '{"label": "negative"}',
            ],
        ),
        (
            {
                "port": (int, "Port number"),
                "ip_addr": (str, "IP address"),
                "flagged": (bool, "Whether the activity is flagged"),
            },
            ['{"port": 8080, "ip_addr": "127.0.0.1", "flagged": false}'] * 3,
        ),
    ],
)
def test_batch_label_good(
    schema: dict[str, tuple[type, str]], llm_responses: list[str]
) -> None:
    """Test that the extractor correctly parses valid batch LLM output."""
    llm = FakeListChatModel(responses=llm_responses)

    extractor = Extractor(schema, llm, keywords=["keyword"])
    result = extractor.batch_label(
        pd.DataFrame(
            {"text": ["Sample text for extraction"] * len(llm_responses)}
        ),
        "text",
    )

    assert result.shape[0] == len(llm_responses)
    assert all(col in result.columns for col in schema.keys())
    for row, llm_response in zip(
        result.to_dict("records"), llm_responses, strict=True
    ):
        row.pop("text")  # Remove the input text column before comparison
        assert row == json.loads(llm_response)


@pytest.mark.parametrize(
    "schema,llm_responses",
    [
        (
            {
                "label": (
                    Literal["positive", "neutral", "negative"],
                    "classification label",
                ),
            },
            [
                '{"label": "good"}',
                'Sure! Here\'s the necessary JSON:\n{"label": "positive"}',
                "{label: positive}",
                "I'm sorry, I cannot provide a valid JSON response.",
            ],
        ),
        (
            {
                "port": (int, "Port number"),
                "ip_addr": (str, "IP address"),
                "flagged": (bool, "Whether the activity is flagged"),
            },
            [
                '{"port": null, "ip_addr": "127.0.0.1", "flagged": true}',
                '{"ip_addr": "127.0.0.1", "flagged": true}',
                '{"port": 8080, "ip_addr": "127.0.0.1", "flagged": "alert"}',
            ],
        ),
    ],
)
def test_batch_all_bad_llm_response(
    schema: dict[str, tuple[type, str]], llm_responses: list[str]
) -> None:
    """Test handling of mal-formatted LLM output."""
    llm = FakeListChatModel(responses=llm_responses)
    extractor = Extractor(schema, llm, keywords=["keyword"])

    result = extractor.batch_label(
        pd.DataFrame({"text": ["Unparseable response"]}),
        "text",
    )

    null_results = result.isnull()
    for col in schema.keys():
        assert col in result.columns
        assert null_results[col].all()


@pytest.mark.parametrize("ignore_errors", [True, False])
def test_batch_error_handling(ignore_errors: bool) -> None:
    """Test that batch error handling works correctly."""
    extractor = Extractor(
        {"label": (str, "classification label")},
        FakeListChatModel(responses=[]),
    )

    extractor.chain = MagicMock()
    extractor.chain.batch.side_effect = [
        [ConnectionError("Connection failed")],
        [{"label": "positive"}],
    ]

    free_text_df = pd.DataFrame({"text": ["Free text"]})

    if ignore_errors:
        result = extractor.batch_label(
            free_text_df,
            "text",
            max_retries=1,
            ignore_errors=ignore_errors,
        )
        assert result["label"].iloc[0] == "positive"
    else:
        with pytest.raises(ConnectionError):
            extractor.batch_label(
                free_text_df,
                "text",
                max_retries=1,
                ignore_errors=ignore_errors,
            )


@pytest.mark.parametrize("max_retries", [0, 1, 2])
def test_batch_error_retries(max_retries: int) -> None:
    """Test that batch retries occur correctly on errors."""
    extractor = Extractor(
        {"label": (str, "classification label")},
        FakeListChatModel(responses=[]),
    )

    # Set canned LLM outputs for testing retries
    llm_responses = [
        [
            OutputParserException("Parsing failed"),
            {"label": "neutral"},
            OutputParserException("Parsing failed"),
        ],
        [
            {"label": "positive"},
            OutputParserException("Parsing failed"),
        ],
        [{"label": "negative"}],
    ]

    extractor.chain = MagicMock()
    extractor.chain.batch.side_effect = llm_responses

    free_text_df = pd.DataFrame(
        {
            "text": [
                "Free text",
                "More free text",
                "Even more free text",
            ]
        }
    )

    # Build ground-truth output
    expected_labels = [pd.NA] * len(free_text_df)  # Assume all NA to start
    llm_response_iter = itertools.chain.from_iterable(
        llm_responses[: max_retries + 1]  # Limit output number by max_retries
    )
    for idx, label in itertools.cycle(enumerate(expected_labels)):
        if label is not pd.NA:
            continue
        try:
            item = next(llm_response_iter)
            if isinstance(item, dict) and "label" in item:
                expected_labels[idx] = item["label"]
        except StopIteration:
            break

    result = extractor.batch_label(
        free_text_df,
        "text",
        max_retries=max_retries,
    )

    assert extractor.chain.batch.call_count == max_retries + 1
    assert result["label"].isnull().sum() == 2 - max_retries
    assert result["label"].tolist() == expected_labels


def test_batch_empty_dataframe() -> None:
    """An empty input returns an empty frame with the result column."""
    extractor = Extractor(
        {"label": (str, "classification label")},
        FakeListChatModel(responses=[]),
    )

    result = extractor.batch_label(pd.DataFrame({"text": []}), "text")

    assert result.empty
    assert list(result.columns) == ["text", "label"]
