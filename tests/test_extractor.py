"""Unit tests for the Extractor class."""

from unittest.mock import MagicMock

import pandas as pd
import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pytest_mock.plugin import MockerFixture

from laurium.decoder_models.extractor import Extractor

SCHEMA = {"label": (str, "classification label")}


def _make_extractor(mocker: MockerFixture) -> Extractor:
    """Create an Extractor with only its batch dependencies configured."""
    mocker.patch.object(Extractor, "__init__", return_value=None)
    extractor = Extractor.__new__(Extractor)
    extractor.chain = MagicMock()
    extractor.schema_dtypes = {"label": str}
    extractor.failed_result_template = {"label": pd.NA}
    return extractor


def test_init_accepts_chat_model_and_builds_schema() -> None:
    """The constructor stores the model and builds the requested schema."""
    llm = FakeListChatModel(responses=['{"label": "positive"}'])

    extractor = Extractor(SCHEMA, llm, keywords=["refund"])

    assert extractor.llm is llm
    assert extractor.schema_dtypes == {"label": str}
    assert extractor.schema_desc == {"label": "classification label"}
    assert extractor.pydantic_model.model_fields["label"].description == (
        "classification label"
    )
    assert extractor.failed_result_template == {"label": pd.NA}


def test_init_creates_llm_from_configuration(mocker: MockerFixture) -> None:
    """Dictionary configuration is passed to the LLM factory."""
    llm = FakeListChatModel(responses=['{"label": "positive"}'])
    create_llm = mocker.patch(
        "laurium.decoder_models.llm.create_llm", return_value=llm
    )

    extractor = Extractor(
        SCHEMA,
        {"llm_platform": "ollama", "model_name": "test-model"},
        keywords=[],
    )

    assert extractor.llm is llm
    create_llm.assert_called_once_with(
        llm_platform="ollama", model_name="test-model"
    )


def test_prompt_contains_keywords_and_schema() -> None:
    """The generated prompt includes keyword and schema guidance."""
    extractor = Extractor(
        SCHEMA,
        FakeListChatModel(responses=['{"label": "positive"}']),
        keywords=["refund"],
    )

    prompt_text = extractor.prompt.format(text="Customer received a refund")

    assert "Pay special attention to these keywords: refund" in prompt_text
    assert "label: classification label" in prompt_text
    assert "Analyze this text: Customer received a refund" in prompt_text


def test_init_rejects_unsupported_llm() -> None:
    """Unsupported LLM values fail with a useful error."""
    with pytest.raises(ValueError, match="llm must be either a dict"):
        Extractor(SCHEMA, object(), keywords=[])


def test_init_accepts_prompt_without_keywords() -> None:
    """The optional prompt customization does not require keywords."""
    extractor = Extractor(
        SCHEMA,
        FakeListChatModel(responses=['{"label": "positive"}']),
    )

    assert extractor.prompt is not None


def test_label_delegates_to_chain(mocker: MockerFixture) -> None:
    """Label forwards one text and returns the chain result."""
    extractor = _make_extractor(mocker)
    extractor.chain.invoke.return_value = {"label": "positive"}

    result = extractor.label("The issue was resolved.")

    assert result == {"label": "positive"}
    extractor.chain.invoke.assert_called_once_with("The issue was resolved.")


def test_label_propagates_parser_errors(mocker: MockerFixture) -> None:
    """Parser errors from a single extraction are not swallowed."""
    extractor = _make_extractor(mocker)
    error = OutputParserException("invalid output")
    extractor.chain.invoke.side_effect = error

    with pytest.raises(OutputParserException) as raised:
        extractor.label("Unparseable response")

    assert raised.value is error


def test_batch_label_appends_results_and_forwards_concurrency(
    mocker: MockerFixture,
) -> None:
    """Successful batch output is aligned with the input rows."""
    extractor = _make_extractor(mocker)
    extractor.chain.batch.return_value = [
        {"label": "positive"},
        {"label": "negative"},
    ]
    frame = pd.DataFrame(
        {"text": ["Good service", "Poor service"], "case_id": [4, 9]},
        index=[10, 20],
    )

    result = extractor.batch_label(frame, "text", max_concurrency=2)

    assert result.to_dict("records") == [
        {"text": "Good service", "case_id": 4, "label": "positive"},
        {"text": "Poor service", "case_id": 9, "label": "negative"},
    ]
    assert result.index.tolist() == [0, 1]
    extractor.chain.batch.assert_called_once_with(
        ["Good service", "Poor service"],
        {"max_concurrency": 2},
        return_exceptions=True,
    )


def test_batch_label_retries_parser_failure_and_preserves_success(
    mocker: MockerFixture,
) -> None:
    """Only failed positions are retried and later results replace them."""
    extractor = _make_extractor(mocker)
    extractor.chain.batch.side_effect = [
        [{"label": "positive"}, OutputParserException("invalid output")],
        [{"label": "negative"}],
    ]
    frame = pd.DataFrame({"text": ["Good", "Bad"]})

    result = extractor.batch_label(frame, "text", max_concurrency=3)

    assert result["label"].tolist() == ["positive", "negative"]
    assert extractor.chain.batch.call_args_list[1].args[0] == ["Bad"]


def test_batch_label_raises_non_parser_error_when_not_ignoring(
    mocker: MockerFixture,
) -> None:
    """Non-parser errors are raised when ignore_errors is false."""
    extractor = _make_extractor(mocker)
    error = RuntimeError("service unavailable")
    extractor.chain.batch.return_value = [error]

    with pytest.raises(RuntimeError) as raised:
        extractor.batch_label(pd.DataFrame({"text": ["Hello"]}), "text")

    assert raised.value is error


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(
            OutputParserException("invalid output"),
            id="parser-error",
        ),
        pytest.param(RuntimeError("service unavailable"), id="runtime-error"),
    ],
)
def test_batch_label_fills_exhausted_errors_when_ignoring(
    mocker: MockerFixture, failure: Exception
) -> None:
    """Ignored failures are retried, then replaced with null fields."""
    extractor = _make_extractor(mocker)
    extractor.chain.batch.side_effect = [
        [failure],
        [failure],
    ]

    result = extractor.batch_label(
        pd.DataFrame({"text": ["Hello"]}),
        "text",
        max_retries=1,
        ignore_errors=True,
    )

    assert pd.isna(result.loc[0, "label"])
    assert extractor.chain.batch.call_count == 2


def test_batch_label_preserves_nullable_integer_dtype_for_null_result(
    mocker: MockerFixture,
) -> None:
    """An integer result column remains nullable when one extraction fails."""
    extractor = _make_extractor(mocker)
    extractor.schema_dtypes = {"count": int}
    extractor.failed_result_template = {"count": pd.NA}
    extractor.chain.batch.return_value = [
        {"count": 3},
        RuntimeError("service unavailable"),
    ]
    frame = pd.DataFrame({"text": ["Three items", "Unknown"]})

    result = extractor.batch_label(
        frame, "text", max_retries=0, ignore_errors=True
    )

    assert result["count"].dtype == pd.Int64Dtype()
    assert result["count"].tolist() == [3, pd.NA]


def test_batch_label_zero_retries_fills_initial_failure(
    mocker: MockerFixture,
) -> None:
    """Zero retries means only the initial batch attempt is made."""
    extractor = _make_extractor(mocker)
    extractor.chain.batch.return_value = [
        OutputParserException("invalid output")
    ]

    result = extractor.batch_label(
        pd.DataFrame({"text": ["Hello"]}), "text", max_retries=0
    )

    assert pd.isna(result.loc[0, "label"])
    extractor.chain.batch.assert_called_once()


def test_batch_label_empty_dataframe(mocker: MockerFixture) -> None:
    """An empty input returns an empty frame with the result column."""
    extractor = _make_extractor(mocker)
    extractor.chain.batch.return_value = []

    result = extractor.batch_label(pd.DataFrame({"text": []}), "text")

    assert result.empty
    assert list(result.columns) == ["text", "label"]
