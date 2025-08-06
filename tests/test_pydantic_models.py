"""Test bundled Pydantic models."""

from typing import Literal

import pydantic
import pytest

from laurium.decoder_models import pydantic_models


@pytest.mark.parametrize(
    "labels",
    [
        ["positive", "neutral", "negative"],
    ],
)
def test_create_label_model(labels):
    """Test `create_label_model` function."""
    # Create label model
    model = pydantic_models.create_label_model(labels)

    # Check that we have a Pydantic model
    assert issubclass(model, pydantic.BaseModel)

    # Check that we have two fields, "label" and "explanation"
    fields = model.model_fields
    assert len(fields) == 2
    assert "label" in fields
    assert "explanation" in fields

    # Check field datatypes
    assert fields["label"].annotation is Literal[*labels]  # type: ignore
    assert fields["explanation"].annotation is str


@pytest.fixture()
def sentiment_model():
    """Create a basic label model for testing."""
    model = pydantic_models.create_label_model(
        ["positive", "neutral", "negative"]
    )
    yield model


@pytest.mark.parametrize(
    "good_input",
    [
        {
            "label": "positive",
            "explanation": "The review says the movie was excellent.",
        },
        {
            "label": "neutral",
            "explanation": "The review did not comment on the movie.",
        },
    ],
)
def test_label_model_validation_pass(sentiment_model, good_input):
    """Test that model validation works for good input."""
    # Test examples that should pass
    sentiment_model(**good_input)


@pytest.mark.parametrize(
    "bad_input",
    [
        {  # incorrect label
            "label": "Positive",
            "explanation": "The review says the movie was excellent",
        },
        {  # missing field
            "label": "neutral",
        },
        {  # blank label and explanation
            "label": "",
            "explanation": "",
        },
        {  # type mismatch
            "label": "negative",
            "explanation": False,
        },
        {  # extra info
            "label": "positive",
            "explanation": "The review says the movie was excellent",
            "note_id": 1,
        },
    ],
)
def test_label_model_validation_fail(sentiment_model, bad_input):
    """Test that model validation fails for invalid input."""
    with pytest.raises(pydantic.ValidationError):
        sentiment_model(**bad_input)
