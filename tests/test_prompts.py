"""Unit tests for the prompts module."""

from typing import Literal

import pytest

from laurium.decoder_models.prompts import (
    create_prompt,
    create_system_message,
    format_examples,
    format_schema_for_prompt,
    format_type_for_prompt,
)
from laurium.decoder_models.pydantic_models import make_dynamic_example_model

Example = make_dynamic_example_model(
    schema={"text": "str", "label": "str", "explanation": "str"},
    descriptions={
        "text": "input text to be analyzed",
        "label": "expected classification label",
        "explanation": "expected explanation",
    },
    model_name="Example",
)


@pytest.fixture
def basic_example_data():
    """Fixture providing a simple example for testing."""
    return format_examples(
        [
            Example(
                text=(
                    "Customer received replacement device within 24 hours "
                    "of reporting the defect."
                ),
                label="1",
                explanation=(
                    "Issue was resolved quickly with appropriate replacement"
                ),
            )
        ]
    )


@pytest.fixture
def example_human_template():
    """Fixture providing a standard human example template."""
    return "Review customer feedback: {text}"


@pytest.fixture
def example_assistant_template():
    """Fixture providing a standard assistant example template."""
    return '{{"label": {label}, "explanation": "{explanation}"}}'


@pytest.mark.parametrize(
    "examples,expected",
    [
        # Empty list
        ([], []),
        # Single example
        (
            [
                Example(
                    text=(
                        "Customer confirmed the software update fixed "
                        "the login issue completely."
                    ),
                    label="1",
                    explanation=(
                        "Evidence of successful issue resolution and customer"
                        "satisfaction"
                    ),
                )
            ],
            [
                {
                    "text": (
                        "Customer confirmed the software update fixed "
                        "the login issue completely."
                    ),
                    "label": "1",
                    "explanation": (
                        "Evidence of successful issue resolution and customer"
                        "satisfaction"
                    ),
                }
            ],
        ),
        # Multiple examples with different labels
        (
            [
                Example(
                    text=(
                        "Support team offered refund but customer declined "
                        "and requested store credit instead."
                    ),
                    label="0",
                    explanation="Resolution was offered but customer chose"
                    "alternative",
                ),
                Example(
                    text="Customer received full refund within 3 business"
                    "days.",
                    label="1",
                    explanation="Financial compensation was processed"
                    "successfully",
                ),
            ],
            [
                {
                    "text": (
                        "Support team offered refund but customer declined "
                        "and requested store credit instead."
                    ),
                    "label": "0",
                    "explanation": "Resolution was offered but customer chose"
                    "alternative",
                },
                {
                    "text": "Customer received full refund within 3 business"
                    "days.",
                    "label": "1",
                    "explanation": "Financial compensation was processed"
                    "successfully",
                },
            ],
        ),
    ],
)
def test_format_examples(examples, expected):
    """Test that format_examples correctly converts Example objects to dict.

    This test verifies that the format_examples function properly transforms
    Example objects into dictionaries suitable for few-shot learning templates.

    Parameters
    ----------
    examples : list[Example]
        List of Example objects to be formatted
    expected : list[dict]
        Expected dictionary output after formatting

    Notes
    -----
    Tests various scenarios including:
    - Empty list handling
    - Single example formatting
    - Multiple examples with different labels
    - Preservation of all fields (text, label, explanation)
    """
    assert format_examples(examples) == expected


@pytest.mark.parametrize(
    "base_message,keywords,expected",
    [
        # No keywords
        (
            "You are a customer service analyst reviewing feedback.",
            None,
            "You are a customer service analyst reviewing feedback.",
        ),
        # Single keyword
        (
            "You are a customer service analyst reviewing feedback.",
            ["refund"],
            "You are a customer service analyst reviewing feedback.\n"
            "Pay special attention to these keywords: refund",
        ),
        # Multiple keywords
        (
            "You are a customer service analyst reviewing feedback.",
            ["replacement", "warranty"],
            "You are a customer service analyst reviewing feedback.\n"
            "Pay special attention to these keywords: replacement, warranty",
        ),
    ],
)
def test_create_system_message(base_message, keywords, expected):
    """Test create_system_message correctly combines base msg with keywords.

    This test verifies that the system message is properly formatted with
    optional keywords when provided.

    Parameters
    ----------
    base_message : str
        The base system message to use
    keywords : list[str] | None
        Optional list of keywords to include in the message
    expected : str
        Expected combined message output

    Notes
    -----
    Tests various scenarios including:
    - No keywords (base message only)
    - Single keyword addition
    - Multiple keywords formatting
    """
    assert create_system_message(base_message, keywords) == expected


@pytest.mark.parametrize(
    "schema,descriptions,expected_parts",
    [
        # Simple types
        (
            {"ai_label": int},
            {"ai_label": "Sentiment classification"},
            [
                "For each text, extract:",
                "- ai_label: Sentiment classification",
                "Expected output format:",
                '"ai_label": <int>',
            ],
        ),
        # Literal types
        (
            {"sentiment": Literal["positive", "negative"]},
            {"sentiment": "Customer's emotional tone"},
            [
                "For each text, extract:",
                "- sentiment: Customer's emotional tone",
                "Expected output format:",
                '"sentiment": "positive"|"negative"',
            ],
        ),
        # Multiple fields
        (
            {
                "sentiment": Literal["positive", "negative"],
                "urgency": Literal[1, 2, 3, 4, 5],
                "category": Literal[
                    "IT", "Support", "Product", "Sales", "Other"
                ],
            },
            {
                "sentiment": "Customer's emotional tone",
                "urgency": "Priority level",
                "category": "Issue type",
            },
            [
                "For each text, extract:",
                "- sentiment: Customer's emotional tone",
                "- urgency: Priority level",
                "- category: Issue type",
                "Expected output format:",
                '"sentiment": "positive"|"negative"',
                '"urgency": 1|2|3|4|5',
                '"category": "IT"|"Support"|"Product"|"Sales"|"Other"',
            ],
        ),
        # Fields without descriptions
        (
            {"ai_label": int, "confidence": float},
            {},
            [
                "For each text, extract:",
                "- ai_label",
                "- confidence",
                "Expected output format:",
                '"ai_label": <int>',
                '"confidence": <float>',
            ],
        ),
    ],
)
def test_format_schema_for_prompt(schema, descriptions, expected_parts):
    """Test format_schema_for_prompt creates correct LLM-friendly format."""
    result = format_schema_for_prompt(schema, descriptions)

    for part in expected_parts:
        assert part in result


@pytest.mark.parametrize(
    "system_message,use_examples,final_query,test_input",
    [
        # Without examples
        (
            "You are a customer service analyst reviewing feedback.",
            False,
            "Analyze this complaint: {text}",
            "Customer with billing issue",
        ),
        # With examples
        (
            "You are a complaint resolution classifier.",
            True,
            "Review this feedback: {text}",
            "Customer with delivery delay",
        ),
    ],
)
def test_few_shot_examples(
    system_message,
    use_examples,
    final_query,
    test_input,
    basic_example_data,
    example_human_template,
    example_assistant_template,
):
    """Test that create_prompt correctly builds prompts.

    This test verifies that the prompt creation process correctly:
    1. Includes the system message at the beginning
    2. Incorporates few-shot examples when provided
    3. Adds the final query at the end
    4. Properly formats all message content with input values

    Parameters
    ----------
    system_message : str
        System message to include in the prompt
    use_examples : bool
        Whether to include few-shot examples
    final_query : str
        Template for the final query message
    test_input : str
        Input text to use when formatting the prompt
    basic_example_data : list[dict]
        Example data from fixture
    example_human_template : str
        Template for human messages in examples
    example_assistant_template : str
        Template for assistant messages in examples

    Notes
    -----
    Tests both with and without examples to ensure:
    - Correct message structure and ordering
    - Proper content formatting
    - All example data is included in the formatted messages
    - Message count is appropriate for the given inputs
    """
    # Use examples based on parameter
    examples = basic_example_data if use_examples else []

    # Create prompt
    prompt = create_prompt(
        system_message=system_message,
        examples=examples,
        example_human_template=example_human_template,
        example_assistant_template=example_assistant_template,
        final_query=final_query,
    )

    # Format messages
    formatted = prompt.format_messages(text=test_input)

    # Check first message is system message with correct content
    assert formatted[0].type == "system"
    assert formatted[0].content == system_message

    # Check last message is human message with correct content
    assert formatted[-1].type == "human"
    assert formatted[-1].content == final_query.format(text=test_input)

    if use_examples:
        # With examples, we should have messages between system and final query
        middle_messages = formatted[1:-1]
        assert len(middle_messages) > 0

        # Verify that all example data appears in the formatted messages
        all_content = " ".join(msg.content for msg in middle_messages)

        for example in examples:
            # Check that key example data appears in the formatted content
            # This is more flexible than checking exact formats
            for key, value in example.items():
                # Convert value to string to handle different types
                str_value = str(value)
                assert str_value in all_content, (
                    f"{key}={str_value} not found in formatted messages"
                )
    else:
        # Without examples, there should be no messages between system and
        # final query
        assert len(formatted) == 2


def test_create_prompt_with_schema():
    """Test create_prompt correctly integrates schema into system message."""
    system_message = "Analyze customer feedback."
    schema = {"sentiment": Literal["positive", "negative"], "urgency": int}
    descriptions = {
        "sentiment": "Customer's emotional tone",
        "urgency": "Priority level 1-5",
    }

    prompt = create_prompt(
        system_message=system_message,
        examples=[],
        example_human_template="Feedback: {text}",
        example_assistant_template='{"sentiment": "{sentiment}"}',
        final_query="Analyze: {text}",
        schema=schema,
        descriptions=descriptions,
    )

    formatted = prompt.format_messages(text="test input")
    system_content = formatted[0].content

    # Check that schema formatting was added to system message
    assert "For each text, extract:" in system_content
    assert "sentiment: Customer's emotional tone" in system_content
    assert "urgency: Priority level 1-5" in system_content
    assert "Expected output format:" in system_content
    assert '"sentiment": "positive"|"negative"' in system_content
    assert '"urgency": <int>' in system_content


@pytest.mark.parametrize(
    "field_type,expected",
    [
        # Simple types
        (int, "<int>"),
        (str, "<str>"),
        (float, "<float>"),
        (bool, "<bool>"),
        # Literal types with strings
        (Literal["positive", "negative"], '"positive"|"negative"'),
        (Literal["high", "medium", "low"], '"high"|"medium"|"low"'),
        # Literal types with numbers
        (Literal[1, 2, 3, 4, 5], "1|2|3|4|5"),
        # Literal types with mixed types
        (Literal["yes", "no", 1, 0], '"yes"|"no"|1|0'),
        # Single literal value
        (Literal["only"], '"only"'),
    ],
)
def test_format_type_for_prompt(field_type, expected):
    """Test format_type_for_prompt handles various type formats correctly.

    This test verifies that the function properly formats:
    - Simple types (int, str, float, bool) as <type>
    - Literal types as pipe-separated values
    - Mixed literal types with proper string conversion
    - Single literal values

    Parameters
    ----------
    field_type : type or Literal
        The type to format
    expected : str
        Expected formatted output
    """
    assert format_type_for_prompt(field_type) == expected
