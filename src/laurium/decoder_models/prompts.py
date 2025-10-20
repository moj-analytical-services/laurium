"""Module for creating and managing prompts for extraction tasks."""

import typing

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from pydantic import BaseModel as Example


def format_examples(examples: list[Example] | None = None) -> list[dict]:
    """Convert Example objects to format needed for few-shot learning.

    Transforms the list of Example objects into a list of dictionaries
    suitable for few-shot learning templates.

    Parameters
    ----------
    examples : list[Example] | None
        List of few-shot examples for demonstration, by default None

    Returns
    -------
    list[dict]
        List of dictionaries containing formatted examples
    """
    return [ex.model_dump() for ex in examples]


def create_system_message(
    base_message: str, keywords: list[str] | None
) -> str:
    """Create the system message with optional keywords.

    Combines the base system message with keywords if they exist.

    Parameters
    ----------
    base_message : str
        Base message for the system prompt

    keywords : list[str]
        List of keywords to include in prompt

    Returns
    -------
    str
        Complete system message including keywords if provided
    """
    keywords_text = (
        f"\nPay special attention to these keywords: {', '.join(keywords)}"
        if keywords
        else ""
    )
    return f"{base_message}{keywords_text}"


def format_type_for_prompt(field_type: typing.Any):
    """Format a type for display in prompt template.

    Parameters
    ----------
    field_type : typing.Any
        The type to format (int, str, Literal, etc.)

    Returns
    -------
    str
        Formatted type string for prompt display
    """
    if typing.get_origin(field_type) is typing.Literal:
        args = typing.get_args(field_type)
        return "|".join(
            f'"{arg}"' if isinstance(arg, str) else str(arg) for arg in args
        )

    return f"<{field_type.__name__}>"


def format_schema_for_prompt(
    schema: dict[str, typing.Any],
    descriptions: dict[str, str],
) -> str:
    """Format schema and descriptions for inclusion in prompt.

    Creates a clear, LLM-friendly format that separates field descriptions
    from the expected JSON output structure with data types or literal values.

    Parameters
    ----------
    schema : dict[str, typing.Any]
        Dictionary mapping field names to their types (int, str, Literal etc)
    descriptions : dict[str, str]
        Dictionary mapping field names to their descriptions

    Returns
    -------
    str
        Formatted string with field descriptions and JSON structure with types

    Examples
    --------
    >>> from typing import Literal
    >>> schema = {"sentiment": Literal["positive", "negative"],
    ...           "urgency": Literal[1, 2, 3, 4, 5]}
    >>> descriptions = {"sentiment": "Customer's emotional tone",
    ...                 "urgency": "Priority level 1-5"}
    >>> result = format_schema_for_prompt(schema, descriptions)
    >>> print(result)
    For each text, extract:
    - sentiment: Customer's emotional tone
    - urgency: Priority level 1-5

    Expected output format:
    {{
        "sentiment": "positive"|"negative",
        "urgency": 1|2|3|4|5
    }}
    """
    # Create field descriptions and type mappings in one loop
    field_descriptions = []
    type_mappings = []

    for field_name, field_type in schema.items():
        # Build description
        if description := descriptions.get(field_name):
            field_descriptions.append(f"    - {field_name}: {description}")
        else:
            field_descriptions.append(f"    - {field_name}")

        # Build type mapping
        formatted_type = format_type_for_prompt(field_type)
        type_mappings.append(f'    "{field_name}": {formatted_type}')

    descriptions_text = "For each text, extract:\n" + "\n".join(
        field_descriptions
    )

    json_format = (
        "Expected output format:\n{{\n" + ",\n".join(type_mappings) + "\n}}"
    )

    return f"{descriptions_text}\n\n{json_format}"


def create_prompt(
    system_message: str,
    examples: list[dict],
    example_human_template: str,
    example_assistant_template: str,
    final_query: str,
    schema: dict[str, typing.Any] | None = None,
    descriptions: dict[str, str] | None = None,
) -> ChatPromptTemplate:
    """Create the complete chat prompt template.

    Parameters
    ----------
    system_message : str
        System message for the prompt
    examples : list[dict]
        List of example dictionaries containing structured example of
        interaction between human and assistant
    example_human_template : str
        Human message template for the example prompt
    example_assistant_template : str
        Assistant message template for the example prompt
    final_query : str
        Final query message for the prompt
    schema : dict[str, typing.Any] | None, optional
        Dictionary mapping field names to their types (simple types or
        Literal), by default None
    descriptions : dict[str, str] | None, optional
        Dictionary mapping field names to their descriptions,
        by default None

    Returns
    -------
    ChatPromptTemplate
        Complete chat template ready for use with system message, optional
        examples, and final query

    Notes
    -----
    The function combines a system message, optional few-shot examples, and a
    final query into a complete chat prompt template. If examples are provided,
    they will be included as few-shot examples in the final template.

    If schema and descriptions are provided, they will be automatically
    formatted and appended to the system message to provide field extraction
    guidance and expected JSON output format to the LLM.
    """
    # Add schema formatting to system message if provided
    if schema is not None and descriptions is not None:
        schema_text = format_schema_for_prompt(schema, descriptions)
        system_message = f"{system_message}\n\n{schema_text}"

    messages = [("system", system_message)]

    if examples:
        messages.append(
            FewShotChatMessagePromptTemplate(
                examples=examples,
                example_prompt=ChatPromptTemplate.from_messages(
                    [
                        ("human", example_human_template),
                        ("assistant", example_assistant_template),
                    ]
                ),
            )
        )

    messages.append(("human", final_query))

    return ChatPromptTemplate.from_messages(messages)
