"""Module for creating and managing prompts for extraction tasks."""

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


def create_prompt(
    system_message: str,
    examples: list[dict],
    example_human_template: str,
    example_assistant_template: str,
    final_query: str,
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
    """
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
