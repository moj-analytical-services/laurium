"""Test LLM functions."""

import os

import pytest
import requests

from laurium.decoder_models.llm import create_llm


def is_ollama_available(api_url: str = "http://localhost:11434/v1") -> bool:
    """Check whether the Ollama server is active."""
    try:
        response = requests.get(api_url[:-3])
    except requests.exceptions.ConnectionError:
        return False
    return response.status_code == 200


@pytest.mark.parametrize(
    "args",
    [
        pytest.param(
            {
                "llm_platform": "bedrock",
                "model_name": "anthropic.claude-3-haiku-20240307-v1:0",
            },
            marks=pytest.mark.skipif(
                not os.environ.get("AWS_ROLE_ARN"),
                reason="No AWS credentials found",
            ),
        ),
        pytest.param(
            {
                "llm_platform": "bedrock",
                "model_name": (
                    "arn:aws:bedrock:eu-west-1:593291632749:inference-profile/"
                    "eu.meta.llama3-2-3b-instruct-v1:0"
                ),
                "aws_model_family": "meta",
            },
            marks=pytest.mark.skipif(
                not os.environ.get("AWS_ROLE_ARN"),
                reason="No AWS credentials found",
            ),
        ),
        pytest.param(
            {
                "llm_platform": "ollama",
                "model_name": "qwen2.5",
            },
            marks=pytest.mark.skipif(
                not is_ollama_available(),
                reason="No ollama instance available",
            ),
        ),
    ],
)
def test_create_llm(args):
    """Test the `create_llm` function for a range of models and providers."""
    llm = create_llm(**args)
    # Based on https://python.langchain.com/docs/integrations/chat/bedrock/
    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French."
            "Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]
    ai_msg = llm.invoke(messages)
    assert isinstance(ai_msg.content, str)
    assert ai_msg.content != ""
