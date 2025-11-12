"""Module for creating and configuring LLM instances."""

import boto3
import botocore
from botocore.exceptions import ClientError
from langchain_anthropic import ChatAnthropic
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def list_bedrock_models(region_name: str = "eu-west-1") -> list[dict]:
    """Retrieve a list of available foundation models from AWS Bedrock.

    Parameters
    ----------
    region_name : str
        The AWS region to check for models.

    Returns
    -------
    list[dict]
        List of model summaries, each containing model details

    Raises
    ------
    ClientError
        If there's an error communicating with AWS Bedrock
    """
    try:
        bedrock_client = boto3.client(
            service_name="bedrock",
            region_name=region_name,
        )
        response = bedrock_client.list_foundation_models()
        models = response["modelSummaries"]
        return models
    except ClientError as e:
        raise ClientError(
            error_response={
                "Error": {
                    "Message": f"Failed to list Bedrock models: {str(e)}"
                }
            },
            operation_name="list_bedrock_models",
        ) from e


def create_llm(
    llm_platform: str = "ollama",
    model_name: str = "qwen2",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    api_key: str | None = None,
    aws_region_name: str = "eu-west-1",
    aws_model_family: str = "anthropic",
    aws_max_pool_connections: int = 20,
) -> BaseChatModel:
    """Create and configure an LLM instance based on the specified provider.

    This class provides a unified interface for creating chat-based LLM
    instances from different providers (Ollama, Anthropic, OpenAI, AWS
    Bedrock).

    Parameters
    ----------
    llm_platform : str, optional
        The LLM provider to use ('ollama', 'bedrock', 'anthropic', 'openai'),
        by default 'ollama'
    model_name : str, optional
        Name of the specific model to use. By default 'qwen2'.
    temperature : float, optional
        Temperature parameter for controlling randomness in LLM responses, by
        default 0.0
    max_tokens : int, optional
        The maximum number of response tokens to generate. Default is 2048.
    api_key : str, optional
        API key for cloud providers (required for OpenAI and Anthropic)
    aws_region_name : str, optional
        AWS region for Bedrock, by default 'eu-west-1'
    aws_model_family : str, optional
        AWS Bedrock model provider, needed for models identified by ARN.
        Default is 'anthropic'.
    aws_max_pool_connections: int, optional
        The maximum number of connections to make to AWS Bedrock.
        Default is 20.

    Returns
    -------
    BaseChatModel
        Configured chat model instance from the specified provider

    Raises
    ------
    ValueError
        If an unsupported provider is specified
    ClientError
        If there's an error creating a Bedrock chat model
    """
    model_kwargs = {"temperature": temperature}

    if llm_platform == "ollama":
        return ChatOllama(model=model_name, **model_kwargs)

    elif llm_platform == "bedrock":
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=aws_region_name,
            config=botocore.client.Config(
                max_pool_connections=aws_max_pool_connections,
            ),
        )
        model_kwargs["max_tokens"] = max_tokens

        try:
            return ChatBedrock(
                provider=aws_model_family,
                model_id=model_name,
                client=bedrock_runtime,
                model_kwargs=model_kwargs,
            )
        except ClientError as e:
            raise ClientError(
                error_response={
                    "Error": {
                        "Message": (
                            f"Failed to create Bedrock chat model: {str(e)}"
                        )
                    }
                },
                operation_name="create_llm_chatbedrock",
            ) from e

    elif llm_platform == "anthropic":
        model_kwargs["anthropic_api_key"] = api_key
        return ChatAnthropic(model=model_name, **model_kwargs)
    elif llm_platform == "openai":
        model_kwargs["openai_api_key"] = api_key
        return ChatOpenAI(model=model_name, **model_kwargs)
    else:
        raise NotImplementedError(f"Unsupported provider: {llm_platform}")
