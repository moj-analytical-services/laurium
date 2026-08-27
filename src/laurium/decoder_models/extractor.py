"""Labeller class for annotating text using an LLM."""

from typing import Any

import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from laurium.decoder_models import prompts, pydantic_models


class Extractor:
    def __init__(
        self,
        schema: dict[str, tuple[Any, str]],
        llm: dict[str, Any] | BaseChatModel,
        prompt: str = (
            "You are an expert annotator. Extract information from the text "
            "following the schema provided below."
        ),
        **prompt_kwargs: dict[str, Any],
    ):
        # Set up schema and Pydantic model
        self.schema_dtypes = {
            key: field_type for key, (field_type, _) in schema.items()
        }
        self.schema_desc = {key: desc for key, (_, desc) in schema.items()}
        self.pydantic_model = pydantic_models.make_dynamic_example_model(
            schema=self.schema_dtypes,
            descriptions=self.schema_desc,
            model_name="DynamicExampleModel",
        )

        # Set up LLM
        if isinstance(llm, dict):
            from laurium.decoder_models.llm import create_llm

            self.llm = create_llm(**llm)
        elif isinstance(llm, BaseChatModel):
            self.llm = llm
        else:
            raise ValueError(
                "llm must be either a dict or an instance of BaseChatModel"
            )

        # Set up prompt
        self.prompt = self._create_prompt(prompt, **prompt_kwargs)

        # Create parser
        self.parser = PydanticOutputParser(pydantic_object=self.pydantic_model)

        # Create chain
        self.chain = (
            {"text": lambda x: x} | self.prompt | self.llm | self.parser | dict
        )

    def _create_prompt(
        self, prompt: str, **prompt_kwargs: dict[str, Any]
    ) -> str:
        system_message = prompts.create_system_message(
            base_message=prompt,
            keywords=prompt_kwargs.pop("keywords"),
        )

        return prompts.create_prompt(
            system_message=system_message,
            final_query="Analyze this text: {text}",
            schema=self.schema_dtypes,
            descriptions=self.schema_desc,
            **prompt_kwargs,
        )

    def label(self, text: str) -> dict:
        return self.chain.invoke(text)

    def batch_label(
        self,
        df: pd.DataFrame,
        text_column: str,
        max_concurrency: int = 8,
    ) -> pd.DataFrame:
        texts = df[text_column].tolist()

        batch_results = self.chain.batch(
            texts,
            {"max_concurrency": max_concurrency},
        )

        results_df = pd.DataFrame(batch_results)
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)
