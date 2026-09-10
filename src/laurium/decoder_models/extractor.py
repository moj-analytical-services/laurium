"""Labeller class for annotating text using an LLM."""

from typing import Any

import pandas as pd
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from laurium.decoder_models import prompts, pydantic_models


class Extractor:
    """
    AI extractor for extracting structured information from text.

    Parameters
    ----------
    schema : dict[str, tuple[Any, str]]
        A dictionary defining the desired output from the model, where
        each key is a field name, and the value is a tuple containing
        the field type and its description.
    llm : dict[str, Any] | BaseChatModel
        Either a dictionary of parameters to create an LLM instance or
        a pre-configured language model instance (see
        `laurium.decoder_models.llm.create_llm`).
    prompt : str, optional
        The base prompt to use for the extraction task.
        Default is "You are an expert annotator. Annotate the following
        text."
    **prompt_kwargs : dict[str, Any]
        Additional arguments to customize the prompt creation, such as
        keywords for the system message.
    """

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

        # Create null result template (for failed extractions)
        self.failed_result_template = {
            key: pd.NA for key in self.schema_dtypes
        }

    def _create_prompt(
        self, prompt: str, **prompt_kwargs: dict[str, Any]
    ) -> str:
        """
        Build the prompt for the labelling task.

        This takes the base prompt, along with any additional arguments
        (such as keywords), and puts together a complete prompt to
        provide context to the language model for the labelling task.

        Parameters
        ----------
        prompt : str
            The base prompt to use for the labelling task.
        **prompt_kwargs : dict[str, Any]
            Additional arguments to customize the prompt creation, such
            as keywords for the system message.

        Returns
        -------
        str
            The fully constructed prompt.
        """
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
        """
        Label a single piece of text.

        Parameters
        ----------
        text : str
            The text to be labelled.

        Returns
        -------
        dict
            The output label as a dictionary.
        """
        return self.chain.invoke(text)

    def batch_label(
        self,
        df: pd.DataFrame,
        text_column: str,
        max_concurrency: int = 8,
        max_retries: int = 3,
        ignore_errors: bool = False,
    ) -> pd.DataFrame:
        """
        Batch label a column of text in a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame containing the texts to be labelled.
        text_column : str
            The name of the column to be labelled.
        max_concurrency : int, optional
            The maximum number of concurrent requests to the language
            model. Default is 8.
        max_retries : int, optional
            The maximum number of retries for failed requests.
            Default is 3.
        ignore_errors : bool, optional
            Whether to ignore errors (other than parsing errors) during
            batch processing. Default is False.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with additional columns corresponding
            to the labelled outputs for each record.
        """
        texts = df[text_column].tolist()

        batch_results = self.chain.batch(
            texts,
            {"max_concurrency": max_concurrency},
            return_exceptions=True,
        )

        # Retry failed requests up to max_retries times
        for _ in range(max_retries):
            # Pull out indices of failed results
            failed_indices = []
            for idx, result in enumerate(batch_results):
                if isinstance(result, dict):
                    # Success
                    continue
                if isinstance(result, OutputParserException):
                    # Ill-formatted output, select for retry
                    failed_indices.append(idx)
                    continue

                # If we're here, we received some other type of error
                if not ignore_errors:
                    raise result
                failed_indices.append(idx)  # If ignoring errors, retry index

            if not failed_indices:
                # No failed results, exit the retry loop
                break

            # Retry failed requests
            retry_texts = [texts[idx] for idx in failed_indices]
            retry_results = self.chain.batch(
                retry_texts,
                {"max_concurrency": max_concurrency},
                return_exceptions=True,
            )

            for idx, result in zip(failed_indices, retry_results, strict=True):
                batch_results[idx] = result

        # Exhausted retries, fill in any remaining failures with null result
        for idx, result in enumerate(batch_results):
            if not isinstance(result, dict):
                batch_results[idx] = self.failed_result_template.copy()

        results_df = pd.DataFrame(
            batch_results, columns=self.schema_dtypes
        ).convert_dtypes()
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)
