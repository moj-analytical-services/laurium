"""Module for batch processing text samples with LLM extraction."""

import logging

import pandas as pd
from httpx import ConnectError
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate


class BatchExtractor:
    """A class for batch processing text samples using LLM extraction.

    This class handles the batch processing of text samples through a language
    model and fallback to individual processing when batch processing fails.

    Parameters
    ----------
    llm : BaseChatModel
        The language model to use for extraction.
    prompt : ChatPromptTemplate
        The prompt template for the extraction task.
    parser : PydanticOutputParser
        Parser for structured output.
    batch_size : int, optional
        The number of samples to process in each batch, default is 1000.
    max_concurrency : int, optional
        Maximum number of concurrent operations, default is 5.
    max_retries : int, optional
        Maximum number of retries for fallback individual processing,
        must be at least 1, default is 1.

    Attributes
    ----------
    batch_size : int
        The size of batches for processing.
    max_concurrency : int
        Maximum number of concurrent operations allowed.
    max_retries : int
        Maximum number of retries for fallback individual processing.
    prompt : ChatPromptTemplate
        The configured prompt template.
    logger : logging.Logger
        Logger instance for the class.
    parser : PydanticOutputParser
        Parser for structured output.
    chain : Chain
        The processing chain combining prompt, LLM, and parser.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        parser: PydanticOutputParser,
        batch_size: int = 1000,
        max_concurrency: int = 5,
        max_retries: int = 1,
    ):
        """Initialize the BatchExtractor.

        Parameters
        ----------
        llm : BaseChatModel
            The language model to use for extraction tasks.
        prompt : ChatPromptTemplate
            The prompt template that defines the extraction task.
        parser : PydanticOutputParser
            Parser for structured output.
        batch_size : int, optional
            Number of samples to process in each batch, default is 1000.
        max_concurrency : int, optional
            Maximum number of concurrent operations allowed, default is 5.
        max_retries : int, optional
            Maximum number of retries for fallback individual processing,
            must be at least 1, default is 1.

        Notes
        -----
        The initializer sets up the processing chain combining the prompt
        template, language model, and Pydantic output parser for structured
        data extraction.
        """
        if max_retries < 1:
            raise ValueError(
                f"max_retries must be at least 1, got {max_retries}. "
            )

        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.prompt = prompt
        self.logger = logging.getLogger(__name__)
        self.parser = parser
        self.chain = {"text": lambda x: x} | prompt | llm | parser

    def _process_batch(self, texts: list[str]) -> list[dict]:
        """
        Process a batch of texts with fallback to individual processing.

        Parameters
        ----------
        texts : list[str]
            List of text samples to process.

        Returns
        -------
        list[dict]
            A list of dictionaries containing extracted attributes from each
            text example.

        Notes
        -----
        If batch processing fails, the method falls back to individual
        processing for failed items while preserving successful results.
        """
        # Initialises a dictionary with all fields set to None for failed
        # extractions
        failure_result = {
            field: None for field in self.parser.pydantic_object.model_fields
        }
        try:
            # First attempt: Process entire batch
            batch_results = self.chain.batch(
                [{"text": text} for text in texts],
                {"max_concurrency": self.max_concurrency},
            )
            processed_results = []
            for result in batch_results:
                # If processing successful unpack dictionary of attributes
                # otherwise set fields to None
                if result:
                    processed_results.append(result.__dict__)
                else:
                    processed_results.append(failure_result)
            # Check if batch processing was fully successful
            if not any(
                result is failure_result for result in processed_results
            ):
                return processed_results

        except OutputParserException as e:
            self.logger.error(f"Batch processing failed completely: {str(e)}")
            processed_results = [failure_result] * len(texts)
        except ConnectError as e:
            raise RuntimeError(
                "Could not connect to Ollama - have you run `ollama serve`?"
            ) from e

        # Individual processing for failed items
        for i, text in enumerate(texts):
            if (
                processed_results[i] is failure_result
            ):  # Only process failed items
                for attempt in range(self.max_retries):
                    try:
                        result = self.chain.invoke({"text": text})
                        if result:
                            processed_results[i] = result.__dict__
                            break

                    except Exception as e:
                        if attempt == (self.max_retries - 1):
                            self.logger.error(
                                f"Individual processing failed for text {i} "
                                f"after {self.max_retries} attempts: {str(e)}"
                            )

                        else:
                            self.logger.warning(
                                f"Attempt {attempt + 1}/{self.max_retries} "
                                f"failed for text {i}: {str(e)}. Retrying..."
                            )

        return processed_results

    def _process_texts(self, texts: list[str]) -> tuple[list]:
        """Process a list of texts in batches.

        Parameters
        ----------
        texts : list[str]
            List of text samples to process.

        Returns
        -------
        list[dict]
            A list containing dictionaries of extracted attributes
        """
        all_results = []
        # Create batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            results = self._process_batch(batch)
            all_results.extend(results)
        return all_results

    def process_chunk(
        self,
        chunk_df: pd.DataFrame,
        text_column: str,
    ) -> pd.DataFrame:
        """Process a single chunk of data and return the processed DataFrame.

        Parameters
        ----------
        chunk_df : pd.DataFrame
            Input DataFrame containing the text samples.
        text_column : str
            Name of the column containing text samples.

        Returns
        -------
        pd.DataFrame
            A copy of the input DataFrame with additional columns for
            structured output attributes. Column names are the parser field
            names

        Notes
        -----
        The method preserves all original columns and adds or updates
        the specified output attribute columns.
        """
        texts = chunk_df[text_column].tolist()
        extracted_results = self._process_texts(texts)

        # Create a DataFrame from the extracted results and combine with the
        # original input DataFrame
        extracted_results_df = pd.DataFrame(extracted_results)
        merged_df = pd.concat([chunk_df, extracted_results_df], axis=1)
        return merged_df
