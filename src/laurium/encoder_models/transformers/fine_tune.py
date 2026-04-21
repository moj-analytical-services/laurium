"""
Fine-tuning sub-module.

This module provides functionality for fine-tuning transformer models using
the HuggingFace transformers library for natural language inference (NLI)
and text classification tasks.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from laurium.components.evaluate_metrics import compute_metrics


@dataclass
class DataConfig:
    """Configuration for dataset preparation.

    Parameters
    ----------
    text_column : str | None, default=None
        Name of the column containing the input text for single-text
        classification.
    label_column : str, default="label"
        Name of the column containing the labels.
    premise_column : str | None, default=None
        For NLI tasks, name of the column containing the premise.
    hypothesis_column : str | None, default=None
        For NLI tasks, name of the column containing the hypothesis.

    Raises
    ------
        ValueError
            If neither text_column nor premise/hypothesis columns are specified
            in data_config.
    """

    text_column: str | None = None
    label_column: str = "label"
    premise_column: str | None = None
    hypothesis_column: str | None = None

    def __post_init__(self):
        """Validate data_config after initialization."""
        if not (
            self.text_column
            or (self.premise_column and self.hypothesis_column)
        ):
            raise ValueError(
                "Either text_column or premise_column/hypothesis_column must "
                "be specified in data_config"
            )


class FineTuner:
    """Fine tune transformer model."""

    def __init__(
        self,
        metrics: list[str],
        model_init: dict[str, Any],
        training_args: dict[str, Any],
        tokenizer_init: dict[str, Any],
        tokenizer_args: dict[str, Any],
        data_config: DataConfig,
        peft_config: LoraConfig | None = None,
    ):
        """
        Initialize a fine-tuning pipeline for NLI or text classification tasks.

        Parameters
        ----------
        metrics: list[str]
            List of metrics for evaluation
        model_init : dict[str, Any]
            Arguments for model initialization including model name/path and
            other parameters passed to
            AutoModelForSequenceClassification.from_pretrained().
        training_args : dict[str, Any]
            Training arguments passed to TrainingArguments constructor to
            configure the training process.
        tokenizer_init : dict[str, Any]
            Arguments for tokenizer initialization passed to
            AutoTokenizer.from_pretrained().
        tokenizer_args : dict[str, Any]
            Arguments for tokenizer processing (padding, truncation, etc.) used
            during dataset preparation.
        data_config : DataConfig
            Configuration for dataset preparation specifying column names.
        peft_config : LoraConfig | None, default=None
            If provided, converts the model to PEFT with LoRA for
            parameter-efficient fine-tuning.
        """
        self.metrics = metrics

        # Store configurations for model creation
        self.model_init_args = model_init
        self.peft_config = peft_config

        # Initialize model, tokenizer and data collector
        self.model = self._create_model()
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_init)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # store configurations
        self.tokenizer_args = tokenizer_args
        self.data_config = data_config
        self.training_args = TrainingArguments(**training_args)

    def _create_model(self) -> PreTrainedModel:
        """
        Create a model instance with the stored configuration.

        This abstraction is needed as a fresh instance of a model is needed
        for hyperparameter tuning.

        Returns
        -------
        PreTrainedModel
            Model instance with PEFT applied if configured.
        """
        model = AutoModelForSequenceClassification.from_pretrained(
            **self.model_init_args
        )
        if self.peft_config is not None:
            model = get_peft_model(model, self.peft_config)
        return model

    def process_dataframe_to_tokenized_dataset(
        self, df: pd.DataFrame
    ) -> Dataset:
        """
        Process a pandas DataFrame into a tokenized HuggingFace Dataset.

        This function converts the DataFrame into a Dataset and tokenizes it.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing text and labels.

        Returns
        -------
        Dataset
            Prepared and tokenized dataset ready for training/evaluation.

        Raises
        ------
        ValueError
            If the input DataFrame is None.
        """
        if df is None:
            raise ValueError(
                "Input DataFrame cannot be None. Please provide valid data "
                "for tokenization."
            )

        return self.tokenize_dataset(Dataset.from_pandas(df))

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset based on configuration.

        Parameters
        ----------
        dataset : Dataset
            HuggingFace dataset to tokenize.

        Returns
        -------
        Dataset
            Tokenized dataset with input_ids, attention_mask, and other
            tokenizer outputs.

        Raises
        ------
        ValueError
            If neither text_column nor premise/hypothesis columns are specified
            in data_config.
        """
        if (
            self.data_config.premise_column
            and self.data_config.hypothesis_column
        ):
            return self.tokenize_nli_task(dataset)
        elif self.data_config.text_column:
            return self.tokenize_single_text(dataset)
        raise ValueError(
            "Either text_column or premise_column/hypothesis_column must be "
            "specified in data_config"
        )

    def tokenize_nli_task(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for NLI task with premise and hypothesis pairs.

        Parameters
        ----------
        dataset : Dataset
            Input dataset containing premise and hypothesis columns.

        Returns
        -------
        Dataset
            Tokenized dataset with input_ids, attention_mask, and other
            tokenizer outputs.
        """
        return dataset.map(
            lambda x: self.tokenizer(
                x[self.data_config.premise_column],
                x[self.data_config.hypothesis_column],
                **self.tokenizer_args,
            ),
            batched=True,
        )

    def tokenize_single_text(self, dataset: Dataset) -> Dataset:
        """Tokenize dataset for single text classification.

        Parameters
        ----------
        dataset : Dataset
            Input dataset containing a text column.

        Returns
        -------
        Dataset
            Tokenized dataset with input_ids, attention_mask, and other
            tokenizer outputs.
        """
        return dataset.map(
            lambda x: self.tokenizer(
                x[self.data_config.text_column],
                **self.tokenizer_args,
            ),
            batched=True,
        )

    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None,
        model_init_fn: Callable[[], PreTrainedModel] | None = None,
    ) -> Trainer:
        """Create a HuggingFace Trainer instance.

        Parameters
        ----------
        train_dataset : Dataset
            Tokenized training dataset.
        eval_dataset : Dataset | None
            Tokenized evaluation dataset, or None if not available.
        model_init_fn : callable, optional
            Model initialization function for hyperparameter search.
            If provided, model=None will be set in Trainer.

        Returns
        -------
        Trainer
            Configured trainer instance ready for training or evaluation.
        """
        if model_init_fn is not None:
            return Trainer(
                model=None,
                model_init=model_init_fn,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_collator,
                compute_metrics=partial(compute_metrics, metrics=self.metrics),
            )
        else:
            return Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=self.data_collator,
                compute_metrics=partial(compute_metrics, metrics=self.metrics),
            )

    def fine_tune_model(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> Trainer:
        """Train the model on the provided datasets.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data as a pandas DataFrame.
        eval_df : pd.DataFrame | None, default=None
            Evaluation data as a pandas DataFrame, or None if not available.

        Returns
        -------
        Trainer
            Fine-tuned trainer instance ready for evaluation.
        """
        train_dataset = self.process_dataframe_to_tokenized_dataset(train_df)
        if eval_df is None:
            if self.training_args.eval_strategy.lower() != "no":
                # Must provide an eval dataset if eval strategy specified
                raise ValueError(
                    "eval_strategy='no' in training_args if eval_df provided"
                )
            eval_dataset = None
        else:
            eval_dataset = self.process_dataframe_to_tokenized_dataset(eval_df)
        trainer = self.create_trainer(train_dataset, eval_dataset)
        trainer.train()
        return trainer

    def create_trainer_for_search(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
    ) -> Trainer:
        """Create a trainer configured for hyperparameter search.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training data as a pandas DataFrame.
        eval_df : pd.DataFrame
            Evaluation data as a pandas DataFrame.

        Returns
        -------
        Trainer
            Trainer instance ready for hyperparameter search.
        """
        train_dataset = self.process_dataframe_to_tokenized_dataset(train_df)
        eval_dataset = self.process_dataframe_to_tokenized_dataset(eval_df)

        return self.create_trainer(
            train_dataset,
            eval_dataset,
            model_init_fn=self._create_model,
        )
