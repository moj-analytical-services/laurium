"""Train setfit model."""

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments


@dataclass
class ColumnMapping:
    """Dataclass for column_mapping validation."""

    mapping: dict

    @classmethod
    def from_dict(cls, data: dict):
        """Validate values from dict and returns the dict."""
        allowed_values = {"text", "label"}
        values = set(data.values())
        if not values == allowed_values:
            raise ValueError(
                "Invalid value(s): Only 'text' and 'label' allowed."
            )
        return data


class SetFit:
    """Setfit model trainer."""

    def __init__(
        self,
        model_init: dict[str, Any],
        training_args: dict[str, Any],
        metric: Callable | str = "accuracy",
        mapping: dict[str, str] = None,
    ):
        """Initialise setfit pipeline.

        Parameters
        ----------
        model_init : dict[str, Any]
            Arguments for model initialization including model name/path passed to SetFitModel.
        training_args : dict[str, Any]
            Training arguments passed to TrainingArguments constructor to
            configure the training process.
        metrics : Callable | str, default = "accuracy"
            metrics to use for evaluating setfit model

        """
        self.model_args = model_init
        self.model = self._create_model()
        self.training_args = TrainingArguments(**training_args)
        self.metric = metric

        if mapping:
            self.mapping = ColumnMapping.from_dict(mapping)
        else:
            self.mapping = mapping

    def _create_model(self, params=None) -> SetFitModel:
        """
        Create a model instance with the stored configuration.

        This abstraction is needed as a fresh instance of a model is needed
        for hyperparameter tuning.

        `params` is passed by hp search, can be ignored if not used.

        Returns
        -------
        SetFitModel
            Model instance.
        """
        return SetFitModel.from_pretrained(**self.model_args)

    def create_setfit_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        model_init_fn: Callable[[], SetFitModel] | None = None,
    ) -> Trainer:
        """Create trainer for setfit.

        Parameters
        ----------
        train_dataset: Dataset
            training dataset used for training.
        eval_dataset: Dataset
            validation dataset for training
        model_init_fn : Callable, optional
            Model initialization function for hyperparameter search.
            If provided, model=None will be set in Trainer.
        """
        if model_init_fn is not None:
            return Trainer(
                model=None,
                model_init=model_init_fn,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                metric=self.metric,
                column_mapping=self.mapping,
            )
        else:
            return Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                metric=self.metric,
                column_mapping=self.mapping,
            )

    def setfit_model_train(
        self, train_df: pd.DataFrame, eval_df: pd.DataFrame
    ):
        """Train model on datasets.

        Parameters
        ----------
        train_df: pd.DataFrame
            dataframe containing training data
        eval_df: pd.DataFrame
            dataframe containing eval data
        """
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)
        trainer = self.create_setfit_trainer(train_dataset, eval_dataset)
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
        train_dataset = Dataset.from_pandas(train_df)
        eval_dataset = Dataset.from_pandas(eval_df)

        return self.create_setfit_trainer(
            train_dataset,
            eval_dataset,
            model_init_fn=self._create_model,
        )
