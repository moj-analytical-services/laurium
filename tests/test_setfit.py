"""Pytest for setfit script."""

import pandas as pd
import pytest
from datasets import Dataset

from laurium.components.setfit_eval_metrics import compute_metrics
from laurium.encoder_models.setfit.setfit import SetFit


@pytest.fixture
def train_eval_data():
    """Define data for both train and eval sets."""
    return pd.DataFrame(
        {
            "text_col": ["This is a Positive text", "This is a Negative text"],
            "label_col": [1, 0],
        }
    )


@pytest.fixture
def set_fit_trainer(train_eval_data):
    """Initialise setfit trainer."""
    setfit_model_init = {
        "pretrained_model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_differentiable_head": False,
    }

    # Training arguments for parameter tuning
    setfit_training_args = {
        "sampling_strategy": "oversampling",
        "max_steps": -1,
        "end_to_end": False,
        "head_learning_rate": 1e-2,
        "body_learning_rate": [2e-5, 1e-5],
        "num_epochs": [1, 16],
        "logging_steps": 50,
        "eval_steps": 50,
        "save_steps": 100,
        "metric_for_best_model": "eval_embedding_loss",
    }

    # Initialize fine-tuner
    return SetFit(
        metric=compute_metrics,
        model_init=setfit_model_init,
        training_args=setfit_training_args,
        mapping={
            train_eval_data.columns[0]: "text",
            train_eval_data.columns[1]: "label",
        },
    )


@pytest.fixture
def set_fit_trainer_no_eval(train_eval_data):
    """Initialise setfit trainer without eval."""
    setfit_model_init = {
        "pretrained_model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_differentiable_head": False,
    }

    # Training arguments for parameter tuning
    setfit_training_args = {
        "sampling_strategy": "oversampling",
        "max_steps": -1,
        "end_to_end": False,
        "head_learning_rate": 1e-2,
        "body_learning_rate": [2e-5, 1e-5],
        "num_epochs": [1, 16],
        "load_best_model_at_end": False,
        "eval_strategy": "no",
    }

    # Initialize fine-tuner
    return SetFit(
        metric=compute_metrics,
        model_init=setfit_model_init,
        training_args=setfit_training_args,
        mapping={
            train_eval_data.columns[0]: "text",
            train_eval_data.columns[1]: "label",
        },
    )


def test_create_trainer_for_search(set_fit_trainer, train_eval_data):
    """Tests the create_trainer_for_search function creates trainer correctly.

    Parameters
    ----------
    set_fit_trainer: SetFit
        Initialisation of SetFit class.
    """
    trainer = set_fit_trainer.create_trainer_for_search(
        train_eval_data, train_eval_data
    )

    assert trainer.model_init is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None


def test_create_setfit_regular(set_fit_trainer, train_eval_data):
    """Tests the regular create_setfit_trainer function works as before.

    Parameters
    ----------
    set_fit_trainer: SetFit
        Initialisation of SetFit class.
    """
    train_eval_dataset = Dataset.from_pandas(train_eval_data)

    trainer = set_fit_trainer.create_setfit_trainer(
        train_eval_dataset, train_eval_dataset
    )
    assert trainer.model is not None
    assert trainer.model_init is None


def test_setfit_no_eval(set_fit_trainer_no_eval, train_eval_data):
    """Tests the regular create_setfit_trainer function works without eval dataset.

    Parameters
    ----------
    set_fit_trainer_no_eval: SetFit
        Initialisation of SetFit class without eval dataset.
    """
    train_eval_dataset = Dataset.from_pandas(train_eval_data)

    trainer = set_fit_trainer_no_eval.create_setfit_trainer(train_eval_dataset)
    assert trainer.model is not None
    assert trainer.model_init is None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is None
