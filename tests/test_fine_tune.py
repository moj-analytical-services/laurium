"""Tests for fine tuning scripts."""

from dataclasses import is_dataclass

import pandas as pd
import pytest
from datasets import Dataset
from peft import LoraConfig, TaskType

from laurium.encoder_models.fine_tune import DataConfig, FineTuner


@pytest.fixture
def data_config_text():
    """Return data config class with label and text columns."""
    return DataConfig(text_column="text", label_column="label")


@pytest.fixture
def data_config_pre_hyp():
    """Return data config class with label, hypothesis and premise columns."""
    return DataConfig(
        label_column="label",
        premise_column="premise",
        hypothesis_column="hypothesis",
    )


@pytest.fixture
def def_finetuner(data_config_text, data_config_pre_hyp):
    """Define and return FineTuner class with/without peft and validation."""
    model_init = {
        "pretrained_model_name_or_path": "distilbert/distilbert-base-uncased",
    }

    tokenizer_init = {
        "pretrained_model_name_or_path": "distilbert/distilbert-base-uncased",
        "num_labels": 3,
    }

    tokenizer_args = {
        "max_length": 128,
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": "longest_first",
    }

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification
        r=8,  # Rank
        lora_alpha=32,  # Alpha parameter for LoRA
        lora_dropout=0.1,  # Dropout probability for LoRA layers
        # Target the correct modules for DebertaV2
        target_modules=["q_lin", "k_lin", "v_lin"],
        bias="none",  # Don't train bias parameters
        modules_to_save=["classifier"],  # Save the classifier layer too
    )

    metrics = ["f1"]

    def __finetuner(peft=True, do_eval=True):
        """Allow for return with the option to include peft/validation."""
        training_args = {
            "output_dir": "./results",
            "learning_rate": 2e-5,
            "per_device_train_batch_size": 16,
            "num_train_epochs": 1,
            "weight_decay": 0.01,
            "save_strategy": "epoch",
            "report_to": "none",
            "eval_strategy": "no",
        }
        if do_eval:
            training_args["per_device_eval_batch_size"] = 16
            training_args["eval_strategy"] = "epoch"

        if peft:
            return FineTuner(
                metrics,
                model_init,
                training_args,
                tokenizer_init,
                tokenizer_args,
                data_config_text,
                peft_config,
            )
        else:
            return FineTuner(
                metrics,
                model_init,
                training_args,
                tokenizer_init,
                tokenizer_args,
                data_config_pre_hyp,
            )

    return __finetuner


class TestDataConfig:
    """Tests data config validation."""

    def test_data_config_text(self, data_config_text):
        """Tests data config for the case it contains label and text."""
        assert is_dataclass(data_config_text)
        assert data_config_text.label_column == "label"
        assert data_config_text.text_column == "text"

    def test_data_config_hyp_prem(self, data_config_pre_hyp):
        """Tests data config when it contains label hypothesis, and premise."""
        assert is_dataclass(data_config_pre_hyp)
        assert data_config_pre_hyp.label_column == "label"
        assert data_config_pre_hyp.hypothesis_column == "hypothesis"
        assert data_config_pre_hyp.premise_column == "premise"

    def test_empty_data_config(self):
        """Tests empty dataconfig."""
        with pytest.raises(ValueError):
            DataConfig()

    def test_label_data_config(self):
        """Tests dataconfig that only contains label column."""
        with pytest.raises(ValueError):
            DataConfig(label_column="label")

    def test_no_hyp_data_config(self):
        """Tests dataconfig that contains only label and premises column."""
        with pytest.raises(ValueError):
            DataConfig(label_column="label", premise_column="premise")

    def test_no_prem_data_config(self):
        """Tests dataconfig that contains only label and hypothesis column."""
        with pytest.raises(ValueError):
            DataConfig(label_column="label", hypothesis_column="hypothesis")


def test_finetuner_init(def_finetuner):
    """Tests the FineTuner class initialises correctly.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft and evaluation.
    """
    finetuner = def_finetuner(peft=True)
    assert finetuner.model is not None
    assert finetuner.tokenizer is not None
    assert finetuner.data_collator is not None
    assert hasattr(finetuner.model, "peft_config")


def test_tokenize_nli_task(def_finetuner):
    """Tests the tokenize_nli_task function runs as desired.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft and evaluation.
    """
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "label": [1, 0],
                "premise": [
                    "This is a Positive text",
                    "This is a Negative text",
                ],
                "hypothesis": ["Positive text", "Positive text"],
            }
        )
    )
    tokenized_data = def_finetuner(peft=False).tokenize_nli_task(dataset)
    assert len(tokenized_data) == 2
    assert "label" in tokenized_data.features
    assert "hypothesis" in tokenized_data.features
    assert "premise" in tokenized_data.features
    assert "input_ids" in tokenized_data.features
    assert "attention_mask" in tokenized_data.features


def test_tokenize_single_text(def_finetuner):
    """Tests the tokenize_single_text function runs as desired.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft and evaluation.
    """
    dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "text": ["This is a Positive text", "This is a Negative text"],
                "label": [1, 0],
            }
        )
    )
    tokenized_data = def_finetuner(peft=True).tokenize_single_text(dataset)
    assert len(tokenized_data) == 2
    assert "text" in tokenized_data.features
    assert "label" in tokenized_data.features
    assert "input_ids" in tokenized_data.features
    assert "attention_mask" in tokenized_data.features


def test_create_trainer_for_search(def_finetuner):
    """Tests the create_trainer_for_search function creates trainer correctly.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft.
    """
    train_df = pd.DataFrame(
        {
            "text": ["This is a Positive text", "This is a Negative text"],
            "label": [1, 0],
        }
    )
    eval_df = pd.DataFrame(
        {
            "text": ["This is a Positive text", "This is a Negative text"],
            "label": [1, 0],
        }
    )

    trainer = def_finetuner(peft=True).create_trainer_for_search(
        train_df, eval_df
    )

    assert trainer.model_init is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None


def test_create_trainer_regular(def_finetuner):
    """Tests the regular create_trainer function works as before.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft and evaluation.
    """
    finetuner = def_finetuner(peft=True)

    train_dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "premise": ["This is premise"],
                "hypothesis": ["This is hypothesis"],
                "label": [1],
            }
        )
    )

    eval_dataset = Dataset.from_pandas(
        pd.DataFrame(
            {
                "premise": ["This is eval premise"],
                "hypothesis": ["This is eval hypothesis"],
                "label": [0],
            }
        )
    )

    trainer = finetuner.create_trainer(train_dataset, eval_dataset)

    assert trainer.model is not None
    assert trainer.model_init is None


def test_finetune_trainer_without_eval(def_finetuner):
    """Tests whether finetuner works without evaluation.

    Parameters
    ----------
    def_finetuner: FineTuner
        Initialisation of FineTuner class with/without peft and evaluation.
    """
    finetuner = def_finetuner(do_eval=False)

    train_df = pd.DataFrame(
        {
            "text": ["This is a Positive text", "This is a Negative text"],
            "label": [1, 0],
        }
    )

    trainer = finetuner.fine_tune_model(train_df)

    assert trainer.model is not None
    assert trainer.model_init is None
    with pytest.raises(ValueError) as excinfo:
        trainer = def_finetuner(do_eval=True).fine_tune_model(train_df)
    assert "eval_strategy='no' in training_args if eval_df provided" in str(
        excinfo.value
    )
