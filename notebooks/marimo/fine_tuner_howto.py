import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    from datasets import load_dataset

    from laurium.encoder_models.fine_tune import DataConfig, FineTuner

    return DataConfig, FineTuner, load_dataset


@app.cell
def _(DataConfig, FineTuner, load_dataset):
    # Model configuration
    classifier_model_init = {
        "pretrained_model_name_or_path": "bert-base-cased",
        "num_labels": 2,
        "local_files_only": False,
    }

    classifier_tokenizer_init = {
        "pretrained_model_name_or_path": "bert-base-cased",
        "use_fast": True,
    }
    # Tokenizer configuration
    classifier_tokenizer_args = {
        "max_length": 128,
        "return_tensors": "pt",
        "padding": "max_length",
        "truncation": "longest_first",
    }
    # Training arguments for parameter tuning
    classifier_training_args = {
        "output_dir": "./results",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": 5,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "report_to": "none",
        "eval_strategy": "epoch",
    }

    # Data configuration
    classifier_data_config = DataConfig(
        text_column="text", label_column="label"
    )
    # Initialize fine-tuner
    classifier_fine_tuner = FineTuner(
        model_init=classifier_model_init,
        training_args=classifier_training_args,
        tokenizer_init=classifier_tokenizer_init,
        tokenizer_args=classifier_tokenizer_args,
        data_config=classifier_data_config,
    )

    # Prepare data and splits
    classifier_tomatoes = load_dataset("rotten_tomatoes")
    classifier_train_data, classifier_test_data = (
        classifier_tomatoes["train"],
        classifier_tomatoes["test"],
    )

    # Tokenize data
    classifier_tokenized_train = classifier_fine_tuner.tokenize_single_text(
        classifier_train_data
    )
    classifier_tokenized_test = classifier_fine_tuner.tokenize_single_text(
        classifier_test_data
    )

    # Create trainer and evaluate
    classifier_bert_trainer = classifier_fine_tuner.create_trainer(
        ["f1", "accuracy", "precision", "recall"],
        classifier_tokenized_train,
        classifier_tokenized_test,
    )
    print(classifier_bert_trainer.evaluate())
    return


if __name__ == "__main__":
    app.run()
