import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(
        """
        # FineTuner How-To Guide

        This notebook demonstrates how to use the `FineTuner` class for fine-tuning transformer models with optional hyperparameter search.
        """
    )
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Setup

        First, we'll import the necessary components and load a sample dataset (Rotten Tomatoes movie reviews).
        """
    )
    return


@app.cell
def _():
    from datasets import load_dataset

    from laurium.encoder_models.fine_tune import DataConfig, FineTuner

    return DataConfig, FineTuner, load_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Configuration

        Configure the model, tokenizer, training arguments, and data settings. The `FineTuner` class requires:

        - **metrics**: List of evaluation metrics to compute
        - **model_init**: Model configuration (name, number of labels, etc.)
        - **training_args**: Training hyperparameters and settings
        - **tokenizer_init**: Tokenizer configuration
        - **tokenizer_args**: Text processing settings (max length, padding, etc.)
        - **data_config**: Column mapping for your dataset
        """
    )
    return


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
        metrics=["f1", "accuracy", "precision", "recall"],
        model_init=classifier_model_init,
        training_args=classifier_training_args,
        tokenizer_init=classifier_tokenizer_init,
        tokenizer_args=classifier_tokenizer_args,
        data_config=classifier_data_config,
    )

    # Prepare data and splits
    classifier_tomatoes = load_dataset("rotten_tomatoes")
    classifier_train_df = classifier_tomatoes["train"].to_pandas()
    classifier_test_df = classifier_tomatoes["test"].to_pandas()

    return (classifier_fine_tuner, classifier_train_df, classifier_test_df)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Example 1: Regular Fine-tuning

        The simplest approach - fine-tune with your predefined hyperparameters using `fine_tune_model()`.
        """
    )
    return


@app.cell
def _(classifier_fine_tuner, classifier_train_df, classifier_test_df):
    # Example 1: Regular fine-tuning
    print("=== Regular Fine-tuning ===")

    # Use a small subset for demonstration
    small_train_df = classifier_train_df.sample(n=100, random_state=42)
    small_test_df = classifier_test_df.sample(n=50, random_state=42)

    # Fine-tune the model
    trainer = classifier_fine_tuner.fine_tune_model(
        train_df=small_train_df, eval_df=small_test_df
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Regular fine-tuning results: {results}")

    return trainer, small_train_df, small_test_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Example 2: Hyperparameter Search

        For automated hyperparameter optimization, use `create_trainer_for_search()` to get a trainer configured for search, then call `hyperparameter_search()` with:
        - **hp_space**: Function defining parameter ranges using Optuna's trial API
        - **n_trials**: Number of optimization trials to run
        - **direction**: "minimize" or "maximize" the objective
        - **compute_objective**: Function to extract the metric to optimize from evaluation results
        """
    )
    return


@app.cell
def _(classifier_fine_tuner, small_train_df, small_test_df):
    # Example 2: Hyperparameter search
    print("\n=== Hyperparameter Search ===")

    # Create a trainer configured for hyperparameter search
    search_trainer = classifier_fine_tuner.create_trainer_for_search(
        train_df=small_train_df, eval_df=small_test_df
    )

    # Define hyperparameter space
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-5, 5e-5, log=True
            ),
            "per_device_train_batch_size": trial.suggest_categorical(
                "batch_size", [8, 16]
            ),
            "num_train_epochs": trial.suggest_int("epochs", 2, 4),
        }

    # Run hyperparameter search
    best_trial = search_trainer.hyperparameter_search(
        hp_space=hp_space,
        n_trials=3,  # Small number for demo
        direction="maximize",
        compute_objective=lambda metrics: metrics.get("eval_f1", 0),
    )

    print(f"Best hyperparameters: {best_trial.hyperparameters}")
    print(f"Best objective value: {best_trial.objective}")

    return search_trainer, best_trial


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## Summary

        The `FineTuner` class provides two main workflows:

        1. **Regular fine-tuning**: `fine_tune_model()` - Use your predefined hyperparameters
        2. **Hyperparameter search**: `create_trainer_for_search()` + `hyperparameter_search()` - Let Optuna find optimal hyperparameters

        Both approaches use the same model and tokenizer instances, making it easy to compare results and switch between methods.
        """
    )
    return


if __name__ == "__main__":
    app.run()
